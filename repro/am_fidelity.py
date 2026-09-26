#!/usr/bin/env python3
"""Higher-fidelity AM training for Lite-GD baseline reproduction.

The original Kool et al. Attention Model trains with REINFORCE and a rollout
baseline (with exponential warmup).  The earlier Lite-GD cross-graph baseline
used only a batch-mean cost baseline, which has much higher variance on the
finite MCRP benchmark.  This module restores rollout/self-critical baselines
without changing the AM architecture, legality mask, split, or evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from crossgraph_data import CrossGraphExact
from baseline_carpool import (
    CarpoolCases, CarpoolAM, make_batches, route_cost, evaluate, seed_all
)


def _greedy_cost(model, b):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        seq, _, _ = model(
            b["coords"], b["event"], b["n_events"],
            teacher=None, sample=False
        )
        c = route_cost(b["cost"], seq)
    if was_training:
        model.train()
    return c


def _clone_eval(model):
    z = copy.deepcopy(model)
    z.eval()
    for p in z.parameters():
        p.requires_grad_(False)
    return z


def train(args, data, device):
    seed_all(args.seed)
    tr, va, te = data.split(args.split_seed)
    cases = CarpoolCases(data, tr)

    model = CarpoolAM(
        args.dim, args.heads, args.am_layers, args.ff, "batch"
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed + 3001)

    baseline_model = _clone_eval(model)
    baseline_val = float("inf")
    ema = None
    best = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses, sample_costs, advantages = [], [], []

        for ids in make_batches(
            tr, data, args.batch, seed=args.seed, epoch=ep, shuffle=True
        ):
            b = cases.collate(ids, device)

            # Baseline is computed before the policy update for this batch.
            if args.baseline == "self_critical":
                bl = _greedy_cost(model, b).detach()
            elif args.baseline in ("rollout_copy", "rollout_copy_norm"):
                if ep <= args.warmup_epochs:
                    bl = None  # exponential scalar warmup below
                else:
                    bl = _greedy_cost(baseline_model, b).detach()
            else:
                raise ValueError(args.baseline)

            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.rollouts_per_batch):
                seq, ll, _ = model(
                    b["coords"], b["event"], b["n_events"],
                    sample=True, generator=gen
                )
                cost = route_cost(b["cost"], seq)

                if bl is None:
                    cur_mean = cost.detach().mean()
                    if ema is None:
                        ema = cur_mean
                    else:
                        ema = args.exp_beta * ema + (1.0 - args.exp_beta) * cur_mean
                    bl_use = ema
                else:
                    bl_use = bl

                adv = (cost - bl_use).detach()
                if args.baseline == "rollout_copy_norm":
                    denom = torch.as_tensor(bl_use, device=cost.device, dtype=cost.dtype).abs().clamp_min(1.0)
                    adv = adv / denom

                loss = (adv * ll).mean() / args.rollouts_per_batch
                loss.backward()
                step_loss += float(loss.detach())
                sample_costs.append(float(cost.detach().mean()))
                advantages.append(float(adv.abs().mean()))

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            losses.append(step_loss)

        # Validation is cheap (100 pilot cases), so evaluate every epoch for a
        # faithful rollout baseline update and keep a coarser history printout.
        vm = evaluate(model, cases, va, data, device, args.eval_batch)
        vscore = float(vm["mean_case_gap"])

        # Rollout baseline update: after warmup, keep a frozen copy of the
        # strongest validation policy so the control variate does not chase the
        # current stochastic actor within a batch.
        baseline_updated = False
        if args.baseline.startswith("rollout_copy"):
            if ep == args.warmup_epochs or vscore < baseline_val - args.baseline_update_eps:
                baseline_model = _clone_eval(model)
                baseline_val = vscore
                baseline_updated = True

        if vscore < best - 1e-6:
            best = vscore
            best_epoch = ep
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if ep == 1 or ep % args.eval_every == 0 or baseline_updated:
            rec = {
                "epoch": ep,
                "train_reinforce": float(np.mean(losses)),
                "sample_route_cost": float(np.mean(sample_costs)),
                "mean_abs_advantage": float(np.mean(advantages)),
                "val": vm,
                "baseline_updated": baseline_updated,
                "baseline_val_mean_case_gap": baseline_val if np.isfinite(baseline_val) else None,
            }
            history.append(rec)
            print("AM_FIDELITY_VAL", json.dumps(rec, sort_keys=True), flush=True)

        if bad_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("no AM checkpoint selected")
    model.load_state_dict(best_state)
    test = evaluate(model, cases, te, data, device, args.eval_batch)

    result = {
        "model": "am",
        "training_protocol": args.baseline,
        "rollouts_per_batch": args.rollouts_per_batch,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "stop_epoch": ep,
        "val_mean_case_gap": best,
        "history": history,
        "test": test,
    }
    return model, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--apsp", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path)

    ap.add_argument("--baseline", choices=[
        "self_critical", "rollout_copy", "rollout_copy_norm"
    ], default="rollout_copy")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--exp-beta", type=float, default=0.8)
    ap.add_argument("--baseline-update-eps", type=float, default=1e-3)
    ap.add_argument("--rollouts-per-batch", type=int, default=4)

    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--am-layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args()

    data = CrossGraphExact(args.benchmark_dir, args.apsp, require_routes=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, result = train(args, data, device)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": "am",
            "state_dict": model.state_dict(),
            "protocol": args.baseline,
            "seed": args.seed,
            "split_seed": args.split_seed,
        }, args.checkpoint)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("AM_FIDELITY_RESULT", json.dumps(result["test"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
