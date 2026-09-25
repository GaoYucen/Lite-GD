#!/usr/bin/env python3
"""Clean learning baselines for the recovered Lite-GD carpool benchmark.

Model backbones are adapted from the verified GroupOpt/ASCC implementations in
GaoYucen/Ptr-net-GT (branch ctqs-phase-c2-20260921, commit
b8d2fd49a27abb3ea53da18ef696f49d58195dbd):
  - src/groupopt/models/ptrnet.py
  - src/groupopt/models/am.py
  - src/groupopt/objectives/reinforce.py

Only the problem layer is changed: driver-fixed multi-candidate carpooling with
pickup-before-own-dropoff precedence and directed road-network route costs.
PointerNet keeps supervised exact-sequence training; AM keeps REINFORCE.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from historical_full_model import HistoricalExact
from crossgraph_data import CrossGraphExact
from evaluation import summarize_route_rows


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CarpoolCases:
    def __init__(self, data: HistoricalExact, train_ids):
        self.data = data
        self.cache = {}
        pts = []
        get_case = data.case_light if hasattr(data, "case_light") else data.case_tensors
        for cid in train_ids:
            pts.append(get_case(cid)["points"])
        cat = np.concatenate(pts, axis=0).astype(np.float64)
        self.coord_mean = cat.mean(axis=0).astype(np.float32)
        self.coord_std = np.maximum(cat.std(axis=0), 1e-6).astype(np.float32)

    def item(self, cid: int) -> dict:
        cid = int(cid)
        if cid in self.cache:
            return self.cache[cid]
        get_case = self.data.case_light if hasattr(self.data, "case_light") else self.data.case_tensors
        x = get_case(cid)
        flat = x["flat"]
        n = len(flat)
        cost = np.zeros((n, n), dtype=np.float64)
        for i, (ea, ra, _, _) in enumerate(flat):
            for j, (eb, rb, _, _) in enumerate(flat):
                if i != j:
                    cost[i, j] = self.data.point_dist(ea, ra, eb, rb)
        out = {
            "cid": cid,
            "coords": ((x["points"] - self.coord_mean) / self.coord_std).astype(np.float32),
            "event": np.asarray([z[2] for z in flat], dtype=np.int64),
            "target": np.asarray(x["target"], dtype=np.int64),
            "n_events": int(x["n_events"]),
            "opt": float(x["opt"]),
            "cost": cost,
            "passengers": int(x["n_events"] // 2),
        }
        self.cache[cid] = out
        return out

    def collate(self, ids, device: torch.device) -> dict:
        rows = [self.item(i) for i in ids]
        ns = {len(x["coords"]) for x in rows}
        if len(ns) != 1:
            raise ValueError("batches must be passenger-count homogeneous")
        return {
            "cid": [x["cid"] for x in rows],
            "coords": torch.tensor(np.stack([x["coords"] for x in rows]), device=device),
            "event": torch.tensor(np.stack([x["event"] for x in rows]), device=device),
            "target": torch.tensor(np.stack([x["target"] for x in rows]), device=device),
            "n_events": torch.tensor([x["n_events"] for x in rows], device=device),
            "opt": torch.tensor([x["opt"] for x in rows], dtype=torch.float64, device=device),
            "cost": torch.tensor(np.stack([x["cost"] for x in rows]), dtype=torch.float64, device=device),
            "passengers": rows[0]["passengers"],
        }


def make_batches(ids, data: HistoricalExact, batch_size: int, *, seed: int, epoch: int, shuffle: bool):
    rng = np.random.default_rng(seed + 1009 * epoch)
    batches = []

    # Historical cases can have a variable number of retained candidates even
    # for the same passenger count. Keep every real candidate and bucket by
    # both semantic length and candidate count instead of padding fake points.
    def signature(cid):
        if hasattr(data, "case_light"):
            x = data.case_light(int(cid))
            return (int(x["n_events"] // 2), int(len(x["flat"])))
        case = data.case_by_id[int(cid)]
        n_points = 1 + sum(len(g) for g in case.candidate_groups)
        return (int(case.passenger_count), int(n_points))

    groups = {}
    for cid in ids:
        groups.setdefault(signature(cid), []).append(int(cid))

    for sig in sorted(groups):
        z = np.asarray(groups[sig], dtype=np.int64)
        if shuffle:
            z = rng.permutation(z)
        for s in range(0, len(z), batch_size):
            batches.append(z[s:s + batch_size].tolist())
    if shuffle and len(batches) > 1:
        rng.shuffle(batches)
    return batches


def legal_mask(event: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """True means illegal. Driver has event=-1 and is always masked."""
    bsz, n = event.shape
    bad = event < 0
    max_events = done.size(1)
    for e in range(max_events):
        em = event == e
        illegal = done[:, e][:, None]
        if e % 2 == 1:
            illegal = illegal | (~done[:, e - 1][:, None])
        bad = bad | (em & illegal)
    return bad


def update_done(done: torch.Tensor, event: torch.Tensor, chosen: torch.Tensor) -> None:
    ev = event.gather(1, chosen[:, None]).squeeze(1)
    done.scatter_(1, ev[:, None], True)


def route_cost(cost: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Cost from fixed driver index 0 through selected candidate indices."""
    bsz = seq.size(0)
    prev = torch.zeros(bsz, dtype=torch.long, device=seq.device)
    total = cost.new_zeros(bsz)
    ar = torch.arange(bsz, device=seq.device)
    for t in range(seq.size(1)):
        cur = seq[:, t]
        total = total + cost[ar, prev, cur]
        prev = cur
    return total


class PointerAttention(nn.Module):
    def __init__(self, dim: int, tanh_clipping: float = 10.0):
        super().__init__()
        self.node = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.v = nn.Parameter(torch.empty(dim))
        self.tanh_clipping = tanh_clipping
        nn.init.uniform_(self.v, -1.0 / math.sqrt(dim), 1.0 / math.sqrt(dim))

    def forward(self, query, nodes, bad):
        z = torch.tanh(self.node(nodes) + self.query(query)[:, None, :])
        logits = torch.matmul(z, self.v)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        return logits.masked_fill(bad, -1e9)


class CarpoolPointerNet(nn.Module):
    """ASCC clean PointerNet backbone with a carpool precedence decoder."""

    def __init__(self, dim=128, layers=1):
        super().__init__()
        self.dim = dim
        self.input_projection = nn.Linear(2, dim)
        self.encoder = nn.LSTM(dim, dim, num_layers=layers, batch_first=True)
        self.decoder_cell = nn.LSTMCell(dim, dim)
        self.project_context = nn.Linear(3 * dim, dim)
        self.pointer = PointerAttention(dim)

    def forward(self, coords, event, n_events, *, teacher=None, sample=False, generator=None):
        emb_in = self.input_projection(coords)
        nodes, (h, c) = self.encoder(emb_in)
        h = h[-1]
        c = c[-1]
        graph = nodes.mean(dim=1)
        current = nodes[:, 0]
        max_events = int(n_events.max().item())
        done = torch.zeros(coords.size(0), max_events, dtype=torch.bool, device=coords.device)
        seq, logps, losses = [], [], []
        for t in range(max_events):
            q = self.project_context(torch.cat([graph, h, current], dim=-1))
            bad = legal_mask(event, done)
            logits = self.pointer(q, nodes, bad)
            logp = torch.log_softmax(logits, dim=-1)
            if teacher is not None:
                chosen = teacher[:, t]
                losses.append(nn.functional.cross_entropy(logits, chosen))
            elif sample:
                chosen = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
            else:
                chosen = logits.argmax(dim=1)
            logps.append(logp.gather(1, chosen[:, None]).squeeze(1))
            seq.append(chosen)
            update_done(done, event, chosen)
            current = nodes.gather(1, chosen[:, None, None].expand(-1, 1, self.dim)).squeeze(1)
            h, c = self.decoder_cell(current, (h, c))
        seq = torch.stack(seq, dim=1)
        ll = torch.stack(logps, dim=1).sum(dim=1)
        loss = torch.stack(losses).mean() if losses else None
        return seq, ll, loss


class Normalization(nn.Module):
    def __init__(self, dim: int, kind: str):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim) if kind == "batch" else nn.LayerNorm(dim)

    def forward(self, x):
        if isinstance(self.norm, nn.BatchNorm1d):
            b, n, d = x.shape
            return self.norm(x.reshape(b * n, d)).reshape(b, n, d)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim=128, heads=8, ff=512, normalization="batch"):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=True)
        self.n1 = Normalization(dim, normalization)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
        self.n2 = Normalization(dim, normalization)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class GraphAttentionEncoder(nn.Module):
    """Same graph-self-attention structure used by the verified ASCC AM host."""

    def __init__(self, input_dim=2, dim=128, heads=8, layers=3, ff=512, normalization="batch"):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([EncoderLayer(dim, heads, ff, normalization) for _ in range(layers)])

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x, x.mean(dim=1)


class CarpoolAM(nn.Module):
    """Kool/ASCC-style Attention Model with the shared carpool problem mask."""

    def __init__(self, dim=128, heads=8, layers=3, ff=512, normalization="batch", tanh_clipping=10.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.tanh_clipping = tanh_clipping
        self.encoder = GraphAttentionEncoder(2, dim, heads, layers, ff, normalization)
        self.project_graph = nn.Linear(dim, dim, bias=False)
        self.project_nodes = nn.Linear(dim, 3 * dim, bias=False)
        self.project_step = nn.Linear(2 * dim, dim, bias=False)
        self.project_out = nn.Linear(dim, dim, bias=False)

    def _split(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, coords, event, n_events, *, teacher=None, sample=False, generator=None):
        nodes, graph = self.encoder(coords)
        graph_q = self.project_graph(graph)
        k, v, logit_k = self.project_nodes(nodes).chunk(3, dim=-1)
        k = self._split(k)
        v = self._split(v)
        first = nodes[:, 0]
        current = first
        max_events = int(n_events.max().item())
        done = torch.zeros(coords.size(0), max_events, dtype=torch.bool, device=coords.device)
        seq, logps, losses = [], [], []
        scale_h = math.sqrt(self.head_dim)
        scale_d = math.sqrt(self.dim)
        for t in range(max_events):
            q = graph_q + self.project_step(torch.cat([first, current], dim=-1))
            qh = q.view(q.size(0), self.heads, 1, self.head_dim)
            bad = legal_mask(event, done)
            compat = torch.matmul(qh, k.transpose(-2, -1)).squeeze(2) / scale_h
            compat = compat.masked_fill(bad[:, None, :], -1e9)
            attn = torch.softmax(compat, dim=-1)
            glimpse = torch.matmul(attn.unsqueeze(2), v).squeeze(2)
            glimpse = self.project_out(glimpse.reshape(q.size(0), self.dim))
            logits = torch.einsum("bd,bnd->bn", glimpse, logit_k) / scale_d
            logits = torch.tanh(logits) * self.tanh_clipping
            logits = logits.masked_fill(bad, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            if teacher is not None:
                chosen = teacher[:, t]
                losses.append(nn.functional.cross_entropy(logits, chosen))
            elif sample:
                chosen = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
            else:
                chosen = logits.argmax(dim=1)
            logps.append(logp.gather(1, chosen[:, None]).squeeze(1))
            seq.append(chosen)
            update_done(done, event, chosen)
            current = nodes.gather(1, chosen[:, None, None].expand(-1, 1, self.dim)).squeeze(1)
        seq = torch.stack(seq, dim=1)
        ll = torch.stack(logps, dim=1).sum(dim=1)
        loss = torch.stack(losses).mean() if losses else None
        return seq, ll, loss


def evaluate(model, cases: CarpoolCases, ids, data: HistoricalExact, device, batch_size=64):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_ids in make_batches(ids, data, batch_size, seed=0, epoch=0, shuffle=False):
            b = cases.collate(batch_ids, device)
            seq, _, _ = model(b["coords"], b["event"], b["n_events"], teacher=None, sample=False)
            pred_cost = route_cost(b["cost"], seq)
            pred_event = b["event"].gather(1, seq)
            true_event = b["event"].gather(1, b["target"])
            for i in range(len(batch_ids)):
                t = int(b["n_events"][i].item())
                pred = seq[i, :t]
                true = b["target"][i, :t]
                pe = pred_event[i, :t]
                te = true_event[i, :t]
                rows.append({
                    "pred": float(pred_cost[i].item()),
                    "opt": float(b["opt"][i].item()),
                    "exact": int(torch.equal(pred, true)),
                    "pointer_hits": int((pred == true).sum().item()),
                    "steps": t,
                    "event_exact": int(torch.equal(pe, te)),
                    "event_hits": int((pe == te).sum().item()),
                    "candidate_hits": int(sum(
                        int(pred[pe == e][0].item() == true[te == e][0].item())
                        for e in torch.unique(te)
                    )),
                })
    return summarize_route_rows(rows)


def train_pointer(args, data, cases, tr, va, te, device):
    # A6000's system cuDNN is older than the PyTorch build. Runtime speed is
    # not a primary baseline metric here, so use PyTorch's native CUDA LSTM
    # kernels without changing the model or optimization protocol.
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    model = CarpoolPointerNet(args.dim, args.ptr_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids in make_batches(tr, data, args.batch, seed=args.seed, epoch=epoch, shuffle=True):
            b = cases.collate(ids, device)
            _, _, loss = model(b["coords"], b["event"], b["n_events"], teacher=b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate(model, cases, va, data, device, args.eval_batch)
            score = float(val["mean_case_gap"])
            history.append({"epoch": epoch, "train_ce": float(np.mean(losses)), "val": val})
            print("PTR_VAL", json.dumps(history[-1], sort_keys=True), flush=True)
            if score < best - 1e-6:
                best = score
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += args.eval_every
            if bad_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def train_am(args, data, cases, tr, va, te, device):
    model = CarpoolAM(args.dim, args.heads, args.am_layers, args.ff, "batch").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    gen = torch.Generator(device=device).manual_seed(args.seed + 3001)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, costs = [], []
        for ids in make_batches(tr, data, args.batch, seed=args.seed, epoch=epoch, shuffle=True):
            b = cases.collate(ids, device)
            seq, ll, _ = model(b["coords"], b["event"], b["n_events"], sample=True, generator=gen)
            cost = route_cost(b["cost"], seq)
            advantage = (cost - cost.mean()).detach()
            loss = (advantage * ll).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            costs.append(float(cost.mean().detach()))
        if epoch == 1 or epoch % args.eval_every == 0:
            val = evaluate(model, cases, va, data, device, args.eval_batch)
            score = float(val["mean_case_gap"])
            history.append({
                "epoch": epoch,
                "train_reinforce": float(np.mean(losses)),
                "train_route_cost": float(np.mean(costs)),
                "val": val,
            })
            print("AM_VAL", json.dumps(history[-1], sort_keys=True), flush=True)
            if score < best - 1e-6:
                best = score
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += args.eval_every
            if bad_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def greedy_evaluate(cases: CarpoolCases, ids, data: HistoricalExact):
    rows = []
    for cid in ids:
        x = cases.item(cid)
        event = x["event"]
        cost = x["cost"]
        done = set()
        current = 0
        seq = []
        for _ in range(x["n_events"]):
            legal_events = [e for e in range(x["n_events"]) if e not in done and (e % 2 == 0 or e - 1 in done)]
            candidates = [j for j in range(1, len(event)) if int(event[j]) in legal_events]
            chosen = min(candidates, key=lambda j: (float(cost[current, j]), int(event[j]), j))
            seq.append(chosen)
            done.add(int(event[chosen]))
            current = chosen
        pred = float(sum(cost[a, z] for a, z in zip([0] + seq[:-1], seq)))
        target = x["target"].tolist()
        pe = [int(event[j]) for j in seq]
        te = [int(event[j]) for j in target]
        pm = {int(event[j]): j for j in seq}
        tm = {int(event[j]): j for j in target}
        rows.append({
            "pred": pred,
            "opt": x["opt"],
            "exact": int(seq == target),
            "pointer_hits": sum(a == z for a, z in zip(seq, target)),
            "steps": x["n_events"],
            "event_exact": int(pe == te),
            "event_hits": sum(a == z for a, z in zip(pe, te)),
            "candidate_hits": sum(pm[e] == tm[e] for e in tm),
        })
    return summarize_route_rows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path)
    ap.add_argument("--orders", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--exact", type=Path)
    ap.add_argument("--benchmark-dir", type=Path)
    ap.add_argument("--apsp", type=Path)
    ap.add_argument("--model", choices=["ptrnet", "am", "greedy"], required=True)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ptr-layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--am-layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    seed_all(args.seed)
    if args.benchmark_dir is not None:
        if args.apsp is None:
            raise ValueError("--apsp is required with --benchmark-dir")
        data = CrossGraphExact(args.benchmark_dir, args.apsp, require_routes=False)
    else:
        if any(x is None for x in (args.links,args.orders,args.labels,args.exact)):
            raise ValueError("historical mode requires --links --orders --labels --exact")
        data = HistoricalExact(args.links, args.orders, args.labels, args.exact)
    tr, va, te = data.split(args.split_seed)
    cases = CarpoolCases(data, tr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "greedy":
        result = {
            "model": "DisGreedy-corrected",
            "split_seed": args.split_seed,
            "test": greedy_evaluate(cases, te, data),
        }
    else:
        trainer = train_pointer if args.model == "ptrnet" else train_am
        model, history = trainer(args, data, cases, tr, va, te, device)
        test = evaluate(model, cases, te, data, device, args.eval_batch)
        result = {
            "model": args.model,
            "source": "ASCC clean backbone adapted to shared Lite-GD carpool problem layer",
            "source_commit": "b8d2fd49a27abb3ea53da18ef696f49d58195dbd",
            "split_seed": args.split_seed,
            "training_seed": args.seed,
            "train_cases": len(tr),
            "validation_cases": len(va),
            "test_cases": len(te),
            "history": history,
            "test": test,
        }
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "split_seed": args.split_seed,
                "training_seed": args.seed,
            }, args.checkpoint)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("BASELINE_RESULT", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
