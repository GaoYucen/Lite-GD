#!/usr/bin/env python3
"""PointerNet / AM / Greedy baselines for generated cross-graph benchmarks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline_carpool import CarpoolPointerNet, CarpoolAM, route_cost, seed_all
from evaluation import summarize_route_rows


class CrossgraphCases:
    def __init__(self, graph: Path, matrix: Path, benchmark: Path):
        g = np.load(graph, mmap_mode="r")
        self.src = np.asarray(g["src"], dtype=np.int32)
        self.dst = np.asarray(g["dst"], dtype=np.int32)
        self.weight = np.asarray(g["weight"], dtype=np.float64)
        self.xy = np.asarray(g["coordinates"], dtype=np.float64)
        self.D = np.load(matrix, mmap_mode="r")

        z = np.load(benchmark, mmap_mode="r")
        self.ptr = np.asarray(z["case_ptr"], dtype=np.int64)
        self.edge = np.asarray(z["edge_idx"], dtype=np.int32)
        self.ratio = np.asarray(z["ratio"], dtype=np.float32)
        self.event = np.asarray(z["event"], dtype=np.int8)
        self.local = np.asarray(z["local_idx"], dtype=np.int8)
        self.target = np.asarray(z["target"], dtype=np.int16)
        self.opt = np.asarray(z["opt_length"], dtype=np.float64)
        self.split = np.asarray(z["split"], dtype=np.int8)
        self.n = len(self.opt)
        self.cache = {}

        train_ids = np.flatnonzero(self.split == 0)
        pts = []
        # Coordinate normalization is computed from training candidates only.
        for cid in train_ids:
            a, b = int(self.ptr[cid]), int(self.ptr[cid+1])
            e = self.edge[a:b].astype(np.int64)
            r = self.ratio[a:b].astype(np.float64)
            p = self.xy[self.src[e]] * (1-r[:, None]) + self.xy[self.dst[e]] * r[:, None]
            pts.append(p)
        cat = np.concatenate(pts, axis=0)
        self.coord_mean = cat.mean(0)
        self.coord_std = np.maximum(cat.std(0), 1e-6)

    def ids(self, split):
        code = {"train": 0, "validation": 1, "test": 2}[split]
        return np.flatnonzero(self.split == code).astype(np.int64).tolist()

    def item(self, cid):
        cid = int(cid)
        if cid in self.cache:
            return self.cache[cid]
        a, b = int(self.ptr[cid]), int(self.ptr[cid+1])
        e = self.edge[a:b].astype(np.int64)
        r = self.ratio[a:b].astype(np.float64)
        ev = self.event[a:b].astype(np.int64)
        pts = self.xy[self.src[e]] * (1-r[:, None]) + self.xy[self.dst[e]] * r[:, None]
        pts = ((pts - self.coord_mean) / self.coord_std).astype(np.float32)

        base = ((1-r) * self.weight[e])[:, None]
        tail = (r * self.weight[e])[None, :]
        core = self.D[self.dst[e][:, None], self.src[e][None, :]]
        C = base + core + tail
        same = e[:, None] == e[None, :]
        forward = r[None, :] >= r[:, None]
        direct = (r[None, :] - r[:, None]) * self.weight[e][:, None]
        C = np.where(same & forward, np.minimum(C, direct), C).astype(np.float64)

        out = {
            "cid": cid, "coords": pts, "event": ev,
            "target": self.target[cid].astype(np.int64),
            "n_events": int(self.target.shape[1]),
            "opt": float(self.opt[cid]), "cost": C,
        }
        self.cache[cid] = out
        return out

    def collate(self, ids, device):
        rows = [self.item(i) for i in ids]
        n = {len(x["coords"]) for x in rows}
        if len(n) != 1:
            # Variable candidate counts imply variable flattened length; pad.
            N = max(n)
        else:
            N = next(iter(n))
        B = len(rows)
        coords = torch.zeros(B, N, 2, dtype=torch.float32, device=device)
        event = torch.full((B, N), -2, dtype=torch.long, device=device)
        valid = torch.zeros(B, N, dtype=torch.bool, device=device)
        cost = torch.zeros(B, N, N, dtype=torch.float64, device=device)
        target = torch.stack([torch.tensor(x["target"], dtype=torch.long) for x in rows]).to(device)
        for i, x in enumerate(rows):
            m = len(x["coords"])
            coords[i, :m] = torch.tensor(x["coords"], dtype=torch.float32, device=device)
            event[i, :m] = torch.tensor(x["event"], dtype=torch.long, device=device)
            valid[i, :m] = True
            cost[i, :m, :m] = torch.tensor(x["cost"], dtype=torch.float64, device=device)
        # Models use event<0 to mask driver/padding, so -2 padding is naturally illegal.
        return {
            "cid": [x["cid"] for x in rows],
            "coords": coords, "event": event, "valid": valid, "cost": cost,
            "target": target,
            "n_events": torch.full((B,), target.shape[1], dtype=torch.long, device=device),
            "opt": torch.tensor([x["opt"] for x in rows], dtype=torch.float64, device=device),
        }


def make_batches(ids, data, batch, seed, epoch, shuffle):
    # Group by the true flattened candidate count.  This keeps both the LSTM
    # encoder and AM self-attention free of synthetic padding tokens.
    rng = np.random.default_rng(seed + 1009*epoch)
    buckets = {}
    for cid in ids:
        n = int(data.ptr[int(cid)+1] - data.ptr[int(cid)])
        buckets.setdefault(n, []).append(int(cid))
    out = []
    for n in sorted(buckets):
        x = np.asarray(buckets[n], dtype=np.int64)
        if shuffle:
            x = rng.permutation(x)
        for i in range(0, len(x), batch):
            out.append(x[i:i+batch].tolist())
    if shuffle and len(out) > 1:
        rng.shuffle(out)
    return out


def model_forward(model, b, teacher=None, sample=False, generator=None):
    # Padding candidates use event=-2, which the shared legality mask rejects.
    return model(
        b["coords"], b["event"], b["n_events"],
        teacher=teacher, sample=sample, generator=generator
    )


def evaluate(model, data, ids, device, batch=64):
    model.eval()
    rows = []
    with torch.no_grad():
        for q in make_batches(ids, data, batch, 0, 0, False):
            b = data.collate(q, device)
            seq, _, _ = model_forward(model, b)
            pred_cost = route_cost(b["cost"], seq)
            pred_event = b["event"].gather(1, seq)
            true_event = b["event"].gather(1, b["target"])
            for i in range(len(q)):
                t = int(b["n_events"][i])
                pred = seq[i, :t]
                true = b["target"][i, :t]
                pe = pred_event[i, :t]
                te = true_event[i, :t]
                pm = {int(pe[j]): int(pred[j]) for j in range(t)}
                tm = {int(te[j]): int(true[j]) for j in range(t)}
                rows.append({
                    "pred": float(pred_cost[i]),
                    "opt": float(b["opt"][i]),
                    "exact": int(torch.equal(pred, true)),
                    "pointer_hits": int((pred == true).sum()),
                    "steps": t,
                    "event_exact": int(torch.equal(pe, te)),
                    "event_hits": int((pe == te).sum()),
                    "candidate_hits": int(sum(pm[e] == tm[e] for e in tm)),
                })
    return summarize_route_rows(rows)


def greedy(data, ids):
    rows = []
    for cid in ids:
        x = data.item(cid)
        ev = x["event"]; C = x["cost"]; t = x["n_events"]
        cur = 0; done = set(); seq = []
        for _ in range(t):
            legal = [e for e in range(t) if e not in done and (e % 2 == 0 or e-1 in done)]
            candidates = [j for j in range(1, len(ev)) if int(ev[j]) in legal]
            j = min(candidates, key=lambda z: (float(C[cur, z]), int(ev[z]), z))
            seq.append(j); done.add(int(ev[j])); cur = j
        target = x["target"].tolist()
        pe = [int(ev[j]) for j in seq]; te = [int(ev[j]) for j in target]
        pm = {int(ev[j]): j for j in seq}; tm = {int(ev[j]): j for j in target}
        L = float(sum(C[a, z] for a, z in zip([0] + seq[:-1], seq)))
        rows.append({
            "pred": L, "opt": x["opt"], "exact": int(seq == target),
            "pointer_hits": sum(a == z for a, z in zip(seq, target)), "steps": t,
            "event_exact": int(pe == te), "event_hits": sum(a == z for a, z in zip(pe, te)),
            "candidate_hits": sum(pm[e] == tm[e] for e in tm),
        })
    return summarize_route_rows(rows)


def train_ptr(args, data, tr, va, device):
    model = CarpoolPointerNet(args.dim, args.ptr_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf"); state = None; bad = 0; hist = []
    for ep in range(1, args.epochs+1):
        model.train(); vals = []
        for ids in make_batches(tr, data, args.batch, args.seed, ep, True):
            b = data.collate(ids, device)
            _, _, loss = model_forward(model, b, teacher=b["target"])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            vals.append(float(loss.detach()))
        if ep == 1 or ep % args.eval_every == 0:
            v = evaluate(model, data, va, device, args.eval_batch)
            score = float(v["mean_case_gap"])
            hist.append({"epoch": ep, "train_ce": float(np.mean(vals)), "val": v})
            print("XG_PTR_VAL", json.dumps(hist[-1], sort_keys=True), flush=True)
            if score < best - 1e-6:
                best = score; bad = 0
                state = copy.deepcopy({k:x.detach().cpu() for k,x in model.state_dict().items()})
            else:
                bad += args.eval_every
            if bad >= args.patience:
                break
    model.load_state_dict(state)
    return model, hist


def train_am(args, data, tr, va, device):
    model = CarpoolAM(args.dim, args.heads, args.am_layers, args.ff, "batch").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed + 3001)
    best = float("inf"); state = None; bad = 0; hist = []
    for ep in range(1, args.epochs+1):
        model.train(); vals=[]; costs=[]
        for ids in make_batches(tr, data, args.batch, args.seed, ep, True):
            b = data.collate(ids, device)
            seq, ll, _ = model_forward(model, b, sample=True, generator=gen)
            c = route_cost(b["cost"], seq)
            advantage = (c - c.mean()).detach()
            loss = (advantage * ll).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            vals.append(float(loss.detach())); costs.append(float(c.mean()))
        if ep == 1 or ep % args.eval_every == 0:
            v = evaluate(model, data, va, device, args.eval_batch)
            score = float(v["mean_case_gap"])
            hist.append({"epoch": ep, "train_reinforce": float(np.mean(vals)),
                         "train_route_cost": float(np.mean(costs)), "val": v})
            print("XG_AM_VAL", json.dumps(hist[-1], sort_keys=True), flush=True)
            if score < best - 1e-6:
                best = score; bad = 0
                state = copy.deepcopy({k:x.detach().cpu() for k,x in model.state_dict().items()})
            else:
                bad += args.eval_every
            if bad >= args.patience:
                break
    model.load_state_dict(state)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["greedy","ptrnet","am"], required=True)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ptr-layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--am-layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    seed_all(args.seed)
    data = CrossgraphCases(args.graph, args.matrix, args.benchmark)
    tr, va, te = data.ids("train"), data.ids("validation"), data.ids("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "greedy":
        result = {"model": "DisGreedy-corrected", "test": greedy(data, te)}
    else:
        trainer = train_ptr if args.model == "ptrnet" else train_am
        model, history = trainer(args, data, tr, va, device)
        result = {
            "model": args.model, "training_seed": args.seed,
            "train_cases": len(tr), "validation_cases": len(va), "test_cases": len(te),
            "history": history, "test": evaluate(model, data, te, device, args.eval_batch),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("CROSSGRAPH_BASELINE_RESULT", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
