#!/usr/bin/env python3
"""Strict-split diagnostic baseline for Lite-GD archived Chengdu samples.

This is deliberately NOT claimed as the paper Lite-GD model.  It is a clean
controlled experiment for auditing the legacy PointerNet representation:
  * deterministic 8/1/1 split;
  * training from scratch;
  * validation-only model selection;
  * untouched test set;
  * paper precedence mask vs legacy all-pickups-first mask;
  * route length evaluated on the directed Chengdu road graph.
"""
from __future__ import annotations
import argparse, copy, json, math, random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_directed_distance(link_file: Path, node_file: Path) -> np.ndarray:
    links = pd.read_csv(link_file)
    nodes = pd.read_csv(node_file)
    n = int(max(nodes.Node.max(), links.Node_Start.max(), links.Node_End.max())) + 1
    best: Dict[Tuple[int, int], float] = {}
    for u, v, w in links[["Node_Start", "Node_End", "Length"]].itertuples(index=False, name=None):
        k = (int(u), int(v))
        w = float(w)
        if k not in best or w < best[k]:
            best[k] = w
    rows = np.fromiter((k[0] for k in best.keys()), dtype=np.int32)
    cols = np.fromiter((k[1] for k in best.keys()), dtype=np.int32)
    vals = np.fromiter(best.values(), dtype=np.float64)
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return dijkstra(graph, directed=True, return_predecessors=False)


def group_of(index: int) -> int:
    if index == 0:
        return -1
    return (index - 1) // 5


def paper_legal(seq: Sequence[int]) -> bool:
    groups = [group_of(int(i)) for i in seq[1:]]
    if len(groups) != 4 or len(set(groups)) != 4:
        return False
    # group 0/2 = passenger-1 pickup/drop; group 1/3 = passenger-2 pickup/drop
    return groups.index(0) < groups.index(2) and groups.index(1) < groups.index(3)


def route_length(D: np.ndarray, point_ids: Sequence[int], seq: Sequence[int]) -> float:
    ids = np.asarray(point_ids, dtype=np.int64)
    s = np.asarray(seq, dtype=np.int64)
    return float(sum(D[ids[s[i]], ids[s[i + 1]]] for i in range(len(s) - 1)))


def canonical_target(sample) -> np.ndarray:
    """Repair archived pointer ambiguity using the semantic Opt_Seq group labels.

    The legacy generator sometimes stored the index of an identical road node
    from the wrong candidate group when that node appeared multiple times in
    the 21-entry list.  The route node itself is correct, but such an index can
    violate the group's decoder mask.  We keep the selected road node and map it
    back to the candidate group specified by Opt_Seq.
    """
    ids = np.asarray(sample["Points_id"], dtype=np.int64)
    old = np.asarray(sample["Solutions"], dtype=np.int64)
    order = np.asarray(sample["Opt_Seq"], dtype=np.int64)
    out = [0]
    for step in range(1, len(order)):
        group = int(order[step])  # groups are 1..4 in the archived 2-passenger data
        node = int(ids[int(old[step])])
        lo = 1 + (group - 1) * 5
        hi = lo + 5
        hits = np.flatnonzero(ids[lo:hi] == node)
        if len(hits) == 0:
            raise ValueError(
                f"cannot canonicalize step={step} group={group} node={node}; "
                f"old_pointer={int(old[step])}"
            )
        out.append(int(lo + hits[0]))
    return np.asarray(out, dtype=np.int64)


class CaseDataset(Dataset):
    def __init__(self, raw: np.ndarray, indices: np.ndarray, feature_mode: str,
                 mean: np.ndarray, std: np.ndarray):
        self.raw = raw
        self.indices = np.asarray(indices)
        self.feature_mode = feature_mode
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def _features(self, s) -> np.ndarray:
        x = np.asarray(s["Points"], dtype=np.float32)
        if self.feature_mode == "coords":
            x = x[:, :2]
        elif self.feature_mode == "legacy23":
            x = x[:, :23]
        else:
            raise ValueError(self.feature_mode)
        return (x - self.mean) / self.std

    def __getitem__(self, j: int):
        s = self.raw[int(self.indices[j])]
        return {
            "x": torch.from_numpy(self._features(s)).float(),
            "target": torch.as_tensor(canonical_target(s), dtype=torch.long),
            "points_id": torch.as_tensor(s["Points_id"], dtype=torch.long),
            "opt_length": torch.as_tensor(float(s["Opt_Length"]), dtype=torch.float32),
            "raw_index": int(self.indices[j]),
        }


class PointerDecoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 96, hidden_dim: int = 192):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.embed = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            emb_dim, hidden_dim // 2, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.h0 = nn.Linear(hidden_dim, hidden_dim)
        self.c0 = nn.Linear(hidden_dim, hidden_dim)
        self.cell = nn.LSTMCell(emb_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    @staticmethod
    def _mask(selected_groups: torch.Tensor, step: int, mode: str,
              batch: int, device: torch.device) -> torch.Tensor:
        # True means forbidden.  Candidate layout: 0 car; groups 0..3 are
        # [1:6], [6:11], [11:16], [16:21].
        m = torch.ones(batch, 21, dtype=torch.bool, device=device)
        m[:, 1:] = False
        # Mask groups already selected.
        for g in range(4):
            done = selected_groups[:, g]
            if done.any():
                lo = 1 + 5 * g
                m[done, lo:lo + 5] = True

        if mode == "paper":
            # Drop-off g=2 only after pickup g=0; drop-off g=3 only after g=1.
            not_p1 = ~selected_groups[:, 0]
            not_p2 = ~selected_groups[:, 1]
            m[not_p1, 11:16] = True
            m[not_p2, 16:21] = True
        elif mode == "legacy":
            # Intended semantics of old hand-written mask (off-by-one repaired).
            if step < 2:
                m[:, 11:21] = True
            else:
                m[:, 1:11] = True
        else:
            raise ValueError(mode)
        return m

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None,
                mask_mode: str = "paper", teacher_forcing: bool = True):
        b, n, _ = x.shape
        assert n == 21
        emb = self.embed(x)
        enc, _ = self.encoder(emb)
        h = torch.tanh(self.h0(enc[:, 0]))
        c = torch.tanh(self.c0(enc[:, 0]))
        dec_in = emb[:, 0]
        selected_groups = torch.zeros(b, 4, dtype=torch.bool, device=x.device)
        logits_steps, chosen_steps = [], []

        for step in range(4):
            h, c = self.cell(dec_in, (h, c))
            logits = self.v(torch.tanh(self.key(enc) + self.query(h).unsqueeze(1))).squeeze(-1)
            mask = self._mask(selected_groups, step, mask_mode, b, x.device)
            logits = logits.masked_fill(mask, -1e9)
            greedy = logits.argmax(dim=1)
            logits_steps.append(logits)

            if self.training and teacher_forcing and target is not None:
                chosen = target[:, step + 1]
            else:
                chosen = greedy
            chosen_steps.append(greedy)

            g = torch.div(chosen - 1, 5, rounding_mode="floor").clamp(0, 3)
            selected_groups.scatter_(1, g[:, None], True)
            dec_in = emb[torch.arange(b, device=x.device), chosen]

        logits = torch.stack(logits_steps, dim=1)
        pred = torch.stack(chosen_steps, dim=1)
        pred = torch.cat([torch.zeros(b, 1, dtype=torch.long, device=x.device), pred], dim=1)
        return logits, pred


def compute_norm(raw: np.ndarray, train_idx: np.ndarray, feature_mode: str):
    xs = []
    for i in train_idx:
        a = np.asarray(raw[int(i)]["Points"], dtype=np.float32)
        xs.append(a[:, :2] if feature_mode == "coords" else a[:, :23])
    x = np.concatenate(xs, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


@torch.no_grad()
def evaluate(model, loader, D, raw, device, mask_mode):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, token_n = 0.0, 0
    exact, token_ok, case_n, illegal = 0, 0, 0, 0
    pred_lengths, target_lengths, stored_opts = [], [], []
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["target"].to(device)
        logits, pred = model(x, target=None, mask_mode=mask_mode, teacher_forcing=False)
        loss_sum += float(ce(logits.reshape(-1, 21), y[:, 1:].reshape(-1)))
        token_n += y[:, 1:].numel()
        p = pred.cpu().numpy()
        yy = y.cpu().numpy()
        exact += int(np.sum(np.all(p == yy, axis=1)))
        token_ok += int(np.sum(p[:, 1:] == yy[:, 1:]))
        case_n += len(p)
        for j in range(len(p)):
            idx = int(batch["raw_index"][j])
            s = raw[idx]
            pl = route_length(D, s["Points_id"], p[j])
            tl = route_length(D, s["Points_id"], yy[j])
            pred_lengths.append(pl); target_lengths.append(tl)
            stored_opts.append(float(s["Opt_Length"]))
            illegal += 0 if paper_legal(p[j]) else 1
    pred_lengths = np.asarray(pred_lengths)
    target_lengths = np.asarray(target_lengths)
    stored_opts = np.asarray(stored_opts)
    return {
        "ce": loss_sum / max(token_n, 1),
        "exact_acc_pct": exact / max(case_n, 1) * 100,
        "pointer_acc_pct": token_ok / max(case_n * 4, 1) * 100,
        "avg_pred_length": float(pred_lengths.mean()),
        "avg_target_graph_length": float(target_lengths.mean()),
        "avg_stored_opt_length": float(stored_opts.mean()),
        "gap_vs_target_graph_pct": float((pred_lengths.mean() / target_lengths.mean() - 1) * 100),
        "gap_vs_stored_opt_pct": float((pred_lengths.mean() / stored_opts.mean() - 1) * 100),
        "target_graph_vs_stored_mae": float(np.mean(np.abs(target_lengths - stored_opts))),
        "paper_illegal_count": int(illegal),
        "n": int(case_n),
    }


def train_one(raw, train_idx, val_idx, test_idx, feature_mode, mask_mode, D,
              device, seed, epochs, batch_size, lr, emb_dim, hidden_dim, patience):
    seed_all(seed)
    mean, std = compute_norm(raw, train_idx, feature_mode)
    ds_train = CaseDataset(raw, train_idx, feature_mode, mean, std)
    ds_val = CaseDataset(raw, val_idx, feature_mode, mean, std)
    ds_test = CaseDataset(raw, test_idx, feature_mode, mean, std)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    input_dim = 2 if feature_mode == "coords" else 23
    model = PointerDecoder(input_dim, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["target"].to(device)
            logits, _ = model(x, target=y, mask_mode=mask_mode, teacher_forcing=True)
            loss = loss_fn(logits.reshape(-1, 21), y[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss); nb += 1
        val = evaluate(model, val_loader, D, raw, device, mask_mode)
        history.append((epoch, running/max(nb,1), val["ce"], val["exact_acc_pct"], val["gap_vs_target_graph_pct"]))
        if val["ce"] + 1e-6 < best:
            best = val["ce"]; stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print("progress", feature_mode, mask_mode, "epoch", epoch,
                  "train_ce", history[-1][1], "val", val, flush=True)
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "feature_mode": feature_mode,
        "mask_mode": mask_mode,
        "seed": seed,
        "epochs_ran": history[-1][0],
        "val": evaluate(model, val_loader, D, raw, device, mask_mode),
        "test": evaluate(model, test_loader, D, raw, device, mask_mode),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--links", default="sim_data/chengdu_link-mod.txt")
    ap.add_argument("--nodes", default="sim_data/chengdu_node-mod.txt")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--configs", default="coords:paper,legacy23:paper,legacy23:legacy")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seed_all(args.seed)
    raw = np.load(args.data, allow_pickle=True)
    n = len(raw)
    perm = np.random.default_rng(args.seed).permutation(n)
    ntr, nv = int(n * .8), int(n * .1)
    train_idx, val_idx, test_idx = perm[:ntr], perm[ntr:ntr+nv], perm[ntr+nv:]
    print("split", len(train_idx), len(val_idx), len(test_idx),
          "train_head", train_idx[:10].tolist(), flush=True)

    D = build_directed_distance(Path(args.links), Path(args.nodes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "torch", torch.__version__, flush=True)

    repaired = 0
    for s in raw:
        repaired += int(not np.array_equal(canonical_target(s), np.asarray(s["Solutions"], dtype=np.int64)))
    print("canonical_pointer_repairs", repaired, flush=True)

    target_patterns = {}
    for name, idx in [("train",train_idx),("val",val_idx),("test",test_idx)]:
        pats={}
        for i in idx:
            t=tuple(int(x) for x in raw[int(i)]["Opt_Seq"])
            pats[t]=pats.get(t,0)+1
        target_patterns[name]={"patterns":{str(k):v for k,v in sorted(pats.items())},
                               "interleaved":sum(v for k,v in pats.items() if list(k[1:3]) not in ([1,2],[2,1]))}
    results={"split_seed":args.seed,"target_patterns":target_patterns,"runs":[]}
    for spec in args.configs.split(","):
        feature, mask = spec.split(":")
        print("RUN", feature, mask, flush=True)
        results["runs"].append(train_one(
            raw, train_idx, val_idx, test_idx, feature, mask, D, device,
            args.seed, args.epochs, args.batch_size, args.lr,
            args.emb_dim, args.hidden_dim, args.patience
        ))
        print("RESULT", json.dumps(results["runs"][-1], ensure_ascii=False), flush=True)

    text=json.dumps(results,indent=2,ensure_ascii=False)
    print("FINAL_JSON\n"+text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
