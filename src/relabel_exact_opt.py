#!/usr/bin/env python3
"""Regenerate exact paper-precedence labels for the archived 21-point Chengdu set.

The archived tensor has fixed layout:
  index 0: driver
  1..5:   passenger-1 pickup candidates
  6..10:  passenger-2 pickup candidates
  11..15: passenger-1 drop-off candidates
  16..20: passenger-2 drop-off candidates

Unlike the legacy labels, the exact solver permits every group order satisfying
pickup_i < dropoff_i. For two passengers there are six legal group orders.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


GROUPS = [
    np.arange(1, 6, dtype=np.int64),
    np.arange(6, 11, dtype=np.int64),
    np.arange(11, 16, dtype=np.int64),
    np.arange(16, 21, dtype=np.int64),
]
LEGAL_GROUP_ORDERS = [
    p for p in itertools.permutations(range(4))
    if p.index(0) < p.index(2) and p.index(1) < p.index(3)
]


def build_directed_distance(link_file: Path, node_file: Path) -> np.ndarray:
    links = pd.read_csv(link_file)
    nodes = pd.read_csv(node_file)
    n = int(max(nodes.Node.max(), links.Node_Start.max(), links.Node_End.max())) + 1
    best = {}
    for u, v, w in links[["Node_Start", "Node_End", "Length"]].itertuples(index=False, name=None):
        key = (int(u), int(v)); w = float(w)
        if key not in best or w < best[key]:
            best[key] = w
    rows = np.fromiter((x[0] for x in best), dtype=np.int32)
    cols = np.fromiter((x[1] for x in best), dtype=np.int32)
    vals = np.fromiter(best.values(), dtype=np.float64)
    return dijkstra(csr_matrix((vals, (rows, cols)), shape=(n, n)), directed=True)


def exact_one(D: np.ndarray, ids: np.ndarray):
    best = np.inf
    best_order = None
    best_local = None

    for order in LEGAL_GROUP_ORDERS:
        a, b, c, d = [GROUPS[g] for g in order]
        cost = (
            D[ids[0], ids[a]][:, None, None, None]
            + D[ids[a][:, None], ids[b][None, :]][:, :, None, None]
            + D[ids[b][:, None], ids[c][None, :]][None, :, :, None]
            + D[ids[c][:, None], ids[d][None, :]][None, None, :, :]
        )
        flat = int(np.argmin(cost))
        value = float(cost.reshape(-1)[flat])
        if value < best - 1e-9:
            best = value
            best_order = order
            best_local = np.unravel_index(flat, cost.shape)

    pointers = [0]
    for step, group in enumerate(best_order):
        pointers.append(int(GROUPS[group][best_local[step]]))
    # Opt_Seq uses archived group ids 1..4, with 0 denoting the driver.
    opt_seq = [0] + [int(g + 1) for g in best_order]
    return np.asarray(pointers, dtype=np.int64), np.asarray(opt_seq, dtype=np.int64), best


def route_length(D: np.ndarray, ids: np.ndarray, pointers: np.ndarray) -> float:
    return float(sum(D[ids[pointers[i]], ids[pointers[i + 1]]] for i in range(len(pointers) - 1)))


def as_like(old, values, is_length=False):
    if torch.is_tensor(old):
        if is_length:
            return torch.tensor(float(values), dtype=torch.float64)
        return torch.as_tensor(values, dtype=old.dtype)
    if isinstance(old, np.ndarray):
        return np.asarray(values, dtype=old.dtype)
    if is_length:
        return float(values)
    return np.asarray(values, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--links", default="sim_data/chengdu_link-mod.txt", type=Path)
    ap.add_argument("--nodes", default="sim_data/chengdu_node-mod.txt", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", default="", type=Path)
    args = ap.parse_args()

    raw = np.load(args.input, allow_pickle=True)
    D = build_directed_distance(args.links, args.nodes)

    repaired = np.empty(len(raw), dtype=object)
    old_gap = []
    old_equal = 0
    old_interleaved = 0
    exact_interleaved = 0
    changed_seq = 0
    old_recalc_err = []
    exact_recalc_err = []
    examples = []

    for i, sample in enumerate(raw):
        s = dict(sample)
        ids = np.asarray(s["Points_id"], dtype=np.int64)
        old_ptr = np.asarray(s["Solutions"], dtype=np.int64)
        old_len = float(s["Opt_Length"])
        old_calc = route_length(D, ids, old_ptr)
        new_ptr, new_seq, new_len = exact_one(D, ids)
        new_calc = route_length(D, ids, new_ptr)

        old_recalc_err.append(abs(old_calc - old_len))
        exact_recalc_err.append(abs(new_calc - new_len))
        gap = (old_calc - new_len) / max(new_len, 1e-12) * 100.0
        old_gap.append(gap)
        tol = max(0.01, 1e-7 * max(1.0, new_len))
        old_equal += int(abs(old_calc - new_len) <= tol)

        old_seq = np.asarray(s["Opt_Seq"], dtype=np.int64)
        old_groups = [int(x - 1) for x in old_seq[1:]]
        new_groups = [int((x - 1) // 5) for x in new_ptr[1:]]
        old_first_drop = min(old_groups.index(2), old_groups.index(3))
        new_first_drop = min(new_groups.index(2), new_groups.index(3))
        old_interleaved += int(old_first_drop < 2)
        exact_interleaved += int(new_first_drop < 2)
        changed_seq += int(not np.array_equal(old_ptr, new_ptr))

        if len(examples) < 12 and gap > 1e-4:
            examples.append({
                "case": i,
                "old_pointers": old_ptr.tolist(),
                "exact_pointers": new_ptr.tolist(),
                "old_length": old_calc,
                "exact_length": new_len,
                "old_gap_pct": gap,
                "old_groups": old_groups,
                "exact_groups": new_groups,
            })

        s["Solutions"] = as_like(sample["Solutions"], new_ptr)
        s["Opt_Seq"] = as_like(sample["Opt_Seq"], new_seq)
        s["Opt_Length"] = as_like(sample["Opt_Length"], new_len, is_length=True)
        repaired[i] = s

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, repaired, allow_pickle=True)

    gaps = np.asarray(old_gap)
    summary = {
        "cases": int(len(raw)),
        "legal_group_orders": [list(x) for x in LEGAL_GROUP_ORDERS],
        "old_route_recalc_mae": float(np.mean(old_recalc_err)),
        "old_equal_exact_count": int(old_equal),
        "old_equal_exact_fraction": float(old_equal / len(raw)),
        "old_vs_exact_gap_mean_pct": float(np.mean(gaps)),
        "old_vs_exact_gap_median_pct": float(np.median(gaps)),
        "old_vs_exact_gap_p95_pct": float(np.percentile(gaps, 95)),
        "old_vs_exact_gap_max_pct": float(np.max(gaps)),
        "old_vs_exact_gap_min_pct": float(np.min(gaps)),
        "old_positive_gap_cases": int(np.sum(gaps > 1e-6)),
        "old_interleaved_cases": int(old_interleaved),
        "exact_interleaved_cases": int(exact_interleaved),
        "changed_pointer_sequence_cases": int(changed_seq),
        "exact_route_recalc_max_error": float(np.max(exact_recalc_err)),
        "examples": examples,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
