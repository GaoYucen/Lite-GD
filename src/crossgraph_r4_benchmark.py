#!/usr/bin/env python3
"""Convert the frozen r4 cross-graph NPZ into the canonical Lite-GD benchmark layout.

This adapter does not resample or relabel cases. It only serializes the already
generated r4 arrays into the metadata/cases format consumed by crossgraph_data.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np


def _as_int(x):
    return int(np.asarray(x).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--cases-npz", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    z = np.load(args.cases_npz, allow_pickle=False)
    required = {
        "case_ptr", "edge_idx", "ratio", "event", "local_idx", "target",
        "opt_length", "split", "passengers", "seed", "split_seed",
        "route_edge_ptr", "route_edge_idx", "route_node_ptr", "route_node_idx",
    }
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"r4 NPZ missing route/case arrays: {missing}")

    case_ptr = np.asarray(z["case_ptr"], dtype=np.int64)
    edge_idx = np.asarray(z["edge_idx"], dtype=np.int64)
    ratio = np.asarray(z["ratio"], dtype=np.float64)
    event = np.asarray(z["event"], dtype=np.int64)
    local_idx = np.asarray(z["local_idx"], dtype=np.int64)
    target = np.asarray(z["target"], dtype=np.int64)
    opt = np.asarray(z["opt_length"], dtype=np.float64)
    split = np.asarray(z["split"], dtype=np.int8)
    passengers = np.asarray(z["passengers"], dtype=np.int64)
    rep = np.asarray(z["route_edge_ptr"], dtype=np.int64)
    re = np.asarray(z["route_edge_idx"], dtype=np.int64)
    rnp = np.asarray(z["route_node_ptr"], dtype=np.int64)
    rn = np.asarray(z["route_node_idx"], dtype=np.int64)

    n = len(opt)
    if len(case_ptr) != n + 1 or target.shape != (n, 4):
        raise ValueError(f"unexpected r4 shapes: cases={n} case_ptr={case_ptr.shape} target={target.shape}")
    if len(rep) != n + 1 or len(rnp) != n + 1:
        raise ValueError("route pointer arrays do not align with cases")
    if not np.all(passengers == 2):
        raise ValueError("r4 canonical adapter expects two-passenger cases")

    rows = []
    for cid in range(n):
        a, b = int(case_ptr[cid]), int(case_ptr[cid + 1])
        ce = edge_idx[a:b]
        cr = ratio[a:b]
        cv = event[a:b]
        cl = local_idx[a:b]
        if len(ce) < 5 or int(cv[0]) != -1:
            raise ValueError(f"case {cid}: malformed driver/candidate layout")

        groups = []
        for ev in range(4):
            ids = np.flatnonzero(cv == ev)
            ids = ids[np.argsort(cl[ids], kind="stable")]
            groups.append([
                {"edge": int(ce[j]), "ratio": float(cr[j])}
                for j in ids
            ])

        t = target[cid].astype(np.int64)
        if np.any(t <= 0) or np.any(t >= len(ce)):
            raise ValueError(f"case {cid}: target out of range")
        selected_events = cv[t]
        if sorted(map(int, selected_events)) != [0, 1, 2, 3]:
            raise ValueError(f"case {cid}: target event semantics invalid: {selected_events.tolist()}")

        er = re[int(rep[cid]):int(rep[cid + 1])]
        nr = rn[int(rnp[cid]):int(rnp[cid + 1])]
        rows.append({
            "case_id": cid,
            "driver": {"edge": int(ce[0]), "ratio": float(cr[0])},
            "candidate_groups": groups,
            "exact_event_sequence": [int(x) for x in selected_events],
            "exact_flat_indices": [int(x) for x in t],
            "exact_local_indices": [int(cl[x]) for x in t],
            "exact_selected_edges": [int(ce[x]) for x in t],
            "exact_selected_ratios": [float(cr[x]) for x in t],
            "exact_length": float(opt[cid]),
            "exact_edge_route": [int(x) for x in er],
            "exact_node_route": [int(x) for x in nr],
        })

    ids = np.arange(n, dtype=np.int64)
    split_obj = {
        "train": ids[split == 0].astype(int).tolist(),
        "validation": ids[split == 1].astype(int).tolist(),
        "test": ids[split == 2].astype(int).tolist(),
    }
    if [len(split_obj[k]) for k in ("train", "validation", "test")] != [int(.8*n), int(.1*n), n-int(.9*n)]:
        raise ValueError(f"unexpected split counts: { {k: len(v) for k, v in split_obj.items()} }")

    summary = json.loads(args.summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.graph, args.out_dir / "graph.npz")

    # Keep a route-free logical copy for audit and the route-attached copy used
    # by Lite-GD pretraining. Candidate/target/split content is identical.
    base_rows = []
    for r in rows:
        q = dict(r)
        q.pop("exact_edge_route")
        q.pop("exact_node_route")
        base_rows.append(q)
    with gzip.open(args.out_dir / "cases.json.gz", "wt", encoding="utf-8") as f:
        json.dump(base_rows, f, separators=(",", ":"))
    with gzip.open(args.out_dir / "cases_with_routes.json.gz", "wt", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))

    meta = {
        "name": f"{args.dataset}-pilot1k-r4",
        "dataset": args.dataset,
        "protocol_version": "r4-frozen-20260925",
        "cases": n,
        "passengers": 2,
        "generator_seed": _as_int(z["seed"]),
        "split_seed": _as_int(z["split_seed"]),
        "split": split_obj,
        "source_cases_npz": str(args.cases_npz),
        "source_graph": str(args.graph),
        "generated_certification": summary,
        "route_attachment": {
            "cases_with_routes_file": "cases_with_routes.json.gz",
            "full_route_supervision": True,
            "route_edge_entries": int(len(re)),
            "route_node_entries": int(len(rn)),
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Round-trip checks against source arrays.
    check_edges = []
    check_ratios = []
    check_events = []
    check_locals = []
    check_targets = []
    for r in rows:
        flat = [(r["driver"]["edge"], r["driver"]["ratio"], -1, -1)]
        for ev, g in enumerate(r["candidate_groups"]):
            flat.extend((x["edge"], x["ratio"], ev, li) for li, x in enumerate(g))
        check_edges.extend(x[0] for x in flat)
        check_ratios.extend(x[1] for x in flat)
        check_events.extend(x[2] for x in flat)
        check_locals.extend(x[3] for x in flat)
        check_targets.append(r["exact_flat_indices"])

    if not np.array_equal(np.asarray(check_edges, dtype=edge_idx.dtype), edge_idx):
        raise AssertionError("edge_idx round-trip mismatch")
    if not np.allclose(np.asarray(check_ratios, dtype=ratio.dtype), ratio, rtol=0, atol=1e-7):
        raise AssertionError("ratio round-trip mismatch")
    if not np.array_equal(np.asarray(check_events, dtype=event.dtype), event):
        raise AssertionError("event round-trip mismatch")
    if not np.array_equal(np.asarray(check_locals, dtype=local_idx.dtype), local_idx):
        raise AssertionError("local_idx round-trip mismatch")
    if not np.array_equal(np.asarray(check_targets, dtype=target.dtype), target):
        raise AssertionError("target round-trip mismatch")

    print("R4_BENCHMARK_READY", json.dumps({
        "dataset": args.dataset,
        "out_dir": str(args.out_dir),
        "cases": n,
        "split_counts": {k: len(v) for k, v in split_obj.items()},
        "route_edge_entries": int(len(re)),
        "route_node_entries": int(len(rn)),
        "source_npz": str(args.cases_npz),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
