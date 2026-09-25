#!/usr/bin/env python3
"""Standardize native directed road graphs and optionally build exact APSP.

Designed for the already-audited distance-project Jinan/Shenzhen assets.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def mapped_edges(original_node_ids: np.ndarray, raw_edges: Path):
    original = np.asarray(original_node_ids, dtype=np.int64)
    max_id = int(original.max())
    mapping = np.full(max_id + 1, -1, dtype=np.int64)
    mapping[original] = np.arange(len(original), dtype=np.int64)
    best = {}
    with raw_edges.open(newline="") as fh:
        for row in csv.DictReader(fh):
            u0, v0 = int(row["Origin"]), int(row["Destination"])
            if u0 > max_id or v0 > max_id:
                continue
            u, v = int(mapping[u0]), int(mapping[v0])
            if u < 0 or v < 0 or u == v:
                continue
            w = float(row["Length"])
            key = (u, v)
            if key not in best or w < best[key]:
                best[key] = w
    src = np.fromiter((k[0] for k in best), dtype=np.int32, count=len(best))
    dst = np.fromiter((k[1] for k in best), dtype=np.int32, count=len(best))
    weight = np.fromiter(best.values(), dtype=np.float64, count=len(best))
    return src, dst, weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol-npz", required=True, type=Path)
    ap.add_argument("--raw-edges", required=True, type=Path)
    ap.add_argument("--graph-out", required=True, type=Path)
    ap.add_argument("--matrix-out", type=Path)
    ap.add_argument("--predecessor-out", type=Path)
    args = ap.parse_args()

    z = np.load(args.protocol_npz, mmap_mode="r")
    coords = np.asarray(z["coordinates"], dtype=np.float32)
    original = np.asarray(z["original_node_ids"], dtype=np.int64)
    src, dst, weight = mapped_edges(original, args.raw_edges)
    n = len(coords)

    A = csr_matrix((weight, (src, dst)), shape=(n, n))
    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.graph_out,
        src=src,
        dst=dst,
        weight=weight,
        coordinates=coords,
        original_node_ids=original,
    )

    summary = {
        "dataset": args.dataset,
        "nodes": int(n),
        "directed_arcs": int(len(src)),
        "weight_min_m": float(weight.min()),
        "weight_median_m": float(np.median(weight)),
        "weight_max_m": float(weight.max()),
        "graph_out": str(args.graph_out),
    }

    if args.matrix_out is not None or args.predecessor_out is not None:
        if args.predecessor_out is not None:
            D, P = dijkstra(A, directed=True, return_predecessors=True)
        else:
            D = dijkstra(A, directed=True)
            P = None
        if not np.isfinite(D).all():
            bad = int((~np.isfinite(D)).sum())
            raise RuntimeError(f"graph is not strongly connected: {bad} nonfinite APSP entries")
        if args.matrix_out is not None:
            args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.matrix_out, D)
            summary["matrix_out"] = str(args.matrix_out)
            summary["matrix_shape"] = list(D.shape)
            summary["matrix_dtype"] = str(D.dtype)
            summary["matrix_bytes"] = int(D.nbytes)
        if args.predecessor_out is not None:
            args.predecessor_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.predecessor_out, P.astype(np.int32, copy=False))
            summary["predecessor_out"] = str(args.predecessor_out)
            summary["predecessor_shape"] = list(P.shape)
            summary["predecessor_bytes"] = int(P.astype(np.int32, copy=False).nbytes)

    print("CROSSGRAPH_PREPARE", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
