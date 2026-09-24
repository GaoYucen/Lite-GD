from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import heldout_pointer as hp


def exact_solution(dist: np.ndarray, point_ids: np.ndarray) -> tuple[float, np.ndarray]:
    ids = point_ids.astype(int)
    best = math.inf
    best_solution = None
    groups = [np.asarray(list(g), dtype=int) for g in hp.GROUPS]

    for order in hp.LEGAL_GROUP_ORDERS:
        a, b, c, d = [groups[g] for g in order]
        cost = (
            dist[ids[0], ids[a]][:, None, None, None]
            + dist[ids[a][:, None], ids[b][None, :]][:, :, None, None]
            + dist[ids[b][:, None], ids[c][None, :]][None, :, :, None]
            + dist[ids[c][:, None], ids[d][None, :]][None, None, :, :]
        )
        flat = int(np.argmin(cost))
        value = float(cost.reshape(-1)[flat])
        if value < best - 1e-10:
            pos = np.unravel_index(flat, cost.shape)
            chosen = [int(a[pos[0]]), int(b[pos[1]]), int(c[pos[2]]), int(d[pos[3]])]
            best = value
            best_solution = np.asarray([0] + chosen, dtype=np.int64)

    assert best_solution is not None
    return best, best_solution


def correct_dataset(data: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, dict]:
    corrected = np.empty(len(data), dtype=object)
    old_gaps = []
    old_route_abs_error = []
    old_exact = 0
    old_legal = 0
    changed_index_sequence = 0
    changed_node_sequence = 0

    for i, sample in enumerate(data):
        ids = np.asarray(sample["Points_id"], dtype=int)
        old_solution = np.asarray(sample["Solutions"], dtype=int)
        old_length = float(sample["Opt_Length"])
        old_calc = hp.route_length(dist, ids, old_solution)
        best, solution = exact_solution(dist, ids)

        old_route_abs_error.append(abs(old_calc - old_length))
        gap = (old_length - best) / max(best, 1e-12) * 100.0
        old_gaps.append(gap)
        if abs(old_length - best) <= max(1e-3, 1e-6 * max(1.0, best)):
            old_exact += 1
        old_legal += int(hp.is_legal(old_solution))
        changed_index_sequence += int(not np.array_equal(old_solution, solution))
        old_nodes = ids[old_solution]
        new_nodes = ids[solution]
        changed_node_sequence += int(not np.array_equal(old_nodes, new_nodes))

        replacement = dict(sample)
        replacement["Solutions"] = solution
        replacement["Opt_Length"] = float(best)
        corrected[i] = replacement

    gaps = np.asarray(old_gaps, dtype=float)
    report = {
        "cases": len(data),
        "historical_route_mean_abs_error": float(np.mean(old_route_abs_error)),
        "historical_route_max_abs_error": float(np.max(old_route_abs_error)),
        "historical_exact_optimum_count": int(old_exact),
        "historical_exact_optimum_fraction": float(old_exact / len(data)),
        "historical_label_gap_mean_pct": float(np.mean(gaps)),
        "historical_label_gap_median_pct": float(np.median(gaps)),
        "historical_label_gap_p95_pct": float(np.percentile(gaps, 95)),
        "historical_label_gap_max_pct": float(np.max(gaps)),
        "historical_gap_gt_0_1pct_count": int(np.sum(gaps > 0.1)),
        "historical_gap_gt_1pct_count": int(np.sum(gaps > 1.0)),
        "historical_gap_gt_5pct_count": int(np.sum(gaps > 5.0)),
        "historical_legal_route_fraction": float(old_legal / len(data)),
        "changed_index_sequence_count": int(changed_index_sequence),
        "changed_node_sequence_count": int(changed_node_sequence),
    }
    return corrected, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", choices=["base", "angle"], default=["base", "angle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get("LITEGD_CACHE", "/workspace/.server-control/litegd-repro-cache"))
    raw = hp.load_historical_data(repo, cache_root)
    dist = hp.road_distance_matrix(repo, cache_root)
    data, label_audit = correct_dataset(raw, dist)
    print("HISTORICAL_LABEL_AUDIT", json.dumps(label_audit, sort_keys=True))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device", device)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0))

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(data))
    train_idx = perm[:800]
    val_idx = perm[800:900]
    test_idx = perm[900:1000]
    np.savez(args.output / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    baseline = hp.nearest_legal_baseline(data, test_idx, dist)
    print("NEAREST_LEGAL_TEST", json.dumps(asdict(baseline), sort_keys=True))

    results = []
    histories = {}
    for variant in args.variants:
        for seed in args.seeds:
            result, extra = hp.train_one(
                data=data,
                dist=dist,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                variant=variant,
                seed=seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            results.append(result)
            histories[f"{variant}-{seed}"] = extra
            print("RESULT", json.dumps(result, sort_keys=True))

    summary = {
        "protocol": {
            "source": "historical chengdu_data.npy recovered from Git history",
            "target_repair": "exact enumeration over all legal 2-passenger candidate routes",
            "split": "fixed 800/100/100 train/validation/test",
            "split_seed": args.split_seed,
            "train_from_scratch": True,
            "strict_rule_mask": True,
            "selection_metric": "validation mean route gap to recomputed exact OPT",
        },
        "historical_label_audit": label_audit,
        "nearest_legal_test": asdict(baseline),
        "runs": results,
        "aggregate": {},
    }

    metric_names = list(asdict(hp.Metrics(0, 0, 0, 0, 0, 0, 0)).keys())
    for variant in args.variants:
        subset = [r for r in results if r["variant"] == variant]
        metrics = {}
        for key in metric_names:
            vals = [r["test"][key] for r in subset]
            metrics[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        summary["aggregate"][variant] = metrics

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    print("FINAL_SUMMARY", json.dumps(summary["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
