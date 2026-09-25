#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch


def summarize_route_rows(rows: List[dict]) -> dict:
    """Summarize route quality under one exact road-distance metric."""
    if not rows:
        return {}
    pred = np.asarray([float(r["pred"]) for r in rows], dtype=np.float64)
    opt = np.asarray([float(r["opt"]) for r in rows], dtype=np.float64)
    signed_gap = (pred / opt - 1.0) * 100.0
    abs_rel = np.abs(pred - opt) / opt * 100.0

    # "Objective accuracy" means the predicted solution reaches the exact
    # objective value, independent of whether its candidate sequence equals
    # one particular optimal label. The tolerance is only for floating point.
    objective = np.isclose(pred, opt, rtol=1e-9, atol=1e-6)
    out = {
        "n": int(len(rows)),
        "gap": (float(np.mean(pred)) / float(np.mean(opt)) - 1.0) * 100.0,
        "mean_case_gap": float(np.mean(signed_gap)),
        "median_case_gap": float(np.median(signed_gap)),
        "p95_case_gap": float(np.percentile(signed_gap, 95)),
        "objective_accuracy": 100.0 * float(np.mean(objective)),
        "objective_optimal_cases": int(objective.sum()),
        "negative_gap_cases": int(np.sum(signed_gap < -1e-7)),
        "max_negative_gap": float(np.min(signed_gap)),
    }
    for tol in (0.1, 0.5, 1.0, 3.0, 5.0, 10.0):
        key = str(tol).replace(".", "p")
        out[f"within_{key}pct"] = 100.0 * float(np.mean(abs_rel <= tol + 1e-12))

    if "exact" in rows[0]:
        n = len(rows)
        steps = sum(int(r["steps"]) for r in rows)
        exact = np.asarray([bool(r["exact"]) for r in rows])
        out.update({
            "exact": 100.0 * float(np.mean(exact)),
            "pointer": 100.0 * sum(int(r["pointer_hits"]) for r in rows) / steps,
            "event_order_exact": 100.0 * sum(int(r["event_exact"]) for r in rows) / n,
            "event_step_acc": 100.0 * sum(int(r["event_hits"]) for r in rows) / steps,
            "candidate_by_event_acc": 100.0 * sum(int(r["candidate_hits"]) for r in rows) / steps,
            "objective_optimal_but_sequence_different": 100.0 * float(np.mean(objective & ~exact)),
            "objective_optimal_but_sequence_different_cases": int(np.sum(objective & ~exact)),
        })
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentiles_ms(samples_ns: List[int]) -> dict:
    x = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {
        "samples": int(len(x)),
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def benchmark_inference(
    model: torch.nn.Module,
    batch1_cpu: List[dict],
    batchn_cpu: List[dict],
    move_fn: Callable,
    device: torch.device,
    infer_fn: Callable,
    *,
    warmup: int = 20,
    rounds: int = 5,
) -> dict:
    """Benchmark route-selection inference.

    core_* excludes host-to-device copy and assumes a collated request is
    already resident on the execution device. pipeline_* includes move_fn but
    still excludes dataset loading, road-graph/APSP construction and label
    generation. Throughput is measured with the supplied larger batches.
    """
    model.eval()
    dev_batches = [move_fn(b, device) for b in batch1_cpu]
    if not dev_batches:
        return {}

    with torch.inference_mode():
        for i in range(warmup):
            infer_fn(model, dev_batches[i % len(dev_batches)])
        _sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        core = []
        for _ in range(rounds):
            for b in dev_batches:
                _sync(device)
                t0 = time.perf_counter_ns()
                infer_fn(model, b)
                _sync(device)
                core.append(time.perf_counter_ns() - t0)

        pipeline = []
        for _ in range(rounds):
            for b in batch1_cpu:
                _sync(device)
                t0 = time.perf_counter_ns()
                bb = move_fn(b, device)
                infer_fn(model, bb)
                _sync(device)
                pipeline.append(time.perf_counter_ns() - t0)

        total_ns = 0
        total_items = 0
        for _ in range(max(3, rounds)):
            for b in batchn_cpu:
                bb = move_fn(b, device)
                _sync(device)
                t0 = time.perf_counter_ns()
                infer_fn(model, bb)
                _sync(device)
                total_ns += time.perf_counter_ns() - t0
                total_items += int(b["n_events"].shape[0])

    out = {
        "device": str(device),
        "scope": (
            "route-selection inference; core excludes host-to-device copy; "
            "pipeline includes tensor move; both exclude road-graph/APSP init, "
            "dataset I/O and exact-label generation"
        ),
        "core_batch1": _percentiles_ms(core),
        "pipeline_batch1": _percentiles_ms(pipeline),
        "throughput_items_per_s": float(total_items / (total_ns / 1e9)),
        "throughput_items": int(total_items),
    }
    if device.type == "cuda":
        out["peak_allocated_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
        out["peak_reserved_mib"] = float(torch.cuda.max_memory_reserved(device) / 2**20)
    return out
