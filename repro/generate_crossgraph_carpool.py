#!/usr/bin/env python3
"""Generate reproducible 2-passenger carpool benchmarks on directed road graphs.

The generator preserves the recovered Chengdu task scale while changing the
underlying road graph:
- driver and pickup candidate ratios: 0.001
- dropoff candidate ratios: 0.999
- pickup/dropoff candidate counts sampled from the exact recovered 2p histogram
- candidate-group spatial radii sampled from recovered Chengdu quantiles
- driver->pickup and pickup->dropoff physical spans sampled from recovered Chengdu
  quantiles
- exact labels computed with the same directed point-on-edge metric and
  pickup-before-own-dropoff precedence.

The first cross-graph stage intentionally fixes passenger_count=2 so graph size
is the main experimental variable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


# Recovered two-passenger candidate-count histograms.
PICKUP_COUNT_VALUES = np.asarray([4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
PICKUP_COUNT_WEIGHTS = np.asarray([12, 56, 197, 551, 752, 319, 5], dtype=np.float64)
DROPOFF_COUNT_VALUES = np.asarray([4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
DROPOFF_COUNT_WEIGHTS = np.asarray([6, 50, 199, 583, 744, 303, 7], dtype=np.float64)

# Piecewise empirical quantile summaries from the retained 996-case Chengdu set.
# We use physical (haversine) rather than road distance for spatial sampling; the
# exact label itself always uses directed road distance.
GROUP_RADIUS_Q = np.asarray([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00])
GROUP_RADIUS_M = np.asarray([
    69.5274471967395, 273.4844473275673, 363.77908771142233,
    489.83045044965695, 705.3136049787797, 1165.3971789299417,
    1711.8907134083697, 6558.969899001407,
])
DRIVER_PICKUP_Q = np.asarray([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00])
DRIVER_PICKUP_M = np.asarray([
    0.0, 2789.5155315690236, 4659.260689989655, 7280.602573530552,
    10416.409246490495, 13693.316666344826, 15454.944033816122,
    26243.67536362974,
])
PICKUP_DROPOFF_Q = np.asarray([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00])
PICKUP_DROPOFF_M = np.asarray([
    274.73736176285996, 3848.3052125156855, 5683.097324132784,
    8207.717335146343, 10845.963782741135, 13467.392445967787,
    15106.558558838176, 27142.97241140127,
])

EARTH_M = 6371000.0


def sample_piecewise(rng: np.random.Generator, q: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(rng.random(), q, values))


def weighted_choice(rng: np.random.Generator, values: np.ndarray, weights: np.ndarray) -> int:
    p = weights / weights.sum()
    return int(rng.choice(values, p=p))


def haversine_m(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lon1, lat1 = np.deg2rad(a[..., 0]), np.deg2rad(a[..., 1])
    lon2, lat2 = np.deg2rad(b[..., 0]), np.deg2rad(b[..., 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_M * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def summarize(xs: Iterable[float]) -> dict:
    a = np.asarray(list(xs), dtype=np.float64)
    if not len(a):
        return {}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def project_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    lon0 = float(np.mean(coords[:, 0]))
    lat0 = float(np.mean(coords[:, 1]))
    x = (coords[:, 0] - lon0) * (111320.0 * math.cos(math.radians(lat0)))
    y = (coords[:, 1] - lat0) * 110540.0
    return np.stack([x, y], axis=1)


def load_city_graph(npz_path: Path, raw_edges: Path):
    z = np.load(npz_path, mmap_mode="r")
    coords = np.asarray(z["coordinates"], dtype=np.float64)
    original = np.asarray(z["original_node_ids"], dtype=np.int64)
    mapping = {int(v): i for i, v in enumerate(original)}

    best: Dict[Tuple[int, int], float] = {}
    with raw_edges.open(newline="") as fh:
        for row in csv.DictReader(fh):
            u0, v0 = int(row["Origin"]), int(row["Destination"])
            if u0 not in mapping or v0 not in mapping:
                continue
            u, v = mapping[u0], mapping[v0]
            if u == v:
                continue
            w = float(row["Length"])
            key = (u, v)
            if key not in best or w < best[key]:
                best[key] = w

    keys = sorted(best)
    src = np.asarray([x[0] for x in keys], dtype=np.int32)
    dst = np.asarray([x[1] for x in keys], dtype=np.int32)
    weight = np.asarray([best[x] for x in keys], dtype=np.float64)
    return coords, src, dst, weight, original


def build_or_load_allpairs(path: Path, n_nodes: int, src, dst, weight):
    if path.exists():
        d = np.load(path, mmap_mode="r")
        if d.shape != (n_nodes, n_nodes):
            raise ValueError(f"allpairs shape mismatch: {d.shape} != {(n_nodes,n_nodes)}")
        return d, False
    path.parent.mkdir(parents=True, exist_ok=True)
    a = csr_matrix((weight, (src, dst)), shape=(n_nodes, n_nodes))
    d = dijkstra(a, directed=True)
    if not np.isfinite(d).all():
        raise RuntimeError("graph is not strongly connected after LSCC mapping")
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(tmp, d)
    tmp.replace(path)
    return np.load(path, mmap_mode="r"), True


class Sampler:
    def __init__(self, coords, src, dst, weight, dist, seed):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.src = np.asarray(src, dtype=np.int32)
        self.dst = np.asarray(dst, dtype=np.int32)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.dist = dist
        self.rng = np.random.default_rng(seed)

        self.xy = project_xy(self.coords)
        self.node_tree = cKDTree(self.xy)
        self.src_xy = self.xy[self.src]
        self.dst_xy = self.xy[self.dst]
        self.src_tree = cKDTree(self.src_xy)
        self.dst_tree = cKDTree(self.dst_xy)

    def point_coord(self, edge: int, ratio: float) -> np.ndarray:
        u, v = int(self.src[edge]), int(self.dst[edge])
        r = float(ratio)
        return self.coords[u] + r * (self.coords[v] - self.coords[u])

    def point_dist(self, ea: int, ra: float, eb: int, rb: float) -> float:
        ea, eb = int(ea), int(eb)
        ra, rb = float(ra), float(rb)
        z = ((1.0 - ra) * self.weight[ea]
             + float(self.dist[int(self.dst[ea]), int(self.src[eb])])
             + rb * self.weight[eb])
        if ea == eb and rb >= ra:
            z = min(z, (rb - ra) * self.weight[ea])
        return float(z)

    def choose_anchor_node(self, base_node: int, target_m: float) -> int:
        # Draw random bearings. Retry because finite city boundaries can clip the
        # requested Chengdu-like spatial span.
        bxy = self.xy[int(base_node)]
        best_node = int(base_node)
        best_err = float("inf")
        for _ in range(24):
            theta = self.rng.uniform(0, 2 * math.pi)
            target = bxy + target_m * np.asarray([math.cos(theta), math.sin(theta)])
            _, node = self.node_tree.query(target, k=1)
            node = int(node)
            actual = haversine_m(self.coords[int(base_node)], self.coords[node])
            err = abs(actual - target_m)
            if err < best_err:
                best_node, best_err = node, err
            if target_m < 250 or (0.65 * target_m <= actual <= 1.35 * target_m):
                return node
        return best_node

    def candidate_edges(self, anchor_node: int, pickup: bool, count: int, radius_m: float,
                        exclude: set[int]) -> List[int]:
        tree = self.src_tree if pickup else self.dst_tree
        anchor = self.xy[int(anchor_node)]
        pool = tree.query_ball_point(anchor, r=float(radius_m))
        pool = [int(e) for e in pool if int(e) not in exclude]
        if len(pool) >= count:
            # Random sampling within the empirical radius prevents every group
            # from collapsing to only the nearest road.
            chosen = self.rng.choice(np.asarray(pool, dtype=np.int64), size=count, replace=False)
            return [int(x) for x in chosen]

        # Boundary/sparse fallback: nearest unique edges.
        k = min(len(self.src), max(count * 5, count + 8))
        _, idx = tree.query(anchor, k=k)
        idx = np.atleast_1d(idx)
        out = []
        for e in idx:
            e = int(e)
            if e not in exclude and e not in out:
                out.append(e)
            if len(out) == count:
                break
        if len(out) != count:
            raise RuntimeError("insufficient candidate edges")
        return out


def exact_dp(flat, groups, point_cost):
    # flat entries are (edge,ratio,event,local_index); index 0 is driver.
    events = sorted(groups)
    states = {(0, 0): (0.0, [])}
    for _ in events:
        nxt = {}
        for (mask, last), (cost, path) in states.items():
            for e in events:
                if mask >> e & 1:
                    continue
                if e % 2 == 1 and not (mask >> (e - 1) & 1):
                    continue
                nm = mask | (1 << e)
                for j in groups[e]:
                    z = cost + point_cost[last, j]
                    key = (nm, j)
                    prev = nxt.get(key)
                    if prev is None or z < prev[0]:
                        nxt[key] = (float(z), path + [j])
        states = nxt
    _, (best, path) = min(states.items(), key=lambda kv: kv[1][0])
    return float(best), path


def generate_one(s: Sampler):
    driver_edge = int(s.rng.integers(0, len(s.src)))
    driver_ratio = 0.001
    driver_node = int(s.src[driver_edge])

    anchors = {}
    for p in range(2):
        dp = sample_piecewise(s.rng, DRIVER_PICKUP_Q, DRIVER_PICKUP_M)
        pickup_node = s.choose_anchor_node(driver_node, dp)
        pd = sample_piecewise(s.rng, PICKUP_DROPOFF_Q, PICKUP_DROPOFF_M)
        drop_node = s.choose_anchor_node(pickup_node, pd)
        anchors[2 * p] = pickup_node
        anchors[2 * p + 1] = drop_node

    flat = [(driver_edge, driver_ratio, -1, -1)]
    groups: Dict[int, List[int]] = {}
    actual_radius = []
    event_counts = []
    excluded = {driver_edge}
    for e in range(4):
        pickup = (e % 2 == 0)
        count = weighted_choice(
            s.rng,
            PICKUP_COUNT_VALUES if pickup else DROPOFF_COUNT_VALUES,
            PICKUP_COUNT_WEIGHTS if pickup else DROPOFF_COUNT_WEIGHTS,
        )
        radius = sample_piecewise(s.rng, GROUP_RADIUS_Q, GROUP_RADIUS_M)
        edges = s.candidate_edges(anchors[e], pickup, count, radius, excluded=set())
        ratio = 0.001 if pickup else 0.999
        ids = []
        pts = []
        for li, edge in enumerate(edges):
            ids.append(len(flat))
            flat.append((int(edge), ratio, e, li))
            pts.append(s.point_coord(edge, ratio))
        groups[e] = ids
        event_counts.append(count)
        center = np.mean(np.asarray(pts), axis=0)
        actual_radius.append(max(haversine_m(p, center) for p in pts))

    n = len(flat)
    pc = np.zeros((n, n), dtype=np.float64)
    for i, (ea, ra, _, _) in enumerate(flat):
        for j, (eb, rb, _, _) in enumerate(flat):
            if i != j:
                pc[i, j] = s.point_dist(ea, ra, eb, rb)
    if not np.isfinite(pc).all():
        raise RuntimeError("non-finite point cost")

    opt, path = exact_dp(flat, groups, pc)
    local = [int(flat[j][3]) for j in path]
    order = [int(flat[j][2]) for j in path]

    driver_coord = s.point_coord(driver_edge, driver_ratio)
    dp_geo = []
    pd_geo = []
    chosen_by_event = {int(flat[j][2]): j for j in path}
    for p in range(2):
        pi = chosen_by_event[2 * p]
        di = chosen_by_event[2 * p + 1]
        dp_geo.append(haversine_m(driver_coord, s.point_coord(flat[pi][0], flat[pi][1])))
        pd_geo.append(haversine_m(s.point_coord(flat[pi][0], flat[pi][1]),
                                  s.point_coord(flat[di][0], flat[di][1])))

    return {
        "driver_edge": driver_edge,
        "driver_ratio": driver_ratio,
        "event_counts": event_counts,
        "event_edges": [[int(flat[j][0]) for j in groups[e]] for e in range(4)],
        "event_ratios": [[float(flat[j][1]) for j in groups[e]] for e in range(4)],
        "exact_event_order": order,
        "exact_local_indices": local,
        "exact_length": opt,
        "_radius": actual_radius,
        "_driver_pickup_geo": dp_geo,
        "_pickup_dropoff_geo": pd_geo,
        "_flat": flat,
        "_point_cost": pc,
        "_target_flat": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph-npz", required=True, type=Path)
    ap.add_argument("--raw-edges", required=True, type=Path)
    ap.add_argument("--allpairs", required=True, type=Path)
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    coords, src, dst, weight, original = load_city_graph(args.graph_npz, args.raw_edges)
    dist, built = build_or_load_allpairs(args.allpairs, len(coords), src, dst, weight)
    sampler = Sampler(coords, src, dst, weight, dist, args.seed)

    records = []
    failures = 0
    while len(records) < args.cases:
        try:
            records.append(generate_one(sampler))
        except RuntimeError:
            failures += 1
            if failures > max(100, args.cases):
                raise

    max_k = max(max(r["event_counts"]) for r in records)
    max_points = max(len(r["_flat"]) for r in records)
    n = len(records)
    driver_edge = np.asarray([r["driver_edge"] for r in records], dtype=np.int32)
    driver_ratio = np.asarray([r["driver_ratio"] for r in records], dtype=np.float32)
    event_counts = np.asarray([r["event_counts"] for r in records], dtype=np.int8)
    event_edges = np.full((n, 4, max_k), -1, dtype=np.int32)
    event_ratios = np.zeros((n, 4, max_k), dtype=np.float32)
    exact_event_order = np.asarray([r["exact_event_order"] for r in records], dtype=np.int8)
    exact_local_indices = np.asarray([r["exact_local_indices"] for r in records], dtype=np.int8)
    exact_length = np.asarray([r["exact_length"] for r in records], dtype=np.float64)
    point_count = np.asarray([len(r["_flat"]) for r in records], dtype=np.int16)
    point_edge = np.full((n, max_points), -1, dtype=np.int32)
    point_ratio = np.zeros((n, max_points), dtype=np.float32)
    point_event = np.full((n, max_points), -2, dtype=np.int8)
    point_coords = np.zeros((n, max_points, 2), dtype=np.float32)
    point_cost = np.zeros((n, max_points, max_points), dtype=np.float32)
    target_flat = np.full((n, 4), -1, dtype=np.int16)
    for i, r in enumerate(records):
        for e in range(4):
            k = r["event_counts"][e]
            event_edges[i, e, :k] = r["event_edges"][e]
            event_ratios[i, e, :k] = r["event_ratios"][e]
        m=len(r["_flat"])
        for j,(edge,ratio,event,_) in enumerate(r["_flat"]):
            point_edge[i,j]=int(edge)
            point_ratio[i,j]=float(ratio)
            point_event[i,j]=int(event)
            point_coords[i,j]=sampler.point_coord(int(edge),float(ratio))
        point_cost[i,:m,:m]=r["_point_cost"].astype(np.float32)
        target_flat[i]=np.asarray(r["_target_flat"],dtype=np.int16)

    split = np.full(n, 2, dtype=np.int8)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n)
    a, b = int(.8 * n), int(.9 * n)
    split[perm[:a]] = 0
    split[perm[a:b]] = 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_out = args.out_dir / "graph.npz"
    cases_out = args.out_dir / "cases.npz"
    np.savez_compressed(
        graph_out,
        coordinates=coords.astype(np.float32),
        src=src, dst=dst, weight=weight.astype(np.float64),
        original_node_ids=original,
    )
    np.savez_compressed(
        cases_out,
        driver_edge=driver_edge,
        driver_ratio=driver_ratio,
        event_counts=event_counts,
        event_edges=event_edges,
        event_ratios=event_ratios,
        exact_event_order=exact_event_order,
        exact_local_indices=exact_local_indices,
        exact_length=exact_length,
        point_count=point_count,
        point_edge=point_edge,
        point_ratio=point_ratio,
        point_event=point_event,
        point_coords=point_coords,
        point_cost=point_cost,
        target_flat=target_flat,
        split=split,
    )

    radii = [x for r in records for x in r["_radius"]]
    dpgeo = [x for r in records for x in r["_driver_pickup_geo"]]
    pdgeo = [x for r in records for x in r["_pickup_dropoff_geo"]]
    cert = {
        "dataset": args.dataset,
        "cases": n,
        "passengers": 2,
        "graph_nodes": int(len(coords)),
        "graph_edges": int(len(src)),
        "allpairs_path": str(args.allpairs),
        "allpairs_built_in_run": bool(built),
        "generation_seed": args.seed,
        "split_seed": args.split_seed,
        "failed_attempts": failures,
        "candidate_count": summarize(event_counts.reshape(-1)),
        "candidate_group_radius_haversine_m": summarize(radii),
        "driver_to_exact_pickup_haversine_m": summarize(dpgeo),
        "exact_pickup_to_own_dropoff_haversine_m": summarize(pdgeo),
        "exact_route_directed_road_m": summarize(exact_length),
        "files": {"graph": str(graph_out), "cases": str(cases_out)},
        "source_protocol": {
            "candidate_count_hist_pickup": dict(zip(map(str, PICKUP_COUNT_VALUES.tolist()), map(int, PICKUP_COUNT_WEIGHTS.tolist()))),
            "candidate_count_hist_dropoff": dict(zip(map(str, DROPOFF_COUNT_VALUES.tolist()), map(int, DROPOFF_COUNT_WEIGHTS.tolist()))),
            "driver_ratio": 0.001,
            "pickup_ratio": 0.001,
            "dropoff_ratio": 0.999,
            "chengdu_candidate_radius_median": 489.83045044965695,
            "chengdu_driver_pickup_median": 7280.602573530552,
            "chengdu_pickup_dropoff_median": 8207.717335146343,
        },
    }
    (args.out_dir / "certification.json").write_text(json.dumps(cert, indent=2) + "\n")
    print("CROSSGRAPH_GENERATED", json.dumps(cert, sort_keys=True))


if __name__ == "__main__":
    main()
