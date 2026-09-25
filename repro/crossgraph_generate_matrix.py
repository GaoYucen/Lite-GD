#!/usr/bin/env python3
"""Generate exact two-passenger carpool benchmarks on a directed road graph.

The generator preserves the recovered Chengdu task scale while varying the road
network.  It uses projected-metre node coordinates, historical-scale candidate
groups, pickup/drop edge-ratio semantics, and one exact directed-distance metric.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# Quantiles recovered from the 996-case historical Chengdu edge+ratio benchmark.
Q = {
    "group_radius_m": [(0.00, 69.5274471967), (0.10, 273.4844473276),
                       (0.25, 363.7790877114), (0.50, 489.8304504497),
                       (0.75, 705.3136049788), (0.90, 1165.3971789299),
                       (0.95, 1711.8907134084), (1.00, 6558.9698990014)],
    "driver_pick_geo_m": [(0.00, 0.0), (0.10, 2789.5155315690),
                          (0.25, 4659.2606899897), (0.50, 7280.6025735306),
                          (0.75, 10416.4092464905), (0.90, 13693.3166663448),
                          (0.95, 15454.9440338161), (1.00, 26243.6753636297)],
    "pickup_drop_geo_m": [(0.00, 274.7373617629), (0.10, 3848.3052125157),
                          (0.25, 5683.0973241328), (0.50, 8207.7173351463),
                          (0.75, 10845.9637827411), (0.90, 13467.3924459678),
                          (0.95, 15106.5585588382), (1.00, 27142.9724114013)],
}

ROAD_ACCEPT = {
    "driver_pick_m": (2500.0, 21000.0),
    "pickup_drop_m": (3500.0, 20500.0),
}


def qsample(rng, pairs):
    u = float(rng.random())
    ps = np.asarray([x[0] for x in pairs])
    vs = np.asarray([x[1] for x in pairs])
    return float(np.interp(u, ps, vs))


def sample_count(rng):
    # Moment-matched approximation to historical count distribution:
    # range 4..10, mean 7.56, sd 1.01.
    return int(np.clip(np.rint(rng.normal(7.56097, 1.00972)), 4, 10))


def summarize(x):
    a = np.asarray(x, dtype=np.float64)
    if not len(a):
        return {}
    return {
        "n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)), "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
    }


class Road:
    def __init__(self, graph: Path, matrix: Path, predecessor: Path | None = None):
        z = np.load(graph, mmap_mode="r")
        self.src = np.asarray(z["src"], dtype=np.int32)
        self.dst = np.asarray(z["dst"], dtype=np.int32)
        self.weight = np.asarray(z["weight"], dtype=np.float64)
        self.xy = np.asarray(z["coordinates"], dtype=np.float64)
        self.D = np.load(matrix, mmap_mode="r")
        self.P = np.load(predecessor, mmap_mode="r") if predecessor is not None else None
        if self.D.shape != (len(self.xy), len(self.xy)):
            raise ValueError(f"matrix shape {self.D.shape} != graph node count {len(self.xy)}")
        self.pick_xy = self.xy[self.src] * 0.999 + self.xy[self.dst] * 0.001
        self.drop_xy = self.xy[self.src] * 0.001 + self.xy[self.dst] * 0.999
        self.mid_xy = (self.xy[self.src] + self.xy[self.dst]) / 2.0
        self.pick_tree = cKDTree(self.pick_xy)
        self.drop_tree = cKDTree(self.drop_xy)
        self.pair_to_edge = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.src, self.dst))}

    def trace_segment(self, edge_a, ratio_a, edge_b, ratio_b):
        if self.P is None:
            raise RuntimeError("predecessor matrix was not loaded")
        ea, eb = int(edge_a), int(edge_b)
        ra, rb = float(ratio_a), float(ratio_b)
        if ea == eb and rb >= ra:
            route_edges = [ea]
        else:
            s, t = int(self.dst[ea]), int(self.src[eb])
            nodes = [t]
            cur = t
            while cur != s:
                cur = int(self.P[s, cur])
                if cur < 0:
                    raise RuntimeError(f"missing predecessor for {s}->{t}")
                nodes.append(cur)
            nodes.reverse()
            middle = []
            for u, v in zip(nodes[:-1], nodes[1:]):
                try:
                    middle.append(int(self.pair_to_edge[(int(u), int(v))]))
                except KeyError as exc:
                    raise RuntimeError(f"route arc {u}->{v} missing from standardized graph") from exc
            route_edges = [ea] + middle + [eb]
        route_nodes = []
        for e in route_edges:
            route_nodes.extend([int(self.src[e]), int(self.dst[e])])
        return route_edges, route_nodes

    def point_xy(self, edge, ratio):
        e = int(edge); r = float(ratio)
        return self.xy[self.src[e]] * (1-r) + self.xy[self.dst[e]] * r

    def point_cost_matrix(self, edges, ratios):
        e = np.asarray(edges, dtype=np.int64)
        r = np.asarray(ratios, dtype=np.float64)
        base = ((1-r) * self.weight[e])[:, None]
        tail = (r * self.weight[e])[None, :]
        core = self.D[self.dst[e][:, None], self.src[e][None, :]]
        C = base + core + tail
        same = e[:, None] == e[None, :]
        forward = r[None, :] >= r[:, None]
        direct = (r[None, :] - r[:, None]) * self.weight[e][:, None]
        C = np.where(same & forward, np.minimum(C, direct), C)
        return np.asarray(C, dtype=np.float64)


def target_center(rng, origin, radius):
    theta = float(rng.uniform(0, 2*math.pi))
    return origin + radius * np.asarray([math.cos(theta), math.sin(theta)])


def sample_group(rng, tree, point_xy, center, k, radius):
    # Snap the synthetic event centre onto the road-support used by this
    # pickup/dropoff candidate type before drawing the local candidate set.
    # This avoids artificially inflating group radius when a geometric target
    # falls inside a road-sparse block.
    _, anchor = tree.query(center, k=1)
    center = point_xy[int(anchor)]
    R = max(0.85 * float(radius), 220.0)  # calibrated once on Jinan pilot; frozen for confirmations
    ids = tree.query_ball_point(center, r=R)
    while len(ids) < k and R < 8000:
        R *= 1.4
        ids = tree.query_ball_point(center, r=R)
    if len(ids) < k:
        _, q = tree.query(center, k=k)
        ids = np.atleast_1d(q).astype(int).tolist()
    ids = np.asarray(sorted(set(map(int, ids))), dtype=np.int64)
    if len(ids) > k:
        d = np.linalg.norm(point_xy[ids] - center[None, :], axis=1)
        # Spread the candidates over the available local disk instead of taking
        # only the nearest roads.
        bins = np.argsort(d)
        pool = ids[bins]
        take = np.linspace(0, len(pool)-1, k).round().astype(int)
        ids = pool[take]
    if len(ids) < k:
        _, q = tree.query(center, k=k)
        ids = np.atleast_1d(q).astype(np.int64)
    return np.asarray(ids[:k], dtype=np.int64)


def exact_dp(C, events):
    events = np.asarray(events, dtype=np.int64)
    n_events = 4
    by_event = {e: np.flatnonzero(events == e).tolist() for e in range(n_events)}
    dp = {(0, 0): (0.0, [])}
    for _ in range(n_events):
        nd = {}
        for (mask, last), (cost, seq) in dp.items():
            for e in range(n_events):
                if mask >> e & 1:
                    continue
                if e % 2 == 1 and not (mask >> (e-1) & 1):
                    continue
                nm = mask | (1 << e)
                for j in by_event[e]:
                    z = cost + float(C[last, j])
                    key = (nm, int(j))
                    old = nd.get(key)
                    if old is None or z < old[0] - 1e-12:
                        nd[key] = (z, seq + [int(j)])
        dp = nd
    best = min(dp.values(), key=lambda x: x[0])
    return float(best[0]), best[1]


def group_radius(points):
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(axis=0)
    return float(np.linalg.norm(p-c, axis=1).max())


def build_case(rng, road: Road, max_attempts=200):
    m = len(road.src)
    for _ in range(max_attempts):
        driver = int(rng.integers(0, m))
        driver_ratio = 0.5
        driver_xy = road.point_xy(driver, driver_ratio)

        groups = []
        centers = {}
        ok = True
        for passenger in range(2):
            pick_center = target_center(rng, driver_xy, qsample(rng, Q["driver_pick_geo_m"]))
            drop_center = target_center(rng, pick_center, qsample(rng, Q["pickup_drop_geo_m"]))
            centers[2*passenger] = pick_center
            centers[2*passenger+1] = drop_center

        for e in range(4):
            k = sample_count(rng)
            rad = qsample(rng, Q["group_radius_m"])
            if e % 2 == 0:
                ids = sample_group(rng, road.pick_tree, road.pick_xy, centers[e], k, rad)
                ratio = 0.001
            else:
                ids = sample_group(rng, road.drop_tree, road.drop_xy, centers[e], k, rad)
                ratio = 0.999
            if len(np.unique(ids)) < k:
                ok = False
                break
            groups.append((ids, ratio))
        if not ok:
            continue

        edges = [driver]
        ratios = [driver_ratio]
        events = [-1]
        locals_ = [-1]
        for e, (ids, ratio) in enumerate(groups):
            for li, edge in enumerate(ids):
                edges.append(int(edge)); ratios.append(float(ratio))
                events.append(e); locals_.append(li)

        C = road.point_cost_matrix(edges, ratios)
        opt, target = exact_dp(C, events)
        chosen = {events[j]: j for j in target}
        good = True
        dp_road = []
        od_road = []
        for p in range(2):
            pi, di = chosen[2*p], chosen[2*p+1]
            dp = float(C[0, pi])
            od = float(C[pi, di])
            dp_road.append(dp); od_road.append(od)
            lo, hi = ROAD_ACCEPT["driver_pick_m"]
            if not (lo <= dp <= hi):
                good = False
            lo, hi = ROAD_ACCEPT["pickup_drop_m"]
            if not (lo <= od <= hi):
                good = False
        if not good or not np.isfinite(opt):
            continue

        route_edges = None
        route_nodes = None
        if road.P is not None:
            route_edges = []
            route_nodes = []
            seq = [0] + target
            for a, b in zip(seq[:-1], seq[1:]):
                ee, nn = road.trace_segment(edges[a], ratios[a], edges[b], ratios[b])
                route_edges.extend(ee)
                route_nodes.extend(nn)

        return {
            "edges": np.asarray(edges, dtype=np.int32),
            "ratios": np.asarray(ratios, dtype=np.float32),
            "events": np.asarray(events, dtype=np.int8),
            "locals": np.asarray(locals_, dtype=np.int8),
            "target": np.asarray(target, dtype=np.int16),
            "opt": float(opt),
            "driver_pick_road": dp_road,
            "pickup_drop_road": od_road,
            "group_radii": [
                group_radius(road.pick_xy[g[0]] if e % 2 == 0 else road.drop_xy[g[0]])
                for e, g in enumerate(groups)
            ],
            "candidate_counts": [len(g[0]) for g in groups],
            "route_edges": None if route_edges is None else np.asarray(route_edges, dtype=np.int32),
            "route_nodes": None if route_nodes is None else np.asarray(route_nodes, dtype=np.int32),
        }
    raise RuntimeError("unable to sample a valid matched-scale case")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--predecessor", type=Path)
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--split-seed", type=int, default=20260925)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    road = Road(args.graph, args.matrix, args.predecessor)
    cases = [build_case(rng, road) for _ in range(args.cases)]

    ptr = [0]
    E=[]; R=[]; V=[]; L=[]
    T=[]; O=[]
    cand=[]; radii=[]; dproad=[]; odroad=[]
    for c in cases:
        E.append(c["edges"]); R.append(c["ratios"]); V.append(c["events"]); L.append(c["locals"])
        ptr.append(ptr[-1] + len(c["edges"]))
        T.append(c["target"]); O.append(c["opt"])
        cand.extend(c["candidate_counts"]); radii.extend(c["group_radii"])
        dproad.extend(c["driver_pick_road"]); odroad.extend(c["pickup_drop_road"])

    perm = np.random.default_rng(args.split_seed).permutation(args.cases)
    split = np.empty(args.cases, dtype=np.int8)
    a, b = int(.8*args.cases), int(.9*args.cases)
    split[perm[:a]]=0; split[perm[a:b]]=1; split[perm[b:]]=2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        case_ptr=np.asarray(ptr, dtype=np.int64),
        edge_idx=np.concatenate(E).astype(np.int32),
        ratio=np.concatenate(R).astype(np.float32),
        event=np.concatenate(V).astype(np.int8),
        local_idx=np.concatenate(L).astype(np.int8),
        target=np.stack(T).astype(np.int16),
        opt_length=np.asarray(O, dtype=np.float64),
        split=split,
        passengers=np.full(args.cases, 2, dtype=np.int8),
        seed=np.int64(args.seed),
        split_seed=np.int64(args.split_seed),
    )
    if args.predecessor is not None:
        rep=[0]; rnp=[0]; re=[]; rn=[]
        for case in cases:
            a=np.asarray(case["route_edges"],dtype=np.int32)
            b=np.asarray(case["route_nodes"],dtype=np.int32)
            re.append(a); rn.append(b)
            rep.append(rep[-1]+len(a)); rnp.append(rnp[-1]+len(b))
        payload.update(
            route_edge_ptr=np.asarray(rep,dtype=np.int64),
            route_edge_idx=np.concatenate(re).astype(np.int32),
            route_node_ptr=np.asarray(rnp,dtype=np.int64),
            route_node_idx=np.concatenate(rn).astype(np.int32),
        )
    np.savez_compressed(args.out, **payload)

    summary = {
        "dataset": args.dataset,
        "cases": args.cases,
        "passengers": 2,
        "split_counts": {
            "train": int((split==0).sum()), "validation": int((split==1).sum()),
            "test": int((split==2).sum())
        },
        "candidate_count_per_event": summarize(cand),
        "candidate_group_radius_m": summarize(radii),
        "driver_to_exact_pickup_directed_road_m": summarize(dproad),
        "exact_pickup_to_own_dropoff_directed_road_m": summarize(odroad),
        "exact_route_directed_road_m": summarize(O),
        "output": str(args.out),
        "protocol": {
            "event_semantics": "0=P0,1=D0,2=P1,3=D1",
            "pickup_ratio": 0.001,
            "dropoff_ratio": 0.999,
            "driver_ratio": 0.5,
            "precedence": "each pickup precedes its own dropoff",
            "primary_goal": "graph-scale reproduction with two passengers fixed",
            "full_route_supervision": bool(args.predecessor is not None),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print("CROSSGRAPH_GENERATED", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
