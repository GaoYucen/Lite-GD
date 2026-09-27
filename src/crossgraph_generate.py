#!/usr/bin/env python3
"""Generate scientifically matched cross-graph Lite-GD carpool benchmarks.

The generator changes road-graph scale while preserving the recovered Chengdu
workload's main physical characteristics:
  * two passengers (first scalability stage);
  * empirical pickup/dropoff candidate-count distributions;
  * point-on-directed-edge representation (edge, ratio);
  * empirical event spatial spans and candidate-group radii;
  * exact pickup-before-own-dropoff labels.

Jinan/Shenzhen can be labeled with SciPy directed Dijkstra for pilot datasets or
an all-pairs matrix for the full benchmark. FLA generation is separated from
large-scale exact CH labeling; the generated candidate instances are identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


EARTH_M = 6371000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_protocol(path: Path) -> dict:
    d = json.loads(path.read_text())
    if d.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    return d


def sample_hist(rng: np.random.Generator, hist: dict) -> float:
    vals = np.asarray([float(k) for k in hist], dtype=np.float64)
    w = np.asarray([float(hist[k]) for k in hist], dtype=np.float64)
    w /= w.sum()
    return float(rng.choice(vals, p=w))


def sample_quantile(rng: np.random.Generator, spec: dict) -> float:
    q = np.asarray(spec["q"], dtype=np.float64)
    v = np.asarray(spec["v"], dtype=np.float64)
    return float(np.interp(rng.random(), q, v))


def lonlat_to_local_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])
    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))
    x = EARTH_M * math.cos(lat0) * (lon - lon0)
    y = EARTH_M * (lat - lat0)
    return np.column_stack((x, y))


def local_xy_to_lonlat(xy: np.ndarray, ref_lonlat: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    ref = np.asarray(ref_lonlat, dtype=np.float64)
    lon0 = math.radians(float(np.mean(ref[:, 0])))
    lat0 = math.radians(float(np.mean(ref[:, 1])))
    lon = lon0 + xy[:, 0] / (EARTH_M * math.cos(lat0))
    lat = lat0 + xy[:, 1] / EARTH_M
    return np.column_stack((np.rad2deg(lon), np.rad2deg(lat)))


@dataclass
class RoadGraph:
    name: str
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    node_xy: np.ndarray
    node_lonlat: np.ndarray
    original_node_ids: np.ndarray

    def __post_init__(self):
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        self.node_xy = np.asarray(self.node_xy, dtype=np.float64)
        self.node_lonlat = np.asarray(self.node_lonlat, dtype=np.float64)
        self.original_node_ids = np.asarray(self.original_node_ids, dtype=np.int64)
        if not (len(self.src) == len(self.dst) == len(self.weight)):
            raise ValueError("edge array shape mismatch")
        if np.any(self.weight <= 0):
            raise ValueError("non-positive road weight")
        n = len(self.node_xy)
        if np.any(self.src < 0) or np.any(self.src >= n) or np.any(self.dst < 0) or np.any(self.dst >= n):
            raise ValueError("edge endpoint outside node range")
        self.edge_mid_xy = (self.node_xy[self.src] + self.node_xy[self.dst]) / 2.0
        self.edge_tree = cKDTree(self.edge_mid_xy)
        self.adj = csr_matrix((self.weight, (self.src, self.dst)), shape=(n, n))

    @property
    def n_nodes(self):
        return len(self.node_xy)

    @property
    def n_edges(self):
        return len(self.src)

    def point_xy(self, edge: int, ratio: float) -> np.ndarray:
        return (1.0 - ratio) * self.node_xy[self.src[edge]] + ratio * self.node_xy[self.dst[edge]]

    def point_lonlat(self, edge: int, ratio: float) -> np.ndarray:
        return (1.0 - ratio) * self.node_lonlat[self.src[edge]] + ratio * self.node_lonlat[self.dst[edge]]


def load_city_graph(
    name: str,
    protocol_npz: Path,
    raw_nodes: Path,
    raw_edges: Path,
) -> RoadGraph:
    z = np.load(protocol_npz, mmap_mode="r")
    # The distance project's certified protocol stores local UTM coordinates
    # in metres (graph mean subtracted), not lon/lat.
    coords_xy = np.asarray(z["coordinates"], dtype=np.float64)
    original = np.asarray(z["original_node_ids"], dtype=np.int64)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2 or len(coords_xy) != len(original):
        raise ValueError("invalid projected protocol coordinates")

    # Recover model-facing WGS84 lon/lat exactly from the native node table.
    lonlat_by_id: dict[int, tuple[float, float]] = {}
    with raw_nodes.open(newline="") as f:
        for row in csv.DictReader(f):
            lonlat_by_id[int(row["NodeID"])] = (
                float(row["Longitude"]),
                float(row["Latitude"]),
            )
    try:
        node_lonlat = np.asarray([lonlat_by_id[int(v)] for v in original], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"protocol original node missing from native node table: {exc}") from exc
    if not (
        np.all(np.abs(node_lonlat[:, 0]) <= 180.0)
        and np.all(np.abs(node_lonlat[:, 1]) <= 90.0)
    ):
        raise ValueError("invalid recovered lon/lat")

    max_id = int(original.max())
    mapping = np.full(max_id + 1, -1, dtype=np.int64)
    mapping[original] = np.arange(len(original), dtype=np.int64)
    best: dict[tuple[int, int], float] = {}
    with raw_edges.open(newline="") as f:
        for row in csv.DictReader(f):
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
    src = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
    dst = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
    weight = np.fromiter(best.values(), dtype=np.float64, count=len(best))
    return RoadGraph(
        name=name,
        src=src,
        dst=dst,
        weight=weight,
        node_xy=coords_xy,
        node_lonlat=node_lonlat,
        original_node_ids=original,
    )


def load_fla_graph(graph_npz: Path) -> RoadGraph:
    z = np.load(graph_npz, mmap_mode="r")
    xy = np.asarray(z["coordinates"], dtype=np.float64)
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("FLA generation requires pyproj for EPSG:5070 -> WGS84") from exc
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
    lonlat = np.column_stack((np.asarray(lon), np.asarray(lat)))
    return RoadGraph(
        name="FLA",
        src=z["src"],
        dst=z["dst"],
        weight=z["weight"],
        node_xy=xy,
        node_lonlat=lonlat,
        original_node_ids=z["original_node_ids"],
    )


def choose_target_edge(
    graph: RoadGraph,
    rng: np.random.Generator,
    base_xy: np.ndarray,
    target_distance_m: float,
    *,
    attempts: int = 48,
) -> int:
    # Try multiple directions so boundary points do not collapse toward a city edge.
    best = None
    for _ in range(attempts):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        target = base_xy + target_distance_m * np.array([math.cos(theta), math.sin(theta)])
        dist, edge = graph.edge_tree.query(target, k=1)
        cand = (float(dist), int(edge))
        if best is None or cand < best:
            best = cand
        # Accept if snapping error is small relative to the requested OD span.
        if dist <= max(750.0, 0.12 * max(target_distance_m, 1.0)):
            return int(edge)
    assert best is not None
    return int(best[1])


def choose_candidate_edges(
    graph: RoadGraph,
    rng: np.random.Generator,
    center_edge: int,
    count: int,
    radius_m: float,
) -> np.ndarray:
    center = graph.edge_mid_xy[center_edge]
    ids = graph.edge_tree.query_ball_point(center, r=max(radius_m, 1.0))
    ids = np.asarray(sorted(set(map(int, ids))), dtype=np.int64)
    if len(ids) < count:
        _, near = graph.edge_tree.query(center, k=min(max(count, 1), graph.n_edges))
        ids = np.unique(np.atleast_1d(near).astype(np.int64))
    if len(ids) < count:
        raise RuntimeError("insufficient candidate edges")
    # Keep the center edge, sample the rest uniformly from the local road neighborhood.
    remain = ids[ids != center_edge]
    if count == 1:
        return np.asarray([center_edge], dtype=np.int64)
    if len(remain) < count - 1:
        _, near = graph.edge_tree.query(center, k=min(max(count * 3, count), graph.n_edges))
        remain = np.unique(np.atleast_1d(near).astype(np.int64))
        remain = remain[remain != center_edge]
    chosen = rng.choice(remain, size=count - 1, replace=False)
    out = np.concatenate(([center_edge], np.asarray(chosen, dtype=np.int64)))
    rng.shuffle(out)
    return out


class PointDistanceOracle:
    def __init__(self, graph: RoadGraph, allpairs: Optional[Path] = None):
        self.graph = graph
        self.allpairs = None
        if allpairs is not None:
            self.allpairs = np.load(allpairs, mmap_mode="r")
            if self.allpairs.shape != (graph.n_nodes, graph.n_nodes):
                raise ValueError(
                    f"allpairs shape {self.allpairs.shape} != {(graph.n_nodes, graph.n_nodes)}"
                )

    def node_distances(self, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        sources = np.asarray(sources, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        if self.allpairs is not None:
            return np.asarray(self.allpairs[sources, targets], dtype=np.float64)

        uniq, inv = np.unique(sources, return_inverse=True)
        D = dijkstra(self.graph.adj, directed=True, indices=uniq)
        return np.asarray(D[inv, targets], dtype=np.float64)

    def point_matrix(self, edges: np.ndarray, ratios: np.ndarray) -> np.ndarray:
        g = self.graph
        edges = np.asarray(edges, dtype=np.int64)
        ratios = np.asarray(ratios, dtype=np.float64)
        n = len(edges)
        src_nodes = g.dst[edges][:, None].repeat(n, axis=1)
        dst_nodes = g.src[edges][None, :].repeat(n, axis=0)
        middle = self.node_distances(src_nodes.ravel(), dst_nodes.ravel()).reshape(n, n)
        out = (
            (1.0 - ratios)[:, None] * g.weight[edges][:, None]
            + middle
            + ratios[None, :] * g.weight[edges][None, :]
        )
        same = edges[:, None] == edges[None, :]
        forward = ratios[None, :] >= ratios[:, None]
        direct = (ratios[None, :] - ratios[:, None]) * g.weight[edges][:, None]
        out = np.where(same & forward, np.minimum(out, direct), out)
        np.fill_diagonal(out, 0.0)
        return out


def exact_precedence_dp(event_indices: list[np.ndarray], point_cost: np.ndarray) -> tuple[float, list[int]]:
    n_events = len(event_indices)
    # State key: (event mask, last point index); value: (cost, selected sequence)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
    for _ in range(n_events):
        nxt = {}
        for (mask, last), (cost, seq) in states.items():
            for ev in range(n_events):
                if mask & (1 << ev):
                    continue
                if ev % 2 == 1 and not (mask & (1 << (ev - 1))):
                    continue
                new_mask = mask | (1 << ev)
                for point_idx in event_indices[ev]:
                    z = cost + float(point_cost[last, int(point_idx)])
                    key = (new_mask, int(point_idx))
                    old = nxt.get(key)
                    if old is None or z < old[0] - 1e-12:
                        nxt[key] = (z, seq + (int(point_idx),))
        states = nxt
    if not states:
        raise RuntimeError("exact DP found no legal route")
    best = min(states.values(), key=lambda x: (x[0], x[1]))
    return float(best[0]), list(best[1])


def generate_one(graph: RoadGraph, protocol: dict, rng: np.random.Generator) -> dict:
    h_pick = protocol["candidate_count_histograms"]["pickup"]
    h_drop = protocol["candidate_count_histograms"]["dropoff"]
    quant = protocol["empirical_quantiles_m"]

    driver_edge = int(rng.integers(0, graph.n_edges))
    driver_ratio = 0.001
    driver_xy = graph.point_xy(driver_edge, driver_ratio)

    groups = []
    centers = []
    for passenger in range(2):
        dp = sample_quantile(rng, quant["driver_to_pickup_haversine"])
        pickup_center_edge = choose_target_edge(graph, rng, driver_xy, dp)
        pickup_center_xy = graph.edge_mid_xy[pickup_center_edge]
        dd = sample_quantile(rng, quant["pickup_to_dropoff_haversine"])
        drop_center_edge = choose_target_edge(graph, rng, pickup_center_xy, dd)
        centers.extend([pickup_center_edge, drop_center_edge])

    # Event order is semantic: P0,D0,P1,D1 -> event ids 0,1,2,3.
    for ev, center_edge in enumerate(centers):
        is_pick = ev % 2 == 0
        count = int(sample_hist(rng, h_pick if is_pick else h_drop))
        radius = sample_quantile(rng, quant["candidate_group_radius"])
        edges = choose_candidate_edges(graph, rng, int(center_edge), count, radius)
        if is_pick:
            ratios = np.full(count, 0.001, dtype=np.float64)
        else:
            # Preserve the recovered empirical anomaly rate rather than silently
            # cleaning the historical distribution.
            p_low = 388.0 / (388.0 + 15066.0)
            ratios = np.where(rng.random(count) < p_low, 0.001, 0.999).astype(np.float64)
        groups.append((edges, ratios))

    return {
        "driver_edge": driver_edge,
        "driver_ratio": driver_ratio,
        "groups": groups,
    }


def flatten_case(case: dict):
    edges = [int(case["driver_edge"])]
    ratios = [float(case["driver_ratio"])]
    events = [-1]
    local = [-1]
    event_indices = []
    for ev, (ee, rr) in enumerate(case["groups"]):
        idx = []
        for li, (e, r) in enumerate(zip(ee, rr)):
            idx.append(len(edges))
            edges.append(int(e))
            ratios.append(float(r))
            events.append(ev)
            local.append(li)
        event_indices.append(np.asarray(idx, dtype=np.int64))
    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(ratios, dtype=np.float64),
        np.asarray(events, dtype=np.int64),
        np.asarray(local, dtype=np.int64),
        event_indices,
    )


def certify_and_pack(
    graph: RoadGraph,
    protocol: dict,
    rng: np.random.Generator,
    count: int,
    oracle: PointDistanceOracle,
):
    max_k = max(int(k) for x in protocol["candidate_count_histograms"].values() for k in x)
    candidate_edges = np.full((count, 4, max_k), -1, dtype=np.int32)
    candidate_ratios = np.zeros((count, 4, max_k), dtype=np.float32)
    candidate_valid = np.zeros((count, 4, max_k), dtype=bool)
    driver_edge = np.zeros(count, dtype=np.int32)
    driver_ratio = np.full(count, 0.001, dtype=np.float32)
    exact_event_sequence = np.zeros((count, 4), dtype=np.int8)
    exact_local_indices = np.zeros((count, 4), dtype=np.int8)
    exact_edges = np.zeros((count, 4), dtype=np.int32)
    exact_ratios = np.zeros((count, 4), dtype=np.float32)
    exact_length = np.zeros(count, dtype=np.float64)

    driver_pick_xy = []
    pickup_drop_xy = []
    group_radii = []
    event_counts = []

    for i in range(count):
        case = generate_one(graph, protocol, rng)
        driver_edge[i] = case["driver_edge"]
        for ev, (ee, rr) in enumerate(case["groups"]):
            k = len(ee)
            candidate_edges[i, ev, :k] = ee
            candidate_ratios[i, ev, :k] = rr
            candidate_valid[i, ev, :k] = True
            event_counts.append(k)
            pts = np.stack([graph.point_xy(int(e), float(r)) for e, r in zip(ee, rr)])
            center = pts.mean(axis=0)
            group_radii.append(float(np.linalg.norm(pts - center, axis=1).max()))

        edges, ratios, events, local, event_indices = flatten_case(case)
        point_cost = oracle.point_matrix(edges, ratios)
        if not np.isfinite(point_cost).all():
            raise RuntimeError(f"nonfinite point distance in case {i}")
        opt, seq = exact_precedence_dp(event_indices, point_cost)
        exact_length[i] = opt
        prev_mask = 0
        for t, idx in enumerate(seq):
            ev = int(events[idx])
            exact_event_sequence[i, t] = ev
            exact_local_indices[i, t] = int(local[idx])
            exact_edges[i, t] = int(edges[idx])
            exact_ratios[i, t] = float(ratios[idx])
            if ev % 2 == 1 and not (prev_mask & (1 << (ev - 1))):
                raise AssertionError("precedence violation")
            prev_mask |= 1 << ev
        if prev_mask != 0b1111:
            raise AssertionError("incomplete exact event mask")

        dxy = graph.point_xy(case["driver_edge"], case["driver_ratio"])
        selected = {int(events[idx]): idx for idx in seq}
        for p in range(2):
            pi = selected[2 * p]
            di = selected[2 * p + 1]
            pxy = graph.point_xy(int(edges[pi]), float(ratios[pi]))
            qxy = graph.point_xy(int(edges[di]), float(ratios[di]))
            driver_pick_xy.append(float(np.linalg.norm(pxy - dxy)))
            pickup_drop_xy.append(float(np.linalg.norm(qxy - pxy)))

        if (i + 1) % max(1, min(100, count // 10 or 1)) == 0:
            print(f"GENERATED {i+1}/{count}", flush=True)

    perm = rng.permutation(count)
    ntr = int(0.8 * count)
    nva = int(0.1 * count)
    split = np.full(count, 2, dtype=np.int8)
    split[perm[:ntr]] = 0
    split[perm[ntr:ntr+nva]] = 1

    arrays = {
        "driver_edge": driver_edge,
        "driver_ratio": driver_ratio,
        "candidate_edges": candidate_edges,
        "candidate_ratios": candidate_ratios,
        "candidate_valid": candidate_valid,
        "exact_event_sequence": exact_event_sequence,
        "exact_local_indices": exact_local_indices,
        "exact_edges": exact_edges,
        "exact_ratios": exact_ratios,
        "exact_length": exact_length,
        "split": split,
        "node_lonlat": graph.node_lonlat.astype(np.float32),
        "node_xy_m": graph.node_xy.astype(np.float32),
        "src": graph.src.astype(np.int32),
        "dst": graph.dst.astype(np.int32),
        "weight": graph.weight.astype(np.float64),
        "original_node_ids": graph.original_node_ids.astype(np.int64),
    }
    cert = {
        "cases": count,
        "graph": graph.name,
        "nodes": graph.n_nodes,
        "directed_edges": graph.n_edges,
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
        },
        "candidate_count": summarize(event_counts),
        "candidate_group_radius_m": summarize(group_radii),
        "exact_driver_to_pickup_euclidean_m": summarize(driver_pick_xy),
        "exact_pickup_to_dropoff_euclidean_m": summarize(pickup_drop_xy),
        "exact_route_length": summarize(exact_length),
        "nonfinite_exact_lengths": int(np.sum(~np.isfinite(exact_length))),
        "nonpositive_exact_lengths": int(np.sum(exact_length <= 0)),
    }
    return arrays, cert


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("jinan", "shenzhen", "fla"), required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-npz", type=Path)
    ap.add_argument("--raw-edges", type=Path)
    ap.add_argument("--graph-npz", type=Path)
    ap.add_argument("--allpairs", type=Path)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("output/report already exists")
    protocol = load_protocol(args.protocol)
    rng = np.random.default_rng(args.seed)

    if args.dataset == "fla":
        if args.graph_npz is None:
            raise ValueError("--graph-npz required for FLA")
        graph = load_fla_graph(args.graph_npz)
    else:
        if args.protocol_npz is None or args.raw_edges is None:
            raise ValueError("--protocol-npz and --raw-edges required for city graphs")
        graph = load_city_graph(args.dataset, args.protocol_npz, args.raw_edges)

    if args.dataset == "fla" and args.allpairs is None:
        raise ValueError(
            "FLA exact labels must be produced with the RoutingKit-CH stage; "
            "do not run million-node SciPy Dijkstra from this generator"
        )

    oracle = PointDistanceOracle(graph, args.allpairs)
    arrays, cert = certify_and_pack(graph, protocol, rng, args.count, oracle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    report = {
        "status": "completed_crossgraph_pilot_exact_labels",
        "dataset": args.dataset,
        "seed": args.seed,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "exact_oracle": (
            "precomputed directed all-pairs matrix"
            if args.allpairs is not None
            else "per-case SciPy directed Dijkstra"
        ),
        "certification": cert,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("CROSSGRAPH_GENERATED", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
