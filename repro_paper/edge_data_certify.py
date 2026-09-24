from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


HISTORICAL_REF = "d8f8d5bc00ad1c605adf1c73c92c815cf12c7065"
FILES = [
    "chengdu_order_1000.txt",
    "chengdu_label_1000.txt",
    "chengdu_link.txt",
    "chengdu_node.txt",
]


def recover(repo: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = cache / name
        if dst.exists():
            continue
        spec = f"{HISTORICAL_REF}:sim_data/{name}"
        with dst.open("wb") as f:
            cp = subprocess.run(
                ["git", "-C", str(repo), "show", spec],
                stdout=f,
                stderr=subprocess.PIPE,
                check=False,
            )
        if cp.returncode != 0:
            dst.unlink(missing_ok=True)
            raise RuntimeError(cp.stderr.decode("utf-8", errors="replace"))


def parse_orders(path: Path) -> dict[int, dict]:
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    out = {}
    for i in range(0, len(lines), 3):
        cid = int(lines[i])
        parts = [x for x in lines[i + 1].split(";") if x]
        rparts = [x for x in lines[i + 2].split(";") if x]
        groups = [[int(v) for v in x.split(",") if v] for x in parts[1:]]
        ratios = [[float(v) for v in x.split(",") if v] for x in rparts[1:]]
        if len(groups) != len(ratios) or any(len(a) != len(b) for a, b in zip(groups, ratios)):
            raise ValueError(f"candidate/ratio mismatch in case {cid}")
        out[cid] = {
            "car_edge": int(parts[0]),
            "car_ratio": float(rparts[0]),
            "groups": groups,
            "ratios": ratios,
        }
    return out


def parse_labels(path: Path) -> dict[int, dict]:
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    if len(lines) % 8:
        raise ValueError("label file does not contain fixed 8-line records")
    out = {}
    for i in range(0, len(lines), 8):
        cid = int(lines[i])
        seq = ast.literal_eval(lines[i + 3])
        selected = [int(v) for v in lines[i + 4].split(";") if v]
        node_paths = [
            [int(v) for v in part.split(",") if v]
            for part in lines[i + 5].split(";")
            if part
        ]
        edge_paths = [
            [int(v) for v in part.split(",") if v]
            for part in lines[i + 6].split(";")
            if part
        ]
        if not lines[i + 7].startswith("route_length:"):
            raise ValueError(f"missing route length in case {cid}")
        route_length = float(lines[i + 7].split(":", 1)[1])
        out[cid] = {
            "seq": seq,
            "selected": selected,
            "node_paths": node_paths,
            "edge_paths": edge_paths,
            "route_length": route_length,
        }
    return out


def build_graph(link_path: Path):
    links = pd.read_csv(
        link_path,
        sep=r"\s+",
        header=None,
        names=["link", "u", "v", "length", "flag"],
    )
    edge = {
        int(r.link): (int(r.u), int(r.v), float(r.length))
        for r in links.itertuples(index=False)
    }
    node_ids = sorted(set(links.u.astype(int)).union(set(links.v.astype(int))))
    index = {v: i for i, v in enumerate(node_ids)}
    rows = np.fromiter((index[int(v)] for v in links.u), dtype=int, count=len(links))
    cols = np.fromiter((index[int(v)] for v in links.v), dtype=int, count=len(links))
    weights = links.length.to_numpy(float)
    graph = csr_matrix((weights, (rows, cols)), shape=(len(node_ids), len(node_ids)))
    dist = dijkstra(graph, directed=True, return_predecessors=False)
    return edge, index, dist


def point_distance(edge: dict, index: dict, dist: np.ndarray, a: tuple[int, float], b: tuple[int, float]) -> float:
    ea, ra = a
    eb, rb = b
    ua, va, wa = edge[ea]
    ub, vb, wb = edge[eb]
    generic = (1.0 - ra) * wa + dist[index[va], index[ub]] + rb * wb
    if ea == eb and rb >= ra:
        return float(min((rb - ra) * wa, generic))
    return float(generic)


def candidate_point(order: dict, group: int, edge_id: int) -> tuple[int, float]:
    hits = [j for j, e in enumerate(order["groups"][group]) if e == edge_id]
    if not hits:
        raise KeyError((group, edge_id))
    # Duplicate edge IDs inside one candidate set have identical ratio in the historical data.
    ratios = [order["ratios"][group][j] for j in hits]
    if max(ratios) - min(ratios) > 1e-12:
        raise ValueError(f"same edge has multiple ratios: {group=} {edge_id=} {ratios=}")
    return edge_id, ratios[0]


def possible_type_maps(order: dict, label: dict) -> list[dict[int, int]]:
    # For the two-passenger records, groups 0/1 are pickups and groups 2/3 are drop-offs.
    # Passenger identity can be permuted independently in the pickup and drop-off pair.
    candidates = []
    for pickup_perm in [(0, 1), (1, 0)]:
        for drop_perm in [(2, 3), (3, 2)]:
            mapping = {
                0: pickup_perm[0],
                2: pickup_perm[1],
                1: drop_perm[0],
                3: drop_perm[1],
            }
            ok = True
            for t, edge_id in zip(label["seq"][1:], label["selected"][1:]):
                if edge_id not in order["groups"][mapping[int(t)]]:
                    ok = False
                    break
            if ok:
                candidates.append(mapping)
    return candidates


def legal_group_orders(mapping: dict[int, int]) -> list[tuple[int, ...]]:
    # Types 0/1 are pickup/drop for passenger 1; 2/3 for passenger 2.
    g0, g1, g2, g3 = mapping[0], mapping[1], mapping[2], mapping[3]
    return [
        p
        for p in itertools.permutations(range(4))
        if p.index(g0) < p.index(g1) and p.index(g2) < p.index(g3)
    ]


def exact_optimum(order: dict, mapping: dict[int, int], edge: dict, index: dict, dist: np.ndarray) -> float:
    points = [
        [(eid, ratio) for eid, ratio in zip(order["groups"][g], order["ratios"][g])]
        for g in range(4)
    ]
    car = (order["car_edge"], order["car_ratio"])
    best = math.inf

    for gorder in legal_group_orders(mapping):
        a, b, c, d = [points[g] for g in gorder]
        c0 = np.asarray([point_distance(edge, index, dist, car, x) for x in a])[:, None, None, None]
        c1 = np.asarray([[point_distance(edge, index, dist, x, y) for y in b] for x in a])[:, :, None, None]
        c2 = np.asarray([[point_distance(edge, index, dist, x, y) for y in c] for x in b])[None, :, :, None]
        c3 = np.asarray([[point_distance(edge, index, dist, x, y) for y in d] for x in c])[None, None, :, :]
        best = min(best, float(np.min(c0 + c1 + c2 + c3)))
    return best


def published_point_length(order: dict, label: dict, mapping: dict[int, int], edge: dict, index: dict, dist: np.ndarray) -> float:
    points = [(order["car_edge"], order["car_ratio"])]
    for t, edge_id in zip(label["seq"][1:], label["selected"][1:]):
        points.append(candidate_point(order, mapping[int(t)], edge_id))
    return float(sum(point_distance(edge, index, dist, points[i], points[i + 1]) for i in range(len(points) - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--cache", type=Path, default=Path("/workspace/.server-control/litegd-edge-cache"))
    ap.add_argument("--limit", type=int, default=950, help="Only the first 950 two-passenger records are certified by default.")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    repo = args.repo.resolve()
    recover(repo, args.cache)
    orders = parse_orders(args.cache / "chengdu_order_1000.txt")
    labels = parse_labels(args.cache / "chengdu_label_1000.txt")
    edge, index, dist = build_graph(args.cache / "chengdu_link.txt")

    map_counts = Counter()
    ambiguous = 0
    no_map = 0
    internal_path_errors = []
    point_route_errors = []
    exact = 0
    certified = 0
    opt_gaps = []
    rows = []

    edge_length = {eid: values[2] for eid, values in edge.items()}

    for cid in range(args.limit):
        order = orders[cid]
        label = labels[cid]
        if len(order["groups"]) != 4 or len(label["seq"]) != 5:
            raise ValueError(f"case {cid} is not a two-passenger record")

        maps = possible_type_maps(order, label)
        if not maps:
            no_map += 1
            continue
        if len(maps) > 1:
            ambiguous += 1

        # The complete edge paths provide an independent internal consistency check.
        path_len = sum(edge_length[eid] for segment in label["edge_paths"] for eid in segment)
        internal_path_errors.append(abs(path_len - label["route_length"]))

        # If several passenger-identity maps remain possible because the selected edge is duplicated,
        # certify against every compatible interpretation. Keep the one whose point-distance route best
        # matches the published route length.
        map_results = []
        for mapping in maps:
            pl = published_point_length(order, label, mapping, edge, index, dist)
            opt = exact_optimum(order, mapping, edge, index, dist)
            map_results.append((abs(pl - label["route_length"]), pl, opt, mapping))
        map_results.sort(key=lambda x: x[0])
        route_err, pl, opt, mapping = map_results[0]

        map_key = tuple(mapping[t] for t in (0, 1, 2, 3))
        map_counts[map_key] += 1
        point_route_errors.append(route_err)
        gap = (label["route_length"] - opt) / max(opt, 1e-12) * 100.0
        opt_gaps.append(gap)
        tol = max(1e-4, 1e-7 * max(1.0, opt))
        is_exact = abs(label["route_length"] - opt) <= tol
        exact += int(is_exact)
        certified += 1
        rows.append(
            {
                "case_id": cid,
                "compatible_type_maps": len(maps),
                "mapping_0_1_2_3_to_group": list(map_key),
                "published_length": label["route_length"],
                "recomputed_published_length": pl,
                "exact_optimum": opt,
                "published_gap_pct": gap,
                "exact": bool(is_exact),
            }
        )

    summary = {
        "limit": args.limit,
        "certified_cases": certified,
        "no_semantic_mapping_cases": no_map,
        "ambiguous_mapping_cases": ambiguous,
        "semantic_map_counts": {str(k): v for k, v in map_counts.items()},
        "complete_edge_path_mean_abs_error": float(np.mean(internal_path_errors)) if internal_path_errors else math.nan,
        "complete_edge_path_max_abs_error": float(np.max(internal_path_errors)) if internal_path_errors else math.nan,
        "candidate_point_route_mean_abs_error": float(np.mean(point_route_errors)) if point_route_errors else math.nan,
        "candidate_point_route_max_abs_error": float(np.max(point_route_errors)) if point_route_errors else math.nan,
        "exact_optimum_count": exact,
        "exact_optimum_fraction": float(exact / certified) if certified else 0.0,
        "published_vs_exact_gap_mean_pct": float(np.mean(opt_gaps)) if opt_gaps else math.nan,
        "published_vs_exact_gap_p95_pct": float(np.percentile(opt_gaps, 95)) if opt_gaps else math.nan,
        "published_vs_exact_gap_max_pct": float(np.max(opt_gaps)) if opt_gaps else math.nan,
    }
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    print("WORST")
    for row in sorted(rows, key=lambda r: r["published_gap_pct"], reverse=True)[:20]:
        print(json.dumps(row, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "cases": rows}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
