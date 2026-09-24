#!/usr/bin/env python3
"""Recover the deleted pre-publication Chengdu Lite-GD samples.

The original repository once contained:
  * chengdu_link_feature.txt
  * chengdu_order_1000.txt
  * chengdu_label_1000.txt
but the derived chengdu_case_feature.txt used by MCRP_Net.py was never
committed.  This module reconstructs its semantics deterministically from the
historical source files without relying on the broken prototype model.

Event type convention recovered from the code/paper:
  -1: driver
   0/1: passenger-1 pickup/dropoff
   2/3: passenger-2 pickup/dropoff
   4/5: passenger-3 pickup/dropoff

For edge pretraining we expose both the paper semantic class and a contiguous
cross-entropy class:
  paper  2 = candidate AND optimal       -> ce 3
  paper  1 = optimal, not candidate      -> ce 2
  paper -1 = candidate, not optimal      -> ce 0
  paper  0 = neither candidate nor opt   -> ce 1
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


LINK_COLS = [
    "edge_id", "node_start", "lon_start", "lat_start",
    "node_end", "lon_end", "lat_end", "length",
    "is_driver_default", "ratio_default",
]


@dataclass
class Candidate:
    edge_id: int
    ratio: float
    event_type: int
    group_index: int
    local_index: int
    lon: float
    lat: float
    is_optimal_choice: bool = False


@dataclass
class Case:
    case_id: int
    driver_edge: int
    driver_ratio: float
    candidate_groups: List[List[Candidate]]
    optimal_event_sequence: List[int]
    optimal_selected_edges: List[int]
    optimal_node_route: List[int]
    optimal_edge_route: List[int]
    route_length: float

    @property
    def passenger_count(self) -> int:
        return len(self.candidate_groups) // 2

    @property
    def candidate_count(self) -> int:
        return sum(len(g) for g in self.candidate_groups)


def _split_groups(s: str, cast):
    groups = [g for g in s.strip().split(";") if g != ""]
    return [[cast(x) for x in g.split(",") if x != ""] for g in groups]


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=LINK_COLS)
    if df["edge_id"].duplicated().any():
        raise ValueError("duplicate edge ids")
    return df.set_index("edge_id", drop=False)


def load_orders(path: Path):
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    if len(lines) % 3:
        raise ValueError(f"order file has {len(lines)} nonblank lines, not divisible by 3")
    out = {}
    for i in range(0, len(lines), 3):
        cid = int(lines[i])
        edge_groups = _split_groups(lines[i + 1], int)
        ratio_groups = _split_groups(lines[i + 2], float)
        if len(edge_groups) != len(ratio_groups):
            raise ValueError(f"case {cid}: edge/ratio group mismatch")
        for a, b in zip(edge_groups, ratio_groups):
            if len(a) != len(b):
                raise ValueError(f"case {cid}: candidate edge/ratio count mismatch")
        if len(edge_groups) not in (5, 7):
            raise ValueError(f"case {cid}: expected driver + 4/6 groups, got {len(edge_groups)}")
        if len(edge_groups[0]) != 1:
            raise ValueError(f"case {cid}: driver group is not singleton")
        out[cid] = (edge_groups, ratio_groups)
    return out


def load_labels(path: Path):
    # Each record has eight nonblank semantic lines.  The file contains blank
    # separators, so stripping blank lines makes parsing stable.
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    if len(lines) % 8:
        raise ValueError(f"label file has {len(lines)} nonblank lines, not divisible by 8")
    out = {}
    for i in range(0, len(lines), 8):
        cid = int(lines[i])
        edge_groups = _split_groups(lines[i + 1], int)
        ratio_groups = _split_groups(lines[i + 2], float)
        event_seq = [int(x) for x in ast.literal_eval(lines[i + 3])]
        selected = [int(x) for x in lines[i + 4].split(";") if x != ""]
        node_segments = _split_groups(lines[i + 5], int)
        edge_segments = _split_groups(lines[i + 6], int)
        prefix = "route_length:"
        if not lines[i + 7].startswith(prefix):
            raise ValueError(f"case {cid}: missing route length")
        route_length = float(lines[i + 7][len(prefix):].strip())
        out[cid] = {
            "edge_groups": edge_groups,
            "ratio_groups": ratio_groups,
            "event_seq": event_seq,
            "selected": selected,
            "node_route": [x for seg in node_segments for x in seg],
            "edge_route": [x for seg in edge_segments for x in seg],
            "route_length": route_length,
        }
    return out


def point_on_edge(edge_row, ratio: float) -> Tuple[float, float]:
    # Paper defines ratio as distance from edge source divided by edge length.
    # Road segments are short; linear interpolation in lon/lat is adequate for
    # reconstructing the candidate coordinate that was implicit in the files.
    r = float(np.clip(ratio, 0.0, 1.0))
    lon = float(edge_row.lon_start + r * (edge_row.lon_end - edge_row.lon_start))
    lat = float(edge_row.lat_start + r * (edge_row.lat_end - edge_row.lat_start))
    return lon, lat


def recover_cases(edges: pd.DataFrame, orders, labels) -> List[Case]:
    if set(orders) != set(labels):
        raise ValueError("order/label case id sets differ")
    cases = []
    for cid in sorted(orders):
        order_edges, order_ratios = orders[cid]
        lab = labels[cid]

        # Historical label repeats the order specification.  Verify exact source
        # consistency before reconstructing any derived features.
        if order_edges != lab["edge_groups"]:
            raise ValueError(f"case {cid}: order candidate edges differ from label file")
        # float strings are simple .001/.999 but use tolerance.
        for a, b in zip(order_ratios, lab["ratio_groups"]):
            if len(a) != len(b) or not np.allclose(a, b, atol=1e-12):
                raise ValueError(f"case {cid}: order ratios differ from label file")

        driver_edge = int(order_edges[0][0])
        driver_ratio = float(order_ratios[0][0])
        event_seq = lab["event_seq"]
        selected = lab["selected"]
        if event_seq[0] != -1:
            raise ValueError(f"case {cid}: optimal sequence does not start at driver")
        if selected[0] != driver_edge:
            raise ValueError(f"case {cid}: selected sequence driver mismatch")
        if len(event_seq) != len(selected):
            raise ValueError(f"case {cid}: event/selected lengths differ")

        # event types are exactly one per candidate group after driver
        event_groups = order_edges[1:]
        ratio_groups = order_ratios[1:]
        if sorted(event_seq[1:]) != list(range(len(event_groups))):
            raise ValueError(f"case {cid}: event sequence is not a permutation")

        chosen_by_event = {}
        for event_type, edge_id in zip(event_seq[1:], selected[1:]):
            chosen_by_event[int(event_type)] = int(edge_id)

        candidate_groups: List[List[Candidate]] = []
        for event_type, (group, ratios) in enumerate(zip(event_groups, ratio_groups)):
            if chosen_by_event[event_type] not in group:
                raise ValueError(
                    f"case {cid}: chosen edge {chosen_by_event[event_type]} "
                    f"not in event group {event_type}"
                )
            g = []
            for local_index, (eid, ratio) in enumerate(zip(group, ratios)):
                row = edges.loc[int(eid)]
                lon, lat = point_on_edge(row, float(ratio))
                g.append(Candidate(
                    edge_id=int(eid), ratio=float(ratio),
                    event_type=event_type, group_index=event_type,
                    local_index=local_index, lon=lon, lat=lat,
                    is_optimal_choice=(int(eid) == chosen_by_event[event_type]),
                ))
            candidate_groups.append(g)

        cases.append(Case(
            case_id=cid,
            driver_edge=driver_edge,
            driver_ratio=driver_ratio,
            candidate_groups=candidate_groups,
            optimal_event_sequence=[int(x) for x in event_seq],
            optimal_selected_edges=[int(x) for x in selected],
            optimal_node_route=[int(x) for x in lab["node_route"]],
            optimal_edge_route=[int(x) for x in lab["edge_route"]],
            route_length=float(lab["route_length"]),
        ))
    return cases


def edge_pretrain_labels(case: Case, all_edge_ids: Sequence[int]) -> np.ndarray:
    """Return CE labels in all_edge_ids order."""
    candidate_edges = {case.driver_edge}
    for g in case.candidate_groups:
        candidate_edges.update(c.edge_id for c in g)
    optimal_edges = set(case.optimal_edge_route)
    y = np.empty(len(all_edge_ids), dtype=np.int8)
    for i, e in enumerate(all_edge_ids):
        in_m = int(e) in candidate_edges
        in_o = int(e) in optimal_edges
        if in_m and in_o:
            y[i] = 3
        elif in_o:
            y[i] = 2
        elif in_m:
            y[i] = 0
        else:
            y[i] = 1
    return y


def audit(cases: Sequence[Case], edges: pd.DataFrame):
    all_edge_ids = edges["edge_id"].to_numpy(dtype=np.int64)
    pcounts = [c.passenger_count for c in cases]
    cc = [c.candidate_count for c in cases]
    pickup_first = 0
    legal = 0
    pretrain_counts = np.zeros(4, dtype=np.int64)
    for c in cases:
        events = c.optimal_event_sequence[1:]
        n = c.passenger_count
        # Even event ids are pickup, odd ids are corresponding dropoff.
        seen_drop = False
        pf = True
        for e in events:
            if e % 2:
                seen_drop = True
            elif seen_drop:
                pf = False
        pickup_first += int(pf)
        legal += int(all(events.index(2*p) < events.index(2*p+1) for p in range(n)))
        y = edge_pretrain_labels(c, all_edge_ids)
        pretrain_counts += np.bincount(y, minlength=4)
    return {
        "cases": len(cases),
        "passenger_counts": {str(k): pcounts.count(k) for k in sorted(set(pcounts))},
        "candidate_count_min": int(min(cc)),
        "candidate_count_max": int(max(cc)),
        "candidate_count_mean": float(np.mean(cc)),
        "route_length_mean": float(np.mean([c.route_length for c in cases])),
        "route_length_std": float(np.std([c.route_length for c in cases])),
        "pickup_first_cases": pickup_first,
        "precedence_legal_cases": legal,
        "edge_pretrain_ce_counts_over_case_edges": {
            "candidate_not_opt": int(pretrain_counts[0]),
            "neither": int(pretrain_counts[1]),
            "opt_not_candidate": int(pretrain_counts[2]),
            "candidate_and_opt": int(pretrain_counts[3]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True)
    ap.add_argument("--orders", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--cases-out", default="")
    args = ap.parse_args()

    edges = load_edges(Path(args.links))
    orders = load_orders(Path(args.orders))
    labels = load_labels(Path(args.labels))
    cases = recover_cases(edges, orders, labels)
    summary = audit(cases, edges)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.cases_out:
        serial = []
        for c in cases:
            d = asdict(c)
            serial.append(d)
        Path(args.cases_out).write_text(json.dumps(serial, ensure_ascii=False))


if __name__ == "__main__":
    main()
