#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from historical_full_model import HistoricalExact
from evaluation import summarize_route_rows


def _pct(samples_ns):
    x=np.asarray(samples_ns,dtype=np.float64)/1e6
    return {
        "samples":int(len(x)),
        "mean_ms":float(np.mean(x)),
        "median_ms":float(np.median(x)),
        "p95_ms":float(np.percentile(x,95)),
        "p99_ms":float(np.percentile(x,99)),
        "min_ms":float(np.min(x)),
        "max_ms":float(np.max(x)),
    }


def pairwise(x,data):
    flat=x["flat"];n=len(flat)
    d=np.zeros((n,n),dtype=np.float64)
    for i,(ea,ra,_,_) in enumerate(flat):
        for j,(eb,rb,_,_) in enumerate(flat):
            if i!=j:
                d[i,j]=data.point_dist(ea,ra,eb,rb)
    return d


def exact_dp(x,d):
    flat=x["flat"];events=sorted({int(z[2]) for z in flat if int(z[2])>=0})
    by_event={e:[j for j,z in enumerate(flat) if int(z[2])==e] for e in events}
    dp={(0,0):0.0}
    steps=int(x["n_events"])
    for _ in range(steps):
        nd={}
        for (mask,last),cost in dp.items():
            for e in events:
                if mask>>e & 1: continue
                if e%2==1 and not (mask>>(e-1)&1): continue
                nm=mask|(1<<e)
                for j in by_event[e]:
                    z=cost+d[last,j]
                    k=(nm,j)
                    if k not in nd or z<nd[k]:
                        nd[k]=z
        dp=nd
    return min(dp.values())


def nearest_feasible_greedy(x,d):
    flat=x["flat"];events=sorted({int(z[2]) for z in flat if int(z[2])>=0})
    by_event={e:[j for j,z in enumerate(flat) if int(z[2])==e] for e in events}
    done=set();last=0;cost=0.0
    for _ in range(int(x["n_events"])):
        feasible=[e for e in events if e not in done and (e%2==0 or (e-1) in done)]
        best=None
        for e in feasible:
            for j in by_event[e]:
                z=float(d[last,j])
                key=(z,e,j)
                if best is None or key<best:
                    best=key
        z,e,j=best
        cost+=z;done.add(e);last=j
    return cost


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seed",type=int,default=20260925);ap.add_argument("--rounds",type=int,default=20)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()

    t0=time.perf_counter()
    data=HistoricalExact(a.links,a.orders,a.labels,a.exact)
    init_s=time.perf_counter()-t0
    _,_,te=data.split(a.seed)
    cases=[data.case_tensors(cid) for cid in te]

    # Validate the implementations once before timing.
    exact_rows=[];greedy_rows=[]
    for x in cases:
        d=pairwise(x,data)
        le=exact_dp(x,d);lg=nearest_feasible_greedy(x,d)
        exact_rows.append({"pred":le,"opt":x["opt"]})
        greedy_rows.append({"pred":lg,"opt":x["opt"]})
    exact_quality=summarize_route_rows(exact_rows)
    greedy_quality=summarize_route_rows(greedy_rows)
    if exact_quality["objective_accuracy"] < 99.999:
        raise RuntimeError(f"exact DP validation failed: {exact_quality}")

    # Warmup.
    for x in cases[:min(10,len(cases))]:
        d=pairwise(x,data); exact_dp(x,d); nearest_feasible_greedy(x,d)

    pair_ns=[];exact_ns=[];greedy_ns=[];exact_total_ns=[];greedy_total_ns=[]
    for _ in range(a.rounds):
        for x in cases:
            q0=time.perf_counter_ns();d=pairwise(x,data);q1=time.perf_counter_ns()
            e0=time.perf_counter_ns();exact_dp(x,d);e1=time.perf_counter_ns()
            g0=time.perf_counter_ns();nearest_feasible_greedy(x,d);g1=time.perf_counter_ns()
            pair_ns.append(q1-q0);exact_ns.append(e1-e0);greedy_ns.append(g1-g0)
            exact_total_ns.append((q1-q0)+(e1-e0))
            greedy_total_ns.append((q1-q0)+(g1-g0))

    result={
        "seed":a.seed,
        "test_cases":len(cases),
        "rounds":a.rounds,
        "offline_data_graph_init_s":init_s,
        "scope":{
            "offline":"HistoricalExact initialization includes road APSP and is excluded from online latency",
            "pairwise":"per-request directed point-on-edge distance table from precomputed APSP",
            "algorithm":"search/greedy after pairwise table exists",
            "online_total":"pairwise distance table + route-selection algorithm",
        },
        "exact":{
            "quality":exact_quality,
            "pairwise":_pct(pair_ns),
            "algorithm":_pct(exact_ns),
            "online_total":_pct(exact_total_ns),
        },
        "greedy":{
            "quality":greedy_quality,
            "pairwise":_pct(pair_ns),
            "algorithm":_pct(greedy_ns),
            "online_total":_pct(greedy_total_ns),
        },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2))
    print("RUNTIME_BASELINES",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
