#!/usr/bin/env python3
"""Independent certification for generated cross-graph carpool benchmarks."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def summarize(xs):
    a=np.asarray(xs,dtype=np.float64)
    if not len(a): return {}
    return {
        "n":int(len(a)),"mean":float(a.mean()),"std":float(a.std()),
        "min":float(a.min()),"median":float(np.median(a)),
        "p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),
        "max":float(a.max())
    }


def valid_orders():
    out=[]
    for p in itertools.permutations(range(4)):
        pos={e:i for i,e in enumerate(p)}
        if pos[0]<pos[1] and pos[2]<pos[3]:
            out.append(p)
    assert len(out)==6
    return out


ORDERS=valid_orders()


def groups_for_case(z,i):
    m=int(z["point_count"][i])
    ev=z["point_event"][i,:m]
    return {e:np.flatnonzero(ev==e).astype(np.int64) for e in range(4)}


def route_cost(pc,seq):
    prev=0; total=0.0
    for cur in seq:
        total+=float(pc[prev,int(cur)])
        prev=int(cur)
    return total


def brute_exact(pc,groups):
    best=float("inf")
    for order in ORDERS:
        gl=[groups[e] for e in order]
        for seq in itertools.product(*gl):
            v=route_cost(pc,seq)
            if v<best: best=v
    return best


def greedy(pc,groups):
    done=set();cur=0;seq=[];events=[]
    for _ in range(4):
        legal=[e for e in range(4) if e not in done and (e%2==0 or e-1 in done)]
        best=None
        for e in legal:
            for j in groups[e]:
                key=(float(pc[cur,int(j)]),e,int(j))
                if best is None or key<best: best=key
        _,e,j=best
        seq.append(j);events.append(e);done.add(e);cur=j
    return seq,events,route_cost(pc,seq)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",required=True,type=Path)
    ap.add_argument("--sample-bruteforce",type=int,default=32)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    z=np.load(a.cases,mmap_mode="r")
    n=len(z["exact_length"])
    required=["point_count","point_event","point_cost","target_flat","exact_event_order",
              "exact_length","split","event_counts"]
    missing=[k for k in required if k not in z.files]
    if missing: raise ValueError(f"missing benchmark arrays: {missing}")

    target_diffs=[];brute_diffs=[];illegal=0;greedy_gaps=[];greedy_event_hits=0
    negative_greedy=0
    brute_ids=np.linspace(0,n-1,min(a.sample_bruteforce,n),dtype=int) if n else np.array([],int)
    brute_set=set(map(int,brute_ids))

    for i in range(n):
        m=int(z["point_count"][i])
        pc=np.asarray(z["point_cost"][i,:m,:m],dtype=np.float64)
        groups=groups_for_case(z,i)
        target=np.asarray(z["target_flat"][i],dtype=np.int64)
        order=list(map(int,z["exact_event_order"][i]))
        opt=float(z["exact_length"][i])
        # precedence and one occurrence of every event.
        if sorted(order)!=[0,1,2,3] or order.index(0)>order.index(1) or order.index(2)>order.index(3):
            illegal+=1
        tc=route_cost(pc,target)
        target_diffs.append(abs(tc-opt))
        if i in brute_set:
            be=brute_exact(pc,groups)
            brute_diffs.append(abs(be-opt))
        _,gev,gc=greedy(pc,groups)
        gap=(gc/opt-1.0)*100.0
        greedy_gaps.append(gap)
        if gap < -1e-5: negative_greedy+=1
        greedy_event_hits += sum(int(x==y) for x,y in zip(gev,order))

    split=np.asarray(z["split"])
    split_counts={name:int(np.sum(split==code)) for name,code in [("train",0),("validation",1),("test",2)]}
    result={
        "cases":n,
        "split_counts":split_counts,
        "illegal_exact_event_orders":illegal,
        "target_cost_abs_diff":summarize(target_diffs),
        "bruteforce_cases":len(brute_diffs),
        "bruteforce_opt_abs_diff":summarize(brute_diffs),
        "negative_greedy_gap_cases":negative_greedy,
        "greedy_mean_case_gap":float(np.mean(greedy_gaps)),
        "greedy_event_step_acc":100.0*greedy_event_hits/(4*n),
        "greedy_gap_distribution":summarize(greedy_gaps),
        "passed":bool(
            illegal==0 and negative_greedy==0
            and (max(target_diffs) if target_diffs else 0)<1e-3
            and (max(brute_diffs) if brute_diffs else 0)<1e-3
            and split_counts["train"]==int(.8*n)
            and split_counts["validation"]==int(.1*n)
            and split_counts["test"]==n-int(.9*n)
        )
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("CROSSGRAPH_CERT",json.dumps(result,sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
