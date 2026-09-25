#!/usr/bin/env python3
"""Independent certification for JSON/GZIP cross-graph carpool benchmarks."""
from __future__ import annotations
import argparse,gzip,itertools,json
from pathlib import Path
import numpy as np


ORDERS=[p for p in itertools.permutations(range(4))
        if p.index(0)<p.index(1) and p.index(2)<p.index(3)]
assert len(ORDERS)==6


def flat_case(c):
    flat=[(int(c["driver"]["edge"]),float(c["driver"]["ratio"]),-1,-1)]
    groups={}
    for ev,g in enumerate(c["candidate_groups"]):
        ids=[]
        for li,z in enumerate(g):
            ids.append(len(flat))
            flat.append((int(z["edge"]),float(z["ratio"]),int(ev),int(li)))
        groups[ev]=np.asarray(ids,dtype=np.int64)
    return flat,groups


def point_cost_matrix(src,dst,w,D,flat):
    e=np.asarray([z[0] for z in flat],dtype=np.int64)
    r=np.asarray([z[1] for z in flat],dtype=np.float64)
    C=(1-r)[:,None]*w[e][:,None] + D[np.ix_(dst[e],src[e])] + r[None,:]*w[e][None,:]
    same=e[:,None]==e[None,:];forward=r[None,:]>=r[:,None]
    direct=(r[None,:]-r[:,None])*w[e][:,None]
    C=np.where(same&forward,np.minimum(C,direct),C)
    np.fill_diagonal(C,0.0)
    return C


def route_cost(C,seq):
    cur=0;v=0.0
    for j in seq:
        v+=float(C[cur,int(j)]);cur=int(j)
    return v


def brute(C,groups):
    best=float("inf")
    for order in ORDERS:
        for seq in itertools.product(*(groups[e] for e in order)):
            z=route_cost(C,seq)
            if z<best:best=z
    return best


def summary(a):
    a=np.asarray(a,dtype=float)
    if not len(a):return {}
    return dict(n=int(len(a)),mean=float(a.mean()),max=float(a.max()),
                median=float(np.median(a)),p95=float(np.percentile(a,95)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",type=Path,required=True)
    ap.add_argument("--apsp",type=Path,required=True)
    ap.add_argument("--sample-bruteforce",type=int,default=64)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    g=np.load(a.benchmark_dir/"graph.npz",mmap_mode="r")
    src=np.asarray(g["src"],dtype=np.int64);dst=np.asarray(g["dst"],dtype=np.int64);w=np.asarray(g["weight"],dtype=np.float64)
    D=np.load(a.apsp,mmap_mode="r")
    with gzip.open(a.benchmark_dir/"cases.json.gz","rt",encoding="utf-8") as f:cases=json.load(f)
    n=len(cases);rng=np.random.default_rng(20260925)
    sample=set(map(int,rng.choice(n,size=min(a.sample_bruteforce,n),replace=False)))
    target_err=[];brute_err=[];illegal=0;nonfinite=0
    for i,c in enumerate(cases):
        flat,groups=flat_case(c);C=point_cost_matrix(src,dst,w,D,flat)
        seq=list(map(int,c["exact_flat_indices"]));order=list(map(int,c["exact_event_sequence"]))
        opt=float(c["exact_length"])
        if not np.isfinite(C).all() or not np.isfinite(opt):nonfinite+=1
        if sorted(order)!=[0,1,2,3] or order.index(0)>order.index(1) or order.index(2)>order.index(3):illegal+=1
        target_err.append(abs(route_cost(C,seq)-opt))
        if i in sample:brute_err.append(abs(brute(C,groups)-opt))
        if (i+1)%2000==0:print(f"certified={i+1}",flush=True)
    meta=json.loads((a.benchmark_dir/"metadata.json").read_text())
    split=meta["split"]
    result={
      "cases":n,
      "split_counts":{k:len(v) for k,v in split.items()},
      "illegal_event_orders":illegal,
      "nonfinite_cases":nonfinite,
      "target_cost_abs_error":summary(target_err),
      "bruteforce_cases":len(brute_err),
      "bruteforce_opt_abs_error":summary(brute_err),
    }
    result["passed"]=bool(illegal==0 and nonfinite==0 and max(target_err,default=0)<1e-6 and max(brute_err,default=0)<1e-6
      and len(split["train"])==int(.8*n) and len(split["validation"])==int(.1*n)
      and len(split["test"])==n-int(.9*n))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("CROSSGRAPH_JSON_CERT",json.dumps(result,sort_keys=True))
    if not result["passed"]:raise SystemExit(2)


if __name__=="__main__":main()
