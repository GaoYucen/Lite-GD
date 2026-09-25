#!/usr/bin/env python3
"""Attach exact node/edge routes to a generated cross-graph benchmark."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np


def load_cases(path:Path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return json.load(f)


def flat_case(c):
    flat=[(int(c["driver"]["edge"]),float(c["driver"]["ratio"]),-1,-1)]
    for e,g in enumerate(c["candidate_groups"]):
        for li,z in enumerate(g):
            flat.append((int(z["edge"]),float(z["ratio"]),int(e),int(li)))
    return flat


def point_dist(src,dst,w,D,a,ra,b,rb):
    a=int(a);b=int(b);ra=float(ra);rb=float(rb)
    generic=(1-ra)*float(w[a])+float(D[int(dst[a]),int(src[b])])+rb*float(w[b])
    if a==b and rb>=ra:
        generic=min(generic,(rb-ra)*float(w[a]))
    return float(generic)


def reconstruct(P,src,dst,w,D,pair_edge,a,ra,b,rb):
    a=int(a);b=int(b);ra=float(ra);rb=float(rb)
    au,av=int(src[a]),int(dst[a]);bu,bv=int(src[b]),int(dst[b])
    generic=(1-ra)*float(w[a])+float(D[av,bu])+rb*float(w[b])
    direct=(rb-ra)*float(w[a]) if a==b and rb>=ra else float("inf")
    if direct<=generic+1e-9:
        return [au,av],[a]
    if av==bu:
        mid=[av]
    else:
        if int(P[av,bu])<0: raise RuntimeError(f"unreachable {av}->{bu}")
        rev=[bu];cur=bu;guard=0
        while cur!=av:
            cur=int(P[av,cur]);rev.append(cur);guard+=1
            if cur<0 or guard>len(src)+1: raise RuntimeError(f"bad predecessor path {av}->{bu}")
        mid=list(reversed(rev))
    ep=[a]
    for x,y in zip(mid[:-1],mid[1:]):
        try: ep.append(int(pair_edge[(int(x),int(y))]))
        except KeyError: raise RuntimeError(f"missing graph edge {x}->{y}")
    if b!=ep[-1] or a==b: ep.append(b)
    nodes=[au]+mid
    if nodes[-1]!=bv: nodes.append(bv)
    nodes=[x for i,x in enumerate(nodes) if i==0 or x!=nodes[i-1]]
    ep=[x for i,x in enumerate(ep) if i==0 or x!=ep[i-1]]
    return nodes,ep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--apsp",required=True,type=Path)
    ap.add_argument("--predecessors",required=True,type=Path)
    args=ap.parse_args()

    g=np.load(args.benchmark_dir/"graph.npz",mmap_mode="r")
    src=np.asarray(g["src"],dtype=np.int64);dst=np.asarray(g["dst"],dtype=np.int64);w=np.asarray(g["weight"],dtype=np.float64)
    D=np.load(args.apsp,mmap_mode="r");P=np.load(args.predecessors,mmap_mode="r")
    n=int(len(g["coordinates"]))
    if D.shape!=(n,n) or P.shape!=(n,n): raise RuntimeError("distance/predecessor shape mismatch")
    pair_edge={(int(u),int(v)):i for i,(u,v) in enumerate(zip(src,dst))}
    if len(pair_edge)!=len(src): raise RuntimeError("graph must be ordered-pair deduplicated")

    cases=load_cases(args.benchmark_dir/"cases.json.gz")
    max_abs=0.0;node_counts=[];edge_counts=[]
    for ix,c in enumerate(cases):
        flat=flat_case(c);seq=list(map(int,c["exact_flat_indices"]))
        full_nodes=[];full_edges=[];total=0.0
        prev=0
        for cur in seq:
            ea,ra,_,_=flat[prev];eb,rb,_,_=flat[cur]
            total+=point_dist(src,dst,w,D,ea,ra,eb,rb)
            ns,es=reconstruct(P,src,dst,w,D,pair_edge,ea,ra,eb,rb)
            if full_nodes and ns and full_nodes[-1]==ns[0]: full_nodes.extend(ns[1:])
            else: full_nodes.extend(ns)
            if full_edges and es and full_edges[-1]==es[0]: full_edges.extend(es[1:])
            else: full_edges.extend(es)
            prev=cur
        err=abs(total-float(c["exact_length"]));max_abs=max(max_abs,err)
        if err>1e-6: raise RuntimeError(f"case {c['case_id']} route cost mismatch {err}")
        c["exact_node_route"]=[int(x) for x in full_nodes]
        c["exact_edge_route"]=[int(x) for x in full_edges]
        node_counts.append(len(full_nodes));edge_counts.append(len(full_edges))
        if (ix+1)%1000==0: print(f"routes_attached={ix+1}",flush=True)

    out=args.benchmark_dir/"cases_with_routes.json.gz"
    tmp=args.benchmark_dir/"cases_with_routes.tmp.json.gz"
    with gzip.open(tmp,"wt",encoding="utf-8") as f: json.dump(cases,f,separators=(",",":"))
    os.replace(tmp,out)
    meta=json.loads((args.benchmark_dir/"metadata.json").read_text())
    meta["route_attachment"]={
        "status":"certified","cases":len(cases),"max_abs_route_cost_error":max_abs,
        "node_route_length":{"mean":float(np.mean(node_counts)),"median":float(np.median(node_counts)),"max":int(max(node_counts))},
        "edge_route_length":{"mean":float(np.mean(edge_counts)),"median":float(np.median(edge_counts)),"max":int(max(edge_counts))},
        "cases_with_routes_file":out.name,
    }
    (args.benchmark_dir/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("ROUTE_ATTACHMENT",json.dumps(meta["route_attachment"],sort_keys=True))


if __name__=="__main__":
    main()
