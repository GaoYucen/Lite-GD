#!/usr/bin/env python3
"""Benchmark exact sparse Lite-GD online inference on CPU.

For B=1 and ~30 candidates, GPU launch overhead can dominate.  This benchmark
uses the exact two-hop sparse encoder plus the vectorized target-free decoder
entirely on one CPU thread.  It verifies predictions against the GPU/full
checkpoint semantics through route metrics already validated by sparse_inference.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch

from crossgraph_data import CrossGraphExact
from historical_full_model import Model
from sparse_inference import compact_case,SparseOnlineLiteGD
from evaluation import summarize_route_rows


def stat(xs):
    a=np.asarray(xs,float)
    return {"n":int(len(a)),"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),
            "p95_ms":float(np.percentile(a,95)),"qps":float(1000.0/a.mean())}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--apsp",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()

    data=CrossGraphExact(a.benchmark_dir,a.apsp,require_routes=True)
    _,_,te=data.split(20260925)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=Model(data,64,decoder_arch="road_metric_hier_fast",metric_layers=2,metric_heads=4).cpu().eval()
    model.load_state_dict(ck["state_dict"])

    t=time.perf_counter();online=SparseOnlineLiteGD(model,data);cache_ms=(time.perf_counter()-t)*1000
    cases=[compact_case(data,cid) for cid in te]
    for c in cases[:10]:online.infer_compact(c)

    xs=[];rows=[]
    with torch.no_grad():
      for case in cases:
        t=time.perf_counter();pred,_=online.infer_compact(case);xs.append((time.perf_counter()-t)*1000)
        pred=pred[0,:case["n_events"]].tolist();true=case["target"].tolist();flat=case["flat"]
        L=0.0
        for x,y in zip([0]+pred[:-1],pred):
          ea,ra,_,_=flat[x];eb,rb,_,_=flat[y];L+=data.point_dist(ea,ra,eb,rb)
        pe=[int(flat[j][2]) for j in pred];tevent=[int(flat[j][2]) for j in true]
        pm={int(flat[j][2]):j for j in pred};tm={int(flat[j][2]):j for j in true}
        rows.append({"pred":L,"opt":case["opt"],"exact":int(pred==true),
                     "pointer_hits":sum(x==y for x,y in zip(pred,true)),"steps":case["n_events"],
                     "event_exact":int(pe==tevent),"event_hits":sum(x==y for x,y in zip(pe,tevent)),
                     "candidate_hits":sum(pm[e]==tm[e] for e in tm)})
    out={"runtime_sparse_cpu":stat(xs),"quality":summarize_route_rows(rows),
         "offline_cache_build_ms":cache_ms,"threads":1,"device":"cpu"}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("CPU_SPARSE_RUNTIME",json.dumps(out,sort_keys=True))


if __name__=="__main__":main()
