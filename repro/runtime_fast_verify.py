#!/usr/bin/env python3
"""Verify exact behavioral compatibility and latency of fast Lite-GD decoding."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch

from crossgraph_data import CrossGraphExact
from historical_full_model import Model,loaders,move
from pretrain_candidate_balanced import eval_decomposed


def stat(xs):
    a=np.asarray(xs,float)
    return {"n":len(a),"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),
            "p95_ms":float(np.percentile(a,95)),"qps":float(1000/a.mean())}


def bench_model(model,cpu_batches,device,warm=10):
    gpu_batches=[move(b,device) for b in cpu_batches]
    model.eval()
    with torch.no_grad():
        for b in gpu_batches[:warm]: model.decoder_loss(b,teacher=False)
        torch.cuda.synchronize();model_ms=[]
        for b in gpu_batches:
            torch.cuda.synchronize();t=time.perf_counter()
            model.decoder_loss(b,teacher=False)
            torch.cuda.synchronize();model_ms.append((time.perf_counter()-t)*1000)
    return stat(model_ms)


def bench_split(model,cpu_batches,device,warm=10):
    gpu_batches=[move(b,device) for b in cpu_batches]
    model.eval()
    enc=[];dec=[]
    with torch.no_grad():
        for b in gpu_batches[:warm]:
            _,eh=model.encode(b)
            model.decoder(eh,b["edge_idx"],b["event"],b["coords"],b["valid"],
                          b["target"],b["n_events"],b["road_cost"],False)
        torch.cuda.synchronize()
        for b in gpu_batches:
            torch.cuda.synchronize();t=time.perf_counter()
            _,eh=model.encode(b)
            torch.cuda.synchronize();enc.append((time.perf_counter()-t)*1000)
            t=time.perf_counter()
            model.decoder(eh,b["edge_idx"],b["event"],b["coords"],b["valid"],
                          b["target"],b["n_events"],b["road_cost"],False)
            torch.cuda.synchronize();dec.append((time.perf_counter()-t)*1000)
    return {"encode":stat(enc),"decoder":stat(dec)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--apsp",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--seed",type=int,default=4321)
    a=ap.parse_args()

    data=CrossGraphExact(a.benchmark_dir,a.apsp,require_routes=True)
    tr,va,te=data.split(20260925)
    dev=torch.device("cuda")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)

    old=Model(data,64,decoder_arch="road_metric_hier",metric_layers=2,metric_heads=4).to(dev)
    fast=Model(data,64,decoder_arch="road_metric_hier_fast",metric_layers=2,metric_heads=4).to(dev)
    old.load_state_dict(ck["state_dict"]);fast.load_state_dict(ck["state_dict"])
    old.eval();fast.eval()

    # Exact greedy-prediction equivalence on full test split.
    mismatches=0;max_loss_abs=0.0
    with torch.no_grad():
        for cid in te:
            b=move(next(iter(loaders(data,[cid],1,shuffle=False,seed=a.seed))),dev)
            lo,po=old.decoder_loss(b,teacher=False)
            lf,pf=fast.decoder_loss(b,teacher=False)
            mismatches+=int(not torch.equal(po,pf))
            max_loss_abs=max(max_loss_abs,abs(float(lo)-float(lf)))

    test_old=list(loaders(data,te,8,shuffle=False,seed=a.seed))
    q_old=eval_decomposed(old,test_old,data,dev)
    q_fast=eval_decomposed(fast,test_old,data,dev)

    ids=list(te[:min(100,len(te))])
    cpu_batches=[next(iter(loaders(data,[cid],1,shuffle=False,seed=a.seed))) for cid in ids]
    runtime_old=bench_model(old,cpu_batches,dev)
    runtime_fast=bench_model(fast,cpu_batches,dev)
    split_fast=bench_split(fast,cpu_batches,dev)

    out={
      "ok":mismatches==0,
      "prediction_mismatches":mismatches,
      "max_loss_abs_diff":max_loss_abs,
      "quality_old":q_old["overall"],
      "quality_fast":q_fast["overall"],
      "runtime_old":runtime_old,
      "runtime_fast":runtime_fast,
      "speedup":runtime_old["mean_ms"]/runtime_fast["mean_ms"],
      "runtime_fast_split":split_fast,
      "device":torch.cuda.get_device_name(0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("FAST_VERIFY",json.dumps(out,sort_keys=True))


if __name__=="__main__":main()
