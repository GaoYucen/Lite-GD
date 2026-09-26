#!/usr/bin/env python3
"""Train/evaluate scalable Lite-GD on Qingdao-SCC."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from qingdao_dataset import QingdaoSCCExact
from qingdao_local_model import QingdaoLiteGD
from baseline_carpool import make_batches, seed_all
from evaluation import summarize_route_rows


def collate(data,ids,device):
    rows=[data.case_light(int(i)) for i in ids]
    ns={len(x["flat"]) for x in rows}
    if len(ns)!=1:raise ValueError("equal-candidate batches required")
    N=next(iter(ns));T=max(x["n_events"] for x in rows);B=len(rows)
    coords=torch.tensor(np.stack([x["points"] for x in rows]),dtype=torch.float32,device=device)
    event=torch.tensor(np.stack([x["event"] for x in rows]),dtype=torch.long,device=device)
    target=torch.full((B,T),-100,dtype=torch.long,device=device)
    road=torch.tensor(np.stack([x["cost"] for x in rows]),dtype=torch.float32,device=device)
    ne=torch.tensor([x["n_events"] for x in rows],dtype=torch.long,device=device)
    valid=torch.ones((B,N),dtype=torch.bool,device=device)
    for i,x in enumerate(rows):target[i,:len(x["target"])]=torch.as_tensor(x["target"],device=device)
    return {"coords":coords,"event":event,"target":target,"road_cost":road,
            "n_events":ne,"valid":valid,"raw":rows}


def evaluate(model,data,ids,device,batch_size):
    model.eval();rows=[]
    with torch.no_grad():
        for bid in make_batches(ids,data,batch_size,seed=0,epoch=0,shuffle=False):
            b=collate(data,bid,device);seq=model.predict_batch(data,bid,b)
            ar=torch.arange(len(bid),device=device)
            for i,cid in enumerate(bid):
                x=b["raw"][i];t=x["n_events"];pred=seq[i,:t].detach().cpu().tolist()
                true=x["target"].tolist();C=x["cost"];event=x["event"]
                L=float(sum(C[a,z] for a,z in zip([0]+pred[:-1],pred)))
                pe=[int(event[j]) for j in pred];te=[int(event[j]) for j in true]
                pm={int(event[j]):j for j in pred};tm={int(event[j]):j for j in true}
                rows.append({"pred":L,"opt":x["opt"],"exact":int(pred==true),
                    "pointer_hits":sum(a==z for a,z in zip(pred,true)),"steps":t,
                    "event_exact":int(pe==te),"event_hits":sum(a==z for a,z in zip(pe,te)),
                    "candidate_hits":sum(pm[e]==tm[e] for e in tm)})
    return summarize_route_rows(rows)


def subset(ids,n,seed):
    ids=np.asarray(ids,dtype=np.int64)
    if n<=0 or n>=len(ids):return ids.tolist()
    rng=np.random.default_rng(seed);return rng.choice(ids,size=n,replace=False).tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--seed",type=int,default=4321)
    ap.add_argument("--train-limit",type=int,default=0)
    ap.add_argument("--val-limit",type=int,default=0)
    ap.add_argument("--test-limit",type=int,default=0)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--metric-layers",type=int,default=2)
    ap.add_argument("--metric-heads",type=int,default=4)
    ap.add_argument("--litegd-arch",choices=["legacy","road_metric","road_metric_hier"],default="road_metric_hier")
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--eval-batch",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--eval-every",type=int,default=2)
    a=ap.parse_args()

    seed_all(a.seed)
    data=QingdaoSCCExact(a.benchmark_dir)
    tr,va,te=data.split()
    tr=subset(tr,a.train_limit,a.seed+11)
    va=subset(va,a.val_limit,a.seed+13)
    te=subset(te,a.test_limit,a.seed+17)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=QingdaoLiteGD(data,a.hidden,a.metric_layers,a.metric_heads,a.litegd_arch).to(dev)

    # Warm RF cache for the selected pilot/full split and report actual local sizes.
    tprep=time.perf_counter();sample_stats=[]
    for cid in sorted(set(tr+va+te)):
        rf=data.receptive_field(cid)
        if len(sample_stats)<500:
            sample_stats.append((len(rf["C"]),len(rf["E1"]),len(rf["E0"]),len(rf["N1"])))
    prep_s=time.perf_counter()-tprep
    rfstat={
      "candidate_unique_edges_mean":float(np.mean([x[0] for x in sample_stats])),
      "e1_mean":float(np.mean([x[1] for x in sample_stats])),
      "e0_mean":float(np.mean([x[2] for x in sample_stats])),
      "n1_mean":float(np.mean([x[3] for x in sample_stats])),
    }

    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=float("inf");best_state=None;best_epoch=0;bad=0;history=[]
    start=time.perf_counter()
    for ep in range(1,a.epochs+1):
        model.train();ls=[]
        for bid in make_batches(tr,data,a.batch,seed=a.seed,epoch=ep,shuffle=True):
            b=collate(data,bid,dev)
            loss,_=model.forward_batch(data,bid,b,teacher=True)
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            ls.append(float(loss.detach()))
        if ep==1 or ep%a.eval_every==0:
            vm=evaluate(model,data,va,dev,a.eval_batch)
            score=float(vm["gap"])
            rec={"epoch":ep,"train_ce":float(np.mean(ls)),"val":vm}
            history.append(rec);print("QINGDAO_VAL",json.dumps(rec,sort_keys=True),flush=True)
            if score<best-1e-6:
                best=score;best_epoch=ep;bad=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad+=a.eval_every
            if bad>=a.patience:break
    if best_state is None:raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    test=evaluate(model,data,te,dev,a.eval_batch)
    elapsed=time.perf_counter()-start
    out={"model":"Lite-GD-Qingdao-local-exact","litegd_arch":a.litegd_arch,"protocol":data.meta["protocol"],
      "seed":a.seed,"train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te),
      "hidden":a.hidden,"metric_layers":a.metric_layers,"metric_heads":a.metric_heads,
      "best_epoch":best_epoch,"val_gap":best,"test":test,"history":history,
      "rf_cache_prepare_s":prep_s,"rf_stats_sample500":rfstat,"elapsed_s":elapsed,
      "params":sum(p.numel() for p in model.parameters())}
    if dev.type=="cuda":
        out["peak_gpu_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
        out["device"]=torch.cuda.get_device_name(0)
    if a.checkpoint:
        a.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":best_state,"config":out},a.checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("QINGDAO_RESULT",json.dumps({k:out[k] for k in ["best_epoch","val_gap","test","elapsed_s","rf_stats_sample500"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
