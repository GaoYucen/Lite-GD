#!/usr/bin/env python3
"""Train the fixed Lite-GD learning baselines on generated cross-graph benchmarks."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from crossgraph_data import CrossGraphExact
from baseline_carpool import (
    CarpoolCases, train_pointer, train_am, evaluate as evaluate_baseline, seed_all as baseline_seed_all
)
from historical_full_model import Model, loaders, move, seed_all
from pretrain_candidate_balanced import candidate_aux_loss, eval_decomposed
from pretrain_diagnostics import pretrain_metrics


def train_litegd(args,data,device):
    seed_all(args.seed)
    tr,va,te=data.split(args.split_seed)
    train=loaders(data,tr,args.litegd_batch,shuffle=True,seed=args.seed)
    val=loaders(data,va,args.litegd_batch,shuffle=False,seed=args.seed)
    test=loaders(data,te,args.litegd_batch,shuffle=False,seed=args.seed)

    model=Model(data,args.hidden,decoder_arch=args.litegd_arch,
                metric_layers=args.metric_layers,metric_heads=args.metric_heads).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.pre_lr,weight_decay=1e-4)
    curve=[]
    for ep in range(1,args.pre_epochs+1):
        model.train();vals=[]
        for b in train:
            b=move(b,device)
            loss,nl,e4,cl=candidate_aux_loss(model,b,args.candidate_pos_weight,args.candidate_lambda)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            vals.append((float(loss.detach()),float(nl),float(e4),float(cl)))
        if ep in (1,5,args.pre_epochs):
            pm=pretrain_metrics(model,val,device)
            rec={"epoch":ep,"loss":float(np.mean([x[0] for x in vals])),
                 "node_loss":float(np.mean([x[1] for x in vals])),
                 "edge4_loss":float(np.mean([x[2] for x in vals])),
                 "candidate_aux_loss":float(np.mean([x[3] for x in vals])),
                 "val":pm}
            curve.append(rec);print("CROSS_PRE",json.dumps(rec,sort_keys=True),flush=True)

    # Same fine-tuning protocol as the fixed-split historical candidate-aware run.
    train_ft=loaders(data,tr,args.litegd_batch,shuffle=True,seed=args.seed)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float("inf");state=None;bad=0;best_ep=0;history=[]
    for ep in range(1,args.epochs+1):
        model.train();tls=[]
        for b in train_ft:
            b=move(b,device);loss,_=model.decoder_loss(b,teacher=True)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            tls.append(float(loss.detach()))
        model.eval();vl=[]
        with torch.no_grad():
            for b in val:
                b=move(b,device);loss,_=model.decoder_loss(b,teacher=True);vl.append(float(loss.detach()))
        v=float(np.mean(vl))
        history.append({"epoch":ep,"train_ce":float(np.mean(tls)),"val_ce":v})
        if ep==1 or ep%5==0:print("CROSS_FT",json.dumps(history[-1],sort_keys=True),flush=True)
        if v<best-1e-5:
            best=v;bad=0;best_ep=ep
            state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        else:
            bad+=1
        if bad>=args.patience:break
    if state is None:raise RuntimeError("no Lite-GD checkpoint selected")
    model.load_state_dict(state)
    quality=eval_decomposed(model,test,data,device)
    return model,{"litegd_arch":args.litegd_arch,"metric_layers":args.metric_layers,
                  "metric_heads":args.metric_heads,"pretrain_curve":curve,
                  "finetune_history":history,"best_epoch":best_ep,
                  "val_decoder_ce":best,"test":quality}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",type=Path,required=True)
    ap.add_argument("--apsp",type=Path,required=True)
    ap.add_argument("--model",choices=["ptrnet","am","litegd"],required=True)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--split-seed",type=int,default=20260925)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path)

    # PointerNet / AM fixed Phase-A protocol.
    ap.add_argument("--dim",type=int,default=128)
    ap.add_argument("--ptr-layers",type=int,default=1)
    ap.add_argument("--heads",type=int,default=8)
    ap.add_argument("--am-layers",type=int,default=3)
    ap.add_argument("--ff",type=int,default=512)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--eval-batch",type=int,default=64)
    ap.add_argument("--baseline-epochs",type=int,default=120)
    ap.add_argument("--eval-every",type=int,default=5)
    ap.add_argument("--baseline-patience",type=int,default=25)
    ap.add_argument("--baseline-lr",type=float,default=1e-4)

    # Candidate-aware Lite-GD fixed Phase-A protocol.
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--litegd-arch",choices=["legacy","road_metric","road_metric_hier"],default="legacy")
    ap.add_argument("--metric-layers",type=int,default=2)
    ap.add_argument("--metric-heads",type=int,default=4)
    ap.add_argument("--litegd-batch",type=int,default=8)
    ap.add_argument("--pre-epochs",type=int,default=8)
    ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--candidate-pos-weight",type=float,default=3.0)
    ap.add_argument("--candidate-lambda",type=float,default=1.0)
    ap.add_argument("--epochs",type=int,default=55)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--patience",type=int,default=10)
    args=ap.parse_args()

    require_routes=args.model=="litegd"
    data=CrossGraphExact(args.benchmark_dir,args.apsp,require_routes=require_routes)
    tr,va,te=data.split(args.split_seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model in ("ptrnet","am"):
        baseline_seed_all(args.seed)
        cases=CarpoolCases(data,tr)
        # Adapt argparse names without changing the tested training functions.
        args.lr=args.baseline_lr;args.epochs=args.baseline_epochs;args.patience=args.baseline_patience
        trainer=train_pointer if args.model=="ptrnet" else train_am
        model,history=trainer(args,data,cases,tr,va,te,device)
        test=evaluate_baseline(model,cases,te,data,device,args.eval_batch)
        result={"model":args.model,"benchmark":str(args.benchmark_dir),"training_seed":args.seed,
                "split_seed":args.split_seed,"train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te),
                "history":history,"test":test}
    else:
        model,extra=train_litegd(args,data,device)
        result={"model":"litegd","benchmark":str(args.benchmark_dir),"training_seed":args.seed,
                "split_seed":args.split_seed,"train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te),**extra}

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"model":args.model,"seed":args.seed,"state_dict":model.state_dict(),
                    "hidden":args.hidden,"litegd_arch":args.litegd_arch if args.model=="litegd" else None,
                    "metric_layers":args.metric_layers if args.model=="litegd" else None,
                    "metric_heads":args.metric_heads if args.model=="litegd" else None},args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("CROSSGRAPH_TRAIN_RESULT",json.dumps(result["test"],sort_keys=True),flush=True)


if __name__=="__main__":main()
