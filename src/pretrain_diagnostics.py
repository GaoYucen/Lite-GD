#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from historical_full_model import (
    HistoricalExact, Model, loaders, move, seed_all
)

EDGE_NAMES=["candidate_only","neither","route_only","candidate_and_route"]

def f1_from_counts(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f=2*p*r/(p+r) if p+r else 0.0
    return {"precision":100*p,"recall":100*r,"f1":100*f}

def pretrain_metrics(model,loader,dev):
    model.eval()
    ntp=nfp=nfn=ntn=0
    conf=np.zeros((4,4),dtype=np.int64)
    with torch.no_grad():
        for b in loader:
            b=move(b,dev)
            nh,eh=model.encode(b)
            npred=model.node_head(nh).argmax(-1)
            ny=b["node_y"]
            ntp+=int(((npred==1)&(ny==1)).sum())
            nfp+=int(((npred==1)&(ny==0)).sum())
            nfn+=int(((npred==0)&(ny==1)).sum())
            ntn+=int(((npred==0)&(ny==0)).sum())

            epred=model.edge_head(eh).argmax(-1)
            ey=b["edge_y"]
            z=(ey.reshape(-1)*4+epred.reshape(-1)).cpu()
            conf += torch.bincount(z,minlength=16).reshape(4,4).numpy()

    node=f1_from_counts(ntp,nfp,nfn)
    node["accuracy"]=100*(ntp+ntn)/max(1,ntp+ntn+nfp+nfn)
    node["positive_rate"]=100*(ntp+nfn)/max(1,ntp+ntn+nfp+nfn)

    per={}
    fs=[]
    for k,name in enumerate(EDGE_NAMES):
        tp=int(conf[k,k]);fp=int(conf[:,k].sum()-tp);fn=int(conf[k,:].sum()-tp)
        m=f1_from_counts(tp,fp,fn)
        m["support"]=int(conf[k,:].sum())
        per[name]=m;fs.append(m["f1"])
    edge_acc=100*np.trace(conf)/max(1,conf.sum())
    return {"node":node,"edge_accuracy":edge_acc,"edge_macro_f1":float(np.mean(fs)),
            "edge_per_class":per,"edge_confusion":conf.tolist()}

def eval_breakdown(model,loader,data,dev):
    model.eval()
    groups={2:dict(pred=[],opt=[],case_gap=[],exact=0,ptr=0,steps=0,n=0),
            3:dict(pred=[],opt=[],case_gap=[],exact=0,ptr=0,steps=0,n=0)}
    with torch.no_grad():
        for b in loader:
            raw=b["raw"];bb=move(b,dev);_,p=model.decoder_loss(bb,teacher=False);p=p.cpu()
            for i,x in enumerate(raw):
                t=x["n_events"];q=t//2;pred=p[i,:t].tolist();true=x["target"].tolist()
                seq=[0]+pred;flat=x["flat"];L=0.0
                for a,z in zip(seq[:-1],seq[1:]):
                    ea,ra,_,_=flat[a];eb,rb,_,_=flat[z]
                    L+=data.point_dist(ea,ra,eb,rb)
                g=groups[q];g["n"]+=1;g["pred"].append(L);g["opt"].append(x["opt"])
                g["case_gap"].append((L/x["opt"]-1)*100)
                g["exact"]+=int(pred==true);g["ptr"]+=sum(a==z for a,z in zip(pred,true));g["steps"]+=t
    out={}
    all_pred=[];all_opt=[];all_case=[];exact=ptr=steps=n=0
    for q,g in groups.items():
        if not g["n"]:continue
        d={
          "n":g["n"],
          "gap":(float(np.mean(g["pred"]))/float(np.mean(g["opt"]))-1)*100,
          "mean_case_gap":float(np.mean(g["case_gap"])),
          "exact":100*g["exact"]/g["n"],
          "pointer":100*g["ptr"]/g["steps"],
          "avg_pred_length":float(np.mean(g["pred"])),
          "avg_opt_length":float(np.mean(g["opt"]))
        }
        out[f"p{q}"]=d
        all_pred+=g["pred"];all_opt+=g["opt"];all_case+=g["case_gap"]
        exact+=g["exact"];ptr+=g["ptr"];steps+=g["steps"];n+=g["n"]
    out["overall"]={
      "n":n,
      "gap":(float(np.mean(all_pred))/float(np.mean(all_opt))-1)*100,
      "mean_case_gap":float(np.mean(all_case)),
      "exact":100*exact/n,
      "pointer":100*ptr/steps
    }
    return out

def finetune(model,data,tr,va,te,seed,args,strategy,dev):
    train=loaders(data,tr,args.batch,shuffle=True,seed=seed)
    val=loaders(data,va,args.batch,shuffle=False,seed=seed)
    test=loaders(data,te,args.batch,shuffle=False,seed=seed)
    if strategy=="scratch":
        pass
    elif strategy=="pretrain_same_lr":
        pass
    elif strategy=="pretrain_encoder_low":
        pass
    else:
        raise ValueError(strategy)

    if strategy=="pretrain_encoder_low":
        opt=torch.optim.AdamW([
            {"params":model.encoder.parameters(),"lr":args.encoder_lr},
            {"params":model.decoder.parameters(),"lr":args.lr},
        ],weight_decay=1e-4)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    best=1e30;state=None;bad=0;best_ep=0
    for ep in range(1,args.epochs+1):
        model.train()
        for b in train:
            b=move(b,dev);loss,_=model.decoder_loss(b,teacher=True)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval();vl=[]
        with torch.no_grad():
            for b in val:
                b=move(b,dev);loss,_=model.decoder_loss(b,teacher=True);vl.append(float(loss.detach()))
        v=float(np.mean(vl))
        if v<best-1e-5:
            best=v;bad=0;best_ep=ep
            state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        else:
            bad+=1
        if bad>=args.patience:break
    model.load_state_dict(state)
    return {"strategy":strategy,"best_epoch":best_ep,"stop_epoch":ep,
            "val_decoder_ce":best,"test":eval_breakdown(model,test,data,dev)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--hidden",type=int,default=64);ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--pre-epochs",type=int,default=8);ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--epochs",type=int,default=55);ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--encoder-lr",type=float,default=1e-4);ap.add_argument("--patience",type=int,default=10)
    args=ap.parse_args()

    seed_all(args.seed)
    data=HistoricalExact(args.links,args.orders,args.labels,args.exact)
    tr,va,te=data.split(args.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("seed",args.seed,"device",dev,"split",len(tr),len(va),len(te))

    # Pre-train exactly once, then clone the same representation for both
    # fine-tuning strategies.
    pre=Model(data,args.hidden).to(dev)
    train_pre=loaders(data,tr,args.batch,shuffle=True,seed=args.seed)
    val_pre=loaders(data,va,args.batch,shuffle=False,seed=args.seed)
    opt=torch.optim.AdamW(pre.parameters(),lr=args.pre_lr,weight_decay=1e-4)
    pre_curve=[]
    for ep in range(1,args.pre_epochs+1):
        pre.train();ls=[]
        for b in train_pre:
            b=move(b,dev);loss,nl,el=pre.pretrain_loss(b,balanced=True)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(pre.parameters(),1.0);opt.step()
            ls.append((float(loss.detach()),float(nl),float(el)))
        if ep in (1,5,args.pre_epochs):
            pm=pretrain_metrics(pre,val_pre,dev)
            rec={"epoch":ep,"loss":float(np.mean([x[0] for x in ls])),
                 "node_loss":float(np.mean([x[1] for x in ls])),
                 "edge_loss":float(np.mean([x[2] for x in ls])),"val":pm}
            pre_curve.append(rec);print("PRE_DIAG",json.dumps(rec,sort_keys=True))
    pre_state={k:v.detach().cpu().clone() for k,v in pre.state_dict().items()}

    rows=[]
    # Scratch gets the same seed/split for a direct per-passenger diagnostic.
    seed_all(args.seed)
    scratch=Model(data,args.hidden).to(dev)
    rows.append(finetune(scratch,data,tr,va,te,args.seed,args,"scratch",dev))

    for strat in ("pretrain_same_lr","pretrain_encoder_low"):
        seed_all(args.seed)
        m=Model(data,args.hidden).to(dev);m.load_state_dict(pre_state)
        rows.append(finetune(m,data,tr,va,te,args.seed,args,strat,dev))
        print("FT_DIAG",json.dumps(rows[-1],sort_keys=True))

    doc={"seed":args.seed,"split":{"train":len(tr),"val":len(va),"test":len(te)},
         "pretrain_curve":pre_curve,"fine_tune":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(doc,indent=2))
    print("DIAGNOSTIC_SUMMARY",json.dumps(doc,sort_keys=True))

if __name__=="__main__":
    main()
