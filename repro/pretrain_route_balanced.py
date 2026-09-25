#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from historical_full_model import HistoricalExact, Model, loaders, move, seed_all
from pretrain_diagnostics import pretrain_metrics, eval_breakdown

def expanded_ids(data, ids, triple_repeat):
    out=[]
    for cid in ids:
        c=data.case_by_id[int(cid)]
        r=triple_repeat if int(c.passenger_count)==3 else 1
        out.extend([int(cid)]*r)
    return out

def route_aux_pretrain_loss(model,b,route_weight=4.0,route_lambda=1.0):
    nh,eh=model.encode(b)

    # Node route-membership objective: same paper-aligned binary target.
    node_logits=model.node_head(nh)
    ny=b["node_y"];pos=(ny==1)
    posw=(~pos).sum().float()/pos.sum().clamp_min(1).float()
    nw=torch.tensor([1.0,float(min(posw,30))],device=ny.device)
    node_loss=nn.functional.cross_entropy(node_logits.reshape(-1,2),ny.reshape(-1),weight=nw)

    # Original paper four-class edge target.
    logits=model.edge_head(eh);ey=b["edge_y"]
    keep=(ey!=1)
    rand=torch.rand_like(ey.float());special=keep.sum(1,keepdim=True).clamp_min(1)
    prob=(5*special/ey.size(1)).clamp(max=1).float()
    keep=keep | ((ey==1)&(rand<prob))
    edge4_loss=nn.functional.cross_entropy(logits[keep],ey[keep])

    # Auxiliary loss derived from the SAME four classes:
    # 0/1 => not on exact route; 2/3 => on exact route.
    not_route=torch.logsumexp(logits[...,0:2],dim=-1)
    on_route=torch.logsumexp(logits[...,2:4],dim=-1)
    route_logits=torch.stack([not_route,on_route],dim=-1)
    route_y=((ey==2)|(ey==3)).long()
    rw=torch.tensor([1.0,float(route_weight)],device=ey.device)
    route_loss=nn.functional.cross_entropy(route_logits[keep],route_y[keep],weight=rw)

    loss=node_loss+edge4_loss+route_lambda*route_loss
    return loss,node_loss.detach(),edge4_loss.detach(),route_loss.detach()

def run(seed,args):
    seed_all(seed)
    data=HistoricalExact(args.links,args.orders,args.labels,args.exact)
    tr,va,te=data.split(seed)
    train_ids=expanded_ids(data,tr,args.triple_repeat)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("seed",seed,"device",dev,"split",len(tr),len(va),len(te),
          "train_expanded",len(train_ids),"triple_repeat",args.triple_repeat)

    model=Model(data,args.hidden).to(dev)
    pretrain=loaders(data,train_ids,args.batch,shuffle=True,seed=seed)
    val=loaders(data,va,args.batch,shuffle=False,seed=seed)
    test=loaders(data,te,args.batch,shuffle=False,seed=seed)

    opt=torch.optim.AdamW(model.parameters(),lr=args.pre_lr,weight_decay=1e-4)
    pre_curve=[]
    for ep in range(1,args.pre_epochs+1):
        model.train();vals=[]
        for b in pretrain:
            b=move(b,dev)
            loss,nl,e4,rl=route_aux_pretrain_loss(model,b,args.route_weight,args.route_lambda)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            vals.append((float(loss.detach()),float(nl),float(e4),float(rl)))
        if ep in (1,5,args.pre_epochs):
            pm=pretrain_metrics(model,val,dev)
            rec={"epoch":ep,"loss":float(np.mean([x[0] for x in vals])),
                 "node_loss":float(np.mean([x[1] for x in vals])),
                 "edge4_loss":float(np.mean([x[2] for x in vals])),
                 "route_aux_loss":float(np.mean([x[3] for x in vals])),
                 "val":pm}
            pre_curve.append(rec);print("BAL_PRE",json.dumps(rec,sort_keys=True))

    # Fine-tune all encoder/decoder parameters at the same LR. Diagnostics
    # already showed a lower encoder LR hurts both representative seeds.
    train_ft=loaders(data,train_ids,args.batch,shuffle=True,seed=seed)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=1e30;state=None;bad=0;best_ep=0
    for ep in range(1,args.epochs+1):
        model.train()
        for b in train_ft:
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
        else: bad+=1
        if bad>=args.patience:break

    model.load_state_dict(state)
    result={
      "seed":seed,"triple_repeat":args.triple_repeat,
      "route_weight":args.route_weight,"route_lambda":args.route_lambda,
      "best_epoch":best_ep,"stop_epoch":ep,"val_decoder_ce":best,
      "pretrain_curve":pre_curve,
      "test":eval_breakdown(model,test,data,dev)
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print("BAL_RESULT",json.dumps(result,sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--triple-repeat",type=int,default=1)
    ap.add_argument("--route-weight",type=float,default=4.0);ap.add_argument("--route-lambda",type=float,default=1.0)
    ap.add_argument("--hidden",type=int,default=64);ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--pre-epochs",type=int,default=8);ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--epochs",type=int,default=55);ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--patience",type=int,default=10)
    a=ap.parse_args();run(a.seed,a)

if __name__=="__main__":main()
