#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from historical_full_model import HistoricalExact, Model, loaders, move, seed_all
from pretrain_diagnostics import pretrain_metrics
from pretrain_candidate_balanced import eval_decomposed

def candidate_rep(model,b,eh):
    dec=model.decoder
    B,N=b["edge_idx"].shape;H=eh.size(-1)
    cand=torch.gather(eh,1,b["edge_idx"].unsqueeze(-1).expand(-1,-1,H))
    fc=dec.domain(b["coords"],b["valid"])
    gfc=torch.sigmoid(dec.gfc1(fc)+dec.gfc2(cand))
    ge=torch.sigmoid(dec.ge1(fc)+dec.ge2(cand))
    return torch.tanh(dec.merge(torch.cat([gfc*fc,ge*cand],-1)))

def group_rank_loss(model,rank_head,b):
    _,eh=model.encode(b)
    rep=candidate_rep(model,b,eh)
    score=rank_head(rep).squeeze(-1)
    losses=[]
    B,N=score.shape
    for bi in range(B):
        targets=[int(x) for x in b["target"][bi].tolist() if int(x)>=0]
        for chosen in targets:
            et=int(b["event"][bi,chosen])
            ids=torch.where(b["valid"][bi] & (b["event"][bi]==et))[0]
            if ids.numel()<=1:
                continue
            pos=torch.where(ids==chosen)[0]
            if pos.numel()!=1:
                raise RuntimeError(f"chosen candidate not unique in event mask: b={bi} event={et}")
            losses.append(nn.functional.cross_entropy(
                score[bi,ids].unsqueeze(0),
                pos.to(score.device).long()
            ))
    return torch.stack(losses).mean() if losses else score.sum()*0

def run(seed,args):
    seed_all(seed)
    data=HistoricalExact(args.links,args.orders,args.labels,args.exact)
    tr,va,te=data.split(seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=loaders(data,tr,args.batch,shuffle=True,seed=seed)
    val=loaders(data,va,args.batch,shuffle=False,seed=seed)
    test=loaders(data,te,args.batch,shuffle=False,seed=seed)

    model=Model(data,args.hidden).to(dev)
    rank_head=nn.Sequential(nn.Linear(args.hidden,args.hidden//2),nn.ReLU(),nn.Linear(args.hidden//2,1)).to(dev)
    opt=torch.optim.AdamW(list(model.parameters())+list(rank_head.parameters()),
                          lr=args.pre_lr,weight_decay=1e-4)
    curve=[]
    for ep in range(1,args.pre_epochs+1):
        model.train();rank_head.train();vals=[]
        for b in train:
            b=move(b,dev)
            paper,nl,el=model.pretrain_loss(b,balanced=True)
            rank=group_rank_loss(model,rank_head,b)
            loss=paper+args.rank_lambda*rank
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters())+list(rank_head.parameters()),1.0)
            opt.step()
            vals.append((float(loss.detach()),float(nl),float(el),float(rank.detach())))
        if ep in (1,5,args.pre_epochs):
            pm=pretrain_metrics(model,val,dev)
            rec={"epoch":ep,
                 "loss":float(np.mean([x[0] for x in vals])),
                 "node_loss":float(np.mean([x[1] for x in vals])),
                 "edge4_loss":float(np.mean([x[2] for x in vals])),
                 "group_rank_loss":float(np.mean([x[3] for x in vals])),
                 "val":pm}
            curve.append(rec);print("RANK_PRE",json.dumps(rec,sort_keys=True))

    # Standard decoder fine-tuning; rank head is intentionally discarded.
    train_ft=loaders(data,tr,args.batch,shuffle=True,seed=seed)
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
        else:
            bad+=1
        if bad>=args.patience:break

    model.load_state_dict(state)
    result={"seed":seed,"rank_lambda":args.rank_lambda,
            "best_epoch":best_ep,"stop_epoch":ep,"val_decoder_ce":best,
            "pretrain_curve":curve,"test":eval_decomposed(model,test,data,dev)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print("RANK_RESULT",json.dumps(result,sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--rank-lambda",type=float,default=1.0)
    ap.add_argument("--hidden",type=int,default=64);ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--pre-epochs",type=int,default=8);ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--epochs",type=int,default=55);ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--patience",type=int,default=10)
    a=ap.parse_args();run(a.seed,a)

if __name__=="__main__":main()
