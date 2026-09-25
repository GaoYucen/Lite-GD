#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn

from historical_full_model import HistoricalExact, Model, loaders, move, seed_all
from pretrain_diagnostics import pretrain_metrics

def candidate_aux_loss(model,b,pos_weight=3.0,cand_lambda=1.0):
    nh,eh=model.encode(b)

    # Node paper target.
    node_logits=model.node_head(nh)
    ny=b["node_y"];pos=(ny==1)
    posw=(~pos).sum().float()/pos.sum().clamp_min(1).float()
    nw=torch.tensor([1.0,float(min(posw,30))],device=ny.device)
    node_loss=nn.functional.cross_entropy(node_logits.reshape(-1,2),ny.reshape(-1),weight=nw)

    # Original four-class edge objective and the same background sampling.
    logits=model.edge_head(eh);ey=b["edge_y"]
    keep=(ey!=1)
    rand=torch.rand_like(ey.float());special=keep.sum(1,keepdim=True).clamp_min(1)
    prob=(5*special/ey.size(1)).clamp(max=1).float()
    keep=keep | ((ey==1)&(rand<prob))
    edge4_loss=nn.functional.cross_entropy(logits[keep],ey[keep])

    # Direct candidate discrimination. Only candidate edges participate:
    # class 0 = candidate-only, class 3 = candidate-and-exact-route.
    cm=(ey==0)|(ey==3)
    cand_logits=torch.stack([logits[...,0],logits[...,3]],dim=-1)
    cand_y=(ey==3).long()
    cw=torch.tensor([1.0,float(pos_weight)],device=ey.device)
    cand_loss=nn.functional.cross_entropy(cand_logits[cm],cand_y[cm],weight=cw)

    loss=node_loss+edge4_loss+cand_lambda*cand_loss
    return loss,node_loss.detach(),edge4_loss.detach(),cand_loss.detach()

def _key(flat,idx):
    e,r,ev,li=flat[int(idx)]
    return (int(e),round(float(r),12))

def eval_decomposed(model,loader,data,dev):
    model.eval()
    buckets={2:[],3:[]}
    with torch.no_grad():
        for b in loader:
            raw=b["raw"];bb=move(b,dev);_,p=model.decoder_loss(bb,teacher=False);p=p.cpu()
            for i,x in enumerate(raw):
                t=x["n_events"];q=t//2
                pred=p[i,:t].tolist();true=x["target"].tolist();flat=x["flat"]
                pred_ev=[int(flat[j][2]) for j in pred]
                true_ev=[int(flat[j][2]) for j in true]

                # Since rule masking selects each semantic event exactly once,
                # compare the selected road point for each event independent of
                # event order.
                pm={int(flat[j][2]):_key(flat,j) for j in pred}
                tm={int(flat[j][2]):_key(flat,j) for j in true}
                cand_hits=sum(pm[e]==tm[e] for e in tm)

                L=0.0
                seq=[0]+pred
                for a,z in zip(seq[:-1],seq[1:]):
                    ea,ra,_,_=flat[a];eb,rb,_,_=flat[z]
                    L+=data.point_dist(ea,ra,eb,rb)
                gap=(L/x["opt"]-1)*100
                buckets[q].append({
                    "pred":L,"opt":x["opt"],"gap":gap,
                    "exact":int(pred==true),
                    "pointer_hits":sum(a==z for a,z in zip(pred,true)),"steps":t,
                    "event_exact":int(pred_ev==true_ev),
                    "event_hits":sum(a==z for a,z in zip(pred_ev,true_ev)),
                    "candidate_hits":cand_hits
                })

    def summarize(rows):
        pred=[r["pred"] for r in rows];opt=[r["opt"] for r in rows];g=[r["gap"] for r in rows]
        steps=sum(r["steps"] for r in rows);n=len(rows)
        return {
          "n":n,
          "gap":(float(np.mean(pred))/float(np.mean(opt))-1)*100,
          "mean_case_gap":float(np.mean(g)),
          "exact":100*sum(r["exact"] for r in rows)/n,
          "pointer":100*sum(r["pointer_hits"] for r in rows)/steps,
          "event_order_exact":100*sum(r["event_exact"] for r in rows)/n,
          "event_step_acc":100*sum(r["event_hits"] for r in rows)/steps,
          "candidate_by_event_acc":100*sum(r["candidate_hits"] for r in rows)/steps,
          "within_1pct":100*sum(x<=1 for x in g)/n,
          "within_3pct":100*sum(x<=3 for x in g)/n,
          "within_5pct":100*sum(x<=5 for x in g)/n,
          "within_10pct":100*sum(x<=10 for x in g)/n,
        }

    out={}
    allrows=[]
    for q,rows in buckets.items():
        if rows:
            out[f"p{q}"]=summarize(rows);allrows+=rows
    out["overall"]=summarize(allrows)
    return out

def run(seed,args):
    seed_all(seed)
    data=HistoricalExact(args.links,args.orders,args.labels,args.exact)
    tr,va,te=data.split(seed);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=loaders(data,tr,args.batch,shuffle=True,seed=seed)
    val=loaders(data,va,args.batch,shuffle=False,seed=seed)
    test=loaders(data,te,args.batch,shuffle=False,seed=seed)

    model=Model(data,args.hidden).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.pre_lr,weight_decay=1e-4)
    curve=[]
    for ep in range(1,args.pre_epochs+1):
        model.train();vals=[]
        for b in train:
            b=move(b,dev)
            loss,nl,e4,cl=candidate_aux_loss(model,b,args.candidate_pos_weight,args.candidate_lambda)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            vals.append((float(loss.detach()),float(nl),float(e4),float(cl)))
        if ep in (1,5,args.pre_epochs):
            pm=pretrain_metrics(model,val,dev)
            rec={"epoch":ep,"loss":float(np.mean([x[0] for x in vals])),
                 "node_loss":float(np.mean([x[1] for x in vals])),
                 "edge4_loss":float(np.mean([x[2] for x in vals])),
                 "candidate_aux_loss":float(np.mean([x[3] for x in vals])),
                 "val":pm}
            curve.append(rec);print("CAND_PRE",json.dumps(rec,sort_keys=True))

    # Standard same-LR decoder fine-tuning, because the previous diagnostic
    # showed reduced encoder LR was worse on both representative seeds.
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
        else: bad+=1
        if bad>=args.patience:break

    model.load_state_dict(state)
    result={"seed":seed,"candidate_pos_weight":args.candidate_pos_weight,
            "candidate_lambda":args.candidate_lambda,"best_epoch":best_ep,
            "stop_epoch":ep,"val_decoder_ce":best,"pretrain_curve":curve,
            "test":eval_decomposed(model,test,data,dev)}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print("CAND_RESULT",json.dumps(result,sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--candidate-pos-weight",type=float,default=3.0)
    ap.add_argument("--candidate-lambda",type=float,default=1.0)
    ap.add_argument("--hidden",type=int,default=64);ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--pre-epochs",type=int,default=8);ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--epochs",type=int,default=55);ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--patience",type=int,default=10)
    a=ap.parse_args();run(a.seed,a)

if __name__=="__main__":main()
