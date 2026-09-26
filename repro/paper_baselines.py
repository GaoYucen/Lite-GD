#!/usr/bin/env python3
"""Paper-baseline closure for the cross-graph Lite-GD benchmark.

This module adds the two paper baselines that were missing from the frozen
cross-graph matrix:

* DisGreedy-paper: nearest feasible candidate under geometric (spherical when
  coordinates are lon/lat) distance.  Road distance is used only for final
  evaluation, never for the greedy decision.
* Graph2Route-adapted: preserves the checked-in Graph2Route core pattern
  (2-layer gated GCN over a task-point graph + recurrent pointer decoder) while
  mapping unavailable delivery-time/courier features to the common MCRP input.
  It uses candidate-point geometry and event semantics, not road APSP costs.

All methods share the same frozen 8:1:1 split and directed-road evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from crossgraph_data import CrossGraphExact
from baseline_carpool import (
    CarpoolCases, make_batches, legal_mask, update_done, route_cost, seed_all
)
from evaluation import summarize_route_rows


def _evaluate_sequence(rows, cases, data):
    out=[]
    for cid,seq in rows:
        x=cases.item(cid)
        event=x["event"]; target=x["target"].tolist(); t=x["n_events"]
        seq=list(map(int,seq[:t]))
        pe=[int(event[j]) for j in seq]
        te=[int(event[j]) for j in target]
        pm={int(event[j]):j for j in seq}
        tm={int(event[j]):j for j in target}
        L=float(sum(x["cost"][a,z] for a,z in zip([0]+seq[:-1],seq)))
        out.append({
            "pred":L,"opt":x["opt"],"exact":int(seq==target),
            "pointer_hits":sum(a==z for a,z in zip(seq,target)),"steps":t,
            "event_exact":int(pe==te),"event_hits":sum(a==z for a,z in zip(pe,te)),
            "candidate_hits":sum(pm[e]==tm[e] for e in tm),
        })
    return summarize_route_rows(out)


def _is_lonlat(points):
    p=np.asarray(points,float)
    return bool(
        p.shape[-1]>=2 and
        np.nanmax(np.abs(p[:,0]))<=180.0 and np.nanmax(np.abs(p[:,1]))<=90.0
    )


def _geo_dist_row(cur, pts):
    cur=np.asarray(cur,float);pts=np.asarray(pts,float)
    if _is_lonlat(np.vstack([cur[None,:2],pts[:,:2]])):
        lon1,lat1=np.deg2rad(cur[0]),np.deg2rad(cur[1])
        lon2=np.deg2rad(pts[:,0]);lat2=np.deg2rad(pts[:,1])
        dlon=lon2-lon1;dlat=lat2-lat1
        a=np.sin(dlat/2.0)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return 6371008.8*2*np.arcsin(np.sqrt(np.clip(a,0,1))),"haversine"
    d=pts[:,:2]-cur[None,:2]
    return np.sqrt(np.square(d).sum(1)),"euclidean"


def disgreedy_paper(data, cases, ids):
    """Nearest legal candidate using geometry only; evaluate with road metric."""
    rows=[];metric=None
    get_case=data.case_light if hasattr(data,"case_light") else data.case_tensors
    for cid in ids:
        raw=get_case(int(cid))
        pts=np.asarray(raw["points"],dtype=np.float64)
        event=np.asarray([z[2] for z in raw["flat"]],dtype=np.int64)
        done=set();cur=0;seq=[]
        for _ in range(int(raw["n_events"])):
            legal={e for e in range(int(raw["n_events"]))
                   if e not in done and (e%2==0 or e-1 in done)}
            cand=[j for j in range(1,len(event)) if int(event[j]) in legal]
            dist,metric=_geo_dist_row(pts[cur],pts[cand])
            order=min(range(len(cand)),key=lambda q:(float(dist[q]),int(event[cand[q]]),cand[q]))
            j=cand[order];seq.append(j);done.add(int(event[j]));cur=j
        rows.append((int(cid),seq))
    quality=_evaluate_sequence(rows,cases,data)
    return quality,metric


class GatedGCNLayer(nn.Module):
    """Graph2Route-style residual gated GCN layer."""
    def __init__(self,h):
        super().__init__()
        self.node_u=nn.Linear(h,h);self.node_v=nn.Linear(h,h)
        self.edge_u=nn.Linear(h,h);self.edge_v=nn.Linear(h,h)
        self.node_norm=nn.LayerNorm(h);self.edge_norm=nn.LayerNorm(h)

    def forward(self,x,e):
        # e_ij <- U e_ij + V x_i + V x_j
        vx=self.edge_v(x)
        et=self.edge_u(e)+vx[:,:,None,:]+vx[:,None,:,:]
        gate=torch.sigmoid(et)
        nv=self.node_v(x)
        msg=(gate*nv[:,None,:,:]).sum(2)/gate.sum(2).clamp_min(1e-6)
        xt=self.node_u(x)+msg
        return x+torch.relu(self.node_norm(xt)), e+torch.relu(self.edge_norm(et))


class Graph2RouteCarpool(nn.Module):
    """Static-MCRP adaptation of the checked-in Graph2Route PD architecture."""
    def __init__(self,h=8,layers=2,event_dim=8,tanh_clip=10.0):
        super().__init__();self.h=h;self.tanh_clip=tanh_clip
        self.event_emb=nn.Embedding(7,event_dim)
        self.node_in=nn.Linear(2+event_dim,h)
        self.edge_in=nn.Linear(4,h)
        self.layers=nn.ModuleList([GatedGCNLayer(h) for _ in range(layers)])
        self.graph_proj=nn.Linear(h,h)
        self.start_proj=nn.Linear(3*h,h)
        self.cell=nn.LSTMCell(h,h)
        self.node_key=nn.Linear(h,h,bias=False)
        self.query=nn.Linear(h,h,bias=False)
        self.dynamic=nn.Sequential(nn.Linear(4,h),nn.ReLU(),nn.Linear(h,h))
        self.v=nn.Parameter(torch.empty(h))
        nn.init.uniform_(self.v,-1.0/math.sqrt(h),1.0/math.sqrt(h))

    @staticmethod
    def pair_features(coords):
        d=coords[:,None,:,:]-coords[:,:,None,:]  # i,j = coord_j - coord_i
        dx=d[...,0];dy=d[...,1]
        dist=torch.sqrt(dx.square()+dy.square()+1e-12)
        return torch.stack([dx,dy,dist,dist.square()],dim=-1)

    def encode(self,coords,event):
        ei=(event+1).clamp(min=0,max=6)
        x=torch.relu(self.node_in(torch.cat([coords,self.event_emb(ei)],dim=-1)))
        pf=self.pair_features(coords)
        e=torch.relu(self.edge_in(pf))
        for layer in self.layers:x,e=layer(x,e)
        return x,pf

    def forward(self,coords,event,n_events,teacher=None):
        nodes,pair=self.encode(coords,event)
        B,N,H=nodes.shape
        graph=self.graph_proj(nodes.mean(1))
        current=torch.zeros(B,dtype=torch.long,device=coords.device)
        cur=nodes[:,0]
        h=torch.tanh(self.start_proj(torch.cat([graph,cur,cur],dim=-1)))
        c=torch.zeros_like(h)
        done=torch.zeros(B,int(n_events.max()),dtype=torch.bool,device=coords.device)
        seq=[];losses=[];ar=torch.arange(B,device=coords.device)
        for step in range(int(n_events.max())):
            dyn=self.dynamic(pair[ar,current])
            q=self.query(h)[:,None,:]
            logits=torch.einsum("bnh,h->bn",torch.tanh(self.node_key(nodes)+q+dyn),self.v)
            logits=torch.tanh(logits)*self.tanh_clip
            bad=legal_mask(event,done)
            logits=logits.masked_fill(bad,-1e9)
            if teacher is not None:
                chosen=teacher[:,step]
                losses.append(nn.functional.cross_entropy(logits,chosen))
            else:
                chosen=logits.argmax(1)
            seq.append(chosen);update_done(done,event,chosen)
            cur=nodes[ar,chosen];h,c=self.cell(cur,(h,c));current=chosen
        return torch.stack(seq,1), (torch.stack(losses).mean() if losses else None)


def evaluate_g2r(model,cases,ids,data,device,batch=64):
    model.eval();rows=[]
    with torch.no_grad():
        for q in make_batches(ids,data,batch,seed=0,epoch=0,shuffle=False):
            b=cases.collate(q,device)
            seq,_=model(b["coords"],b["event"],b["n_events"])
            for i,cid in enumerate(q):
                rows.append((int(cid),seq[i].detach().cpu().tolist()))
    return _evaluate_sequence(rows,cases,data)


def train_g2r(args,data,cases,tr,va,device):
    model=Graph2RouteCarpool(args.hidden,args.layers).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float("inf");best_state=None;bad=0;history=[]
    for ep in range(1,args.epochs+1):
        model.train();vals=[]
        for q in make_batches(tr,data,args.batch,seed=args.seed,epoch=ep,shuffle=True):
            b=cases.collate(q,device)
            _,loss=model(b["coords"],b["event"],b["n_events"],teacher=b["target"])
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            vals.append(float(loss.detach()))
        if ep==1 or ep%args.eval_every==0:
            v=evaluate_g2r(model,cases,va,data,device,args.eval_batch)
            score=float(v["mean_case_gap"])
            rec={"epoch":ep,"train_ce":float(np.mean(vals)),"val":v}
            history.append(rec);print("G2R_VAL",json.dumps(rec,sort_keys=True),flush=True)
            if score<best-1e-6:
                best=score;bad=0
                best_state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            else:bad+=args.eval_every
            if bad>=args.patience:break
    if best_state is None:raise RuntimeError("no Graph2Route checkpoint")
    model.load_state_dict(best_state)
    return model,history


def runtime_graph2route(model,cases,ids,data,device,n=100,warm=10):
    ids=list(ids[:min(n,len(ids))])
    batches=[cases.collate([cid],device) for cid in ids]
    model.eval()
    with torch.no_grad():
        for b in batches[:min(warm,len(batches))]:
            model(b["coords"],b["event"],b["n_events"])
        if device.type=="cuda":torch.cuda.synchronize()
        xs=[]
        for b in batches:
            if device.type=="cuda":torch.cuda.synchronize()
            t0=time.perf_counter()
            model(b["coords"],b["event"],b["n_events"])
            if device.type=="cuda":torch.cuda.synchronize()
            xs.append((time.perf_counter()-t0)*1000)
    a=np.asarray(xs,float)
    return {"n":len(a),"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),
            "p95_ms":float(np.percentile(a,95)),"qps":float(1000/a.mean())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--apsp",required=True,type=Path)
    ap.add_argument("--model",choices=["disgreedy","graph2route"],required=True)
    ap.add_argument("--seed",type=int,default=4321)
    ap.add_argument("--split-seed",type=int,default=20260925)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--checkpoint",type=Path)
    # Checked-in Graph2Route defaults are hidden=8, GCN layers=2.
    ap.add_argument("--hidden",type=int,default=8)
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--eval-batch",type=int,default=128)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--eval-every",type=int,default=5)
    ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--lr",type=float,default=1e-2)
    ap.add_argument("--weight-decay",type=float,default=1e-5)
    ap.add_argument("--runtime",action="store_true")
    args=ap.parse_args()

    seed_all(args.seed)
    data=CrossGraphExact(args.benchmark_dir,args.apsp,require_routes=False)
    tr,va,te=data.split(args.split_seed)
    cases=CarpoolCases(data,tr)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model=="disgreedy":
        t0=time.perf_counter();quality,metric=disgreedy_paper(data,cases,te)
        elapsed=time.perf_counter()-t0
        result={"model":"DisGreedy-paper","decision_metric":metric,"split_seed":args.split_seed,
                "test":quality,"runtime":{"test_cases":len(te),"total_s":elapsed,
                "mean_ms_per_query":1000*elapsed/max(len(te),1)}}
    else:
        model,hist=train_g2r(args,data,cases,tr,va,dev)
        test=evaluate_g2r(model,cases,te,data,dev,args.eval_batch)
        result={"model":"Graph2Route-adapted","source":"checked-in Graph2Route gated-GCN + sequential decoder",
                "hidden":args.hidden,"gcn_layers":args.layers,"training_seed":args.seed,
                "split_seed":args.split_seed,"history":hist,"test":test}
        if args.runtime:result["runtime"]=runtime_graph2route(model,cases,te,data,dev)
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":"graph2route","state_dict":model.state_dict(),
                        "hidden":args.hidden,"layers":args.layers,"seed":args.seed},args.checkpoint)

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("PAPER_BASELINE_RESULT",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
