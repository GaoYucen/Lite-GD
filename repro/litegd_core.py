#!/usr/bin/env python3
"""Paper-aligned Lite-GD core diagnostic on the archived simplified Chengdu set.

This experiment intentionally isolates the two central online ingredients that
are actually specified by the WWW/TMC papers and can be tested with the
archived 21-point samples:
  1. a 2-layer road-topology GCN representation;
  2. domain distance/angle feature crossover + the paper gating equations;
  3. sequential attention decoder with the precedence/group rule mask.

It is NOT called a full paper reproduction because the archived simplified
samples place candidates on graph nodes rather than edge+ratio positions and
lack the full-route per-case labels required by the paper's node/edge
pre-training and filter modules.  Those are reconstructed separately from the
older historical files.
"""
from __future__ import annotations
import argparse, copy, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from strict_pointer import (
    build_directed_distance, canonical_target, route_length, seed_all,
)


class CoreDataset(Dataset):
    def __init__(self, raw, indices):
        self.raw=raw
        self.indices=np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self,j):
        idx=int(self.indices[j]); s=self.raw[idx]
        return {
            "coords": torch.as_tensor(np.asarray(s["Points"],dtype=np.float32)[:,:2]),
            "ids": torch.as_tensor(s["Points_id"],dtype=torch.long),
            "target": torch.as_tensor(canonical_target(s),dtype=torch.long),
            "raw_index": idx,
        }


def build_normalized_adjacency(nodes_path: Path, links_path: Path):
    nodes=pd.read_csv(nodes_path)
    links=pd.read_csv(links_path)
    n=int(nodes["Node"].max())+1
    # Aggregate both incoming and outgoing road neighbors.  This mirrors the
    # paper's neighborhood aggregation while avoiding an arbitrary one-way
    # choice for node context; route direction is still used by evaluation.
    pairs=set()
    for u,v in links[["Node_Start","Node_End"]].itertuples(index=False,name=None):
        u=int(u); v=int(v)
        pairs.add((u,v)); pairs.add((v,u))
    for i in range(n): pairs.add((i,i))
    rows=np.fromiter((a for a,b in pairs),dtype=np.int64)
    cols=np.fromiter((b for a,b in pairs),dtype=np.int64)
    deg=np.bincount(rows,minlength=n).astype(np.float32)
    vals=1.0/np.maximum(deg[rows],1.0)
    idx=torch.as_tensor(np.stack([rows,cols]),dtype=torch.long)
    val=torch.as_tensor(vals,dtype=torch.float32)
    A=torch.sparse_coo_tensor(idx,val,(n,n)).coalesce()
    # node feature order: [lat, lon] to match archived Points
    by_id=nodes.set_index("Node")
    lat=by_id["Latitude"].reindex(range(n)).to_numpy(dtype=np.float32)
    lon=by_id["Longitude"].reindex(range(n)).to_numpy(dtype=np.float32)
    X=np.stack([lat,lon],axis=1)
    mean=np.nanmean(X,axis=0,keepdims=True); std=np.nanstd(X,axis=0,keepdims=True)
    X=(np.nan_to_num(X,nan=mean.squeeze())-mean)/np.maximum(std,1e-6)
    return A,torch.as_tensor(X,dtype=torch.float32)


class NodeGCN(nn.Module):
    def __init__(self,in_dim=2,hidden=64,out_dim=64):
        super().__init__()
        self.w1=nn.Linear(in_dim,hidden)
        self.w2=nn.Linear(hidden,out_dim)
    def forward(self,A,X):
        h=torch.sparse.mm(A,X)
        h=torch.relu(self.w1(h))
        h=torch.sparse.mm(A,h)
        return self.w2(h)


def haversine_rows(coords):
    """Pairwise spherical distance matrix in km, BxNxN."""
    lat=torch.deg2rad(coords[:,:,0])
    lon=torch.deg2rad(coords[:,:,1])
    lat1=lat[:,:,None]; lat2=lat[:,None,:]
    lon1=lon[:,:,None]; lon2=lon[:,None,:]
    dlat=lat2-lat1; dlon=lon2-lon1
    a=torch.sin(dlat/2).square()+torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).square()
    return 6371.0088*2*torch.asin(torch.sqrt(torch.clamp(a,0,1)))


def angle_to_group_means(coords):
    """sin/cos bearing to driver/group means; output BxNx10."""
    # groups: driver singleton and four candidate sets of five.
    groups=[(0,1),(1,6),(6,11),(11,16),(16,21)]
    means=torch.stack([coords[:,lo:hi].mean(dim=1) for lo,hi in groups],dim=1) # B,5,2
    lat1=torch.deg2rad(coords[:,:,0])[:,:,None]
    lon1=torch.deg2rad(coords[:,:,1])[:,:,None]
    lat2=torch.deg2rad(means[:,:,0])[:,None,:]
    lon2=torch.deg2rad(means[:,:,1])[:,None,:]
    dlon=lon2-lon1
    y=torch.sin(dlon)*torch.cos(lat2)
    x=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
    bearing=torch.atan2(y,x)
    return torch.cat([torch.sin(bearing),torch.cos(bearing)],dim=-1)


class LiteGDCore(nn.Module):
    def __init__(self,A,node_x,mode="gcn_domain",dim=64,hidden=128):
        super().__init__()
        self.mode=mode
        self.register_buffer("A",A)
        self.register_buffer("node_x",node_x)
        self.gcn=NodeGCN(2,dim,dim)
        self.coord=nn.Sequential(nn.Linear(2,dim),nn.ReLU(),nn.Linear(dim,dim))
        self.dist_proj=nn.Sequential(nn.Linear(21,dim),nn.ReLU(),nn.Linear(dim,dim))
        self.angle_proj=nn.Sequential(nn.Linear(10,dim),nn.ReLU(),nn.Linear(dim,dim))
        self.gf1=nn.Linear(dim,dim); self.gf2=nn.Linear(dim,dim)
        self.ge1=nn.Linear(dim,dim); self.ge2=nn.Linear(dim,dim)
        rep_dim=dim if mode in ("gcn","domain","coord") else dim*2
        self.to_hidden=nn.Linear(rep_dim,hidden)
        self.k=nn.Linear(hidden,hidden,bias=False)
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,1,bias=False)

    @staticmethod
    def mask(selected_groups,device):
        b=selected_groups.shape[0]
        m=torch.zeros(b,21,dtype=torch.bool,device=device)
        m[:,0]=True
        for g in range(4):
            done=selected_groups[:,g]
            if done.any():
                lo=1+5*g
                m[done,lo:lo+5]=True
        # drop p1(group2) until pickup p1(group0); drop p2(group3) until pickup p2(group1)
        m[~selected_groups[:,0],11:16]=True
        m[~selected_groups[:,1],16:21]=True
        return m

    def candidate_repr(self,coords,ids):
        # normalized coords for coord embedding
        cm=coords.mean(dim=1,keepdim=True); cs=coords.std(dim=1,keepdim=True).clamp_min(1e-5)
        cfeat=(coords-cm)/cs
        coord_rep=self.coord(cfeat)
        if self.mode in ("gcn","gcn_domain"):
            all_h=self.gcn(self.A,self.node_x)
            road=all_h[ids]
        else:
            road=coord_rep

        if self.mode in ("domain","gcn_domain"):
            dist=haversine_rows(coords)
            # normalize by each instance's nonzero distance scale
            scale=dist.amax(dim=(1,2),keepdim=True).clamp_min(1e-5)
            d=self.dist_proj(dist/scale)
            a=self.angle_proj(angle_to_group_means(coords))
            fcp=d*a  # paper-suggested Hadamard feature crossover
            if self.mode=="domain":
                return fcp
            gf=torch.sigmoid(self.gf1(fcp)+self.gf2(road))
            ge=torch.sigmoid(self.ge1(fcp)+self.ge2(road))
            return torch.cat([gf*fcp,ge*road],dim=-1)
        return road

    def forward(self,coords,ids,target=None,teacher_forcing=True):
        rep=self.candidate_repr(coords,ids)
        h=self.to_hidden(rep)
        b=h.shape[0]
        current=h[:,0]
        selected=torch.zeros(b,4,dtype=torch.bool,device=h.device)
        logits_list=[]; greedy_list=[]
        for step in range(4):
            score=self.v(torch.tanh(self.k(h)+self.q(current)[:,None,:])).squeeze(-1)
            score=score.masked_fill(self.mask(selected,h.device),-1e9)
            greedy=score.argmax(dim=1)
            logits_list.append(score); greedy_list.append(greedy)
            if teacher_forcing and target is not None:
                chosen=target[:,step+1]
            else:
                chosen=greedy
            group=torch.div(chosen-1,5,rounding_mode="floor").clamp(0,3)
            selected.scatter_(1,group[:,None],True)
            current=h[torch.arange(b,device=h.device),chosen]
        logits=torch.stack(logits_list,dim=1)
        pred=torch.stack(greedy_list,dim=1)
        pred=torch.cat([torch.zeros(b,1,dtype=torch.long,device=h.device),pred],dim=1)
        return logits,pred


@torch.no_grad()
def evaluate(model,loader,D,raw,device):
    model.eval()
    exact=tok=cases=illegal=0
    lengths=[]; opts=[]
    loss_sum=0.0; n_tok=0
    ce=nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        coords=batch["coords"].to(device); ids=batch["ids"].to(device); y=batch["target"].to(device)
        logits_tf,_=model(coords,ids,target=y,teacher_forcing=True)
        _,p=model(coords,ids,target=None,teacher_forcing=False)
        loss_sum+=float(ce(logits_tf.reshape(-1,21),y[:,1:].reshape(-1))); n_tok+=y[:,1:].numel()
        pp=p.cpu().numpy(); yy=y.cpu().numpy()
        exact+=int(np.sum(np.all(pp==yy,axis=1))); tok+=int(np.sum(pp[:,1:]==yy[:,1:])); cases+=len(pp)
        for j in range(len(pp)):
            idx=int(batch["raw_index"][j]); s=raw[idx]
            lengths.append(route_length(D,s["Points_id"],pp[j]))
            opts.append(route_length(D,s["Points_id"],yy[j]))
            # semantic precedence/group uniqueness
            gs=[(int(q)-1)//5 for q in pp[j,1:]]
            if len(set(gs))<4 or gs.index(0)>gs.index(2) or gs.index(1)>gs.index(3): illegal+=1
    return {
      "ce":loss_sum/max(n_tok,1),
      "gap_pct":(float(np.mean(lengths))/float(np.mean(opts))-1)*100,
      "avg_pred_length":float(np.mean(lengths)),
      "avg_opt_length":float(np.mean(opts)),
      "exact_acc_pct":exact/max(cases,1)*100,
      "pointer_acc_pct":tok/max(cases*4,1)*100,
      "illegal":illegal,"n":cases,
    }


def run(raw,train_idx,val_idx,test_idx,A,node_x,D,mode,seed,args,device):
    seed_all(seed)
    tr=DataLoader(CoreDataset(raw,train_idx),batch_size=args.batch_size,shuffle=True,
                  generator=torch.Generator().manual_seed(seed))
    va=DataLoader(CoreDataset(raw,val_idx),batch_size=args.batch_size,shuffle=False)
    te=DataLoader(CoreDataset(raw,test_idx),batch_size=args.batch_size,shuffle=False)
    model=LiteGDCore(A.to(device),node_x.to(device),mode=mode,dim=args.dim,hidden=args.hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    ce=nn.CrossEntropyLoss()
    best=math.inf; state=None; stale=0; last=0
    for epoch in range(1,args.epochs+1):
        model.train()
        for batch in tr:
            coords=batch["coords"].to(device); ids=batch["ids"].to(device); y=batch["target"].to(device)
            logits,_=model(coords,ids,target=y,teacher_forcing=True)
            loss=ce(logits.reshape(-1,21),y[:,1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        vm=evaluate(model,va,D,raw,device); last=epoch
        if vm["ce"]+1e-6<best:
            best=vm["ce"]; state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if epoch==1 or epoch%10==0:
            print("progress",mode,epoch,vm,flush=True)
        if stale>=args.patience: break
    model.load_state_dict(state)
    return {"mode":mode,"seed":seed,"epochs":last,"val":evaluate(model,va,D,raw,device),"test":evaluate(model,te,D,raw,device)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--nodes",default="sim_data/chengdu_node-mod.txt")
    ap.add_argument("--links",default="sim_data/chengdu_link-mod.txt")
    ap.add_argument("--seed",type=int,default=20260925)
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--patience",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--dim",type=int,default=64)
    ap.add_argument("--hidden",type=int,default=128)
    ap.add_argument("--modes",default="coord,gcn,domain,gcn_domain")
    ap.add_argument("--out",default="")
    args=ap.parse_args()
    seed_all(args.seed)
    raw=np.load(args.data,allow_pickle=True); n=len(raw)
    perm=np.random.default_rng(args.seed).permutation(n); a=int(.8*n); b=int(.9*n)
    tr,va,te=perm[:a],perm[a:b],perm[b:]
    repaired=sum(int(not np.array_equal(canonical_target(s),np.asarray(s["Solutions"]))) for s in raw)
    print("split",len(tr),len(va),len(te),"canonical_repairs",repaired,flush=True)
    A,node_x=build_normalized_adjacency(Path(args.nodes),Path(args.links))
    D=build_directed_distance(Path(args.links),Path(args.nodes))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,flush=True)
    out=[]
    for mode in args.modes.split(","):
        print("RUN",mode,flush=True)
        r=run(raw,tr,va,te,A,node_x,D,mode,args.seed,args,device)
        print("RESULT",json.dumps(r),flush=True); out.append(r)
    doc={"seed":args.seed,"runs":out}
    text=json.dumps(doc,indent=2)
    print("FINAL_JSON\n"+text)
    if args.out: Path(args.out).write_text(text)


if __name__=="__main__":
    main()
