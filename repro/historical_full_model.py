#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from historical_chengdu import load_edges, load_orders, load_labels, recover_cases

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class HistoricalExact:
    def __init__(self, links:Path, orders:Path, labels:Path, exact_json:Path):
        self.edges=load_edges(links)
        self.cases=recover_cases(self.edges,load_orders(orders),load_labels(labels))
        exact={int(x["case_id"]):x for x in json.loads(exact_json.read_text())}
        self.exact=exact
        self.case_by_id={int(c.case_id):c for c in self.cases}
        self.edge_ids=self.edges["edge_id"].to_numpy(np.int64)
        self.e2i={int(e):i for i,e in enumerate(self.edge_ids)}
        self.n_edges=len(self.edge_ids)
        self.n_nodes=int(max(self.edges.node_start.max(),self.edges.node_end.max()))+1
        self.src=self.edges.node_start.to_numpy(np.int64)
        self.dst=self.edges.node_end.to_numpy(np.int64)

        # node coordinates from incident edge records
        lon=np.zeros(self.n_nodes,np.float64);lat=np.zeros(self.n_nodes,np.float64);cnt=np.zeros(self.n_nodes)
        for r in self.edges.itertuples(index=False):
            for v,x,y in ((int(r.node_start),float(r.lon_start),float(r.lat_start)),
                          (int(r.node_end),float(r.lon_end),float(r.lat_end))):
                lon[v]+=x;lat[v]+=y;cnt[v]+=1
        cnt=np.maximum(cnt,1);lon/=cnt;lat/=cnt
        self.node_xy=np.stack([lon,lat],1)
        mu=self.node_xy.mean(0);sd=self.node_xy.std(0)+1e-8
        self.node_feat=((self.node_xy-mu)/sd).astype(np.float32)

        eb=np.stack([
          self.edges.lon_start.to_numpy(float),self.edges.lat_start.to_numpy(float),
          self.edges.lon_end.to_numpy(float),self.edges.lat_end.to_numpy(float),
          np.log1p(self.edges.length.to_numpy(float))
        ],1)
        em=eb.mean(0);es=eb.std(0)+1e-8
        self.edge_base=((eb-em)/es).astype(np.float32)

        # road shortest paths for evaluation
        best={}
        for r in self.edges.itertuples(index=False):
            k=(int(r.node_start),int(r.node_end));w=float(r.length)
            if k not in best or w<best[k]:best[k]=w
        rr=np.fromiter((k[0] for k in best),dtype=np.int32)
        cc=np.fromiter((k[1] for k in best),dtype=np.int32)
        vv=np.fromiter(best.values(),dtype=np.float64)
        self.D=dijkstra(csr_matrix((vv,(rr,cc)),shape=(self.n_nodes,self.n_nodes)),directed=True)

        self.max_points=max(1+sum(len(g) for g in c.candidate_groups) for c in self.cases)

    def point_coord(self,eid,ratio):
        r=self.edges.loc[int(eid)];q=float(ratio)
        return np.array([float(r.lon_start)+q*(float(r.lon_end)-float(r.lon_start)),
                         float(r.lat_start)+q*(float(r.lat_end)-float(r.lat_start))],np.float32)

    def point_dist(self,a,ra,b,rb):
        A=self.edges.loc[int(a)];B=self.edges.loc[int(b)]
        z=(1-float(ra))*float(A.length)+self.D[int(A.node_end),int(B.node_start)]+float(rb)*float(B.length)
        if int(a)==int(b) and float(rb)>=float(ra):
            z=min(z,(float(rb)-float(ra))*float(A.length))
        return float(z)

    def split(self,seed):
        rng=np.random.default_rng(seed);out=[[],[],[]]
        for p in sorted(set(c.passenger_count for c in self.cases)):
            ids=np.array([c.case_id for c in self.cases if c.passenger_count==p],dtype=np.int64)
            ids=rng.permutation(ids);a=int(.8*len(ids));b=int(.9*len(ids))
            for z,x in zip(out,[ids[:a],ids[a:b],ids[b:]]):z.extend(map(int,x))
        return out

    def case_tensors(self,cid):
        c=self.case_by_id[int(cid)];ex=self.exact[int(cid)]
        # semantic groups keyed by event type
        groups={int(g[0].event_type):g for g in c.candidate_groups if g}
        n_events=2*c.passenger_count

        role=np.zeros((self.n_edges,7),np.float32)
        rval=np.zeros((self.n_edges,7),np.float32)
        di=self.e2i[int(c.driver_edge)];role[di,0]=1;rval[di,0]=float(c.driver_ratio)
        flat=[(int(c.driver_edge),float(c.driver_ratio),-1,-1)]
        for et in range(n_events):
            for cand in groups[et]:
                ei=self.e2i[int(cand.edge_id)];ch=et+1
                role[ei,ch]=1;rval[ei,ch]=float(cand.ratio)
                flat.append((int(cand.edge_id),float(cand.ratio),et,int(cand.local_index)))

        edge_route={self.e2i[int(e)] for e in ex["exact_edge_route"] if int(e) in self.e2i}
        cand_edges={self.e2i[e] for e,_,_,_ in flat}
        ey=np.ones(self.n_edges,np.int64) # neither
        for i in edge_route: ey[i]=2
        for i in cand_edges: ey[i]=0
        for i in edge_route & cand_edges: ey[i]=3

        ny=np.zeros(self.n_nodes,np.int64)
        for v in ex["exact_node_route"]:
            if 0<=int(v)<self.n_nodes:ny[int(v)]=1

        # exact target indexes in flattened candidate list
        target=[]
        for et,e,r,li in zip(ex["exact_event_sequence"][1:],ex["exact_selected_edges"][1:],
                             ex["exact_selected_ratios"][1:],ex["exact_local_indices"][1:]):
            hits=[i for i,(ee,rr,tt,ll) in enumerate(flat)
                  if tt==int(et) and ll==int(li) and ee==int(e) and abs(rr-float(r))<1e-9]
            if not hits:
                hits=[i for i,(ee,rr,tt,ll) in enumerate(flat) if tt==int(et) and ee==int(e) and abs(rr-float(r))<1e-9]
            if not hits: raise RuntimeError(f"case {cid}: exact target not found")
            target.append(hits[0])

        pts=np.stack([self.point_coord(e,r) for e,r,_,_ in flat])
        return dict(cid=int(cid),role=role,ratio=rval,node_y=ny,edge_y=ey,flat=flat,points=pts,
                    target=np.asarray(target,np.int64),opt=float(ex["exact_length"]),n_events=n_events)

class Cases(Dataset):
    def __init__(self,data,ids):self.data=data;self.ids=list(ids)
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return self.data.case_tensors(self.ids[i])

def collate(batch):
    B=len(batch);E=batch[0]["role"].shape[0];N=max(len(x["flat"]) for x in batch);T=max(x["n_events"] for x in batch)
    role=torch.tensor(np.stack([x["role"] for x in batch]))
    ratio=torch.tensor(np.stack([x["ratio"] for x in batch]))
    node_y=torch.tensor(np.stack([x["node_y"] for x in batch]))
    edge_y=torch.tensor(np.stack([x["edge_y"] for x in batch]))
    edge_idx=torch.zeros(B,N,dtype=torch.long);event=torch.full((B,N),-2,dtype=torch.long)
    coords=torch.zeros(B,N,2);valid=torch.zeros(B,N,dtype=torch.bool);target=torch.full((B,T),-100,dtype=torch.long)
    n_events=torch.tensor([x["n_events"] for x in batch])
    for b,x in enumerate(batch):
        valid[b,:len(x["flat"])]=True;coords[b,:len(x["flat"])]=torch.tensor(x["points"])
        for j,(e,r,t,l) in enumerate(x["flat"]):
            edge_idx[b,j]=x["_e2i"][e] if "_e2i" in x else 0
            event[b,j]=t
        target[b,:len(x["target"])]=torch.tensor(x["target"])
    return dict(role=role,ratio=ratio,node_y=node_y,edge_y=edge_y,edge_idx=edge_idx,event=event,
                coords=coords,valid=valid,target=target,n_events=n_events,raw=batch)

class JointEncoder(nn.Module):
    def __init__(self,node_feat,edge_base,src,dst,h=64):
        super().__init__();self.h=h
        self.register_buffer("node_feat",torch.tensor(node_feat))
        self.register_buffer("edge_base",torch.tensor(edge_base))
        self.register_buffer("src",torch.tensor(src,dtype=torch.long))
        self.register_buffer("dst",torch.tensor(dst,dtype=torch.long))
        deg=np.zeros(len(node_feat),np.float32)
        np.add.at(deg,src,1);np.add.at(deg,dst,1)
        self.register_buffer("node_deg",torch.tensor(np.maximum(deg,1.0)).view(1,-1,1))
        self.node_in=nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h))
        self.edge_in=nn.Sequential(nn.Linear(19,h),nn.ReLU(),nn.Linear(h,h))
        self.node_up=nn.ModuleList([nn.Linear(2*h,h),nn.Linear(2*h,h)])
        self.eq=nn.ModuleList([nn.Linear(h,h,bias=False),nn.Linear(h,h,bias=False)])
        self.ek=nn.ModuleList([nn.Linear(h,h,bias=False),nn.Linear(h,h,bias=False)])
        self.edge_up=nn.ModuleList([nn.Linear(2*h,h),nn.Linear(2*h,h)])

    def forward(self,role,ratio):
        # Case-conditioned joint message passing:
        # edge(candidate type/ratio) -> incident nodes -> edges.
        B=role.size(0)
        n=torch.relu(self.node_in(self.node_feat)).unsqueeze(0).expand(B,-1,-1)
        base=self.edge_base.unsqueeze(0).expand(B,-1,-1)
        e=torch.relu(self.edge_in(torch.cat([base,role,ratio],-1)))

        for l in range(2):
            # Candidate-conditioned edge messages make node representations
            # different across cases, which is required by route-membership
            # node pre-training.
            agg=torch.zeros_like(n)
            agg.index_add_(1,self.src,e)
            agg.index_add_(1,self.dst,e)
            agg=agg/self.node_deg
            n=torch.relu(self.node_up[l](torch.cat([n,agg],-1)))

            # Paper-style endpoint aggregation back into each road edge.
            v=torch.stack([n[:,self.src],n[:,self.dst]],2)  # B,E,2,H
            q=self.eq[l](e).unsqueeze(2)                     # B,E,1,H
            k=self.ek[l](v)                                  # B,E,2,H
            a=torch.softmax((q*k).sum(-1)/math.sqrt(self.h),dim=2)
            endpoint=(a.unsqueeze(-1)*v).sum(2)
            e=torch.relu(self.edge_up[l](torch.cat([e,endpoint],-1)))
        return n,e

class PaperDecoder(nn.Module):
    def __init__(self,h,max_points):
        super().__init__();d=h//2;self.max_points=max_points
        self.fd=nn.Linear(max_points,d);self.fa=nn.Linear(2*max_points,d)
        self.gfc1=nn.Linear(d,d);self.gfc2=nn.Linear(h,d);self.ge1=nn.Linear(d,h);self.ge2=nn.Linear(h,h)
        self.merge=nn.Linear(d+h,h);self.cand=nn.Linear(h,h);self.state=nn.Linear(h,h);self.v=nn.Linear(h,1,bias=False)
        self.gru=nn.GRUCell(h,h)
    def domain(self,coords,valid):
        # coords lon/lat; pairwise haversine + bearing
        lon=torch.deg2rad(coords[...,0]);lat=torch.deg2rad(coords[...,1])
        dlat=lat[:,None,:]-lat[:,:,None];dlon=lon[:,None,:]-lon[:,:,None]
        h=torch.sin(dlat/2)**2+torch.cos(lat[:,:,None])*torch.cos(lat[:,None,:])*torch.sin(dlon/2)**2
        dist=6371000*2*torch.asin(torch.sqrt(h.clamp(0,1)))/10000.0
        y=torch.sin(dlon)*torch.cos(lat[:,None,:])
        x=torch.cos(lat[:,:,None])*torch.sin(lat[:,None,:])-torch.sin(lat[:,:,None])*torch.cos(lat[:,None,:])*torch.cos(dlon)
        ang=torch.atan2(y,x)
        B,N,_=dist.shape;M=self.max_points
        dv=torch.zeros(B,N,M,device=coords.device);av=torch.zeros(B,N,2*M,device=coords.device)
        dv[:,:,:N]=dist;av[:,:,:N]=torch.sin(ang);av[:,:,M:M+N]=torch.cos(ang)
        pair=valid[:,:,None]&valid[:,None,:]
        dv[:,:,:N]*=pair;av[:,:,:N]*=pair;av[:,:,M:M+N]*=pair
        return torch.tanh(self.fd(dv))*torch.tanh(self.fa(av))
    def forward(self,edge_h,edge_idx,event,coords,valid,target,n_events,teacher=True):
        B,N=edge_idx.shape;H=edge_h.size(-1)
        cand=torch.gather(edge_h,1,edge_idx.unsqueeze(-1).expand(-1,-1,H))
        fc=self.domain(coords,valid)
        gfc=torch.sigmoid(self.gfc1(fc)+self.gfc2(cand))
        ge=torch.sigmoid(self.ge1(fc)+self.ge2(cand))
        rep=torch.tanh(self.merge(torch.cat([gfc*fc,ge*cand],-1)))
        state=rep[:,0];done=torch.zeros(B,6,dtype=torch.bool,device=rep.device);loss=[];preds=[]
        for t in range(6):
            score=self.v(torch.tanh(self.cand(rep)+self.state(state)[:,None,:])).squeeze(-1)
            bad=~valid | (event<0)
            for b in range(B):
                if t>=int(n_events[b]):bad[b]=True;continue
                for j in range(N):
                    et=int(event[b,j])
                    if et<0:continue
                    if done[b,et] or (et%2==1 and not done[b,et-1]):bad[b,j]=True
            score=score.masked_fill(bad,-1e9);p=score.argmax(1);preds.append(p)
            active=target[:,t]!=-100
            if active.any():loss.append(nn.functional.cross_entropy(score[active],target[active,t]))
            choose=torch.where(active & teacher,target[:,t],p)
            for b in range(B):
                if active[b]:
                    et=int(event[b,choose[b]])
                    if et>=0:done[b,et]=True
            state=self.gru(rep[torch.arange(B,device=rep.device),choose],state)
        return sum(loss)/max(1,len(loss)),torch.stack(preds,1)

class Model(nn.Module):
    def __init__(self,data,h=64):
        super().__init__()
        self.encoder=JointEncoder(data.node_feat,data.edge_base,data.src,data.dst,h)
        self.node_head=nn.Linear(h,2);self.edge_head=nn.Linear(h,4)
        self.decoder=PaperDecoder(h,data.max_points)
    def encode(self,b):return self.encoder(b["role"],b["ratio"])
    def pretrain_loss(self,b,balanced=True):
        nh,eh=self.encode(b)
        if balanced:
            # Sample all positives/special edges plus <=5x random 'neither' edges.
            node_logits=self.node_head(nh)
            ny=b["node_y"];pos=(ny==1)
            posw=(~pos).sum().float()/pos.sum().clamp_min(1).float()
            nw=torch.tensor([1.0,float(min(posw,30))],device=ny.device)
            nl=nn.functional.cross_entropy(node_logits.reshape(-1,2),ny.reshape(-1),weight=nw)
            logits=self.edge_head(eh);ey=b["edge_y"]
            keep=(ey!=1)
            rand=torch.rand_like(ey.float());special=keep.sum(1,keepdim=True).clamp_min(1)
            prob=(5*special/ey.size(1)).clamp(max=1).float()
            keep=keep | ((ey==1)&(rand<prob))
            el=nn.functional.cross_entropy(logits[keep],ey[keep])
        else:
            node_logits=self.node_head(nh)
            nl=nn.functional.cross_entropy(node_logits.reshape(-1,2),b["node_y"].reshape(-1))
            el=nn.functional.cross_entropy(self.edge_head(eh).reshape(-1,4),b["edge_y"].reshape(-1))
        return nl+el,nl.detach(),el.detach()
    def decoder_loss(self,b,teacher=True):
        _,eh=self.encode(b)
        return self.decoder(eh,b["edge_idx"],b["event"],b["coords"],b["valid"],b["target"],b["n_events"],teacher)

def attach_edge_indices(batch,data):
    for x in batch:
        x["_e2i"]=data.e2i

def move(b,dev):
    return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}

def loaders(data,ids,batch=6):
    ds=Cases(data,ids)
    def cf(items):
        attach_edge_indices(items,data);return collate(items)
    return DataLoader(ds,batch_size=batch,shuffle=True,collate_fn=cf,num_workers=0)

def eval_model(model,loader,data,dev):
    model.eval();gaps=[];exact=0;ptr=0;tot=0;illegal=0
    with torch.no_grad():
        for b in loader:
            raw=b["raw"];b=move(b,dev);_,p=model.decoder_loss(b,teacher=False);p=p.cpu()
            for i,x in enumerate(raw):
                t=x["n_events"];pred=p[i,:t].tolist();true=x["target"].tolist()
                exact+=int(pred==true);ptr+=sum(a==z for a,z in zip(pred,true));tot+=t
                # predicted event validity is guaranteed by mask; reconstruct point route
                seq=[0]+pred
                flat=x["flat"];L=0.0
                for a,z in zip(seq[:-1],seq[1:]):
                    ea,ra,_,_=flat[a];eb,rb,_,_=flat[z];L+=data.point_dist(ea,ra,eb,rb)
                gaps.append((L/x["opt"]-1)*100)
    return dict(gap=float(np.mean(gaps)),gap_std=float(np.std(gaps)),exact=100*exact/len(gaps),pointer=100*ptr/tot,illegal=illegal)

def run(seed,args,data):
    seed_all(seed);tr,va,te=data.split(seed);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=loaders(data,tr,args.batch);val=loaders(data,va,args.batch);test=loaders(data,te,args.batch)
    out=[]
    for use_pre in (False,True):
        seed_all(seed)
        model=Model(data,args.hidden).to(dev)
        if use_pre:
            opt=torch.optim.AdamW(model.parameters(),lr=args.pre_lr,weight_decay=1e-4)
            for ep in range(1,args.pre_epochs+1):
                model.train();ls=[]
                for b in train:
                    b=move(b,dev);loss,_,_=model.pretrain_loss(b,balanced=True)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();ls.append(float(loss))
                if ep==1 or ep%5==0:print("pre",seed,ep,float(np.mean(ls)))
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        best=1e9;state=None;bad=0
        for ep in range(1,args.epochs+1):
            model.train()
            for b in train:
                b=move(b,dev);loss,_=model.decoder_loss(b,teacher=True)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            model.eval();vl=[]
            with torch.no_grad():
                for b in val:
                    b=move(b,dev);loss,_=model.decoder_loss(b,teacher=True);vl.append(float(loss))
            v=float(np.mean(vl))
            if v<best-1e-5:
                best=v;bad=0;state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            else:bad+=1
            if bad>=args.patience:break
        model.load_state_dict(state);m=eval_model(model,test,data,dev)
        row={"seed":seed,"pretrained":use_pre,"epochs":ep,**m};out.append(row);print("RESULT",json.dumps(row,sort_keys=True))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path);ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path);ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--seeds",nargs="+",type=int,default=[20260925]);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--hidden",type=int,default=64);ap.add_argument("--batch",type=int,default=6)
    ap.add_argument("--pre-epochs",type=int,default=12);ap.add_argument("--pre-lr",type=float,default=1e-3)
    ap.add_argument("--epochs",type=int,default=70);ap.add_argument("--lr",type=float,default=5e-4);ap.add_argument("--patience",type=int,default=12)
    a=ap.parse_args()
    data=HistoricalExact(a.links,a.orders,a.labels,a.exact);rows=[]
    for s in a.seeds:rows+=run(s,a,data)
    summary={}
    for p in (False,True):
        z=[r for r in rows if r["pretrained"]==p]
        summary[str(p)]={k:float(np.mean([r[k] for r in z])) for k in ("gap","exact","pointer")}
        summary[str(p)]["gap_seed_std"]=float(np.std([r["gap"] for r in z]))
    doc={"rows":rows,"summary":summary}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(doc,indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True))

if __name__=="__main__":main()
