import argparse, ast, math, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA=Path("/workspace/.server-control/litegd-paper-data")

def nonempty(p): return [x for x in p.read_text().splitlines() if x.strip()]
def parts(s,cast=int):
    x=[z for z in s.split(";") if z]
    return cast(x[0]),[[cast(v) for v in z.split(",") if v] for z in x[1:]]

def recover_group_types(groups,seq,selected):
    q=len(groups)//2;out=[None]*len(groups)
    for parity,grange in ((0,range(q)),(1,range(q,2*q))):
        items=[(t,e) for t,e in zip(seq[1:],selected[1:]) if t%2==parity]
        sols=[]
        def bt(k,used,cur):
            if k==len(items): sols.append(cur.copy());return
            t,e=items[k]
            for g in grange:
                if g not in used and e in groups[g]:
                    used.add(g);cur[t]=g;bt(k+1,used,cur);cur.pop(t);used.remove(g)
        bt(0,set(),{})
        if not sols: raise RuntimeError(f"cannot recover semantics {seq} {selected}")
        for t,g in sols[0].items():out[g]=t
    if any(x is None for x in out):raise RuntimeError("incomplete semantics")
    return out

class Corpus:
    def __init__(self):
        self.nodes=pd.read_csv(DATA/"chengdu_node.txt",sep=r"\s+",header=None,names=["id","lon","lat","flag"])
        self.links=pd.read_csv(DATA/"chengdu_link.txt",sep=r"\s+",header=None,names=["id","u","v","w","flag"])
        self.edge={int(r.id):(int(r.u),int(r.v),float(r.w)) for r in self.links.itertuples(index=False)}
        self.xy={int(r.id):(float(r.lat),float(r.lon)) for r in self.nodes.itertuples(index=False)}
        n=max(max(u,v) for u,v,w in self.edge.values())+1
        self.sd=dijkstra(csr_matrix((self.links.w,(self.links.u,self.links.v)),shape=(n,n)),directed=True)
        ol=nonempty(DATA/"chengdu_order_1000.txt");self.orders={}
        for i in range(0,len(ol),3):
            c=int(ol[i]);car,g=parts(ol[i+1]);cr,r=parts(ol[i+2],float);self.orders[c]=(car,g,cr,r)
        ll=nonempty(DATA/"chengdu_label_1000.txt");self.labels={}
        for i in range(0,len(ll),8):
            c=int(ll[i]);self.labels[c]=(ast.literal_eval(ll[i+3]),[int(x) for x in ll[i+4].split(";") if x],float(ll[i+7].split(":")[1]))
        self.lat_mean=np.mean([x[0] for x in self.xy.values()]);self.lat_std=np.std([x[0] for x in self.xy.values()])
        self.lon_mean=np.mean([x[1] for x in self.xy.values()]);self.lon_std=np.std([x[1] for x in self.xy.values()])
        self.logw_mean=float(np.log1p(self.links.w).mean());self.logw_std=float(np.log1p(self.links.w).std())
    def coord(self,e,r):
        u,v,w=self.edge[e];a=np.array(self.xy[u]);b=np.array(self.xy[v]);return a*(1-r)+b*r
    def road(self,a,ra,b,rb):
        ua,va,wa=self.edge[a];ub,vb,wb=self.edge[b]
        z=(1-ra)*wa+self.sd[va,ub]+rb*wb
        return float(min(z,(rb-ra)*wa) if a==b and rb>=ra else z)

def geo_features(points,groups):
    anchors=[points[0]]
    for g in range(6):
        ids=np.where(groups==g)[0];anchors.append(points[ids].mean(0) if len(ids) else points[0])
    A=np.asarray(anchors);P=points
    lat1=np.radians(P[:,0,None]);lat2=np.radians(A[None,:,0])
    dlat=lat2-lat1;dlon=np.radians(A[None,:,1]-P[:,1,None])
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist=6371000*2*np.arcsin(np.sqrt(np.clip(h,0,1)))/10000.0
    y=np.sin(dlon)*np.cos(lat2)
    x=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    ang=np.arctan2(y,x)
    return dist.astype("float32"),np.concatenate([np.sin(ang),np.cos(ang)],1).astype("float32")

class Cases(Dataset):
    def __init__(self,C,ids):self.C=C;self.ids=list(ids)
    def __len__(self):return len(self.ids)
    def __getitem__(self,k):
        c=self.ids[k];car,G,cr,R=self.C.orders[c];seq,sel,opt=self.C.labels[c];q=len(G)//2
        gtypes=recover_group_types(G,seq,sel);type_to_group={t:g for g,t in enumerate(gtypes)}
        prereq=np.full(6,-1,np.int64)
        for g,t in enumerate(gtypes):
            if t%2==1:prereq[g]=type_to_group[t-1]
        es=[car];rs=[cr];gs=[-1];types=[-1]
        for g,(aa,rr) in enumerate(zip(G,R)):
            es+=aa;rs+=rr;gs += [g]*len(aa);types += [gtypes[g]]*len(aa)
        pts=np.stack([self.C.coord(e,r) for e,r in zip(es,rs)])
        base=[]
        for e,r,g,t,p in zip(es,rs,gs,types,pts):
            u,v,w=self.C.edge[e];one=np.zeros(7,np.float32);one[0 if g<0 else t+1]=1
            base.append([(p[0]-self.C.lat_mean)/self.C.lat_std,(p[1]-self.C.lon_mean)/self.C.lon_std,
                         (math.log1p(w)-self.C.logw_mean)/self.C.logw_std,r]+one.tolist())
        dist,angle=geo_features(pts,np.asarray(gs))
        offsets=[];z=1
        for a in G:offsets.append(z);z+=len(a)
        target=[]
        for typ,e in zip(seq[1:],sel[1:]):
            g=type_to_group[typ];target.append(offsets[g]+G[g].index(e))
        return dict(case=c,base=torch.tensor(base),dist=torch.tensor(dist),angle=torch.tensor(angle),
                    group=torch.tensor(gs),target=torch.tensor(target),q=q,prereq=torch.tensor(prereq),
                    edges=es,ratios=rs,opt=opt)

def collate(batch):
    B=len(batch);N=max(len(x["group"]) for x in batch);T=6
    base=torch.zeros(B,N,11);dist=torch.zeros(B,N,7);angle=torch.zeros(B,N,14)
    group=torch.full((B,N),-2,dtype=torch.long);target=torch.full((B,T),-100,dtype=torch.long)
    q=torch.tensor([x["q"] for x in batch]);prereq=torch.stack([x["prereq"] for x in batch])
    for i,x in enumerate(batch):
        n=len(x["group"]);t=len(x["target"])
        base[i,:n]=x["base"];dist[i,:n]=x["dist"];angle[i,:n]=x["angle"];group[i,:n]=x["group"];target[i,:t]=x["target"]
    return base,dist,angle,group,target,q,prereq,batch

class DomainPointer(nn.Module):
    def __init__(self,h=96):
        super().__init__();d=h//2
        self.edge=nn.Sequential(nn.Linear(11,d),nn.ReLU(),nn.Linear(d,d))
        self.fd=nn.Linear(7,d);self.fa=nn.Linear(14,d)
        self.gfc1=nn.Linear(d,d);self.gfc2=nn.Linear(d,d);self.ge1=nn.Linear(d,d);self.ge2=nn.Linear(d,d)
        self.merge=nn.Linear(2*d,h);self.cand=nn.Linear(h,h);self.state=nn.Linear(h,h);self.v=nn.Linear(h,1,bias=False)
        self.gru=nn.GRUCell(h,h)
    def encode(self,base,dist,angle):
        e=torch.tanh(self.edge(base));fc=torch.tanh(self.fd(dist))*torch.tanh(self.fa(angle))
        gfc=torch.sigmoid(self.gfc1(fc)+self.gfc2(e));ge=torch.sigmoid(self.ge1(fc)+self.ge2(e))
        return torch.tanh(self.merge(torch.cat([gfc*fc,ge*e],-1)))
    def forward(self,base,dist,angle,group,target,q,prereq,teacher=True):
        x=self.encode(base,dist,angle);B,N,H=x.shape;state=x[:,0]
        picked=torch.zeros(B,6,dtype=torch.bool,device=x.device);losses=[];preds=[]
        for t in range(6):
            score=self.v(torch.tanh(self.cand(x)+self.state(state)[:,None,:])).squeeze(-1)
            bad=(group<0)
            for b in range(B):
                if t>=2*int(q[b]):bad[b]=True;continue
                for j in range(N):
                    g=int(group[b,j])
                    if g<0:continue
                    pre=int(prereq[b,g])
                    if picked[b,g] or (pre>=0 and not picked[b,pre]):bad[b,j]=True
            score=score.masked_fill(bad,-1e9);pred=score.argmax(1);preds.append(pred)
            active=target[:,t]!=-100
            if active.any():losses.append(nn.functional.cross_entropy(score[active],target[active,t]))
            choose=torch.where(active & teacher,target[:,t],pred)
            for b in range(B):
                if active[b]:
                    g=int(group[b,choose[b]])
                    if g>=0:picked[b,g]=True
            state=self.gru(x[torch.arange(B,device=x.device),choose],state)
        return sum(losses)/max(1,len(losses)),torch.stack(preds,1)

def split_ids(C,seed=2025):
    rng=random.Random(seed);out=[[],[],[]]
    for m in (4,6):
        ids=[c for c,v in C.orders.items() if len(v[1])==m];rng.shuffle(ids);n=len(ids);a=int(.8*n);b=int(.9*n)
        for dst,z in zip(out,[ids[:a],ids[a:b],ids[b:]]):dst+=z
    return out

def metrics(model,loader,C,dev):
    model.eval();gaps=[];exact=0;n=0;by={2:[],3:[]}
    with torch.no_grad():
        for base,dist,angle,group,target,q,prereq,raw in loader:
            base,dist,angle,group,target,q,prereq=[x.to(dev) for x in (base,dist,angle,group,target,q,prereq)]
            _,p=model(base,dist,angle,group,target,q,prereq,teacher=False);p=p.cpu()
            for b,x in enumerate(raw):
                steps=2*x["q"];idx=p[b,:steps].tolist()
                full_e=[x["edges"][0]]+[x["edges"][i] for i in idx]
                full_r=[x["ratios"][0]]+[x["ratios"][i] for i in idx]
                length=sum(C.road(full_e[j],full_r[j],full_e[j+1],full_r[j+1]) for j in range(steps))
                gap=(length-x["opt"])/x["opt"]*100;gaps.append(gap);by[x["q"]].append(gap)
                exact+=int(torch.equal(p[b,:steps],x["target"]));n+=1
    return dict(n=n,gap=float(np.mean(gaps)),median=float(np.median(gaps)),p95=float(np.percentile(gaps,95)),acc=100*exact/n,
                gap2=float(np.mean(by[2])) if by[2] else None,gap3=float(np.mean(by[3])) if by[3] else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--epochs",type=int,default=80);ap.add_argument("--seed",type=int,default=2025);ap.add_argument("--batch",type=int,default=64);a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    C=Corpus();tr,va,te=split_ids(C,a.seed)
    print("split",len(tr),len(va),len(te),"three_passenger",*[sum(len(C.orders[c][1])==6 for c in z) for z in (tr,va,te)])
    loaders=[DataLoader(Cases(C,z),batch_size=a.batch,shuffle=(i==0),collate_fn=collate,num_workers=0) for i,z in enumerate((tr,va,te))]
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");print("device",dev)
    M=DomainPointer().to(dev);opt=torch.optim.AdamW(M.parameters(),lr=1e-3,weight_decay=1e-4);best=None;best_state=None
    for ep in range(1,a.epochs+1):
        M.train();ls=[]
        for base,dist,angle,group,target,q,prereq,_ in loaders[0]:
            base,dist,angle,group,target,q,prereq=[x.to(dev) for x in (base,dist,angle,group,target,q,prereq)]
            loss,_=M(base,dist,angle,group,target,q,prereq,teacher=True)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(M.parameters(),1.0);opt.step();ls.append(float(loss))
        if ep==1 or ep%10==0:
            vm=metrics(M,loaders[1],C,dev);print("epoch",ep,"loss",np.mean(ls),"val",vm)
            key=(vm["gap"],-vm["acc"])
            if best is None or key<best:best=key;best_state={k:v.detach().cpu().clone() for k,v in M.state_dict().items()}
    if best_state:M.load_state_dict(best_state)
    print("VAL_BEST",metrics(M,loaders[1],C,dev));print("TEST",metrics(M,loaders[2],C,dev))
if __name__=="__main__":main()
