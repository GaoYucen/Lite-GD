#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

GROUPS=[np.arange(1,6),np.arange(6,11),np.arange(11,16),np.arange(16,21)]
PAPER=[p for p in itertools.permutations(range(4)) if p.index(0)<p.index(2) and p.index(1)<p.index(3)]
RESTRICTED=[(0,1,2,3),(0,1,3,2),(1,0,2,3),(1,0,3,2)]

def graph_dist(links,nodes):
    L=pd.read_csv(links);N=pd.read_csv(nodes)
    n=max(int(N.Node.max()),int(L.Node_Start.max()),int(L.Node_End.max()))+1
    best={}
    for u,v,w in L[['Node_Start','Node_End','Length']].itertuples(index=False,name=None):
        k=(int(u),int(v)); w=float(w)
        if k not in best or w<best[k]:best[k]=w
    r=np.fromiter((k[0] for k in best),dtype=np.int32)
    c=np.fromiter((k[1] for k in best),dtype=np.int32)
    v=np.fromiter(best.values(),dtype=np.float64)
    return dijkstra(csr_matrix((v,(r,c)),shape=(n,n)),directed=True)

def route(D,ids,p):
    return float(sum(D[ids[p[i]],ids[p[i+1]]] for i in range(len(p)-1)))

def best_orders(D,ids,orders):
    best=np.inf;bp=None
    for order in orders:
        a,b,c,d=[GROUPS[g] for g in order]
        C=(D[ids[0],ids[a]][:,None,None,None]
           +D[ids[a][:,None],ids[b][None,:]][:,:,None,None]
           +D[ids[b][:,None],ids[c][None,:]][None,:,:,None]
           +D[ids[c][:,None],ids[d][None,:]][None,None,:,:])
        f=int(np.argmin(C));val=float(C.reshape(-1)[f])
        if val<best-1e-9:
            z=np.unravel_index(f,C.shape)
            bp=np.asarray([0]+[int(GROUPS[g][z[i]]) for i,g in enumerate(order)],dtype=np.int64)
            best=val
    return bp,best

def nearest_feasible(D,ids):
    cur=0;p=[0];done=set()
    for _ in range(4):
        allowed=[]
        for g in range(4):
            if g in done:continue
            if g==2 and 0 not in done:continue
            if g==3 and 1 not in done:continue
            for j in GROUPS[g]:allowed.append((float(D[ids[cur],ids[j]]),int(j),g))
        _,j,g=min(allowed)
        p.append(j);done.add(g);cur=j
    return np.asarray(p,dtype=np.int64)

def metric(vals,opts):
    vals=np.asarray(vals);opts=np.asarray(opts)
    per=(vals/opts-1)*100
    return {
      'gap_ratio_of_means_pct':float((vals.mean()/opts.mean()-1)*100),
      'mean_case_gap_pct':float(per.mean()),
      'median_case_gap_pct':float(np.median(per)),
      'p95_case_gap_pct':float(np.percentile(per,95)),
      'exact_length_pct':float((np.abs(vals-opts)<=0.01).mean()*100),
      'avg_length':float(vals.mean()),'avg_opt':float(opts.mean())
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--old-data',default='',type=Path)
    ap.add_argument('--links',default=Path('sim_data/chengdu_link-mod.txt'),type=Path)
    ap.add_argument('--nodes',default=Path('sim_data/chengdu_node-mod.txt'),type=Path)
    ap.add_argument('--seeds',nargs='*',type=int,default=[20260925,20260926,20260927])
    ap.add_argument('--out',default='',type=Path)
    a=ap.parse_args()
    X=np.load(a.data,allow_pickle=True)
    old=np.load(a.old_data,allow_pickle=True) if str(a.old_data) else None
    D=graph_dist(a.links,a.nodes)
    rows=[]
    for seed in a.seeds:
        perm=np.random.default_rng(seed).permutation(len(X));test=perm[int(.9*len(X)):]
        pred={'restricted_exact':[],'nearest_feasible':[],'old_label':[]};opts=[]
        for i in test:
            s=X[int(i)];ids=np.asarray(s['Points_id'],dtype=np.int64)
            opt=float(s['Opt_Length']);opts.append(opt)
            _,z=best_orders(D,ids,RESTRICTED);pred['restricted_exact'].append(z)
            q=nearest_feasible(D,ids);pred['nearest_feasible'].append(route(D,ids,q))
            if old is not None:
                op=np.asarray(old[int(i)]['Solutions'],dtype=np.int64)
                pred['old_label'].append(route(D,ids,op))
        for name,v in pred.items():
            if not v:continue
            r={'seed':seed,'baseline':name,**metric(v,opts)}
            rows.append(r);print(json.dumps(r,sort_keys=True))
    doc={'rows':rows}
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(doc,indent=2))
if __name__=='__main__':main()
