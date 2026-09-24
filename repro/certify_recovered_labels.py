from pathlib import Path
import ast
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DATA = Path("/workspace/.server-control/litegd-paper-data")

def nonempty(path):
    return [x for x in path.read_text().splitlines() if x.strip()]

def parse_parts(line, cast=int):
    p=[x for x in line.split(";") if x]
    return cast(p[0]), [[cast(z) for z in x.split(",") if z] for x in p[1:]]

def load():
    links=pd.read_csv(DATA/"chengdu_link.txt",sep=r"\s+",header=None,names=["id","u","v","w","flag"])
    edge={int(r.id):(int(r.u),int(r.v),float(r.w)) for r in links.itertuples(index=False)}
    n=max(max(u,v) for u,v,w in edge.values())+1
    sd=dijkstra(csr_matrix((links.w,(links.u,links.v)),shape=(n,n)),directed=True)

    ol=nonempty(DATA/"chengdu_order_1000.txt")
    orders={}
    for i in range(0,len(ol),3):
        cid=int(ol[i]); car,groups=parse_parts(ol[i+1]); cr,ratios=parse_parts(ol[i+2],float)
        orders[cid]=(car,groups,cr,ratios)

    ll=nonempty(DATA/"chengdu_label_1000.txt")
    labels={}
    for i in range(0,len(ll),8):
        cid=int(ll[i])
        labels[cid]=(ast.literal_eval(ll[i+3]),[int(x) for x in ll[i+4].split(";") if x],float(ll[i+7].split(":")[1]))
    return edge,sd,orders,labels

def point_distance(edge,sd,a,ra,b,rb):
    ua,va,wa=edge[a]; ub,vb,wb=edge[b]
    val=(1-ra)*wa+sd[va,ub]+rb*wb
    if a==b and rb>=ra:
        val=min(val,(rb-ra)*wa)
    return float(val)

def exact_dp(edge,sd,car,groups,car_ratio,ratios):
    m=len(groups); q=m//2
    pts=[list(zip(groups[g],ratios[g])) for g in range(m)]
    # state: (visited-group mask, final edge, final ratio) -> (cost,path)
    dp={(0,car,car_ratio):(0.0,[])}
    for _ in range(m):
        nd={}
        for (mask,e,r),(cost,path) in dp.items():
            for g in range(m):
                if mask>>g&1: continue
                if g>=q and not (mask>>(g-q)&1): continue
                for ee,rr in pts[g]:
                    key=(mask|1<<g,ee,rr)
                    val=cost+point_distance(edge,sd,e,r,ee,rr)
                    if key not in nd or val<nd[key][0]:
                        nd[key]=(val,path+[(g,ee,rr)])
        dp=nd
    return min(dp.values(),key=lambda x:x[0])

def main():
    edge,sd,orders,labels=load()
    errs=[]; gaps=[]; edge_match=seq_match=0; counts={}
    examples=[]
    for cid,(car,groups,cr,ratios) in orders.items():
        seq,selected,label_len=labels[cid]
        q=len(groups)//2
        counts[len(groups)]=counts.get(len(groups),0)+1
        typemap={2*i:i for i in range(q)}|{2*i+1:q+i for i in range(q)}
        pp=[(car,cr)]
        for typ,e in zip(seq[1:],selected[1:]):
            g=typemap[typ]; j=groups[g].index(e); pp.append((e,ratios[g][j]))
        rec=sum(point_distance(edge,sd,*pp[j],*pp[j+1]) for j in range(len(pp)-1))
        errs.append(abs(rec-label_len))

        best,path=exact_dp(edge,sd,car,groups,cr,ratios)
        btypes=[2*g if g<q else 2*(g-q)+1 for g,_,_ in path]
        bseq=[-1]+btypes
        bedges=[car]+[e for _,e,_ in path]
        edge_match+=int(bedges==selected)
        seq_match+=int(bseq==seq)
        gap=(label_len-best)/best*100
        gaps.append(gap)
        if len(examples)<10 and (abs(gap)>1e-5 or bedges!=selected):
            examples.append((cid,seq,bseq,selected,bedges,label_len,best,gap))

    print("case_group_counts",counts)
    print("published_route_recalc_mae",float(np.mean(errs)))
    print("published_route_recalc_p95",float(np.percentile(errs,95)))
    print("published_route_recalc_max",float(np.max(errs)))
    print("exact_edge_route_matches",edge_match,"/",len(orders))
    print("exact_sequence_matches",seq_match,"/",len(orders))
    print("published_vs_dp_gap_mean_pct",float(np.mean(gaps)))
    print("published_vs_dp_gap_p95_pct",float(np.percentile(gaps,95)))
    print("published_vs_dp_gap_min_pct",float(np.min(gaps)))
    print("published_vs_dp_gap_max_pct",float(np.max(gaps)))
    print("near_zero_gap",sum(abs(x)<=1e-5 for x in gaps),"/",len(gaps))
    for x in examples: print("example",x)

if __name__=="__main__":
    main()
