#!/usr/bin/env python3
"""Build and finalize the Qingdao-SCC Lite-GD reproduction benchmark.

Protocol:
* maximum strongly connected directed road graph;
* all 12,277 multi-link-valid orders are retained;
* vehicle start uses the first map-matched link in its raw group;
* each passenger candidate is represented by one directed link;
* because the uploaded asset does not contain endpoint coordinates / original
  point-on-link ratios, the reproducible primary objective places a candidate
  at the midpoint of its link:
      0.5*w(a) + dist(head(a), tail(b)) + 0.5*w(b)
  with zero self-transition.
This is explicitly labeled the Qingdao-SCC link-midpoint protocol.
"""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
import numpy as np

def read_labels(path):
    out={}
    with open(path) as f:
        for line in f:
            z=line.split()
            if len(z)<2: continue
            out[int(z[0])]=[int(x) for x in z[1].split(",")]
    return out

def parse_cases(base):
    labels=read_labels(base/"carpool_route_point_qingdao_20221010_scc_ds")
    cases=[]
    with open(base/"carpool_route_point_qingdao_20221010_res_add_multi_link_valid_scc_ds") as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)!=3: continue
            cid=int(p[0])
            if cid not in labels: continue
            geos=[tuple(map(float,x.split(","))) for x in p[1].split(";")]
            groups=[[int(v) for v in g.split(",") if v] for g in p[2].split(";")]
            seq=labels[cid]
            if len(seq)!=len(groups) or -1 not in seq: continue
            ne=len(seq)-1
            if ne not in (4,6): continue
            # reorder groups by semantic code: driver, event 0..ne-1
            dpos=seq.index(-1)
            driver_group=groups[dpos]; driver_geo=geos[dpos]
            ev_groups=[];ev_geos=[]
            ok=True
            for e in range(ne):
                if e not in seq:ok=False;break
                k=seq.index(e);ev_groups.append(groups[k]);ev_geos.append(geos[k])
            if not ok or not driver_group or any(not g for g in ev_groups):continue
            raw_order=[x for x in seq if x!=-1]
            cases.append(dict(case_id=cid,driver_link=int(driver_group[0]),
                driver_geo=driver_geo,event_groups=ev_groups,event_geos=ev_geos,
                raw_event_order=raw_order,passengers=ne//2))
    return cases

def dp_opt(cost,event_groups):
    ne=len(event_groups);N=cost.shape[0]
    cand_to_event=np.full(N,-1,np.int16)
    for e,g in enumerate(event_groups):
        cand_to_event[g]=e
    cur={(0,0):(0.0,[])}
    full=(1<<ne)-1
    for step in range(ne):
        nxt={}
        for (mask,last),(val,path) in cur.items():
            for e,g in enumerate(event_groups):
                bit=1<<e
                if mask&bit:continue
                if e%2==1 and not(mask&(1<<(e-1))):continue
                for j in g:
                    nv=val+float(cost[last,j]);key=(mask|bit,int(j))
                    if key not in nxt or nv<nxt[key][0]:
                        nxt[key]=(nv,path+[int(j)])
        cur=nxt
    best=min((v for (m,_),v in cur.items() if m==full),key=lambda x:x[0])
    return best

def fixed_event_opt(cost,event_groups,order):
    cur={0:(0.0,[])}
    for e in order:
        nxt={}
        for last,(val,path) in cur.items():
            for j in event_groups[e]:
                nv=val+float(cost[last,j])
                if j not in nxt or nv<nxt[j][0]:nxt[j]=(nv,path+[int(j)])
        cur=nxt
    return min(cur.values(),key=lambda x:x[0])

def cmd_prepare(a):
    base=a.base
    link_ids=np.load(base/"link_ids.npy")
    src=np.load(base/"src_idx.npy").astype(np.uint32)
    dst=np.load(base/"dst_idx.npy").astype(np.uint32)
    weight=np.load(base/"col5.npy").astype(np.uint32)
    edge_pos={int(e):i for i,e in enumerate(link_ids.tolist())}
    cases=parse_cases(base)
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    src.tofile(out/"tail.u32");dst.tofile(out/"head.u32");weight.tofile(out/"weight.u32")
    meta=[];qsrc=[];qdst=[];offset=0
    for c in cases:
        links=[c["driver_link"]]+[x for g in c["event_groups"] for x in g]
        if any(x not in edge_pos for x in links):continue
        edges=np.asarray([edge_pos[x] for x in links],dtype=np.int64)
        n=len(edges);s=dst[edges];t=src[edges]
        qs=np.repeat(s,n);qt=np.tile(t,n)
        qsrc.append(qs);qdst.append(qt)
        # local event candidate indices
        ev_local=[];k=1
        for g in c["event_groups"]:
            ev_local.append(list(range(k,k+len(g))));k+=len(g)
        coords=[c["driver_geo"]]
        for geo,g in zip(c["event_geos"],c["event_groups"]):
            coords.extend([geo]*len(g))
        meta.append(dict(**c,local_links=links,local_edges=edges.tolist(),
            event_local=ev_local,coords=coords,q_offset=offset,n_candidates=n))
        offset+=n*n
    np.concatenate(qsrc).astype(np.uint32).tofile(out/"query_src.u32")
    np.concatenate(qdst).astype(np.uint32).tofile(out/"query_dst.u32")
    with gzip.open(out/"query_cases.json.gz","wt") as f:json.dump(meta,f)
    m={"protocol":"qingdao-scc-link-midpoint-v1","node_count":int(len(np.load(base/"node_ids.npy"))),
       "edge_count":int(len(link_ids)),"case_count":len(meta),"query_count":int(offset),
       "weight_field":"link_info column 5","ratio_assumption":0.5,
       "driver_rule":"first link in raw driver map-match group"}
    (out/"prepare_meta.json").write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps(m,sort_keys=True))

def cmd_finalize(a):
    base=a.base;out=a.out
    link_ids=np.load(base/"link_ids.npy");weight=np.load(base/"col5.npy").astype(np.float64)
    with gzip.open(out/"query_cases.json.gz","rt") as f:cases=json.load(f)
    d=np.fromfile(out/"query_dist.u32",dtype=np.uint32)
    UINT_MAX=np.iinfo(np.uint32).max
    rows=[];cost_flat=[];cost_offsets=[0];target_flat=[];target_offsets=[0]
    raw_match=0;raw_gap=[];opt_lengths=[]
    for c in cases:
        n=int(c["n_candidates"]);off=int(c["q_offset"])
        nd=d[off:off+n*n].reshape(n,n).astype(np.float64)
        edges=np.asarray(c["local_edges"],dtype=np.int64);w=weight[edges]
        C=.5*w[:,None]+nd+.5*w[None,:]
        np.fill_diagonal(C,0.0)
        C[nd>=UINT_MAX/2]=1e15
        best_len,best_path=dp_opt(C,c["event_local"])
        raw_len,raw_path=fixed_event_opt(C,c["event_local"],c["raw_event_order"])
        pred_event=[]
        cand_event={}
        for e,g in enumerate(c["event_local"]):
            for j in g:cand_event[int(j)]=e
        pred_event=[cand_event[j] for j in best_path]
        match=pred_event==c["raw_event_order"]
        raw_match+=int(match);raw_gap.append((raw_len/best_len-1)*100 if best_len>0 else 0)
        opt_lengths.append(best_len)
        cost_flat.append(C.astype(np.float32).ravel());cost_offsets.append(cost_offsets[-1]+n*n)
        target_flat.extend(best_path);target_offsets.append(len(target_flat))
        rows.append({k:c[k] for k in ["case_id","driver_link","driver_geo","event_groups","event_geos",
                    "raw_event_order","passengers","local_links","local_edges","event_local","coords","n_candidates"]})
        rows[-1].update(exact_length=float(best_len),exact_event_order=pred_event,
                        raw_order_best_length=float(raw_len),raw_event_order_matches_exact=bool(match))
    np.savez_compressed(out/"benchmark_arrays.npz",
        road_cost=np.concatenate(cost_flat).astype(np.float32),
        cost_offsets=np.asarray(cost_offsets,dtype=np.int64),
        target=np.asarray(target_flat,dtype=np.int32),
        target_offsets=np.asarray(target_offsets,dtype=np.int64))
    with gzip.open(out/"cases.json.gz","wt") as f:json.dump(rows,f)
    ids=np.arange(len(rows));rng=np.random.default_rng(20260925);rng.shuffle(ids)
    n=len(ids);ntr=int(.8*n);nva=int(.1*n)
    split={"train":ids[:ntr].tolist(),"validation":ids[ntr:ntr+nva].tolist(),"test":ids[ntr+nva:].tolist()}
    diag={"protocol":"qingdao-scc-link-midpoint-v1","cases":n,
      "raw_event_order_exact_pct":100*raw_match/max(n,1),
      "raw_event_order_mean_gap_pct":float(np.mean(raw_gap)),
      "raw_event_order_median_gap_pct":float(np.median(raw_gap)),
      "mean_exact_length":float(np.mean(opt_lengths)),"median_exact_length":float(np.median(opt_lengths)),
      "split_sizes":{k:len(v) for k,v in split.items()}}
    metadata={"protocol":"qingdao-scc-link-midpoint-v1","split":split,"diagnostic":diag,
      "graph_dir":str(base),"arrays":"benchmark_arrays.npz","cases":"cases.json.gz"}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(diag,sort_keys=True))

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
    for name in ["prepare","finalize"]:
        q=sp.add_parser(name);q.add_argument("--base",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    a=p.parse_args();{"prepare":cmd_prepare,"finalize":cmd_finalize}[a.cmd](a)
if __name__=="__main__":main()
