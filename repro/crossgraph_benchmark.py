#!/usr/bin/env python3
"""Generate cross-graph directed-road carpool benchmarks.

The generator changes road-graph scale while matching the recovered Chengdu
workload's empirical task scale:
- two passengers for the first graph-scale study;
- empirical candidate-count distribution (historical range 4--10);
- empirical candidate-group physical radius;
- empirical driver->pickup and pickup->dropoff geographic spans;
- historical pickup/dropoff edge-ratio semantics;
- exact paper-precedence labels under the same point-on-directed-edge metric.

Stage-1 output contains exact selected candidate sequences and route lengths.
Exact node/edge paths for Lite-GD pretraining are added in a later certification
step, after the generated workload is accepted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from historical_full_model import HistoricalExact, AMBIGUOUS_GROUP_CASES


EARTH_M = 6371000.0


def haversine_m(a, b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    lon1,lat1=np.deg2rad(a[...,0]),np.deg2rad(a[...,1])
    lon2,lat2=np.deg2rad(b[...,0]),np.deg2rad(b[...,1])
    dlon=lon2-lon1; dlat=lat2-lat1
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_M*np.arcsin(np.sqrt(np.clip(h,0,1)))


def project_xy(coords, meta=None):
    coords=np.asarray(coords,dtype=np.float64)
    if meta is None:
        lon0=float(np.mean(coords[:,0])); lat0=float(np.mean(coords[:,1]))
    else:
        lon0,lat0=map(float,meta)
    x=(coords[:,0]-lon0)*111320.0*math.cos(math.radians(lat0))
    y=(coords[:,1]-lat0)*110540.0
    return np.stack([x,y],1), (lon0,lat0)


def summarize(xs):
    a=np.asarray(xs,dtype=np.float64)
    if not len(a): return {}
    return {
        "n":int(len(a)), "mean":float(a.mean()), "std":float(a.std()),
        "min":float(a.min()), "p10":float(np.percentile(a,10)),
        "p25":float(np.percentile(a,25)), "median":float(np.percentile(a,50)),
        "p75":float(np.percentile(a,75)), "p90":float(np.percentile(a,90)),
        "p95":float(np.percentile(a,95)), "max":float(a.max()),
    }


@dataclass
class RoadGraph:
    coords: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray

    def __post_init__(self):
        self.coords=np.asarray(self.coords,dtype=np.float64)
        self.src=np.asarray(self.src,dtype=np.int64)
        self.dst=np.asarray(self.dst,dtype=np.int64)
        self.weight=np.asarray(self.weight,dtype=np.float64)
        self.xy,self.proj_meta=project_xy(self.coords)
        self.node_tree=cKDTree(self.xy)

    @property
    def n_nodes(self): return len(self.coords)
    @property
    def n_edges(self): return len(self.src)

    def point_coords(self, ratios):
        ratios=np.asarray(ratios,dtype=np.float64)
        return self.coords[self.src]*(1-ratios[:,None])+self.coords[self.dst]*ratios[:,None]


def load_native(protocol_npz:Path, raw_edges:Path) -> RoadGraph:
    z=np.load(protocol_npz,mmap_mode="r")
    coords=np.asarray(z["coordinates"],dtype=np.float64)
    original=np.asarray(z["original_node_ids"],dtype=np.int64)
    maxid=int(original.max())
    mapping=np.full(maxid+1,-1,dtype=np.int64)
    mapping[original]=np.arange(len(original),dtype=np.int64)
    best={}
    with raw_edges.open(newline="") as f:
        for row in csv.DictReader(f):
            u0,v0=int(row["Origin"]),int(row["Destination"])
            if u0<0 or v0<0 or u0>maxid or v0>maxid: continue
            u,v=int(mapping[u0]),int(mapping[v0])
            if u<0 or v<0 or u==v: continue
            w=float(row["Length"])
            key=(u,v)
            if key not in best or w<best[key]: best[key]=w
    src=np.fromiter((k[0] for k in best),dtype=np.int64,count=len(best))
    dst=np.fromiter((k[1] for k in best),dtype=np.int64,count=len(best))
    weight=np.fromiter(best.values(),dtype=np.float64,count=len(best))
    return RoadGraph(coords,src,dst,weight)


def load_fla(graph_npz:Path) -> RoadGraph:
    z=np.load(graph_npz,mmap_mode="r")
    return RoadGraph(z["coordinates"],z["src"],z["dst"],z["weight"])


def empirical_protocol(args):
    d=HistoricalExact(args.hist_links,args.hist_orders,args.hist_labels,args.hist_exact)
    counts=[]; radii=[]; dp=[]; pd=[]; driver_ratios=[]; pick_ratios=[]; drop_ratios=[]
    retained=[int(c.case_id) for c in d.cases if int(c.case_id) not in AMBIGUOUS_GROUP_CASES]
    for cid in retained:
        x=d.case_tensors(cid); flat=x["flat"]; pts=np.asarray(x["points"],dtype=np.float64)
        driver_ratios.append(float(flat[0][1]))
        by_event={}
        for j,(_,r,e,_) in enumerate(flat):
            if int(e)>=0:
                by_event.setdefault(int(e),[]).append(j)
                (pick_ratios if int(e)%2==0 else drop_ratios).append(float(r))
        for e,ids in by_event.items():
            counts.append(len(ids))
            gp=pts[ids]; center=gp.mean(0)
            radii.append(float(max(haversine_m(gp,center))))
        chosen={int(flat[j][2]):j for j in x["target"]}
        for p in range(int(x["n_events"])//2):
            pi=chosen[2*p]; di=chosen[2*p+1]
            dp.append(float(haversine_m(pts[0],pts[pi])))
            pd.append(float(haversine_m(pts[pi],pts[di])))
    return {
        "candidate_counts":np.asarray(counts,dtype=np.int64),
        "candidate_radii_m":np.asarray(radii,dtype=np.float64),
        "driver_pickup_geo_m":np.asarray(dp,dtype=np.float64),
        "pickup_dropoff_geo_m":np.asarray(pd,dtype=np.float64),
        "driver_ratios":np.asarray(driver_ratios,dtype=np.float64),
        "pickup_ratios":np.asarray(pick_ratios,dtype=np.float64),
        "dropoff_ratios":np.asarray(drop_ratios,dtype=np.float64),
    }


def choose_node_at_distance(g:RoadGraph, origin_xy, target_m, rng, tries=16):
    best=None
    for _ in range(tries):
        ang=float(rng.uniform(0,2*np.pi))
        q=np.asarray(origin_xy)+target_m*np.array([math.cos(ang),math.sin(ang)])
        dist,idx=g.node_tree.query(q,k=1)
        actual=float(np.linalg.norm(g.xy[int(idx)]-origin_xy))
        err=abs(actual-target_m)
        key=(err,int(idx))
        if best is None or key<best[0]: best=(key,int(idx))
    return best[1]


def candidate_group(g:RoadGraph, event_type:int, center_node:int, k:int, radius_m:float,
                    rng, pickup_tree, drop_tree, pickup_xy, drop_xy, pickup_ratio, drop_ratio):
    is_pick=(event_type%2==0)
    tree=pickup_tree if is_pick else drop_tree
    edge_xy=pickup_xy if is_pick else drop_xy
    center=g.xy[int(center_node)]
    ids=tree.query_ball_point(center,max(50.0,float(radius_m)))
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)<k:
        _,near=tree.query(center,k=min(max(k,1),g.n_edges))
        ids=np.atleast_1d(near).astype(np.int64)
    if len(ids)>k:
        # Include one edge closest to center, then sample the rest across the
        # accepted radius to avoid collapsing every group to a tiny cluster.
        dist=np.linalg.norm(edge_xy[ids]-center,axis=1)
        first=int(ids[int(np.argmin(dist))])
        rest=ids[ids!=first]
        if k>1:
            chosen=rng.choice(rest,size=k-1,replace=False)
            ids=np.concatenate([[first],chosen])
        else:
            ids=np.asarray([first],dtype=np.int64)
    ratio=float(pickup_ratio if is_pick else drop_ratio)
    return [(int(e),ratio) for e in ids[:k]]


def load_or_build_apsp(g:RoadGraph, path:Path|None, save_path:Path|None):
    if path is not None and path.exists():
        D=np.load(path,mmap_mode="r")
        if D.shape!=(g.n_nodes,g.n_nodes):
            raise ValueError(f"APSP shape {D.shape} != {(g.n_nodes,g.n_nodes)}")
        return D, "loaded"
    A=csr_matrix((g.weight,(g.src,g.dst)),shape=(g.n_nodes,g.n_nodes))
    D=dijkstra(A,directed=True)
    if not np.isfinite(D).all():
        raise RuntimeError("target graph is not strongly connected")
    if save_path is not None:
        save_path.parent.mkdir(parents=True,exist_ok=True)
        np.save(save_path,D)
    return D, "computed"


def cost_matrix(g:RoadGraph,D,flat):
    e=np.asarray([z[0] for z in flat],dtype=np.int64)
    r=np.asarray([z[1] for z in flat],dtype=np.float64)
    src=g.src[e]; dst=g.dst[e]; w=g.weight[e]
    C=(1-r)[:,None]*w[:,None] + D[np.ix_(dst,src)] + r[None,:]*w[None,:]
    same=e[:,None]==e[None,:]
    forward=r[None,:]>=r[:,None]
    direct=(r[None,:]-r[:,None])*w[:,None]
    C=np.where(same & forward,np.minimum(C,direct),C)
    np.fill_diagonal(C,0.0)
    return C


def exact_dp(flat, groups, C):
    n_events=len(groups)
    positions={e:list(groups[e]) for e in range(n_events)}
    dp={(0,0):0.0}
    parents=[]
    for _ in range(n_events):
        nd={}; par={}
        for (mask,last),base in dp.items():
            for e in range(n_events):
                if mask>>e & 1: continue
                if e%2==1 and not (mask>>(e-1)&1): continue
                nm=mask|(1<<e)
                for j in positions[e]:
                    z=base+float(C[last,j]); key=(nm,j)
                    if key not in nd or z<nd[key]:
                        nd[key]=z; par[key]=(mask,last)
        dp=nd; parents.append(par)
    key=min(dp,key=dp.get); opt=float(dp[key])
    seq=[None]*n_events
    for t in range(n_events-1,-1,-1):
        seq[t]=int(key[1]); key=parents[t][key]
    return seq,opt


def make_case(g,D,proto,rng,case_id,pickup_tree,drop_tree,pickup_xy,drop_xy):
    q=2; n_events=4
    driver_edge=int(rng.integers(0,g.n_edges))
    driver_ratio=float(rng.choice(proto["driver_ratios"]))
    driver_coord=g.coords[g.src[driver_edge]]*(1-driver_ratio)+g.coords[g.dst[driver_edge]]*driver_ratio
    driver_xy=g.xy[g.src[driver_edge]]*(1-driver_ratio)+g.xy[g.dst[driver_edge]]*driver_ratio

    centers={}
    for p in range(q):
        target=float(rng.choice(proto["driver_pickup_geo_m"]))
        pn=choose_node_at_distance(g,driver_xy,target,rng)
        centers[2*p]=pn
        target2=float(rng.choice(proto["pickup_dropoff_geo_m"]))
        dn=choose_node_at_distance(g,g.xy[pn],target2,rng)
        centers[2*p+1]=dn

    flat=[(driver_edge,driver_ratio,-1,-1)]
    groups={}
    groups_json=[]
    for e in range(n_events):
        k=int(rng.choice(proto["candidate_counts"]))
        rad=float(rng.choice(proto["candidate_radii_m"]))
        pr=float(rng.choice(proto["pickup_ratios"]))
        dr=float(rng.choice(proto["dropoff_ratios"]))
        cand=candidate_group(g,e,centers[e],k,rad,rng,pickup_tree,drop_tree,pickup_xy,drop_xy,pr,dr)
        pos=[]
        gj=[]
        for li,(edge,ratio) in enumerate(cand):
            pos.append(len(flat)); flat.append((edge,ratio,e,li))
            gj.append({"edge":int(edge),"ratio":float(ratio)})
        groups[e]=pos; groups_json.append(gj)

    C=cost_matrix(g,D,flat)
    seq,opt=exact_dp(flat,groups,C)
    events=[int(flat[j][2]) for j in seq]
    local=[int(flat[j][3]) for j in seq]

    return {
        "case_id":int(case_id),
        "driver":{"edge":driver_edge,"ratio":driver_ratio},
        "candidate_groups":groups_json,
        "exact_event_sequence":events,
        "exact_flat_indices":seq,
        "exact_local_indices":local,
        "exact_selected_edges":[int(flat[j][0]) for j in seq],
        "exact_selected_ratios":[float(flat[j][1]) for j in seq],
        "exact_length":opt,
        "_flat":flat,
        "_C":C,
    }


def certify(g,cases):
    counts=[]; radii=[]; dp_geo=[]; dp_road=[]; pd_geo=[]; pd_road=[]; routes=[]
    for c in cases:
        flat=c["_flat"]; C=c["_C"]
        pts=[]
        for edge,ratio,_,_ in flat:
            pts.append(g.coords[g.src[edge]]*(1-ratio)+g.coords[g.dst[edge]]*ratio)
        pts=np.asarray(pts)
        offset=1
        for gj in c["candidate_groups"]:
            ids=list(range(offset,offset+len(gj))); offset+=len(gj)
            counts.append(len(ids)); center=pts[ids].mean(0)
            radii.append(float(max(haversine_m(pts[ids],center))))
        chosen={int(flat[j][2]):int(j) for j in c["exact_flat_indices"]}
        for p in range(2):
            pi=chosen[2*p]; di=chosen[2*p+1]
            dp_geo.append(float(haversine_m(pts[0],pts[pi]))); dp_road.append(float(C[0,pi]))
            pd_geo.append(float(haversine_m(pts[pi],pts[di]))); pd_road.append(float(C[pi,di]))
        routes.append(float(c["exact_length"]))
    return {
        "candidate_count_per_event":summarize(counts),
        "candidate_group_radius_haversine_m":summarize(radii),
        "driver_to_exact_pickup_haversine_m":summarize(dp_geo),
        "driver_to_exact_pickup_directed_road_m":summarize(dp_road),
        "exact_pickup_to_own_dropoff_haversine_m":summarize(pd_geo),
        "exact_pickup_to_own_dropoff_directed_road_m":summarize(pd_road),
        "exact_route_directed_road_m":summarize(routes),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["native","fla"],required=True)
    ap.add_argument("--name",required=True)
    ap.add_argument("--protocol-npz",type=Path)
    ap.add_argument("--raw-edges",type=Path)
    ap.add_argument("--graph-npz",type=Path)
    ap.add_argument("--apsp",type=Path)
    ap.add_argument("--save-apsp",type=Path)
    ap.add_argument("--cases",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=20260925)
    ap.add_argument("--split-seed",type=int,default=20260925)
    ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--hist-links",required=True,type=Path)
    ap.add_argument("--hist-orders",required=True,type=Path)
    ap.add_argument("--hist-labels",required=True,type=Path)
    ap.add_argument("--hist-exact",required=True,type=Path)
    args=ap.parse_args()

    if args.dataset=="native":
        if args.protocol_npz is None or args.raw_edges is None: raise ValueError("native requires protocol npz + raw edges")
        g=load_native(args.protocol_npz,args.raw_edges)
    else:
        if args.graph_npz is None: raise ValueError("fla requires graph npz")
        g=load_fla(args.graph_npz)

    if g.n_nodes>50000 and (args.apsp is None or not args.apsp.exists()):
        raise RuntimeError("large graph requires an external exact-distance oracle; full APSP is intentionally disabled")
    D,apsp_mode=load_or_build_apsp(g,args.apsp,args.save_apsp)
    proto=empirical_protocol(args)
    rng=np.random.default_rng(args.seed)

    # Candidate-point indexes at the historical pickup/dropoff edge ratios.
    # The recovered workload is overwhelmingly 0.001 for pickup and 0.999 for dropoff.
    pick_ref=float(np.median(proto["pickup_ratios"]))
    drop_ref=float(np.median(proto["dropoff_ratios"]))
    pick_coords=g.point_coords(np.full(g.n_edges,pick_ref)); drop_coords=g.point_coords(np.full(g.n_edges,drop_ref))
    pick_xy,_=project_xy(pick_coords,g.proj_meta); drop_xy,_=project_xy(drop_coords,g.proj_meta)
    pickup_tree=cKDTree(pick_xy); drop_tree=cKDTree(drop_xy)

    cases=[]
    for cid in range(args.cases):
        c=make_case(g,D,proto,rng,cid,pickup_tree,drop_tree,pick_xy,drop_xy)
        cases.append(c)
        if (cid+1)%1000==0: print(f"generated={cid+1}",flush=True)

    cert=certify(g,cases)
    ids=np.arange(args.cases,dtype=np.int64); srng=np.random.default_rng(args.split_seed); ids=srng.permutation(ids)
    a=int(.8*len(ids)); b=int(.9*len(ids))
    split={"train":ids[:a].tolist(),"validation":ids[a:b].tolist(),"test":ids[b:].tolist()}

    args.out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out_dir/"graph.npz",
        coordinates=g.coords.astype(np.float32),src=g.src.astype(np.int32),
        dst=g.dst.astype(np.int32),weight=g.weight.astype(np.float64))
    serial=[]
    for c in cases:
        z={k:v for k,v in c.items() if not k.startswith("_")}
        serial.append(z)
    with gzip.open(args.out_dir/"cases.json.gz","wt",encoding="utf-8") as f:
        json.dump(serial,f,separators=(",",":"))
    meta={
        "name":args.name,"graph_nodes":g.n_nodes,"graph_edges":g.n_edges,
        "cases":args.cases,"passengers":2,"generator_seed":args.seed,
        "split_seed":args.split_seed,"split":split,"apsp_mode":apsp_mode,
        "historical_protocol":{
            "candidate_count":summarize(proto["candidate_counts"]),
            "candidate_radius_m":summarize(proto["candidate_radii_m"]),
            "driver_pickup_geo_m":summarize(proto["driver_pickup_geo_m"]),
            "pickup_dropoff_geo_m":summarize(proto["pickup_dropoff_geo_m"]),
            "driver_ratio":summarize(proto["driver_ratios"]),
            "pickup_ratio":summarize(proto["pickup_ratios"]),
            "dropoff_ratio":summarize(proto["dropoff_ratios"]),
        },
        "generated_certification":cert,
    }
    (args.out_dir/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("CROSSGRAPH_GENERATED",json.dumps(meta,sort_keys=True))


if __name__=="__main__":
    main()
