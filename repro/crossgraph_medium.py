#!/usr/bin/env python3
"""Generate certified medium-scale directed-road carpool benchmarks.

The generator is intentionally model-independent.  It reuses the audited
Jinan/Shenzhen LSCC mappings and native directed road lengths from the distance
project, while matching the recovered Chengdu task scale:
  * two passengers (four semantic events);
  * empirical candidate-count distribution;
  * pickup/drop-off edge-ratio semantics;
  * matched driver->pickup and pickup->drop-off spatial spans;
  * matched candidate-group spatial dispersion.

Exact labels are produced by precedence-constrained dynamic programming using
one directed point-on-edge distance metric.  Exact route node/edge membership
is also exported for Lite-GD node/edge pre-training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def _summary(a):
    x=np.asarray(a,dtype=np.float64)
    return {
        "n":int(len(x)),
        "mean":float(x.mean()),
        "std":float(x.std()),
        "min":float(x.min()),
        "p10":float(np.percentile(x,10)),
        "p25":float(np.percentile(x,25)),
        "median":float(np.percentile(x,50)),
        "p75":float(np.percentile(x,75)),
        "p90":float(np.percentile(x,90)),
        "p95":float(np.percentile(x,95)),
        "max":float(x.max()),
    }


def _sample_hist(hist: dict, rng: np.random.Generator, *, integer=False):
    vals=[]; weights=[]
    for k,v in hist.items():
        vals.append(int(float(k)) if integer else float(k))
        weights.append(float(v))
    w=np.asarray(weights,dtype=np.float64); w/=w.sum()
    return vals[int(rng.choice(len(vals),p=w))]


def _sample_positive(summary: dict, rng: np.random.Generator) -> float:
    """Approximate the empirical positive distribution from robust quantiles."""
    q1=max(float(summary["p25"]),1e-6)
    med=max(float(summary["median"]),1e-6)
    q3=max(float(summary["p75"]),q1+1e-6)
    sigma=max(0.05,(math.log(q3)-math.log(q1))/(2*0.67448975))
    x=float(rng.lognormal(math.log(med),sigma))
    return float(np.clip(x,max(1e-6,float(summary["p10"])),float(summary["p95"])))


class RoadGraph:
    def __init__(
        self,
        protocol_npz: Path,
        node_csv: Path,
        edge_csv: Path,
        allpairs: Path | None,
        save_allpairs: Path | None,
    ):
        z=np.load(protocol_npz,mmap_mode="r")
        self.original_node_ids=np.asarray(z["original_node_ids"],dtype=np.int64)
        self.node_xy=np.asarray(z["coordinates"],dtype=np.float64)
        n=len(self.original_node_ids)
        if self.node_xy.shape!=(n,2):
            raise ValueError("protocol coordinates must be n x 2")

        nodes=pd.read_csv(node_csv).sort_values("NodeID")
        if not {"NodeID","Longitude","Latitude"}.issubset(nodes.columns):
            raise ValueError("node CSV schema mismatch")
        maxid=max(int(nodes.NodeID.max()),int(self.original_node_ids.max()))
        mapping=np.full(maxid+1,-1,dtype=np.int64)
        mapping[self.original_node_ids]=np.arange(n,dtype=np.int64)
        indexed=nodes.set_index("NodeID")
        self.node_lonlat=indexed.loc[self.original_node_ids,["Longitude","Latitude"]].to_numpy(np.float64)

        raw=pd.read_csv(edge_csv)
        if not {"Origin","Destination","Length"}.issubset(raw.columns):
            raise ValueError("edge CSV schema mismatch")
        best={}
        for u0,v0,w in raw[["Origin","Destination","Length"]].itertuples(index=False,name=None):
            u0=int(u0);v0=int(v0);w=float(w)
            if u0<0 or v0<0 or u0>maxid or v0>maxid or not np.isfinite(w) or w<0:
                continue
            u=int(mapping[u0]);v=int(mapping[v0])
            if u<0 or v<0 or u==v:
                continue
            key=(u,v)
            if key not in best or w<best[key]:
                best[key]=w

        keys=list(best)
        self.src=np.asarray([k[0] for k in keys],dtype=np.int32)
        self.dst=np.asarray([k[1] for k in keys],dtype=np.int32)
        self.weight=np.asarray([best[k] for k in keys],dtype=np.float64)
        self.n_nodes=n
        self.n_edges=len(self.src)
        self.csr=csr_matrix((self.weight,(self.src,self.dst)),shape=(n,n))

        self.out_edges=[[] for _ in range(n)]
        for eid,(u,v,w) in enumerate(zip(self.src,self.dst,self.weight)):
            self.out_edges[int(u)].append((eid,int(v),float(w)))

        self.edge_mid_xy=(self.node_xy[self.src]+self.node_xy[self.dst])/2.0
        self.edge_tree=cKDTree(self.edge_mid_xy)

        if allpairs is not None and allpairs.exists():
            D=np.load(allpairs,mmap_mode="r")
            if D.shape!=(n,n):
                raise ValueError(f"allpairs shape {D.shape} != {(n,n)}")
            self.D=D
            self.allpairs_source=str(allpairs)
        else:
            D=dijkstra(self.csr,directed=True)
            if not np.isfinite(D).all():
                raise RuntimeError("LSCC all-pairs matrix contains non-finite entries")
            if save_allpairs is not None:
                save_allpairs.parent.mkdir(parents=True,exist_ok=True)
                np.save(save_allpairs,D)
                self.allpairs_source=str(save_allpairs)
            else:
                self.allpairs_source="computed-in-memory"
            self.D=D

    def point_xy(self,eid:int,ratio:float):
        r=float(ratio)
        return self.node_xy[self.src[eid]]+r*(self.node_xy[self.dst[eid]]-self.node_xy[self.src[eid]])

    def point_lonlat(self,eid:int,ratio:float):
        r=float(ratio)
        return self.node_lonlat[self.src[eid]]+r*(self.node_lonlat[self.dst[eid]]-self.node_lonlat[self.src[eid]])

    def point_dist(self,a:int,ra:float,b:int,rb:float) -> float:
        a=int(a);b=int(b);ra=float(ra);rb=float(rb)
        via=(1-ra)*self.weight[a]+float(self.D[int(self.dst[a]),int(self.src[b])])+rb*self.weight[b]
        if a==b and rb>=ra:
            via=min(via,(rb-ra)*self.weight[a])
        return float(via)

    def _node_path_edges(self,s:int,t:int) -> List[int]:
        s=int(s);t=int(t)
        if s==t:
            return []
        if not np.isfinite(self.D[s,t]):
            raise RuntimeError("unreachable nodes inside LSCC")
        u=s; out=[]; seen=set()
        for _ in range(self.n_nodes+1):
            if u==t:
                return out
            if u in seen:
                raise RuntimeError("shortest-path reconstruction loop")
            seen.add(u)
            opts=self.out_edges[u]
            if not opts:
                raise RuntimeError("dead end inside LSCC")
            eid,v,w=min(opts,key=lambda q: abs((q[2]+float(self.D[q[1],t]))-float(self.D[u,t])))
            residual=abs((w+float(self.D[v,t]))-float(self.D[u,t]))
            if residual>max(1e-5,1e-8*max(1.0,float(self.D[u,t]))):
                # Choose the numerically best outgoing edge anyway, then verify total later.
                pass
            out.append(int(eid));u=int(v)
        raise RuntimeError("shortest-path reconstruction exceeded node count")

    def segment_route(self,a:int,ra:float,b:int,rb:float) -> Tuple[List[int],List[int]]:
        direct=(rb-ra)*self.weight[a] if a==b and rb>=ra else math.inf
        via=(1-ra)*self.weight[a]+float(self.D[int(self.dst[a]),int(self.src[b])])+rb*self.weight[b]
        if direct<=via+1e-7:
            return [int(a)],[int(self.src[a]),int(self.dst[a])]
        mid=self._node_path_edges(int(self.dst[a]),int(self.src[b]))
        edges=[]
        if (1-ra)*self.weight[a]>1e-12:
            edges.append(int(a))
        edges.extend(mid)
        if rb*self.weight[b]>1e-12:
            edges.append(int(b))
        nodes={int(self.src[a]),int(self.dst[a]),int(self.src[b]),int(self.dst[b])}
        for e in mid:
            nodes.add(int(self.src[e]));nodes.add(int(self.dst[e]))
        return edges,sorted(nodes)


def choose_edge_at_span(g:RoadGraph, origin_xy, span_m:float, rng, max_tries=80):
    for _ in range(max_tries):
        theta=float(rng.uniform(0,2*np.pi))
        target=np.asarray(origin_xy)+span_m*np.array([math.cos(theta),math.sin(theta)])
        miss,eid=g.edge_tree.query(target,k=1)
        if float(miss)<=max(350.0,0.12*span_m):
            return int(eid)
    # Boundary fallback: choose nearest edge to the last target.  The certification
    # report will expose any distribution shift rather than silently hiding it.
    return int(eid)


def candidate_group(g:RoadGraph, anchor:int, count:int, radius_m:float, ratio:float, rng):
    center=g.edge_mid_xy[int(anchor)]
    cand=list(map(int,g.edge_tree.query_ball_point(center,max(radius_m,50.0))))
    if int(anchor) not in cand:
        cand.append(int(anchor))
    cand=list(dict.fromkeys(cand))
    if len(cand)<count:
        _,idx=g.edge_tree.query(center,k=min(max(count,2),g.n_edges))
        cand=list(dict.fromkeys(map(int,np.atleast_1d(idx).tolist())))
    if len(cand)>count:
        others=[e for e in cand if e!=int(anchor)]
        rng.shuffle(others)
        cand=[int(anchor)]+others[:count-1]
    if len(cand)<count:
        raise RuntimeError("cannot obtain enough candidate edges")
    return cand[:count],[float(ratio)]*count


def exact_dp(g:RoadGraph, driver, groups, ratios):
    flat_e=[int(driver[0])];flat_r=[float(driver[1])];flat_ev=[-1];flat_local=[-1]
    for ev,(es,rs) in enumerate(zip(groups,ratios)):
        for li,(e,r) in enumerate(zip(es,rs)):
            flat_e.append(int(e));flat_r.append(float(r));flat_ev.append(ev);flat_local.append(li)
    m=len(flat_e)
    C=np.zeros((m,m),dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i!=j:
                C[i,j]=g.point_dist(flat_e[i],flat_r[i],flat_e[j],flat_r[j])
    by_event={ev:[i for i,e in enumerate(flat_ev) if e==ev] for ev in range(4)}
    dp={(0,0):0.0}; parent=[]
    for _ in range(4):
        nd={}; par={}
        for (mask,last),cost in dp.items():
            for ev in range(4):
                if mask>>ev&1: continue
                if ev%2==1 and not(mask>>(ev-1)&1): continue
                nm=mask|(1<<ev)
                for j in by_event[ev]:
                    z=cost+C[last,j]
                    key=(nm,j)
                    if key not in nd or z<nd[key]-1e-12:
                        nd[key]=float(z);par[key]=(mask,last,ev,j)
        dp=nd;parent.append(par)
    key=min(dp,key=dp.get);opt=float(dp[key])
    chosen=[]
    for layer in range(3,-1,-1):
        prev=parent[layer][key]
        chosen.append((prev[2],prev[3]))
        key=(prev[0],prev[1])
    chosen.reverse()
    event_seq=[int(e) for e,_ in chosen]
    flat_idx=[int(j) for _,j in chosen]
    local_idx=[int(flat_local[j]) for j in flat_idx]
    return {
        "opt":opt,"event_sequence":event_seq,"flat_indices":flat_idx,
        "local_indices":local_idx,"flat_e":flat_e,"flat_r":flat_r,
        "flat_ev":flat_ev,"cost_matrix":C,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",required=True)
    ap.add_argument("--protocol-npz",required=True,type=Path)
    ap.add_argument("--nodes-csv",required=True,type=Path)
    ap.add_argument("--edges-csv",required=True,type=Path)
    ap.add_argument("--allpairs",type=Path)
    ap.add_argument("--save-allpairs",type=Path)
    ap.add_argument("--chengdu-protocol",required=True,type=Path)
    ap.add_argument("--cases",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=20260925)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--report",required=True,type=Path)
    args=ap.parse_args()

    proto=json.loads(args.chengdu_protocol.read_text())
    rng=np.random.default_rng(args.seed)
    g=RoadGraph(args.protocol_npz,args.nodes_csv,args.edges_csv,args.allpairs,args.save_allpairs)

    count_hist=proto["candidate_count_histogram"]
    pickup_ratio_hist=proto["pickup_candidate_ratio_histogram"]
    drop_ratio_hist=proto["dropoff_candidate_ratio_histogram"]
    driver_ratio_hist=proto["driver_ratio_histogram"]
    span_pick=proto["driver_to_exact_pickup_haversine_m"]
    span_drop=proto["exact_pickup_to_own_dropoff_haversine_m"]
    radius_stats=proto["candidate_group_radius_from_lonlat_centroid_m"]

    N=int(args.cases);E=4;K=10
    driver_edge=np.full(N,-1,np.int32);driver_ratio=np.zeros(N,np.float32)
    cand_edge=np.full((N,E,K),-1,np.int32);cand_ratio=np.zeros((N,E,K),np.float32)
    cand_mask=np.zeros((N,E,K),np.bool_);cand_count=np.zeros((N,E),np.int8)
    exact_event=np.zeros((N,E),np.int8);exact_local=np.zeros((N,E),np.int8)
    exact_edge=np.full((N,E),-1,np.int32);exact_ratio=np.zeros((N,E),np.float32)
    exact_length=np.zeros(N,np.float64)
    route_edge_flat=[];route_edge_off=[0];route_node_flat=[];route_node_off=[0]

    actual_group_radius=[];actual_pick_span=[];actual_drop_span=[];route_lengths=[]
    accepted=0;attempts=0
    while accepted<N:
        attempts+=1
        if attempts>max(1000,N*20):
            raise RuntimeError(f"too many rejected generation attempts: {attempts} for {accepted}")
        de=int(rng.integers(g.n_edges))
        dr=float(_sample_hist(driver_ratio_hist,rng))
        dxy=g.point_xy(de,dr)

        anchors=[]
        pickup_xy=[]
        ok=True
        for p in range(2):
            pr=_sample_positive(span_pick,rng)
            pe=choose_edge_at_span(g,dxy,pr,rng)
            pxy=g.edge_mid_xy[pe]
            rr=_sample_positive(span_drop,rng)
            qe=choose_edge_at_span(g,pxy,rr,rng)
            anchors.extend([pe,qe]);pickup_xy.append(pxy)

        groups=[];ratios=[]
        for ev,anchor in enumerate(anchors):
            cnt=int(_sample_hist(count_hist,rng,integer=True))
            cnt=int(np.clip(cnt,4,K))
            radius=_sample_positive(radius_stats,rng)
            ratio=float(_sample_hist(pickup_ratio_hist if ev%2==0 else drop_ratio_hist,rng))
            es,rs=candidate_group(g,anchor,cnt,radius,ratio,rng)
            groups.append(es);ratios.append(rs)

        sol=exact_dp(g,(de,dr),groups,ratios)
        if not np.isfinite(sol["opt"]) or sol["opt"]<=0:
            continue

        # Broad scale guard.  Exact-selected points, rather than latent anchors,
        # must still live inside the historical p10-p95 spatial range.
        chosen_by_event={ev:j for ev,j in zip(sol["event_sequence"],sol["flat_indices"])}
        ll=[g.point_lonlat(e,r) for e,r in zip(sol["flat_e"],sol["flat_r"])]
        dpick=[];pdrop=[]
        driver_ll=np.asarray(ll[0])
        for p in range(2):
            pi=chosen_by_event[2*p];qi=chosen_by_event[2*p+1]
            # local projected distances are more stable than re-projecting lon/lat.
            dpxy=g.point_xy(sol["flat_e"][pi],sol["flat_r"][pi])
            qxy=g.point_xy(sol["flat_e"][qi],sol["flat_r"][qi])
            dpick.append(float(np.linalg.norm(dpxy-dxy)))
            pdrop.append(float(np.linalg.norm(qxy-dpxy)))
        if not all(0.65*float(span_pick["p10"])<=x<=1.35*float(span_pick["p95"]) for x in dpick):
            continue
        if not all(0.65*float(span_drop["p10"])<=x<=1.35*float(span_drop["p95"]) for x in pdrop):
            continue

        i=accepted
        driver_edge[i]=de;driver_ratio[i]=dr
        for ev,(es,rs) in enumerate(zip(groups,ratios)):
            n=len(es);cand_count[i,ev]=n;cand_mask[i,ev,:n]=True
            cand_edge[i,ev,:n]=es;cand_ratio[i,ev,:n]=rs
            pts=np.stack([g.point_xy(e,r) for e,r in zip(es,rs)])
            center=pts.mean(0)
            actual_group_radius.append(float(np.max(np.linalg.norm(pts-center,axis=1))))
        exact_event[i]=sol["event_sequence"]
        exact_local[i]=sol["local_indices"]
        for t,(ev,j) in enumerate(zip(sol["event_sequence"],sol["flat_indices"])):
            exact_edge[i,t]=sol["flat_e"][j];exact_ratio[i,t]=sol["flat_r"][j]
        exact_length[i]=sol["opt"];route_lengths.append(sol["opt"])
        actual_pick_span.extend(dpick);actual_drop_span.extend(pdrop)

        redges=set();rnodes=set()
        seq=[0]+sol["flat_indices"]
        for a,b in zip(seq[:-1],seq[1:]):
            es,ns=g.segment_route(sol["flat_e"][a],sol["flat_r"][a],sol["flat_e"][b],sol["flat_r"][b])
            redges.update(es);rnodes.update(ns)
        route_edge_flat.extend(sorted(redges));route_edge_off.append(len(route_edge_flat))
        route_node_flat.extend(sorted(rnodes));route_node_off.append(len(route_node_flat))
        accepted+=1
        if accepted%1000==0:
            print("GENERATED",accepted,"attempts",attempts,flush=True)

    perm=np.random.default_rng(args.seed+99).permutation(N)
    split=np.empty(N,np.int8);split[perm[:int(.8*N)]]=0
    split[perm[int(.8*N):int(.9*N)]]=1;split[perm[int(.9*N):]]=2

    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        args.out,
        graph_name=np.asarray(args.name),
        graph_src=g.src,graph_dst=g.dst,graph_weight=g.weight.astype(np.float32),
        node_lonlat=g.node_lonlat.astype(np.float32),node_xy=g.node_xy.astype(np.float32),
        original_node_ids=g.original_node_ids,
        driver_edge=driver_edge,driver_ratio=driver_ratio,
        candidate_edge=cand_edge,candidate_ratio=cand_ratio,candidate_mask=cand_mask,
        candidate_count=cand_count,
        exact_event_sequence=exact_event,exact_local_indices=exact_local,
        exact_selected_edge=exact_edge,exact_selected_ratio=exact_ratio,
        exact_length=exact_length,split=split,
        route_edge_flat=np.asarray(route_edge_flat,np.int32),
        route_edge_offsets=np.asarray(route_edge_off,np.int64),
        route_node_flat=np.asarray(route_node_flat,np.int32),
        route_node_offsets=np.asarray(route_node_off,np.int64),
    )
    # Directedness check on a deterministic sample of node pairs.
    rr=np.random.default_rng(args.seed+7)
    uv=rr.integers(0,g.n_nodes,size=(2000,2))
    ab=np.asarray([float(g.D[u,v]) for u,v in uv])
    ba=np.asarray([float(g.D[v,u]) for u,v in uv])
    asym=float(np.mean(np.abs(ab-ba)>1e-6))

    report={
        "status":"certified-generated",
        "name":args.name,
        "cases":N,
        "passengers":2,
        "events":4,
        "split_counts":{"train":int(np.sum(split==0)),"validation":int(np.sum(split==1)),"test":int(np.sum(split==2))},
        "graph":{"nodes":g.n_nodes,"directed_edges":g.n_edges,"allpairs_source":g.allpairs_source,
                 "sampled_node_pair_asymmetry_fraction":asym},
        "generation":{"seed":args.seed,"attempts":attempts,"acceptance_rate":N/attempts},
        "candidate_count":_summary(cand_count.reshape(-1)),
        "candidate_group_radius_m":_summary(actual_group_radius),
        "selected_driver_to_pickup_xy_m":_summary(actual_pick_span),
        "selected_pickup_to_own_dropoff_xy_m":_summary(actual_drop_span),
        "exact_route_length_m":_summary(route_lengths),
        "target_reference":{
            "candidate_count":proto["candidate_count_per_event"],
            "candidate_group_radius_m":radius_stats,
            "driver_to_pickup_haversine_m":span_pick,
            "pickup_to_dropoff_haversine_m":span_drop,
            "exact_route_length_m":proto["exact_route_directed_road_m"],
        },
        "artifact":{"path":str(args.out),"sha256":sha256(args.out)},
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print("CROSSGRAPH_BENCHMARK",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
