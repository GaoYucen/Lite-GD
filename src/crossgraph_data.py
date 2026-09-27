#!/usr/bin/env python3
"""Generic exact-label dataset interface for generated cross-graph benchmarks."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class CrossGraphExact:
    def __init__(self, benchmark_dir:Path, apsp_path:Path, require_routes:bool=False):
        self.root=Path(benchmark_dir)
        z=np.load(self.root/"graph.npz",mmap_mode="r")
        self.node_xy=np.asarray(z["coordinates"],dtype=np.float64)
        self.src=np.asarray(z["src"],dtype=np.int64)
        self.dst=np.asarray(z["dst"],dtype=np.int64)
        self.weight=np.asarray(z["weight"],dtype=np.float64)
        self.n_nodes=len(self.node_xy);self.n_edges=len(self.src)
        self.edge_ids=np.arange(self.n_edges,dtype=np.int64)
        self.e2i={int(i):int(i) for i in self.edge_ids}
        self.D=np.load(apsp_path,mmap_mode="r")
        if self.D.shape!=(self.n_nodes,self.n_nodes):
            raise ValueError(f"APSP shape {self.D.shape} != {(self.n_nodes,self.n_nodes)}")

        meta=json.loads((self.root/"metadata.json").read_text())
        self.metadata=meta
        route_file=self.root/meta.get("route_attachment",{}).get("cases_with_routes_file","cases_with_routes.json.gz")
        p=route_file if Path(route_file).is_absolute() else self.root/route_file
        if not p.exists():
            if require_routes: raise FileNotFoundError(f"route-labelled cases missing: {p}")
            p=self.root/"cases.json.gz"
        with gzip.open(p,"rt",encoding="utf-8") as f:
            self.cases=json.load(f)
        self.raw_by_id={int(c["case_id"]):c for c in self.cases}
        self.case_by_id={
            int(c["case_id"]):SimpleNamespace(passenger_count=len(c["candidate_groups"])//2)
            for c in self.cases
        }

        mu=self.node_xy.mean(0);sd=self.node_xy.std(0)+1e-8
        self.node_feat=((self.node_xy-mu)/sd).astype(np.float32)
        eb=np.stack([
            self.node_xy[self.src,0],self.node_xy[self.src,1],
            self.node_xy[self.dst,0],self.node_xy[self.dst,1],
            np.log1p(self.weight)
        ],1)
        em=eb.mean(0);es=eb.std(0)+1e-8
        self.edge_base=((eb-em)/es).astype(np.float32)
        self.max_points=max(1+sum(len(g) for g in c["candidate_groups"]) for c in self.cases)
        self._light_cache={}

    def split(self,seed=None):
        s=self.metadata["split"]
        return [list(map(int,s["train"])),list(map(int,s["validation"])),list(map(int,s["test"]))]

    def point_coord(self,eid,ratio):
        e=int(eid);r=float(ratio)
        return (self.node_xy[self.src[e]]*(1-r)+self.node_xy[self.dst[e]]*r).astype(np.float32)

    def point_dist(self,a,ra,b,rb):
        a=int(a);b=int(b);ra=float(ra);rb=float(rb)
        z=(1-ra)*float(self.weight[a])+float(self.D[int(self.dst[a]),int(self.src[b])])+rb*float(self.weight[b])
        if a==b and rb>=ra:
            z=min(z,(rb-ra)*float(self.weight[a]))
        return float(z)

    def case_light(self,cid):
        """Return only candidate-level tensors and cache them safely.

        This avoids caching O(|E|) dense role/label arrays for every case on
        10k-instance cross-graph benchmarks.
        """
        cid=int(cid)
        if cid in self._light_cache:return self._light_cache[cid]
        c=self.raw_by_id[cid]
        n_events=len(c["candidate_groups"])
        de=int(c["driver"]["edge"]);dr=float(c["driver"]["ratio"])
        flat=[(de,dr,-1,-1)]
        for et,g in enumerate(c["candidate_groups"]):
            for li,z in enumerate(g):
                flat.append((int(z["edge"]),float(z["ratio"]),int(et),int(li)))
        target=np.asarray(c["exact_flat_indices"],dtype=np.int64)
        pts=np.stack([self.point_coord(e,r) for e,r,_,_ in flat])
        out=dict(cid=cid,flat=flat,points=pts,target=target,
                 opt=float(c["exact_length"]),n_events=n_events)
        self._light_cache[cid]=out
        return out

    def case_tensors(self,cid):
        cid=int(cid)
        c=self.raw_by_id[cid]
        light=self.case_light(cid)
        flat=light["flat"];n_events=light["n_events"]

        # Dense graph-sized arrays are intentionally ephemeral.  Caching these
        # across 10k cases would consume tens of GB on medium directed graphs.
        role=np.zeros((self.n_edges,7),np.float32)
        rval=np.zeros((self.n_edges,7),np.float32)
        de,dr,_,_=flat[0]
        role[de,0]=1;rval[de,0]=dr
        for e,r,et,_ in flat[1:]:
            ch=int(et)+1;role[int(e),ch]=1;rval[int(e),ch]=float(r)

        edge_y=np.ones(self.n_edges,np.int64)
        node_y=np.zeros(self.n_nodes,np.int64)
        if "exact_edge_route" in c and "exact_node_route" in c:
            route_edges={int(e) for e in c["exact_edge_route"]}
            cand_edges={int(e) for e,_,_,_ in flat}
            for e in route_edges: edge_y[e]=2
            for e in cand_edges: edge_y[e]=0
            for e in route_edges & cand_edges: edge_y[e]=3
            for v in c["exact_node_route"]:
                if 0<=int(v)<self.n_nodes:node_y[int(v)]=1

        return dict(
            cid=cid,role=role,ratio=rval,node_y=node_y,edge_y=edge_y,
            flat=flat,points=light["points"],target=light["target"],
            opt=light["opt"],n_events=n_events
        )
