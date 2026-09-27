#!/usr/bin/env python3
"""Qingdao-SCC benchmark interface for scalable Lite-GD experiments."""
from __future__ import annotations
import gzip,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


class QingdaoSCCExact:
    def __init__(self, benchmark_dir: Path):
        self.root=Path(benchmark_dir)
        self.graph_root=self.root.parent
        self.meta=json.loads((self.root/"metadata.json").read_text())
        with gzip.open(self.root/"cases.json.gz","rt",encoding="utf-8") as f:
            self.cases=json.load(f)
        self.arr=np.load(self.root/"benchmark_arrays.npz",allow_pickle=False)
        self.cost_flat=np.asarray(self.arr["road_cost"],dtype=np.float32)
        self.cost_offsets=np.asarray(self.arr["cost_offsets"],dtype=np.int64)
        self.target_flat=np.asarray(self.arr["target"],dtype=np.int64)
        self.target_offsets=np.asarray(self.arr["target_offsets"],dtype=np.int64)

        self.node_ids=np.load(self.graph_root/"node_ids.npy",mmap_mode="r")
        self.edge_ids=np.load(self.graph_root/"link_ids.npy",mmap_mode="r")
        self.src=np.asarray(np.load(self.graph_root/"src_idx.npy",mmap_mode="r"),dtype=np.int64)
        self.dst=np.asarray(np.load(self.graph_root/"dst_idx.npy",mmap_mode="r"),dtype=np.int64)
        self.road_class=np.asarray(np.load(self.graph_root/"col4.npy",mmap_mode="r"),dtype=np.float32)
        self.weight=np.asarray(np.load(self.graph_root/"col5.npy",mmap_mode="r"),dtype=np.float32)
        self.n_nodes=len(self.node_ids);self.n_edges=len(self.edge_ids)

        indeg=np.bincount(self.dst,minlength=self.n_nodes).astype(np.float32)
        outdeg=np.bincount(self.src,minlength=self.n_nodes).astype(np.float32)
        self.node_deg=(indeg+outdeg).clip(min=1.0).astype(np.float32)
        nf=np.stack([np.log1p(indeg),np.log1p(outdeg)],axis=1)
        self.node_feat=((nf-nf.mean(0))/(nf.std(0)+1e-8)).astype(np.float32)

        # Five static edge channels, preserving the JointEncoder dimensionality
        # without fabricating unavailable endpoint coordinates.
        srcdeg=np.log1p(self.node_deg[self.src])
        dstdeg=np.log1p(self.node_deg[self.dst])
        # Reciprocal-link signal captures a useful directed-road structural cue.
        key=self.src.astype(np.int64)*np.int64(self.n_nodes)+self.dst.astype(np.int64)
        rev=self.dst.astype(np.int64)*np.int64(self.n_nodes)+self.src.astype(np.int64)
        sk=np.sort(key)
        p=np.searchsorted(sk,rev)
        reciprocal=((p<len(sk)) & (sk[np.minimum(p,len(sk)-1)]==rev)).astype(np.float32)
        ef=np.stack([
            np.log1p(self.weight),
            self.road_class,
            srcdeg,
            dstdeg,
            reciprocal
        ],axis=1).astype(np.float32)
        self.edge_base=((ef-ef.mean(0))/(ef.std(0)+1e-8)).astype(np.float32)

        # Incidence CSR for exact two-hop receptive-field extraction.
        nodes=np.concatenate([self.src,self.dst])
        edges=np.concatenate([np.arange(self.n_edges,dtype=np.int64),
                              np.arange(self.n_edges,dtype=np.int64)])
        order=np.argsort(nodes,kind="stable")
        ns=nodes[order];self.inc_edges=edges[order]
        cnt=np.bincount(ns,minlength=self.n_nodes)
        self.inc_ptr=np.concatenate([[0],np.cumsum(cnt)]).astype(np.int64)

        self.max_points=max(int(c["n_candidates"]) for c in self.cases)
        self.raw_by_id={i:c for i,c in enumerate(self.cases)}
        self.case_by_id={i:SimpleNamespace(passenger_count=int(c["passengers"])) for i,c in enumerate(self.cases)}
        self._light_cache={}
        self._rf_cache={}

    def split(self,seed=None):
        s=self.meta["split"]
        return [list(map(int,s["train"])),list(map(int,s["validation"])),list(map(int,s["test"]))]

    def case_light(self,cid):
        cid=int(cid)
        if cid in self._light_cache:return self._light_cache[cid]
        c=self.cases[cid];n=int(c["n_candidates"])
        co0,co1=int(self.cost_offsets[cid]),int(self.cost_offsets[cid+1])
        C=self.cost_flat[co0:co1].reshape(n,n)
        t0,t1=int(self.target_offsets[cid]),int(self.target_offsets[cid+1])
        target=self.target_flat[t0:t1].astype(np.int64,copy=False)
        event=np.full(n,-1,dtype=np.int64)
        for e,g in enumerate(c["event_local"]):
            event[np.asarray(g,dtype=np.int64)]=e
        flat=[]
        for j,(edge,e) in enumerate(zip(c["local_edges"],event.tolist())):
            flat.append((int(edge),0.5,int(e),-1))
        pts=np.asarray(c["coords"],dtype=np.float32)
        out=dict(cid=cid,flat=flat,points=pts,target=target,
                 opt=float(c["exact_length"]),n_events=int(len(c["event_local"])),
                 cost=C,event=event,passengers=int(c["passengers"]))
        self._light_cache[cid]=out
        return out

    def incident_union(self,nodes):
        chunks=[]
        for v in np.asarray(nodes,dtype=np.int64):
            a,b=int(self.inc_ptr[v]),int(self.inc_ptr[v+1])
            if b>a:chunks.append(self.inc_edges[a:b])
        if not chunks:return np.empty(0,dtype=np.int64)
        return np.unique(np.concatenate(chunks))

    def receptive_field(self,cid):
        cid=int(cid)
        if cid in self._rf_cache:return self._rf_cache[cid]
        x=self.case_light(cid)
        cand_pos_edges=np.asarray([z[0] for z in x["flat"]],dtype=np.int64)
        C=np.unique(cand_pos_edges)
        N2=np.unique(np.concatenate([self.src[C],self.dst[C]]))
        E1=self.incident_union(N2)
        N1=np.unique(np.concatenate([self.src[E1],self.dst[E1]]))
        E0=self.incident_union(N1)

        # Sorted arrays make all nested mappings vectorizable via searchsorted.
        c_to_e1=np.searchsorted(E1,C)
        e1_to_e0=np.searchsorted(E0,E1)
        cand_pos_to_C=np.searchsorted(C,cand_pos_edges)
        e1_src_n1=np.searchsorted(N1,self.src[E1])
        e1_dst_n1=np.searchsorted(N1,self.dst[E1])
        C_src_n2=np.searchsorted(N2,self.src[C])
        C_dst_n2=np.searchsorted(N2,self.dst[C])
        n2_to_n1=np.searchsorted(N1,N2)

        # Incidences used by exact aggregation into N1 / N2.
        contrib0_e=[];contrib0_n=[]
        n1pos={int(v):i for i,v in enumerate(N1.tolist())}
        for i,e in enumerate(E0.tolist()):
            s=int(self.src[e]);d=int(self.dst[e])
            if s in n1pos:contrib0_e.append(i);contrib0_n.append(n1pos[s])
            if d in n1pos:contrib0_e.append(i);contrib0_n.append(n1pos[d])
        contrib1_e=[];contrib1_n=[]
        n2pos={int(v):i for i,v in enumerate(N2.tolist())}
        for i,e in enumerate(E1.tolist()):
            s=int(self.src[e]);d=int(self.dst[e])
            if s in n2pos:contrib1_e.append(i);contrib1_n.append(n2pos[s])
            if d in n2pos:contrib1_e.append(i);contrib1_n.append(n2pos[d])

        # Dynamic role/ratio are attached to unique candidate links.  If one
        # link appears under multiple event semantics, preserve all channels.
        role=np.zeros((len(C),7),np.float32)
        ratio=np.zeros((len(C),7),np.float32)
        for j,e in enumerate(cand_pos_edges):
            ci=int(cand_pos_to_C[j]);ev=int(x["event"][j])
            ch=0 if ev<0 else ev+1
            role[ci,ch]=1.0;ratio[ci,ch]=0.5

        out=dict(C=C,N2=N2,E1=E1,N1=N1,E0=E0,
                 c_to_e1=c_to_e1,e1_to_e0=e1_to_e0,cand_pos_to_C=cand_pos_to_C,
                 e1_src_n1=e1_src_n1,e1_dst_n1=e1_dst_n1,
                 C_src_n2=C_src_n2,C_dst_n2=C_dst_n2,n2_to_n1=n2_to_n1,
                 contrib0_e=np.asarray(contrib0_e,np.int64),contrib0_n=np.asarray(contrib0_n,np.int64),
                 contrib1_e=np.asarray(contrib1_e,np.int64),contrib1_n=np.asarray(contrib1_n,np.int64),
                 role=role,ratio=ratio)
        self._rf_cache[cid]=out
        return out
