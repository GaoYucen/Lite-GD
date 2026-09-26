#!/usr/bin/env python3
"""Exact sparse online inference for the two-layer case-conditioned Lite-GD GCN.

The trained JointEncoder receives graph-sized role/ratio tensors although only
the driver/candidate edges are case-dependent.  With exactly two message-passing
layers, the dynamic effect has a bounded two-hop support.  This module caches
the all-zero-case graph once, then recomputes only the exact sparse deltas needed
for candidate-edge embeddings.  No learned weight or decoding rule is changed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch


def compact_case(data, cid):
    x=data.case_light(int(cid))
    flat=x["flat"]; n=len(flat)
    edge=np.asarray([z[0] for z in flat],dtype=np.int64)
    ratio=np.asarray([z[1] for z in flat],dtype=np.float32)
    event=np.asarray([z[2] for z in flat],dtype=np.int64)

    spec=np.unique(edge)  # sorted, stable lookup target
    pos={int(e):i for i,e in enumerate(spec.tolist())}
    role=np.zeros((len(spec),7),np.float32)
    rval=np.zeros((len(spec),7),np.float32)
    for e,r,ev,_ in flat:
        ch=0 if int(ev)<0 else int(ev)+1
        k=pos[int(e)]
        role[k,ch]=1.0;rval[k,ch]=float(r)
    local=np.asarray([pos[int(e)] for e in edge],dtype=np.int64)

    # Exact directed point-on-edge pairwise metric.
    e=edge;r=ratio.astype(np.float64)
    w=np.asarray(data.weight[e],dtype=np.float64)
    src=np.asarray(data.src[e],dtype=np.int64);dst=np.asarray(data.dst[e],dtype=np.int64)
    C=(1.0-r[:,None])*w[:,None]+np.asarray(data.D[np.ix_(dst,src)],dtype=np.float64)+r[None,:]*w[None,:]
    same=e[:,None]==e[None,:];forward=r[None,:]>=r[:,None]
    direct=(r[None,:]-r[:,None])*w[:,None]
    C=np.where(same&forward,np.minimum(C,direct),C)
    np.fill_diagonal(C,0.0)
    finite=C[np.isfinite(C)&(C>0)]
    if not np.isfinite(C).all():
        fallback=(float(finite.max()) if finite.size else 1.0)*4.0
        C[~np.isfinite(C)]=fallback

    return {
      "cid":int(cid),
      "spec_edge":spec,
      "role":role,
      "ratio_role":rval,
      "edge_idx_local":local,
      "event":event,
      "coords":np.asarray(x["points"],dtype=np.float32),
      "road_cost":C.astype(np.float32),
      "target":np.asarray(x["target"],dtype=np.int64),
      "n_events":int(x["n_events"]),
      "opt":float(x["opt"]),
      "flat":flat,
    }


@dataclass
class SparseStatic:
    node0: torch.Tensor
    e0: torch.Tensor
    agg0: torch.Tensor
    n1: torch.Tensor
    e1: torch.Tensor
    agg1: torch.Tensor


class SparseJointEncoderInference:
    def __init__(self,encoder,data):
        self.enc=encoder
        self.data=data
        self.device=encoder.node_feat.device
        self.src_np=np.asarray(data.src,dtype=np.int64)
        self.dst_np=np.asarray(data.dst,dtype=np.int64)
        self._build_incidence()
        self.static=self._build_static()

    def _build_incidence(self):
        E=len(self.src_np);N=int(self.enc.node_feat.shape[0])
        nodes=np.concatenate([self.src_np,self.dst_np])
        edges=np.concatenate([np.arange(E,dtype=np.int64),np.arange(E,dtype=np.int64)])
        order=np.argsort(nodes,kind="stable")
        ns=nodes[order];es=edges[order]
        cnt=np.bincount(ns,minlength=N)
        self.inc_ptr=np.concatenate([[0],np.cumsum(cnt)]).astype(np.int64)
        self.inc_edges=es

    def _incident_union(self,nodes):
        z=[]
        for v in np.asarray(nodes,dtype=np.int64).tolist():
            a,b=int(self.inc_ptr[v]),int(self.inc_ptr[v+1])
            if b>a:z.append(self.inc_edges[a:b])
        if not z:return np.empty(0,dtype=np.int64)
        return np.unique(np.concatenate(z))

    def _edge_update(self,l,e,n_src,n_dst):
        enc=self.enc
        v=torch.stack([n_src,n_dst],dim=1)
        q=enc.eq[l](e).unsqueeze(1)
        k=enc.ek[l](v)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(enc.h),dim=1)
        endpoint=(a.unsqueeze(-1)*v).sum(1)
        return torch.relu(enc.edge_up[l](torch.cat([e,endpoint],dim=-1)))

    @torch.no_grad()
    def _build_static(self):
        enc=self.enc;dev=self.device
        node0=torch.relu(enc.node_in(enc.node_feat))
        zeros=torch.zeros(enc.edge_base.size(0),14,device=dev,dtype=enc.edge_base.dtype)
        e0=torch.relu(enc.edge_in(torch.cat([enc.edge_base,zeros],dim=-1)))

        agg0=torch.zeros_like(node0)
        agg0.index_add_(0,enc.src,e0);agg0.index_add_(0,enc.dst,e0)
        agg0=agg0/enc.node_deg.squeeze(0)
        n1=torch.relu(enc.node_up[0](torch.cat([node0,agg0],dim=-1)))
        e1=self._edge_update(0,e0,n1[enc.src],n1[enc.dst])

        agg1=torch.zeros_like(n1)
        agg1.index_add_(0,enc.src,e1);agg1.index_add_(0,enc.dst,e1)
        agg1=agg1/enc.node_deg.squeeze(0)
        return SparseStatic(node0,e0,agg0,n1,e1,agg1)

    @staticmethod
    def _overlay(base,query_idx,dyn_idx,dyn_val):
        out=base[query_idx].clone()
        if dyn_idx.numel()==0:return out
        pos=torch.searchsorted(dyn_idx,query_idx)
        safe=pos.clamp(max=max(dyn_idx.numel()-1,0))
        hit=(pos<dyn_idx.numel())&(dyn_idx[safe]==query_idx)
        if hit.any():out[hit]=dyn_val[safe[hit]]
        return out

    @torch.no_grad()
    def encode_special(self,case):
        enc=self.enc;st=self.static;dev=self.device
        spec_np=np.asarray(case["spec_edge"],dtype=np.int64)
        touched1_np=np.unique(np.concatenate([self.src_np[spec_np],self.dst_np[spec_np]]))
        e1set_np=self._incident_union(touched1_np)
        touched2_np=np.unique(np.concatenate([self.src_np[e1set_np],self.dst_np[e1set_np]]))

        spec=torch.as_tensor(spec_np,device=dev,dtype=torch.long)
        t1=torch.as_tensor(touched1_np,device=dev,dtype=torch.long)
        e1set=torch.as_tensor(e1set_np,device=dev,dtype=torch.long)
        t2=torch.as_tensor(touched2_np,device=dev,dtype=torch.long)

        role=torch.as_tensor(case["role"],device=dev,dtype=enc.edge_base.dtype)
        ratio=torch.as_tensor(case["ratio_role"],device=dev,dtype=enc.edge_base.dtype)

        # Dynamic initial embeddings only on special edges.
        e0dyn=torch.relu(enc.edge_in(torch.cat([enc.edge_base[spec],role,ratio],dim=-1)))
        d0=e0dyn-st.e0[spec]

        ssrc=torch.as_tensor(np.searchsorted(touched1_np,self.src_np[spec_np]),device=dev,dtype=torch.long)
        sdst=torch.as_tensor(np.searchsorted(touched1_np,self.dst_np[spec_np]),device=dev,dtype=torch.long)
        dn=torch.zeros(len(touched1_np),enc.h,device=dev,dtype=d0.dtype)
        dn.index_add_(0,ssrc,d0);dn.index_add_(0,sdst,d0)
        agg0=st.agg0[t1]+dn/enc.node_deg.squeeze(0)[t1]
        n1dyn=torch.relu(enc.node_up[0](torch.cat([st.node0[t1],agg0],dim=-1)))

        # First edge update changes only edges incident to touched layer-1 nodes.
        e0local=self._overlay(st.e0,e1set,spec,e0dyn)
        src1=torch.as_tensor(self.src_np[e1set_np],device=dev,dtype=torch.long)
        dst1=torch.as_tensor(self.dst_np[e1set_np],device=dev,dtype=torch.long)
        n1s=self._overlay(st.n1,src1,t1,n1dyn)
        n1d=self._overlay(st.n1,dst1,t1,n1dyn)
        e1dyn=self._edge_update(0,e0local,n1s,n1d)
        d1=e1dyn-st.e1[e1set]

        # Second node update changes only endpoints of the changed first-layer edges.
        src2loc=torch.as_tensor(np.searchsorted(touched2_np,self.src_np[e1set_np]),device=dev,dtype=torch.long)
        dst2loc=torch.as_tensor(np.searchsorted(touched2_np,self.dst_np[e1set_np]),device=dev,dtype=torch.long)
        dn2=torch.zeros(len(touched2_np),enc.h,device=dev,dtype=d1.dtype)
        dn2.index_add_(0,src2loc,d1);dn2.index_add_(0,dst2loc,d1)
        agg1=st.agg1[t2]+dn2/enc.node_deg.squeeze(0)[t2]
        n1input=self._overlay(st.n1,t2,t1,n1dyn)
        n2dyn=torch.relu(enc.node_up[1](torch.cat([n1input,agg1],dim=-1)))

        # We only need final second-layer edge embeddings for driver/candidates.
        e1spec=self._overlay(st.e1,spec,e1set,e1dyn)
        srcs=torch.as_tensor(self.src_np[spec_np],device=dev,dtype=torch.long)
        dsts=torch.as_tensor(self.dst_np[spec_np],device=dev,dtype=torch.long)
        n2s=self._overlay(st.n1,srcs,t2,n2dyn)  # base ignored because every endpoint is in t2
        n2d=self._overlay(st.n1,dsts,t2,n2dyn)
        # Fix overlay base: use arbitrary static shape but all spec endpoints must hit t2.
        e2spec=self._edge_update(1,e1spec,n2s,n2d)
        return e2spec, {
          "special_edges":int(len(spec_np)),
          "layer1_edges":int(len(e1set_np)),
          "layer1_nodes":int(len(touched1_np)),
          "layer2_nodes":int(len(touched2_np)),
        }


class SparseOnlineLiteGD:
    def __init__(self,model,data):
        if model.decoder_arch!="road_metric_hier_fast":
            raise ValueError("SparseOnlineLiteGD requires road_metric_hier_fast")
        self.model=model
        self.data=data
        self.encoder=SparseJointEncoderInference(model.encoder,data)
        self.device=model.encoder.node_feat.device

    @torch.no_grad()
    def infer_compact(self,case):
        e2spec,meta=self.encoder.encode_special(case)
        dev=self.device
        edge_idx=torch.as_tensor(case["edge_idx_local"],device=dev,dtype=torch.long)[None,:]
        event=torch.as_tensor(case["event"],device=dev,dtype=torch.long)[None,:]
        coords=torch.as_tensor(case["coords"],device=dev,dtype=torch.float32)[None,:,:]
        valid=torch.ones(1,len(case["event"]),device=dev,dtype=torch.bool)
        target=torch.as_tensor(case["target"],device=dev,dtype=torch.long)[None,:]
        ne=torch.tensor([case["n_events"]],device=dev,dtype=torch.long)
        cost=torch.as_tensor(case["road_cost"],device=dev,dtype=torch.float32)[None,:,:]
        pred=self.model.decoder.infer(e2spec[None,:,:],edge_idx,event,coords,valid,ne,cost)
        return pred,meta
