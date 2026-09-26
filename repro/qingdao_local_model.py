#!/usr/bin/env python3
"""Exact local-receptive-field Lite-GD encoder for million-edge Qingdao."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

from historical_full_model import JointEncoder, HierarchicalRoadMetricDecoderFast


class LocalJointEncoder(JointEncoder):
    """Evaluate the original 2-layer JointEncoder exactly on a candidate RF.

    Only tensors that can influence the requested final candidate-edge
    embeddings are materialized.  The learned operations and global degree
    normalizers are identical to JointEncoder.
    """
    def _edge_update(self,l,e,n_src,n_dst):
        v=torch.stack([n_src,n_dst],dim=1)
        q=self.eq[l](e).unsqueeze(1)
        k=self.ek[l](v)
        a=torch.softmax((q*k).sum(-1)/math.sqrt(self.h),dim=1)
        endpoint=(a.unsqueeze(-1)*v).sum(1)
        return torch.relu(self.edge_up[l](torch.cat([e,endpoint],dim=-1)))

    def _idx(self,x,dev):
        return torch.as_tensor(x,dtype=torch.long,device=dev)

    def forward_case(self,rf):
        dev=self.node_feat.device
        C=self._idx(rf["C"],dev); N2=self._idx(rf["N2"],dev)
        E1=self._idx(rf["E1"],dev); N1=self._idx(rf["N1"],dev); E0=self._idx(rf["E0"],dev)
        c_to_e1=self._idx(rf["c_to_e1"],dev)
        e1_to_e0=self._idx(rf["e1_to_e0"],dev)
        cand_pos_to_C=self._idx(rf["cand_pos_to_C"],dev)
        e1_src_n1=self._idx(rf["e1_src_n1"],dev);e1_dst_n1=self._idx(rf["e1_dst_n1"],dev)
        C_src_n2=self._idx(rf["C_src_n2"],dev);C_dst_n2=self._idx(rf["C_dst_n2"],dev)
        n2_to_n1=self._idx(rf["n2_to_n1"],dev)
        c0e=self._idx(rf["contrib0_e"],dev);c0n=self._idx(rf["contrib0_n"],dev)
        c1e=self._idx(rf["contrib1_e"],dev);c1n=self._idx(rf["contrib1_n"],dev)

        # Candidate dynamic channels inserted into the E0 initial edge tensor.
        role0=torch.zeros((len(E0),7),dtype=self.edge_base.dtype,device=dev)
        ratio0=torch.zeros_like(role0)
        c_to_e0=e1_to_e0[c_to_e1]
        role=torch.as_tensor(rf["role"],dtype=self.edge_base.dtype,device=dev)
        ratio=torch.as_tensor(rf["ratio"],dtype=self.edge_base.dtype,device=dev)
        role0[c_to_e0]=role;ratio0[c_to_e0]=ratio

        node0=torch.relu(self.node_in(self.node_feat[N1]))
        e0=torch.relu(self.edge_in(torch.cat([self.edge_base[E0],role0,ratio0],dim=-1)))

        agg0=torch.zeros((len(N1),self.h),dtype=e0.dtype,device=dev)
        agg0.index_add_(0,c0n,e0[c0e])
        agg0=agg0/self.node_deg[0,N1]
        n1=torch.relu(self.node_up[0](torch.cat([node0,agg0],dim=-1)))

        e1in=e0[e1_to_e0]
        e1=self._edge_update(0,e1in,n1[e1_src_n1],n1[e1_dst_n1])

        agg1=torch.zeros((len(N2),self.h),dtype=e1.dtype,device=dev)
        agg1.index_add_(0,c1n,e1[c1e])
        agg1=agg1/self.node_deg[0,N2]
        n2=torch.relu(self.node_up[1](torch.cat([n1[n2_to_n1],agg1],dim=-1)))

        e1c=e1[c_to_e1]
        e2c=self._edge_update(1,e1c,n2[C_src_n2],n2[C_dst_n2])
        return e2c[cand_pos_to_C]


class QingdaoLiteGD(nn.Module):
    def __init__(self,data,h=64,metric_layers=2,metric_heads=4):
        super().__init__()
        self.encoder=LocalJointEncoder(data.node_feat,data.edge_base,data.src,data.dst,h)
        self.decoder=HierarchicalRoadMetricDecoderFast(
            h,data.max_points,layers=metric_layers,heads=metric_heads)

    def encode_cases(self,data,ids):
        return [self.encoder.forward_case(data.receptive_field(int(cid))) for cid in ids]

    def forward_batch(self,data,ids,b,teacher=True):
        reps=self.encode_cases(data,ids)
        ns={x.shape[0] for x in reps}
        if len(ns)!=1:
            raise ValueError("Qingdao batches must have equal candidate count")
        eh=torch.stack(reps,dim=0)
        B,N,_=eh.shape
        edge_idx=torch.arange(N,device=eh.device,dtype=torch.long)[None,:].expand(B,-1)
        return self.decoder(
            eh,edge_idx,b["event"],b["coords"],b["valid"],b["target"],
            b["n_events"],b["road_cost"],teacher)

    @torch.no_grad()
    def predict_batch(self,data,ids,b):
        reps=self.encode_cases(data,ids)
        eh=torch.stack(reps,dim=0)
        B,N,_=eh.shape
        edge_idx=torch.arange(N,device=eh.device,dtype=torch.long)[None,:].expand(B,-1)
        return self.decoder.infer(
            eh,edge_idx,b["event"],b["coords"],b["valid"],b["n_events"],b["road_cost"])
