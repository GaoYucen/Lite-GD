#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from historical_chengdu import load_edges, load_orders, load_labels, recover_cases

def graph_tables(edges: pd.DataFrame):
    max_node=int(max(edges["node_start"].max(),edges["node_end"].max()))
    n=max_node+1
    best={}
    best_e={}
    for r in edges.itertuples(index=False):
        u=int(r.node_start);v=int(r.node_end);w=float(r.length);e=int(r.edge_id)
        k=(u,v)
        if k not in best or w<best[k]:
            best[k]=w;best_e[k]=e
    rr=np.fromiter((k[0] for k in best),dtype=np.int32)
    cc=np.fromiter((k[1] for k in best),dtype=np.int32)
    vv=np.fromiter(best.values(),dtype=np.float64)
    D,P=dijkstra(csr_matrix((vv,(rr,cc)),shape=(n,n)),directed=True,return_predecessors=True)
    return D,P,best_e

def point_dist(edge_row,D,a_ratio,b_row,b_ratio):
    a_id=int(edge_row.edge_id);b_id=int(b_row.edge_id)
    au=int(edge_row.node_start);av=int(edge_row.node_end);aw=float(edge_row.length)
    bu=int(b_row.node_start);bv=int(b_row.node_end);bw=float(b_row.length)
    ar=float(a_ratio);br=float(b_ratio)
    generic=(1.0-ar)*aw + float(D[av,bu]) + br*bw
    if a_id==b_id and br>=ar:
        return min((br-ar)*aw,generic)
    return generic

def reconstruct_nodes_edges(P, pair_edge, edge_a, ra, edge_b, rb):
    a_id=int(edge_a.edge_id);b_id=int(edge_b.edge_id)
    au=int(edge_a.node_start);av=int(edge_a.node_end)
    bu=int(edge_b.node_start);bv=int(edge_b.node_end)
    if a_id==b_id and rb>=ra:
        return [au,av],[a_id]
    # reconstruct node path av -> bu
    if av==bu:
        mid=[av]
    else:
        if int(P[av,bu])<0:
            raise RuntimeError(f"unreachable {av}->{bu}")
        rev=[bu];cur=bu
        while cur!=av:
            cur=int(P[av,cur]);rev.append(cur)
        mid=list(reversed(rev))
    edge_path=[a_id]
    for x,y in zip(mid[:-1],mid[1:]):
        edge_path.append(int(pair_edge[(x,y)]))
    if b_id!=edge_path[-1] or a_id==b_id:
        edge_path.append(b_id)
    nodes=[au]+mid
    if not nodes or nodes[-1]!=bv:
        nodes.append(bv)
    # stable unique adjacent duplicates
    nodes=[x for i,x in enumerate(nodes) if i==0 or x!=nodes[i-1]]
    edges2=[x for i,x in enumerate(edge_path) if i==0 or x!=edge_path[i-1]]
    return nodes,edges2

def exact_case(case,edges,D,P,pair_edge):
    groups={}
    for g in case.candidate_groups:
        if not g: continue
        et=int(g[0].event_type)
        groups[et]=g
    event_types=sorted(groups)
    n=len(event_types)//2
    expected=list(range(2*n))
    if event_types!=expected:
        raise RuntimeError(f"case {case.case_id}: event types {event_types} != {expected}")
    driver_row=edges.loc[int(case.driver_edge)]
    # DP: (mask, edge, ratio) -> cost/path of (event_type, edge_id, ratio, local_index)
    dp={(0,int(case.driver_edge),float(case.driver_ratio)):(0.0,[])}
    for _ in range(2*n):
        nd={}
        for (mask,eid,ratio),(cost,path) in dp.items():
            erow=edges.loc[eid]
            for et in event_types:
                if mask>>et & 1: continue
                if et%2==1 and not (mask>>(et-1)&1): continue
                for c in groups[et]:
                    brow=edges.loc[int(c.edge_id)]
                    z=cost+point_dist(erow,D,ratio,brow,float(c.ratio))
                    key=(mask|(1<<et),int(c.edge_id),float(c.ratio))
                    if key not in nd or z<nd[key][0]-1e-9:
                        nd[key]=(z,path+[(et,int(c.edge_id),float(c.ratio),int(c.local_index))])
        dp=nd
    (best_key,(best,path))=min(dp.items(),key=lambda kv:kv[1][0])

    full_nodes=[];full_edges=[]
    prev_e=int(case.driver_edge);prev_r=float(case.driver_ratio)
    for et,e,r,li in path:
        ns,es=reconstruct_nodes_edges(P,pair_edge,edges.loc[prev_e],prev_r,edges.loc[e],r)
        if full_nodes and ns and full_nodes[-1]==ns[0]: ns=ns[1:]
        if full_edges and es and full_edges[-1]==es[0]: es=es[1:]
        full_nodes.extend(ns);full_edges.extend(es)
        prev_e=e;prev_r=r

    return {
      "case_id":int(case.case_id),
      "passenger_count":int(case.passenger_count),
      "driver_edge":int(case.driver_edge),
      "driver_ratio":float(case.driver_ratio),
      "exact_length":float(best),
      "exact_event_sequence":[-1]+[int(x[0]) for x in path],
      "exact_selected_edges":[int(case.driver_edge)]+[int(x[1]) for x in path],
      "exact_selected_ratios":[float(case.driver_ratio)]+[float(x[2]) for x in path],
      "exact_local_indices":[-1]+[int(x[3]) for x in path],
      "exact_node_route":[int(x) for x in full_nodes],
      "exact_edge_route":[int(x) for x in full_edges],
      "legacy_length":float(case.route_length),
      "legacy_event_sequence":[int(x) for x in case.optimal_event_sequence],
      "legacy_selected_edges":[int(x) for x in case.optimal_selected_edges],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path)
    ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--summary-out",required=True,type=Path)
    a=ap.parse_args()

    edges=load_edges(a.links)
    cases=recover_cases(edges,load_orders(a.orders),load_labels(a.labels))
    D,P,pair_edge=graph_tables(edges)
    out=[]
    for i,c in enumerate(cases):
        out.append(exact_case(c,edges,D,P,pair_edge))
        if (i+1)%100==0: print("processed",i+1)

    # The historical stored route_length uses a different legacy metric.
    # Do not compare it numerically with the regenerated point-on-edge exact
    # length here. Legacy selected routes are audited under one unified metric
    # by repro/audit_historical_exact.py.
    changed=np.asarray([x["legacy_selected_edges"]!=x["exact_selected_edges"] for x in out])
    inter=np.asarray([
      any(seq.index(2*p+1)<max(seq.index(2*q) for q in range(x["passenger_count"]))
          for p in range(x["passenger_count"]))
      for x in out for seq in [x["exact_event_sequence"]]
    ])
    summary={
      "cases":len(out),
      "two_passenger":sum(x["passenger_count"]==2 for x in out),
      "three_passenger":sum(x["passenger_count"]==3 for x in out),
      "legacy_stored_length_comparable":False,
      "legacy_metric_note":"Use audit_historical_exact.py for same-metric legacy-vs-exact comparison.",
      "changed_selected_sequence_cases":int(changed.sum()),
      "exact_interleaved_cases":int(inter.sum()),
      "exact_length_mean":float(np.mean([x["exact_length"] for x in out])),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False))
    a.summary_out.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print("HISTORICAL_EXACT_SUMMARY",json.dumps(summary,sort_keys=True))

if __name__=="__main__":
    main()
