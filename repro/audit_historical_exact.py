#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from historical_chengdu import load_edges, load_orders, load_labels, recover_cases
from relabel_historical_exact import graph_tables, point_dist

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path)
    ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path)
    ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()

    edges=load_edges(a.links)
    cases=recover_cases(edges,load_orders(a.orders),load_labels(a.labels))
    exact={int(x["case_id"]):x for x in json.loads(a.exact.read_text())}
    D,_,_=graph_tables(edges)
    rows=[];amb=0
    for c in cases:
        groups={int(g[0].event_type):g for g in c.candidate_groups if g}
        pts=[(int(c.driver_edge),float(c.driver_ratio))]
        for et,e in zip(c.optimal_event_sequence[1:],c.optimal_selected_edges[1:]):
            hits=[x for x in groups[int(et)] if int(x.edge_id)==int(e)]
            if not hits:
                raise RuntimeError(f"case {c.case_id}: legacy edge {e} missing for event {et}")
            if len(hits)>1: amb+=1
            x=hits[0];pts.append((int(x.edge_id),float(x.ratio)))
        old=0.0
        for x,y in zip(pts[:-1],pts[1:]):
            old+=point_dist(edges.loc[x[0]],D,x[1],edges.loc[y[0]],y[1])
        opt=float(exact[c.case_id]["exact_length"])
        rows.append({"case_id":c.case_id,"legacy_recalc":old,"exact":opt,"gap_pct":(old/opt-1)*100})
    gaps=np.asarray([x["gap_pct"] for x in rows])
    doc={
      "cases":len(rows),"ambiguous_within_event_selected_edge":amb,
      "legacy_equal_exact_count":int(np.sum(np.abs(gaps)<=1e-8)),
      "legacy_vs_exact_gap_mean_pct":float(gaps.mean()),
      "legacy_vs_exact_gap_median_pct":float(np.median(gaps)),
      "legacy_vs_exact_gap_p95_pct":float(np.percentile(gaps,95)),
      "legacy_vs_exact_gap_max_pct":float(gaps.max()),
      "legacy_vs_exact_gap_min_pct":float(gaps.min()),
      "negative_gap_below_tolerance":int(np.sum(gaps < -1e-7)),
      "rows":rows
    }
    a.out.write_text(json.dumps(doc,indent=2))
    print("HISTORICAL_UNIFIED_METRIC_AUDIT",json.dumps({k:v for k,v in doc.items() if k!="rows"},sort_keys=True))

if __name__=="__main__":main()
