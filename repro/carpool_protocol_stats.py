#!/usr/bin/env python3
"""Characterize the recovered Chengdu carpool workload for cross-graph generation.

This script does not train a model and does not change labels.  It freezes the
physical/task-scale statistics that future Jinan/Shenzhen/FLA carpool instances
should approximately match, so graph scale is not confounded with spatial span.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from historical_full_model import HistoricalExact, AMBIGUOUS_GROUP_CASES


EARTH_M = 6371000.0


def haversine_m(a, b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    lon1,lat1=np.deg2rad(a[...,0]),np.deg2rad(a[...,1])
    lon2,lat2=np.deg2rad(b[...,0]),np.deg2rad(b[...,1])
    dlon=lon2-lon1; dlat=lat2-lat1
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_M*np.arcsin(np.sqrt(np.clip(h,0,1)))


def summarize(xs):
    a=np.asarray(xs,dtype=np.float64)
    if not len(a): return {}
    return {
        "n":int(len(a)),
        "mean":float(a.mean()),
        "std":float(a.std()),
        "min":float(a.min()),
        "p10":float(np.percentile(a,10)),
        "p25":float(np.percentile(a,25)),
        "median":float(np.percentile(a,50)),
        "p75":float(np.percentile(a,75)),
        "p90":float(np.percentile(a,90)),
        "p95":float(np.percentile(a,95)),
        "max":float(a.max()),
    }




def value_counts(xs):
    vals,cnt=np.unique(np.asarray(xs),return_counts=True)
    return {str(float(v) if np.issubdtype(vals.dtype,np.floating) else int(v)):int(n)
            for v,n in zip(vals,cnt)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--links",required=True,type=Path)
    ap.add_argument("--orders",required=True,type=Path)
    ap.add_argument("--labels",required=True,type=Path)
    ap.add_argument("--exact",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()

    data=HistoricalExact(args.links,args.orders,args.labels,args.exact)
    group_counts=[]
    pickup_group_counts=[]; dropoff_group_counts=[]
    within_pair=[]
    group_diameter=[]
    group_radius=[]
    driver_pick_geo=[]; driver_pick_road=[]
    pickup_drop_geo=[]; pickup_drop_road=[]
    route_lengths=[]
    ratios=[]
    driver_ratios=[]
    pickup_ratios=[]
    dropoff_ratios=[]; pickup_ratios=[]; dropoff_ratios=[]; driver_ratios=[]

    per_passenger={2:{"cases":0},3:{"cases":0}}
    retained_ids=[int(c.case_id) for c in data.cases if int(c.case_id) not in AMBIGUOUS_GROUP_CASES]
    for cid in retained_ids:
        x=data.case_tensors(int(cid))
        flat=x["flat"]; pts=np.asarray(x["points"],dtype=np.float64)
        driver_ratios.append(float(flat[0][1]))
        for _,r,e,_ in flat[1:]:
            (pickup_ratios if int(e)%2==0 else dropoff_ratios).append(float(r))
        target=list(map(int,x["target"]))
        n_events=int(x["n_events"])
        q=n_events//2
        per_passenger.setdefault(q,{"cases":0})["cases"]+=1

        driver_ratios.append(float(flat[0][1]))
        by_event={}
        for j,(_,r,e,_) in enumerate(flat):
            if int(e)>=0:
                ev=int(e)
                by_event.setdefault(ev,[]).append(j)
                ratios.append(float(r))
                (pickup_ratios if ev%2==0 else dropoff_ratios).append(float(r))
        for e,idx in by_event.items():
            group_counts.append(len(idx))
            (pickup_group_counts if e%2==0 else dropoff_group_counts).append(len(idx))
            gp=pts[idx]
            if len(gp)>1:
                vals=[]
                for i in range(len(gp)):
                    for j in range(i+1,len(gp)):
                        vals.append(float(haversine_m(gp[i],gp[j])))
                within_pair.extend(vals)
                group_diameter.append(max(vals))
            center=gp.mean(axis=0)
            group_radius.append(float(max(haversine_m(gp,center))))

        chosen={int(flat[j][2]):j for j in target}
        driver=0
        for p in range(q):
            pe=2*p; de=pe+1
            pi=chosen[pe]; di=chosen[de]
            driver_pick_geo.append(float(haversine_m(pts[driver],pts[pi])))
            pickup_drop_geo.append(float(haversine_m(pts[pi],pts[di])))
            ea,ra,_,_=flat[driver]; eb,rb,_,_=flat[pi]
            driver_pick_road.append(float(data.point_dist(ea,ra,eb,rb)))
            ea,ra,_,_=flat[pi]; eb,rb,_,_=flat[di]
            pickup_drop_road.append(float(data.point_dist(ea,ra,eb,rb)))

        L=0.0
        seq=[0]+target
        for a,b in zip(seq[:-1],seq[1:]):
            ea,ra,_,_=flat[a]; eb,rb,_,_=flat[b]
            L+=float(data.point_dist(ea,ra,eb,rb))
        route_lengths.append(L)

    out={
        "dataset":"historical Chengdu recovered edge+ratio exact-label benchmark",
        "cases":int(len(retained_ids)),
        "passenger_counts":per_passenger,
        "candidate_count_per_event":summarize(group_counts),
        "candidate_count_histogram":value_counts(group_counts),
        "pickup_candidate_count_histogram":value_counts(pickup_group_counts),
        "dropoff_candidate_count_histogram":value_counts(dropoff_group_counts),
        "candidate_pairwise_haversine_m":summarize(within_pair),
        "candidate_group_diameter_haversine_m":summarize(group_diameter),
        "candidate_group_radius_from_lonlat_centroid_m":summarize(group_radius),
        "driver_to_exact_pickup_haversine_m":summarize(driver_pick_geo),
        "driver_to_exact_pickup_directed_road_m":summarize(driver_pick_road),
        "exact_pickup_to_own_dropoff_haversine_m":summarize(pickup_drop_geo),
        "exact_pickup_to_own_dropoff_directed_road_m":summarize(pickup_drop_road),
        "exact_route_directed_road_m":summarize(route_lengths),
        "candidate_ratio":summarize(ratios),
        "driver_ratio":summarize(driver_ratios),
        "pickup_candidate_ratio":summarize(pickup_ratios),
        "dropoff_candidate_ratio":summarize(dropoff_ratios),
        "candidate_ratio_histogram":value_counts(ratios),
        "pickup_candidate_ratio":summarize(pickup_ratios),
        "pickup_candidate_ratio_histogram":value_counts(pickup_ratios),
        "dropoff_candidate_ratio":summarize(dropoff_ratios),
        "dropoff_candidate_ratio_histogram":value_counts(dropoff_ratios),
        "driver_ratio":summarize(driver_ratios),
        "driver_ratio_histogram":value_counts(driver_ratios),
        "generator_policy_note":(
            "Future directed-road benchmarks should match the candidate-group physical "
            "dispersion and OD spatial-span distributions, while changing road graph scale."
        ),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2)+"\n")
    print("CARPOOL_PROTOCOL_STATS",json.dumps(out,sort_keys=True))


if __name__=="__main__":
    main()
