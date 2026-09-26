#!/usr/bin/env python3
"""Verify sparse online Lite-GD against full dense inference and benchmark latency."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch

from crossgraph_data import CrossGraphExact
from historical_full_model import Model,loaders,move
from sparse_inference import compact_case,SparseOnlineLiteGD
from evaluation import summarize_route_rows


def stats(xs):
    a=np.asarray(xs,float)
    return {"n":len(a),"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),
            "p95_ms":float(np.percentile(a,95)),"qps":float(1000/a.mean())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--apsp",required=True,type=Path)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    dev=torch.device("cuda")
    data=CrossGraphExact(a.benchmark_dir,a.apsp,require_routes=True)
    _,_,te=data.split(20260925)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    full=Model(data,64,decoder_arch="road_metric_hier_fast",metric_layers=2,metric_heads=4).to(dev)
    full.load_state_dict(ck["state_dict"]);full.eval()
    torch.cuda.synchronize();t0=time.perf_counter()
    sparse=SparseOnlineLiteGD(full,data)
    torch.cuda.synchronize();cache_build_ms=(time.perf_counter()-t0)*1000

    mismatch=0;max_edge_diff=0.0;rows=[];local_meta=[]
    compact=[]
    with torch.no_grad():
        for cid in te:
            case=compact_case(data,cid);compact.append(case)
            b=move(next(iter(loaders(data,[cid],1,shuffle=False,seed=4321))),dev)
            _,eh=full.encode(b)
            pf=full.predict(b)
            ps,meta=sparse.infer_compact(case)
            mismatch+=int(not torch.equal(pf,ps))
            spec=torch.as_tensor(case["spec_edge"],device=dev,dtype=torch.long)
            e2,_=sparse.encoder.encode_special(case)
            max_edge_diff=max(max_edge_diff,float((eh[0,spec]-e2).abs().max()))
            pred=ps[0,:case["n_events"]].cpu().tolist()
            flat=case["flat"];L=0.0
            for x,y in zip([0]+pred[:-1],pred):
                ea,ra,_,_=flat[x];eb,rb,_,_=flat[y]
                L+=data.point_dist(ea,ra,eb,rb)
            true=case["target"].tolist()
            pe=[int(flat[j][2]) for j in pred];tevent=[int(flat[j][2]) for j in true]
            pm={int(flat[j][2]):j for j in pred};tm={int(flat[j][2]):j for j in true}
            rows.append({"pred":L,"opt":case["opt"],"exact":int(pred==true),
                         "pointer_hits":sum(x==y for x,y in zip(pred,true)),"steps":case["n_events"],
                         "event_exact":int(pe==tevent),"event_hits":sum(x==y for x,y in zip(pe,tevent)),
                         "candidate_hits":sum(pm[e]==tm[e] for e in tm)})
            local_meta.append(meta)

    quality=summarize_route_rows(rows)

    # Runtime after all offline caches are built. CPU compact construction and
    # GPU sparse inference are measured separately and together.
    for case in compact[:10]:
        sparse.infer_compact(case)
    torch.cuda.synchronize()
    gpu_ms=[]
    for case in compact:
        torch.cuda.synchronize();t=time.perf_counter()
        sparse.infer_compact(case)
        torch.cuda.synchronize();gpu_ms.append((time.perf_counter()-t)*1000)

    cpu_ms=[]
    for cid in te:
        t=time.perf_counter();compact_case(data,cid);cpu_ms.append((time.perf_counter()-t)*1000)

    # Fresh compact creation + sparse GPU inference: paper-like online request.
    e2e=[]
    for cid in te:
        torch.cuda.synchronize();t=time.perf_counter()
        case=compact_case(data,cid)
        sparse.infer_compact(case)
        torch.cuda.synchronize();e2e.append((time.perf_counter()-t)*1000)

    out={
      "ok":mismatch==0,
      "prediction_mismatches":mismatch,
      "max_candidate_edge_embedding_abs_diff":max_edge_diff,
      "quality":quality,
      "offline_cache_build_ms":cache_build_ms,
      "runtime_gpu_sparse":stats(gpu_ms),
      "runtime_cpu_compact":stats(cpu_ms),
      "runtime_end_to_end_online":stats(e2e),
      "local_support":{
        "special_edges_mean":float(np.mean([m["special_edges"] for m in local_meta])),
        "layer1_edges_mean":float(np.mean([m["layer1_edges"] for m in local_meta])),
        "layer1_nodes_mean":float(np.mean([m["layer1_nodes"] for m in local_meta])),
        "layer2_nodes_mean":float(np.mean([m["layer2_nodes"] for m in local_meta])),
      },
      "device":torch.cuda.get_device_name(0)
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("SPARSE_VERIFY",json.dumps(out,sort_keys=True))


if __name__=="__main__":main()
