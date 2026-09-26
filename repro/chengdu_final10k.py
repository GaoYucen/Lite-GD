#!/usr/bin/env python3
"""Prepare the paper-scale Chengdu road graph for a reconstructed 10k benchmark."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra

def project(coords):
    lon0=float(coords[:,0].mean());lat0=float(coords[:,1].mean())
    x=(coords[:,0]-lon0)*(111320.0*math.cos(math.radians(lat0)))
    y=(coords[:,1]-lat0)*110540.0
    return np.stack([x,y],1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nodes",required=True,type=Path)
    ap.add_argument("--links",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()

    nd=pd.read_csv(a.nodes)
    ed=pd.read_csv(a.links)
    ids=nd["Node"].to_numpy(np.int64)
    if not np.array_equal(ids,np.arange(len(ids))):
        raise RuntimeError("Chengdu node ids are not contiguous")
    coords=nd[["Longitude","Latitude"]].to_numpy(np.float64)
    src=ed["Node_Start"].to_numpy(np.int32)
    dst=ed["Node_End"].to_numpy(np.int32)
    w=ed["Length"].to_numpy(np.float64)
    n=len(coords)

    A=csr_matrix((w,(src,dst)),shape=(n,n))
    nc,lab=connected_components(A,directed=True,connection="strong",return_labels=True)
    sizes=np.bincount(lab)
    if int(sizes.max())!=n:
        raise RuntimeError(f"expected paper Chengdu to be strongly connected, largest={sizes.max()} n={n}")
    if n!=1902 or len(src)!=5940:
        raise RuntimeError(f"paper graph size mismatch nodes={n} edges={len(src)}")

    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/"graph_lonlat.npz",coordinates=coords.astype(np.float32),
                        src=src,dst=dst,weight=w)
    np.savez_compressed(a.out/"graph_projected.npz",coordinates=project(coords).astype(np.float32),
                        src=src,dst=dst,weight=w)
    D,P=dijkstra(A,directed=True,return_predecessors=True)
    if not np.isfinite(D).all():raise RuntimeError("non-finite Chengdu APSP")
    np.save(a.out/"apsp.npy",D)
    np.save(a.out/"predecessor.npy",P.astype(np.int32))
    summary={"ok":True,"nodes":n,"edges":int(len(src)),"strong_components":int(nc),
             "apsp_shape":list(D.shape),"apsp_bytes":int(D.nbytes),
             "paper_match":n==1902 and len(src)==5940}
    (a.out/"graph_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,sort_keys=True))

if __name__=="__main__":main()
