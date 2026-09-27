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
    sizes=np.bincount(lab);largest=int(sizes.argmax());keep=(lab==largest)
    raw_n=n;raw_m=len(src)
    if int(sizes[largest])!=n:
        old_to_new=np.full(n,-1,dtype=np.int64)
        old_to_new[np.flatnonzero(keep)]=np.arange(int(keep.sum()))
        em=keep[src]&keep[dst]
        coords=coords[keep]
        src=old_to_new[src[em]].astype(np.int32)
        dst=old_to_new[dst[em]].astype(np.int32)
        w=w[em]
        n=len(coords)
        A=csr_matrix((w,(src,dst)),shape=(n,n))
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/"graph_lonlat.npz",coordinates=coords.astype(np.float32),
                        src=src,dst=dst,weight=w)
    np.savez_compressed(a.out/"graph_projected.npz",coordinates=project(coords).astype(np.float32),
                        src=src,dst=dst,weight=w)
    D,P=dijkstra(A,directed=True,return_predecessors=True)
    if not np.isfinite(D).all():raise RuntimeError("non-finite Chengdu APSP")
    np.save(a.out/"apsp.npy",D)
    np.save(a.out/"predecessor.npy",P.astype(np.int32))
    summary={"ok":True,"nodes":n,"edges":int(len(src)),"strong_components_raw":int(nc),
             "raw_nodes":int(raw_n),"raw_edges":int(raw_m),
             "largest_scc_nodes":int(sizes[largest]),"largest_scc_edges":int(len(src)),
             "apsp_shape":list(D.shape),"apsp_bytes":int(D.nbytes),
             "paper_table_target":{"nodes":1902,"edges":5940},
             "code_asset_expected":{"nodes":1901,"edges":5941},
             "code_asset_match":raw_n==1901 and raw_m==5941,
             "protocol_note":"Use the checked-in Chengdu code asset and its largest SCC; paper table differs by one node/edge."}
    (a.out/"graph_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,sort_keys=True))

if __name__=="__main__":main()
