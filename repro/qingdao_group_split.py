#!/usr/bin/env python3
"""Create a group-safe Qingdao split without changing cases, costs, or graph.

All rows sharing the same source case_id are assigned to exactly one split.
"""
from __future__ import annotations
import argparse,gzip,json,os,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--seed",type=int,default=20260925)
    a=ap.parse_args()

    meta=json.loads((a.source/"metadata.json").read_text())
    with gzip.open(a.source/"cases.json.gz","rt",encoding="utf-8") as f:
        cases=json.load(f)

    groups=defaultdict(list)
    for i,c in enumerate(cases):
        groups[int(c["case_id"])].append(i)

    gids=np.asarray(sorted(groups),dtype=np.int64)
    rng=np.random.default_rng(a.seed)
    gids=rng.permutation(gids)

    target_train=.8*len(cases);target_val=.1*len(cases)
    train=[];val=[];test=[];nt=nv=0
    phase=0
    for gid in gids:
        rows=groups[int(gid)]
        if phase==0 and nt+len(rows)<=target_train:
            train.extend(rows);nt+=len(rows)
        elif phase==0:
            phase=1
        if phase==1 and nv+len(rows)<=target_val:
            val.extend(rows);nv+=len(rows)
        elif phase==1:
            phase=2
        if phase==2:
            test.extend(rows)

    # The transition group is assigned to the next split by the logic above.
    # If any split is unexpectedly empty, fail rather than inventing a split.
    if not train or not val or not test:
        raise RuntimeError((len(train),len(val),len(test)))

    s={"train":sorted(map(int,train)),"validation":sorted(map(int,val)),"test":sorted(map(int,test))}
    cidsets={k:{int(cases[i]["case_id"]) for i in v} for k,v in s.items()}
    assert not (cidsets["train"]&cidsets["validation"])
    assert not (cidsets["train"]&cidsets["test"])
    assert not (cidsets["validation"]&cidsets["test"])

    a.out.mkdir(parents=True,exist_ok=True)
    for name in ["cases.json.gz","benchmark_arrays.npz"]:
        dst=a.out/name
        if dst.exists() or dst.is_symlink():dst.unlink()
        os.symlink(os.path.relpath(a.source/name,a.out),dst)

    out=dict(meta)
    out["protocol"]="qingdao-scc-link-midpoint-v2-group-safe"
    out["split"]=s
    out["group_split"]={
        "key":"case_id","seed":a.seed,"groups":len(groups),
        "rows":{k:len(v) for k,v in s.items()},
        "unique_case_ids":{k:len(cidsets[k]) for k in s},
        "case_id_overlap":{"train_validation":0,"train_test":0,"validation_test":0},
        "source_benchmark":str(a.source),
    }
    (a.out/"metadata.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"ok":True,**out["group_split"]},sort_keys=True))

if __name__=="__main__":main()
