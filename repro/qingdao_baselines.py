#!/usr/bin/env python3
"""Candidate-level baselines on the Qingdao-SCC exact benchmark."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

from qingdao_dataset import QingdaoSCCExact
from baseline_carpool import CarpoolCases, train_pointer, evaluate, seed_all
from paper_baselines import Graph2RouteCarpool, train_g2r, disgreedy_paper, _evaluate_sequence
from am_fidelity import train as train_am_fidelity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir",required=True,type=Path)
    ap.add_argument("--model",choices=["ptrnet","am","graph2route","disgreedy"],required=True)
    ap.add_argument("--seed",type=int,default=4321)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--eval-batch",type=int,default=128)
    ap.add_argument("--epochs",type=int,default=120)
    ap.add_argument("--eval-every",type=int,default=5)
    ap.add_argument("--patience",type=int,default=25)
    ap.add_argument("--lr",type=float,default=1e-4)
    # Pointer
    ap.add_argument("--dim",type=int,default=128)
    ap.add_argument("--ptr-layers",type=int,default=1)
    # AM
    ap.add_argument("--heads",type=int,default=8)
    ap.add_argument("--am-layers",type=int,default=3)
    ap.add_argument("--ff",type=int,default=512)
    ap.add_argument("--baseline",default="rollout_copy")
    ap.add_argument("--warmup-epochs",type=int,default=1)
    ap.add_argument("--exp-beta",type=float,default=.8)
    ap.add_argument("--baseline-update-eps",type=float,default=1e-3)
    ap.add_argument("--rollouts-per-batch",type=int,default=4)
    ap.add_argument("--max-grad-norm",type=float,default=1.0)
    # G2R
    ap.add_argument("--hidden",type=int,default=8)
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--weight-decay",type=float,default=1e-5)
    a=ap.parse_args()
    a.split_seed=20260925

    seed_all(a.seed)
    data=QingdaoSCCExact(a.benchmark_dir)
    tr,va,te=data.split()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases=CarpoolCases(data,tr)

    if a.model=="disgreedy":
        quality,metric=disgreedy_paper(data,cases,te)
        result={"model":"DisGreedy-paper","decision_metric":metric,"test":quality,
                "train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te)}
        model=None
    elif a.model=="ptrnet":
        model,hist=train_pointer(a,data,cases,tr,va,te,device)
        result={"model":"ptrnet","seed":a.seed,"history":hist,
                "train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te),
                "test":evaluate(model,cases,te,data,device,a.eval_batch)}
    elif a.model=="graph2route":
        model,hist=train_g2r(a,data,cases,tr,va,device)
        result={"model":"graph2route","seed":a.seed,"history":hist,
                "train_cases":len(tr),"validation_cases":len(va),"test_cases":len(te),
                "test":evaluate(model,cases,te,data,device,a.eval_batch)}
    else:
        model,result=train_am_fidelity(a,data,device)
        result["train_cases"]=len(tr);result["validation_cases"]=len(va);result["test_cases"]=len(te)

    if a.checkpoint and model is not None:
        a.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":model.state_dict(),"model":a.model,"seed":a.seed},a.checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("QINGDAO_BASELINE_RESULT",json.dumps({"model":a.model,"test":result["test"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
