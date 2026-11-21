#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Evaluate all previously frozen short-horizon adapters without retraining."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from train_models import ROOT,load_rows,save

def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['previous','upstream','compatibility']:p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    spec=importlib.util.spec_from_file_location('unchanged_prior_neural_adapter',args.previous/'neural_adapter.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=json.loads((args.previous/'PROTOCOL.json').read_text());frozen=json.loads((args.previous/'MODEL_FREEZE.json').read_text())
    with np.load(args.previous/'NORMALIZERS.npz') as z:norm={k:z[k] for k in z.files}
    author=old.author_module(args.upstream,args.compatibility);rows=load_rows('test');results=[]
    for row in frozen['models']:
        model=old.load_frozen(author,row,norm,cfg,args.previous/'models');cases=[]
        for e,z in rows:
            D=z['grad_H_x'][...,2]-z['grad_H_x'][...,3];x=old.features(z['state_y'],D,z['reference_volume'])
            with torch.no_grad():prediction=model(torch.tensor(x),torch.tensor(D)).numpy()
            isfinite=bool(np.isfinite(prediction).all());metrics=old.operator_metrics(model,z) if isfinite else None
            cases.append(dict(seed_group=e['seed_group'],kappa=e['kappa'],predictions_all_finite=isfinite,
                nonfinite_prediction_fraction=float(np.mean(~np.isfinite(prediction))),metrics=finite_json(metrics)))
        results.append(dict(model=row['file'],sha256=row['sha256'],kind=row['kind'],seed=row['seed'],cases=cases))
    save(ROOT/'FROZEN_SHORT_MODEL_TRANSFER.json',dict(models=results,all_six_original_checkpoints_hash_checked=True,
        retrained=False,selected_using_new_tests=False,scope='Temporal extension beyond the original time-0-to-0.005 training support, not a claim about the original native Hamiltonian.'))
    print(json.dumps(dict(models=len(results),cases=sum(len(r['cases']) for r in results),finite_cases=sum(c['predictions_all_finite'] for r in results for c in r['cases']))),flush=True)

if __name__=='__main__':main()
