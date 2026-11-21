#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Separate explicit time-step instability from inaccurate learned mechanics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
from scipy.integrate import DOP853
from train_models import ROOT,load_model,load_rows,save

CFG=dict(target_time=10.,rtol=1e-7,atol=1e-9,max_step=.005,max_seconds=40,max_rhs_calls=40000,seed_group=12,kappa=100,
    purpose='Post-scoring diagnostic using unchanged frozen weights: do not attribute failed RK4 rollouts solely to the vector field without an adaptive step check. This diagnostic is not used for model selection or training.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    save(ROOT/'ADAPTIVE_DIAGNOSTIC_PROTOCOL.json',CFG);protocol=json.loads((ROOT/'PROTOCOL.json').read_text());frozen=json.loads((ROOT/'MODEL_FREEZE.json').read_text())
    entry,z=next((e,z) for e,z in load_rows('test') if e['seed_group']==CFG['seed_group'] and e['kappa']==CFG['kappa'])
    n=len(z['reference_volume']);out=ROOT/'adaptive_diagnostics';out.mkdir(exist_ok=False);results=[]
    sys.path.insert(0,str(ROOT/'dependencies'));import native_fixture
    with tempfile.TemporaryDirectory(prefix='ashebo_adaptive_diagnostic_') as tmp:
        fixture=native_fixture.prepare_native(args.release,tmp);case=native_fixture.make_case(n,fixture,heat_kappa=0)
        for row in frozen['models']:
            model=load_model(row,protocol);initial=np.r_[z['state_y'][0].T.ravel(),np.zeros(n)];count=0;start=time.perf_counter();failure=None;solver=None
            def rhs(t,state):
                nonlocal count
                count+=1
                if count>CFG['max_rhs_calls'] or time.perf_counter()-start>CFG['max_seconds']:raise TimeoutError('Declared adaptive diagnostic computation budget reached')
                y=state[:8*n].reshape(8,n)
                if not np.isfinite(y).all() or np.max(abs(y[:2]))>600 or np.max(abs(y))>10000:raise FloatingPointError('Learned state left the declared finite chart budget')
                dy,dq,_=model.vector(torch.tensor(y[None]),torch.tensor([float(z['kappa'])]));value=np.r_[dy.numpy().ravel(),dq.numpy().ravel()]
                if not np.isfinite(value).all():raise FloatingPointError('Nonfinite adaptive learned RHS')
                return value
            try:
                solver=DOP853(rhs,0,initial,CFG['target_time'],rtol=CFG['rtol'],atol=CFG['atol'],max_step=CFG['max_step'])
                while solver.status=='running':
                    message=solver.step()
                    if solver.status=='failed':raise RuntimeError(message)
            except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
            t=float(solver.t) if solver else 0.;state=solver.y if solver else initial;H=case.energy(state);Q=np.sum(state[8*n:]);content=np.exp(np.asarray(state[:2*n],np.longdouble)).sum()
            finite=bool(np.isfinite(H));drift=float(abs(H+Q-z['H'][0])/max(1,abs(z['H'][0]))) if finite else None
            if drift is not None and not np.isfinite(drift):drift=None
            result=dict(kind=row['kind'],seed_group=12,kappa=100,completed=solver is not None and solver.status=='finished',last_time=t,
                failure=failure,RHS_calls=count,seconds=time.perf_counter()-start,native_H_finite=finite,native_H_plus_heat_relative_drift=drift,
                content_relative_drift=float(abs(content-z['total_inventory'][0])/z['total_inventory'][0]),
                inference='A computation budget exhaustion leaves adaptive long-horizon stability unresolved; it is not proof of mathematical divergence. Held-out operator accuracy is graded separately.')
            np.savez_compressed(out/(row['kind']+'.npz'),time=t,state_y=state[:8*n].reshape(8,n).T,heat_cells=state[8*n:])
            results.append(result);save(ROOT/'ADAPTIVE_DIAGNOSTICS.json',results);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
