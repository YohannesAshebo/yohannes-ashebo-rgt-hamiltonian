#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Qualify one longer native reference without assuming its time window works."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from scipy.integrate import DOP853

ROOT=Path(__file__).resolve().parent


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def integrate(case,cfg,step,folder):
    folder.mkdir();n=case.n;y0=case.initial.copy();H0=case.energy(y0)
    C0=np.exp(np.asarray(y0[:2*n],np.longdouble)).sum()
    times=np.asarray(cfg['saved_times']);saved=[y0];next_save=1;history=[];failure=None
    solver=DOP853(case.rhs,0.,y0,cfg['target_time'],rtol=cfg['rtol'],atol=cfg['atol'],max_step=step)
    began=time.perf_counter();last_report=began;max_energy=0.;max_content=0.
    while solver.status=='running':
        try:
            message=solver.step()
            if solver.status=='failed':raise RuntimeError(message)
            if not np.isfinite(solver.y).all():raise FloatingPointError('Nonfinite state')
            H=case.energy(solver.y);Q=np.asarray(solver.y[8*n:8*n+4],np.longdouble).sum()
            C=np.exp(np.asarray(solver.y[:2*n],np.longdouble)).sum()
            er=float(abs(H+Q-H0)/max(1,abs(H0)));cr=float(abs(C-C0)/C0)
            max_energy=max(max_energy,er);max_content=max(max_content,cr)
            history.append(dict(time=float(solver.t),energy_heat_scaled_error=er,content_relative_error=cr,heat=float(Q)))
            if er>cfg['energy_heat_relative_gate'] or cr>cfg['content_relative_gate']:raise RuntimeError('Declared reference balance gate exceeded')
            if next_save<len(times) and times[next_save]<=solver.t:
                dense=solver.dense_output()
                while next_save<len(times) and times[next_save]<=solver.t:saved.append(dense(times[next_save]));next_save+=1
            now=time.perf_counter()
            if now-last_report>10:
                progress=dict(step=step,time=float(solver.t),target=cfg['target_time'],accepted_steps=len(history),seconds=now-began,max_energy_error=max_energy,max_content_error=max_content)
                save(folder/'PROGRESS.json',progress);np.savez_compressed(folder/'checkpoint.npz',time=solver.t,state=solver.y)
                print(json.dumps(progress),flush=True);last_report=now
            if now-began>cfg['max_wall_seconds_per_run']:raise TimeoutError('Declared qualification time budget reached; reference remains unqualified beyond saved checkpoint')
            if len(history)>=cfg['max_accepted_steps_per_run']:raise RuntimeError('Declared step budget reached')
        except Exception as error:
            failure=dict(type=type(error).__name__,message=str(error),time=float(solver.t));break
    complete=solver.status=='finished' and len(saved)==len(times) and failure is None
    np.savez_compressed(folder/'states.npz',time=times[:len(saved)],state=np.asarray(saved))
    np.savez_compressed(folder/'checkpoint.npz',time=solver.t,state=solver.y)
    save(folder/'accepted_step_audit.json',history)
    result=dict(completed=complete,final_time=float(solver.t),saved_states=len(saved),accepted_steps=len(history),RHS_evaluations=solver.nfev,
        seconds=time.perf_counter()-began,maximum_step=step,max_energy_error=max_energy,max_content_error=max_content,failure=failure)
    save(folder/'RESULT.json',result);print(json.dumps(result),flush=True)
    return result,np.asarray(saved)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    args=p.parse_args();protocol=json.loads((ROOT/'PROTOCOL.json').read_text());cfg=protocol['long_reference']
    sys.path.insert(0,str(args.pilot/'dependencies'));import native_fixture
    spec=importlib.util.spec_from_file_location('frozen_pilot_generator',args.pilot/'generate_pilot.py');generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    old=json.loads((args.pilot/'PROTOCOL.json').read_text())
    results=[];states=[]
    with tempfile.TemporaryDirectory(prefix='ashebo_dct_native_') as tmp:
        fixture=native_fixture.prepare_native(args.release,tmp)
        for i,step in enumerate(cfg['maximum_steps']):
            case=native_fixture.make_case(cfg['cells'],fixture,heat_kappa=cfg['kappa']);generator.perturb(case,cfg['seed_group'],old)
            result,state=integrate(case,cfg,step,ROOT/f'native_window_level{i}')
            results.append(result);states.append(state)
            if not result['completed']:break
    difference=None
    if len(results)==2 and all(r['completed'] for r in results):difference=float(np.max(abs(states[0]-states[1])/np.maximum(1,abs(states[1]))))
    qualified=difference is not None and difference<=cfg['comparison_tolerance']
    save(ROOT/'NATIVE_WINDOW_CHECK.json',dict(qualified=qualified,results=results,max_scaled_state_difference=difference,
        original_equations_changed=False,scope='One manufactured seed at fixed N=128 and kappa100; no historical time-85-to-95 identification, no spatial convergence and no newly trained long-rollout network.'))


if __name__=='__main__':main()
