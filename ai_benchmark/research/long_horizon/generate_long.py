#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Generate, audit and freeze the longer fixed-grid reference dataset."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'dependencies'))
import native_fixture
import frozen_pilot_generator as generator
from qualify_native_window import integrate

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);args=p.parse_args()
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());cfg=protocol['reference']
    old=json.loads((ROOT/'dependencies/PILOT_PROTOCOL.json').read_text())
    for row in json.loads((ROOT/'dependencies/BASELINE_FILE_BLOBS.json').read_text()):
        if row['path'] in {'CITATION.cff','README.md','MANIFEST_SHA256.json'}:continue  # Known DOI/citation metadata only; all native and numerical source still checked.
        raw=(args.release/row['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
    spec=importlib.util.spec_from_file_location('independent_native_audit',args.release/'audit_mechanics.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    out=ROOT/'data';out.mkdir(exist_ok=False);rawout=ROOT/'reference_runs';rawout.mkdir(exist_ok=False)
    manifest=dict(protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),cases=[],time_controls=[],gradient_checks=[],source_files_checked=91,citation_metadata_exempted=["CITATION.cff","README.md","MANIFEST_SHA256.json"],
        physical_measurements=False,spatial_continuum_reference=False)
    maxima={key:0. for key in ['RHS_scaled_error','continuity_scaled_error','independent_energy_scaled_error','power_scaled_error','heat_sum_scaled_error']}
    with tempfile.TemporaryDirectory(prefix='ashebo_long_native_') as tmp:
        fixture=native_fixture.prepare_native(args.release,tmp)
        for split,key in [('train','train_seed_groups'),('validation','validation_seed_groups'),('test','test_seed_groups')]:
            for seed in cfg[key]:
                for k in cfg['test_kappas'] if split=='test' else cfg['train_validation_kappas']:
                    case=native_fixture.make_case(cfg['cells'],fixture,heat_kappa=k);coefficients=generator.perturb(case,seed,old)
                    label=f'seed{seed:02d}_k{k:g}';result,states=integrate(case,cfg,cfg['maximum_step'],rawout/label)
                    if not result['completed']:raise RuntimeError(dict(scenario=label,reference=result))
                    stack={};checks=[]
                    for index,state in enumerate(states):
                        a,c,g,y=generator.label(case,state,k,audit)
                        assert max(c[field] for field in maxima)<1e-9 and c['minimum_heat_cell']>=0,c
                        checks.append(c)
                        for field in maxima:maxima[field]=max(maxima[field],c[field])
                        # Retain the independent derivative and reference-chart labels.
                        # Extensive states and total rates are reconstructed exactly from the chart and known kappa.
                        fields=['state_y','grad_H_x','rhs_conservative_y','rhs_dissipative_unit_y','heat_cells_unit','H','total_inventory','complete_flux','gamma']
                        for field in fields:stack.setdefault(field,[]).append(np.asarray(a[field],dtype=float))
                        flux=a['complete_flux'][1:-1];A=a['interior_area'];paired=np.maximum(0,np.minimum(-flux[:,0]/A,flux[:,1]/A))
                        j=(np.r_[0,paired]+np.r_[paired,0])/2
                        stack.setdefault('unit_mobility',[]).append(np.asarray(j/a['state_x'][:,:2].sum(-1),float))
                        if dict(seed=seed,kappa=k,index=index) in cfg['complex_gradients']:
                            check,_=audit.complex_gradient(case.model,case.law,y,g.ravel());assert check['max_scaled_error']<1e-10,check
                            manifest['gradient_checks'].append(dict(seed=seed,kappa=k,time=cfg['saved_times'][index],**check))
                    file=out/(label+'.npz')
                    np.savez_compressed(file,**{key:np.asarray(value) for key,value in stack.items()},time=cfg['saved_times'],heat_ledgers=states[:,8*case.n:],
                        reference_volume=np.asarray(case.model.V0,float),r=np.asarray(case.model.r,float),kappa=float(k),seed_group=seed,perturbation_coefficients=coefficients)
                    manifest['cases'].append(dict(file=str(file.relative_to(ROOT)),seed_group=seed,kappa=k,split=split,samples=len(states),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                        reference=result,pointwise_checks=checks))
                    if dict(seed=seed,kappa=k) in cfg['time_controls']:
                        control,finer=integrate(case,cfg,cfg['control_step'],rawout/(label+'_half_step'))
                        difference=float(np.max(abs(states-finer)/np.maximum(1,abs(finer)))) if control['completed'] else None
                        passed=difference is not None and difference<=cfg['comparison_tolerance']
                        manifest['time_controls'].append(dict(seed=seed,kappa=k,maximum_scaled_state_difference=difference,pass_gate=passed,reference=control))
                        assert passed,manifest['time_controls'][-1]
                    manifest['pointwise_maxima']=maxima;save(ROOT/'GENERATION_CHECKPOINT.json',manifest)
                    print(json.dumps(dict(saved=label,split=split,cases=len(manifest['cases']),seconds=result['seconds'])),flush=True)
    assert len(manifest['cases'])==50 and len(manifest['gradient_checks'])==3
    manifest['reference_gates_pass']=True;save(ROOT/'MANIFEST.json',manifest)
    print(json.dumps(dict(completed=True,cases=len(manifest['cases']),samples=sum(c['samples'] for c in manifest['cases']),maxima=maxima)),flush=True)

if __name__=='__main__':main()
