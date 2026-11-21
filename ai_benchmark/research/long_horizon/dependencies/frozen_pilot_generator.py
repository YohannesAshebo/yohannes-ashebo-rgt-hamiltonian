#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Generate audited fixed-grid reference trajectories and split operator labels."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
LD=np.longdouble


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def perturb(case,seed,protocol):
    rng=np.random.default_rng(protocol['seed_base']+seed)
    coefficients=rng.normal(size=(8,protocol['perturbations']['modes']))
    r=np.asarray(case.model.r,float);x=r/protocol['outer_radius']
    basis=np.stack([np.cos((k+1)*np.pi*x) for k in range(coefficients.shape[1])])
    values=coefficients@basis/np.sqrt(coefficients.shape[1])
    q=case.initial[:8*case.n].reshape(8,case.n).copy()
    p=protocol['perturbations']
    q[:2]+=p['log_content_amplitude']*values[:2]
    q[2:4]+=p['phase_amplitude']*values[2:4]
    q[4:6]+=p['metric_amplitude']*x*x*values[4:6]
    q[6:8]+=p['momentum_density_amplitude']*case.model.V0*x*x*values[6:8]
    case.initial=np.r_[q.ravel(),np.zeros(5)]
    return coefficients


def label(case,state,kappa,audit):
    n=case.n;m=case.model
    y=np.asarray(state[:8*n],dtype=LD)
    base=m.evaluate(y);extra=case.law.evaluate(y)
    N=np.exp(y[:2*n].reshape(2,n))
    g=m.unpack(base['gradient'])+extra['gradient']
    con=case.law.add_rhs(base['rhs'].copy(),y).reshape(8,n)
    canonical=y.reshape(8,n).copy();canonical[:2]=N
    gx=g.copy();gx[:2]/=N
    conx=con.copy();conx[:2]*=N
    fullflux=base['flux']+extra['extra_flux']
    paired_faces=np.maximum(0,np.minimum(-fullflux[0]/base['A'],fullflux[1]/base['A']))
    paired=(np.r_[0,paired_faces]+np.r_[paired_faces,0])/2
    D=g[2]-g[3]
    rate=paired*D/N.sum(axis=0)
    unit=np.zeros((8,n),dtype=LD);unit[2]=-rate;unit[3]=rate
    q1=rate*D
    total=con+kappa*unit;totalx=conx+kappa*unit
    actual=case.rhs(0,np.asarray(state,float))
    core_rhs=np.asarray(actual[:8*n],dtype=LD).reshape(8,n)
    error=float(np.max(abs(core_rhs-total)/np.maximum(1,abs(total))))
    flux=np.zeros((2,n+1),dtype=LD);flux[:,1:-1]=fullflux
    bare=np.zeros_like(flux);bare[:,1:-1]=base['flux']
    gamma=base['Gamma']+extra['Gamma']
    transport=flux[:,:-1]-flux[:,1:]
    expected=transport+np.stack([-gamma,gamma])
    continuity=float(np.max(abs(conx[:2]-expected)/np.maximum(1,abs(expected))))
    H=base['H']+extra['energy'];independent=audit.scalar_energy(m,case.law,y)
    energy_error=float(abs(H-independent)/max(1,abs(H)))
    power=np.sum(g*core_rhs)
    native_heat=np.sum(np.asarray(actual[8*n:8*n+4],dtype=LD))
    power_error=float(abs(power+native_heat)/max(1,abs(power),abs(native_heat)))
    heat_error=float(abs(native_heat-kappa*q1.sum())/max(1,abs(native_heat)))
    chi=base['chi'];core=0.;exists=False
    if chi[0]>0:
        ids=np.flatnonzero((chi[:-1]>0)&(chi[1:]<=0))
        if len(ids):
            i=int(ids[0]);core=float(m.r[i]+(m.r[i+1]-m.r[i])*chi[i]/(chi[i]-chi[i+1]));exists=True
    arrays=dict(state_y=y.reshape(8,n).T,state_x=canonical.T,
        grad_H_y=g.T,grad_H_x=gx.T,rhs_conservative_y=con.T,rhs_conservative_x=conx.T,
        rhs_dissipative_unit_y=unit.T,rhs_dissipative_unit_x=unit.T,
        rhs_total_y=total.T,rhs_total_x=totalx.T,
        heat_cells_unit=q1,heat_cells=kappa*q1,complete_flux=flux.T,bare_flux=bare.T,
        gamma=gamma,proper_volume=base['V'],interior_area=base['A'],
        H=np.asarray(H),total_inventory=np.asarray(N.sum()),core_radius=np.asarray(core),core_exists=np.asarray(exists))
    check=dict(RHS_scaled_error=error,continuity_scaled_error=continuity,
        independent_energy_scaled_error=energy_error,power_scaled_error=power_error,
        heat_sum_scaled_error=heat_error,minimum_heat_cell=float(np.min(kappa*q1)))
    return arrays,check,g,y


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--r4-audit',type=Path,required=True)
    args=p.parse_args();out=ROOT/'data';out.mkdir(exist_ok=False)
    rawout=ROOT/'reference_runs';rawout.mkdir(exist_ok=False)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    for name,key in [('native_fixture.py','r4_fixture_sha256'),('run_investigation.py','r4_integration_driver_sha256')]:
        assert hashlib.sha256((args.r4_audit/name).read_bytes()).hexdigest()==protocol[key]
    source_rows=json.loads((args.r4_audit/'BASELINE_FILE_BLOBS.json').read_text())
    for row in source_rows:
        raw=(args.release/row['path']).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
    sys.path.insert(0,str(args.r4_audit))
    import native_fixture as helper
    import run_investigation as integrator
    spec=importlib.util.spec_from_file_location('pilot_independent_audit',args.release/'audit_mechanics.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    manifest=dict(name=protocol['name'],version=protocol['version'],baseline_commit=protocol['baseline_commit'],
        protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
        cells=protocol['cells'],cases=[],quality_controls=[],gradient_checks=[],
        pointwise_check_maxima={},source_files_checked=len(source_rows),
        continuum_ground_truth=False,physical_measurements_used=False,public_upload=False)
    maxchecks={key:0. for key in ['RHS_scaled_error','continuity_scaled_error','independent_energy_scaled_error','power_scaled_error','heat_sum_scaled_error']}
    with tempfile.TemporaryDirectory(prefix='ashebo_sciml_pilot_') as temporary:
        fixture=helper.prepare_native(args.release,temporary)
        first=True
        for split,key in [('train','train_seed_groups'),('validation','validation_seed_groups'),('test','test_seed_groups')]:
            for seed in protocol[key]:
                kappas=protocol['test_kappas'] if split=='test' else protocol['training_and_validation_kappas']
                for kappa in kappas:
                    case=helper.make_case(protocol['cells'],fixture,heat_kappa=kappa)
                    coefficients=perturb(case,seed,protocol)
                    scenario=f'seed{seed:02d}_k{kappa:g}'
                    result=integrator.integrate(case,protocol,scenario,protocol['maximum_step'],rawout)
                    assert result['gates_pass'],result
                    with np.load(rawout/result['name']/'states.npz') as z:states=z['state'];times=z['time']
                    stack={};checks=[]
                    for index,state in enumerate(states):
                        arrays,check,gradient,native_state=label(case,state,kappa,audit)
                        assert max(check['RHS_scaled_error'],check['continuity_scaled_error'])<=protocol['pointwise_RHS_and_continuity_scaled_tolerance'],check
                        assert max(check['power_scaled_error'],check['heat_sum_scaled_error'])<=protocol['pointwise_power_scaled_tolerance'],check
                        assert check['independent_energy_scaled_error']<=protocol['independent_scalar_energy_scaled_tolerance'],check
                        assert check['minimum_heat_cell']>=0,check
                        for field in maxchecks:maxchecks[field]=max(maxchecks[field],check[field])
                        checks.append(check)
                        for field,value in arrays.items():stack.setdefault(field,[]).append(np.asarray(value,dtype=bool if field=='core_exists' else float))
                        if dict(seed=seed,kappa=kappa,saved_index=index) in protocol['complex_gradient_cases']:
                            checked,_=audit.complex_gradient(case.model,case.law,native_state,gradient.ravel())
                            assert checked['max_scaled_error']<=protocol['complex_gradient_scaled_tolerance'],checked
                            manifest['gradient_checks'].append(dict(seed=seed,kappa=kappa,saved_index=index,**checked))
                    filename=scenario+'.npz'
                    packed={field:np.asarray(value) for field,value in stack.items()}
                    np.savez_compressed(out/filename,**packed,time=times,heat_ledgers=states[:,8*case.n:],
                        reference_volume=case.model.V0,r=case.model.r,faces=case.model.faces,
                        kappa=kappa,seed_group=seed,perturbation_coefficients=coefficients)
                    entry=dict(file='data/'+filename,seed_group=seed,kappa=kappa,split=split,
                        sample_count=len(states),reference_run='reference_runs/'+result['name'],evolution=result,pointwise_checks=checks)
                    manifest['cases'].append(entry)
                    if first:
                        manifest['model_parameters']=asdict(case.model.p)
                        manifest['exchange_parameters']=dict(omega=float(case.law.omega),j0=float(case.law.j0),p0=float(case.model.p0),caps='continuum_native_caps')
                        first=False
                    if dict(seed=seed,kappa=kappa) in protocol['time_control_cases']:
                        cfg={**protocol,'maximum_step':protocol['time_control_maximum_step']}
                        control=integrator.integrate(case,cfg,scenario+'_time_control',cfg['maximum_step'],rawout)
                        assert control['gates_pass'],control
                        with np.load(rawout/control['name']/'states.npz') as z:finer=z['state']
                        difference=float(np.max(abs(states-finer)/np.maximum(1,abs(finer))))
                        active=min(result['actual_largest_step']/result['maximum_step'],control['actual_largest_step']/control['maximum_step'])>.9
                        assert difference<=protocol['time_control_scaled_tolerance'] and active
                        manifest['quality_controls'].append(dict(seed=seed,kappa=kappa,maximum_scaled_state_difference=difference,
                            caps_active=active,reference_run='reference_runs/'+control['name'],evolution=control,pass_gate=True))
                    manifest['pointwise_check_maxima']=maxchecks
                    save(ROOT/'GENERATION_CHECKPOINT.json',manifest)
    assert len(manifest['cases'])==protocol['expected_dataset_trajectories']
    assert len(manifest['gradient_checks'])==len(protocol['complex_gradient_cases'])
    manifest.update(dataset_reference_gates_pass=True,physical_validation_claimed=False)
    save(ROOT/'MANIFEST.json',manifest)
    print(json.dumps(dict(trajectories=len(manifest['cases']),saved_states=sum(c['sample_count'] for c in manifest['cases']),
        reference_gates_pass=True,pointwise_maxima=maxchecks)),flush=True)


if __name__=='__main__':main()
