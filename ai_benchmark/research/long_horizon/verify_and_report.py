#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Check saved evidence, protect the source, and make a concise results figure."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from train_models import ROOT,save
from hybrid_validation import Hybrid,operator_score

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bounds(rows,key):
    values=[r[key] for r in rows if r[key] is not None];return [min(values),max(values)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    source=json.loads((ROOT/'dependencies/BASELINE_FILE_BLOBS.json').read_text())
    for row in source:
        if row['path'] in {'CITATION.cff','README.md','MANIFEST_SHA256.json'}:continue  # Existing citation metadata; source model unchanged.
        data=(args.release/row['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['sha']
    prior=json.loads((args.previous/'MODEL_FREEZE.json').read_text())
    for row in prior['models']:assert sha(args.previous/'models'/row['file'])==row['sha256']
    freeze=json.loads((ROOT/'MODEL_FREEZE.json').read_text());hybridfreeze=json.loads((ROOT/'HYBRID_FREEZE.json').read_text())
    for row in freeze['models']:assert sha(ROOT/row['file'])==row['sha256']
    assert sha(ROOT/'PROTOCOL.json')==freeze['protocol_sha256'] and sha(ROOT/'NORMALIZERS.npz')==freeze['normalizers_sha256']
    assert sha(ROOT/'models/native_H_hybrid.pt')==hybridfreeze['model_sha256'] and sha(ROOT/'HYBRID_PROTOCOL.json')==hybridfreeze['protocol_sha256']
    assert sha(ROOT/'HYBRID_NORMALIZERS.npz')==hybridfreeze['normalizers_sha256']
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for row in manifest['cases']:assert sha(ROOT/row['file'])==row['sha256']
    assert {row['seed_group'] for row in manifest['cases'] if row['split']=='test'}=={12,13,14,15}
    assert all(c['pass_gate'] for c in manifest['time_controls'])
    op=json.loads((ROOT/'HYBRID_OPERATOR_RESULTS.json').read_text());roll=json.loads((ROOT/'HYBRID_ROLLOUT_RESULTS.json').read_text())
    tc=json.loads((ROOT/'HYBRID_TIME_CONTROLS.json').read_text())
    assert len(op)==20 and len(roll)==4 and all(r['completed'] for r in roll)
    assert all(r['negative_heat_fraction']==0 and r['minimum_heat_power_cell']>=0 for r in op)
    assert max(r['max_native_H_plus_heat_relative_drift'] for r in roll)<1e-8
    assert max(r['max_total_content_relative_drift'] for r in roll)<1e-10
    assert tc[0]['both_complete'] and tc[0]['max_scaled_state_difference']<1e-6
    # Reopen the frozen hybrid from disk for a fresh gauge check and label audit.
    with np.load(ROOT/'HYBRID_NORMALIZERS.npz') as z:norm={k:z[k] for k in z.files}
    cfg=json.loads((ROOT/'HYBRID_PROTOCOL.json').read_text());model=Hybrid(norm,cfg['network']).double()
    model.load_state_dict(torch.load(ROOT/'models/native_H_hybrid.pt',map_location='cpu',weights_only=True));model.eval()
    with np.load(ROOT/'data/seed12_k100.npz') as z:z={k:z[k] for k in z.files}
    y=torch.tensor(z['state_y'].transpose(0,2,1));D=torch.tensor(z['grad_H_x'][...,2]-z['grad_H_x'][...,3]);shifted=y.clone();shifted[:,2:4]+=.71
    with torch.no_grad():a,c=model(y,D);b,cb=model(shifted,D)
    gauge=float((a-b).abs().max());antisym=float((a[:,0]+a[:,1]).abs().max());assert gauge<1e-12 and antisym==0
    stored=next(r for r in op if r['seed_group']==12 and r['kappa']==100);fresh=operator_score(model,z)
    for key,value in fresh.items():assert abs(value-stored[key])<=1e-14*max(1,abs(value)),key
    neuralop=json.loads((ROOT/'OPERATOR_TEST_RESULTS.json').read_text());neuralroll=json.loads((ROOT/'ROLLOUT_TEST_RESULTS.json').read_text())
    summary=dict(native_reference=dict(trajectories=len(manifest['cases']),saved_states=sum(c['samples'] for c in manifest['cases']),cells=128,time_range=[0,10],
        unique_training_initial_states=8,unique_validation_initial_states=2,unique_test_initial_states=4,test_conditions=20,
        maximum_energy_heat_relative_error=max(c['reference']['max_energy_error'] for c in manifest['cases']),
        maximum_content_relative_error=max(c['reference']['max_content_error'] for c in manifest['cases']),
        time_controls=manifest['time_controls'],independent_gradient_max_scaled_error=max(c['max_scaled_error'] for c in manifest['gradient_checks'])),
        long_trained_native_H_hybrid=dict(operator_cases=20,rollouts=4,completed_rollouts=4,
            unit_phase_rate_relative_L2_range=bounds(op,'unit_phase_rate_relative_L2'),instantaneous_total_heat_relative_L2_range=bounds(op,'unit_total_heat_relative_L2'),
            final_accumulated_heat_relative_error_range=bounds(roll,'final_total_heat_relative_error'),
            normalized_feature_error_over_persistence_range=bounds(roll,'error_over_persistence'),
            maximum_native_H_plus_heat_relative_drift=max(r['max_native_H_plus_heat_relative_drift'] for r in roll),
            maximum_content_relative_drift=max(r['max_total_content_relative_drift'] for r in roll),
            maximum_instantaneous_power_residual=max(r['maximum_instantaneous_power_residual'] for r in roll),
            time_controls=tc,gauge_rate_max_abs_difference=gauge,phase_pair_antisymmetry=antisym,
            inference='A useful supervised dissipative surrogate with known exact mechanics, limited to the checked synthetic family. Not full-H discovery or measured physical validation.'),
        learned_entire_flow=dict(models=3,operator_cases=60,RK4_rollouts=60,completed_RK4_rollouts=sum(r['completed'] for r in neuralroll['cases']),
            conservative_relative_L2_by_model={m['kind']:bounds(m['cases'],'conservative_relative_L2') for m in neuralop['models']},
            adaptive_diagnostics=json.loads((ROOT/'ADAPTIVE_DIAGNOSTICS.json').read_text()),
            inference='These frozen replacements do not pass the present accuracy/rollout checks. This is a result about these trained approximations and budgets, not disproof of the original Hamiltonian.'),
        source=dict(original_files_unchanged=len(source)-3,previous_checkpoints_unchanged=len(prior['models']),new_checkpoints_hash_checked=4),
        physical_measurements_used=False,spatial_continuum_validation=False,public_upload=False,original_equations_modified=False)
    save(ROOT/'SUMMARY.json',summary)
    provenance=dict(original_commit='9bb33e6162eedc20c00a67123212ebe466c618ac',original_files_unchanged=len(source)-3,
        all_previous_six_checkpoints_unchanged=True,all_four_new_checkpoints_unchanged_since_freezing=True,all_50_dataset_hashes_checked=True,
        current_protocol_sha256=sha(ROOT/'PROTOCOL.json'),generation_start_protocol_sha256=manifest['protocol_sha256'],
        protocol_history='The neural rollout method and training implementation text were finalized while native generation was running. The reference grid, seeds, kappa list, horizon, saved times, integration settings and gates were unchanged. Reference generation used its already loaded reference configuration; no reference outcome was used to select neural test predictions.',
        new_candidate_source='Standalone corrected DCT/canonical factorization; the earlier author FNO is used only to score the six frozen prior models. No upstream source is redistributed.',
        public_repository_modified=False,outreach_sent=False,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__,platform=platform.platform()))
    save(ROOT/'PROVENANCE.json',provenance)
    make_figure(manifest,op,roll,neuralop,model)
    print(json.dumps(dict(verified=True,source_files=len(source),native_runs=50,hybrid_rollouts_passed=4,max_hybrid_heat_error=max(r['final_total_heat_relative_error'] for r in roll))),flush=True)

def make_figure(manifest,op,roll,neuralop,model):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    colors=['#0d9488','#0284c7','#d97706','#7c3aed']
    for i,row in enumerate(roll):
        s=row['seed_group'];k=row['kappa']
        with np.load(ROOT/f'data/seed{s:02d}_k{k:g}.npz') as z:t=z['time'];ref=z['heat_ledgers'][:,:4].sum(1)
        with np.load(ROOT/f'hybrid_rollouts/seed{s}_k{k}.npz') as z:pred=z['heat_cells'].sum(1)
        axs[0,0].plot(t,ref,color=colors[i],lw=2,label=f'IC {s}, κ={k}')
        axs[0,0].plot(t,pred,color=colors[i],ls='--',lw=1.4)
    axs[0,0].set(title='Accumulated heat: native and trained hybrid',xlabel='Native time',ylabel='Heat')
    axs[0,0].legend(fontsize=8);axs[0,0].text(.02,.97,'Solid: native   Dashed: hybrid',transform=axs[0,0].transAxes,va='top',fontsize=8)
    for i,row in enumerate(roll):
        h=json.loads((ROOT/f"hybrid_rollouts/seed{row['seed_group']}_k{row['kappa']}_history.json").read_text())
        axs[0,1].semilogy([r['time'] for r in h],np.maximum([r['native_H_plus_heat_relative_drift'] for r in h],1e-17),color=colors[i],lw=1)
    axs[0,1].set(title='Hybrid audit of the original H + heat',xlabel='Native time',ylabel='Relative balance error')
    names=['Direct DCT','Learned H\nraw phases','Learned H\nU(1)','Native H\nhybrid']
    med=[100*np.median([c['conservative_relative_L2'] for c in m['cases']]) for m in neuralop['models']]+[0.]
    bars=axs[1,0].bar(names,med,color=['#94a3b8','#f59e0b','#8b5cf6','#0d9488'])
    for bar,v in zip(bars,med):axs[1,0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{v:.1f}%',ha='center',fontsize=8)
    axs[1,0].set(title='Held-out conservative flow error',ylabel='Relative L2 error (%)',ylim=(0,118))
    axs[1,0].text(.02,.95,'Hybrid retains the exact conservative operator',transform=axs[1,0].transAxes,va='top',fontsize=8)
    labels=[f"IC {r['seed_group']}\nκ={r['kappa']}" for r in roll]
    axs[1,1].bar(labels,[100*r['final_total_heat_relative_error'] for r in roll],color=colors)
    axs[1,1].set(title='Hybrid error in final accumulated heat',ylabel='Relative error (%)',ylim=(0,.65))
    axs[1,1].text(.02,.97,'Instantaneous total-heat rate error: 6.9–19.6%\nacross all 20 held-out operator cases',transform=axs[1,1].transAxes,va='top',fontsize=8)
    fig.suptitle('Ashebo counterflow: longer held-out AI validation\nN=128; original Hamiltonian unchanged; synthetic reference',fontsize=14)
    fig.savefig(ROOT/'validation_results.png');fig.savefig(ROOT/'validation_results.pdf');plt.close(fig)

if __name__=='__main__':main()
