#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Train and validate a positive mobility surrogate with the original H intact."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.integrate import DOP853
from train_models import ROOT,load_rows,save,relative
from dependencies.radial_generic import RadialFeatures,DCTBackbone,canonical_from_log

def examples(rows):
    y=np.concatenate([z['state_y'].transpose(0,2,1) for _,z in rows]);D=np.concatenate([z['grad_H_x'][...,2]-z['grad_H_x'][...,3] for _,z in rows])
    target=np.concatenate([np.stack([z['rhs_dissipative_unit_y'][...,2],z['rhs_dissipative_unit_y'][...,3],z['heat_cells_unit']],1) for _,z in rows])
    c=np.concatenate([z['unit_mobility'] for _,z in rows])
    return dict(y=y,D=D,target=target,c=c)

class Hybrid(nn.Module):
    def __init__(self,norm,cfg):
        super().__init__()
        for key,value in norm.items():self.register_buffer(key,torch.tensor(value,dtype=torch.float64))
        self.features=RadialFeatures(norm['volume'],norm['radius'],True)
        self.net=DCTBackbone(12,1,width=cfg['width'],modes=cfg['modes'],layers=cfg['layers'])
        nn.init.zeros_(self.net.head[-1].weight);nn.init.constant_(self.net.head[-1].bias,float(np.log(np.expm1(1.))))
    def forward(self,y,D):
        feature=torch.cat([self.features(canonical_from_log(y)),D[:,None]],1)
        c=self.c_base*F.softplus(self.net((feature-self.mean)/self.scale)[:,0])
        return torch.stack([-c*D,c*D,c*D*D],1),c
    def predict(self,y,D):
        with torch.no_grad():unit,_=self(torch.tensor(y[None]),torch.tensor(D[None]))
        result=unit[0].numpy()
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite hybrid constitutive prediction')
        return result

def norms(data,volume,radius):
    f=RadialFeatures(volume,radius,True)
    with torch.no_grad():fe=torch.cat([f(canonical_from_log(torch.tensor(data['y']))),torch.tensor(data['D'])[:,None]],1).numpy()
    c=data['c'];base=np.maximum(c.mean(0),.01*np.sqrt((c*c).mean()))*(c.max(0)>0)
    return dict(volume=volume,radius=radius,mean=fe.mean((0,2),keepdims=True),scale=np.maximum(fe.std((0,2),keepdims=True),1e-8),c_base=base,
        target_scale=np.maximum(np.sqrt((data['target']**2).mean((0,2))),1e-20),c_scale=np.sqrt((c*c).mean()))

def loss(model,data):
    unit,c=model(data['y'],data['D'])
    return (((unit-data['target'])/model.target_scale[None,:,None])**2).mean()+.1*(((c-data['c'])/model.c_scale)**2).mean()

def meanloss(model,data):
    with torch.no_grad():return float(loss(model,data))

def train(cfg):
    rows=load_rows('train');vr=load_rows('validation');data=examples(rows);vd=examples(vr)
    norm=norms(data,rows[0][1]['reference_volume'],rows[0][1]['r']);np.savez_compressed(ROOT/'HYBRID_NORMALIZERS.npz',**norm)
    data={key:torch.tensor(v) for key,v in data.items()};vd={key:torch.tensor(v) for key,v in vd.items()}
    torch.manual_seed(cfg['network']['seed']);model=Hybrid(norm,cfg['network']).double();tcfg=cfg['training']
    opt=torch.optim.AdamW(model.parameters(),lr=tcfg['learning_rate'],weight_decay=tcfg['weight_decay']);rng=np.random.default_rng(cfg['network']['seed'])
    best=float('inf');best_state=None;best_update=None;history=[];began=time.perf_counter()
    for update in range(tcfg['updates']+1):
        if update:
            idx=rng.choice(len(data['D']),tcfg['batch_size'],False);batch={key:v[idx] for key,v in data.items()}
            opt.zero_grad(set_to_none=True);value=loss(model,batch)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite hybrid training loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg['gradient_clip_norm']);opt.step()
        if update%tcfg['validation_every']==0:
            tl=meanloss(model,data);vl=meanloss(model,vd);row=dict(update=update,training_loss=tl,validation_loss=vl,seconds=time.perf_counter()-began)
            history.append(row);print(json.dumps(dict(hybrid_training=True,**row)),flush=True)
            if vl<best:best=vl;best_state=copy.deepcopy(model.state_dict());best_update=update
    model.load_state_dict(best_state);torch.save(model.state_dict(),ROOT/'models/native_H_hybrid.pt');save(ROOT/'HYBRID_TRAINING.json',history)
    freeze=dict(model_sha256=hashlib.sha256((ROOT/'models/native_H_hybrid.pt').read_bytes()).hexdigest(),
        normalizers_sha256=hashlib.sha256((ROOT/'HYBRID_NORMALIZERS.npz').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'HYBRID_PROTOCOL.json').read_bytes()).hexdigest(),
        best_update=best_update,validation_loss=best,parameters=sum(p.numel() for p in model.parameters()),test_arrays_opened_by_hybrid_training=False)
    save(ROOT/'HYBRID_FREEZE.json',freeze);return model

def operator_score(model,z):
    a=examples([({},z)])
    with torch.no_grad():unit,c=model(torch.tensor(a['y']),torch.tensor(a['D']));unit=unit.numpy();c=c.numpy()
    true=a['target'];Q=unit[:,2].sum(1);Qt=true[:,2].sum(1);k=float(z['kappa'])
    power=(-a['D']*c*a['D']).sum(1)+Q
    return dict(unit_phase_rate_relative_L2=relative(unit[:,:2],true[:,:2]),unit_heat_cells_relative_L2=relative(unit[:,2],true[:,2]),
        unit_total_heat_relative_L2=relative(Q,Qt),unit_mobility_relative_L2=relative(c,a['c']),
        power_residual_max_abs=float(abs(power).max()),minimum_heat_power_cell=float((k*unit[:,2]).min()),negative_heat_fraction=float(np.mean(k*unit[:,2]<-1e-14)))

def integrate(model,case,z,cfg,step,out,label):
    n=case.n;initial=np.r_[z['state_y'][0].T.ravel(),np.zeros(n)];k=float(z['kappa']);times=z['time'];saved=[initial.copy()];next_save=1
    H0=case.energy(initial);N0=np.exp(np.asarray(initial[:2*n],np.longdouble)).sum();history=[];failure=None;began=time.perf_counter();calls=0;maxpower=0.
    def rhs(t,state):
        nonlocal calls,maxpower
        calls+=1
        if calls>cfg['max_rhs_per_case'] or time.perf_counter()-began>cfg['max_seconds_per_case']:raise TimeoutError('Declared hybrid time/RHS budget reached')
        y=state[:8*n].reshape(8,n);con=case.compiled(t,state[:8*n]).reshape(8,n)
        # From canonical J: H_thetaC=NCdot and H_thetaR=NRdot.
        D=np.exp(y[0])*con[0]-np.exp(y[1])*con[1]
        unit=model.predict(y,D) if k else np.zeros((3,n));rate=k*unit
        total=con.copy();total[2]+=rate[0];total[3]+=rate[1]
        maxpower=max(maxpower,float(abs((np.exp(y[0])*con[0]*rate[0]+np.exp(y[1])*con[1]*rate[1]).sum()+rate[2].sum())))
        value=np.r_[total.ravel(),rate[2]]
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite hybrid vector field')
        return value
    solver=None
    try:
        solver=DOP853(rhs,0,initial,float(times[-1]),max_step=step,rtol=cfg['rtol'],atol=cfg['atol'])
        while solver.status=='running':
            message=solver.step()
            if solver.status=='failed':raise RuntimeError(message)
            H=case.energy(solver.y);Q=np.asarray(solver.y[8*n:],np.longdouble).sum();N=np.exp(np.asarray(solver.y[:2*n],np.longdouble)).sum()
            history.append(dict(time=float(solver.t),native_H_plus_heat_relative_drift=float(abs(H+Q-H0)/max(1,abs(H0))),content_relative_drift=float(abs(N-N0)/N0)))
            if next_save<len(times) and times[next_save]<=solver.t:
                dense=solver.dense_output()
                while next_save<len(times) and times[next_save]<=solver.t:saved.append(dense(times[next_save]));next_save+=1
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error),time=float(solver.t) if solver else 0.)
    completed=solver is not None and solver.status=='finished' and len(saved)==len(times) and failure is None
    saved=np.asarray(saved);y=saved[:,:8*n].reshape(len(saved),8,n);Q=saved[:,8*n:]
    score=dict(completed=completed,failure=failure,seconds=time.perf_counter()-began,RHS_calls=calls,accepted_steps=len(history),maximum_step=step,last_saved_time=float(times[len(saved)-1]),
        maximum_instantaneous_power_residual=maxpower,max_native_H_plus_heat_relative_drift=max((r['native_H_plus_heat_relative_drift'] for r in history),default=0),
        max_total_content_relative_drift=max((r['content_relative_drift'] for r in history),default=0))
    if completed:
        # One common physical feature chart; all scales are training-only.
        target=torch.tensor(z['state_y'].transpose(0,2,1));pred=torch.tensor(y)
        with torch.no_grad():pf=model.features(canonical_from_log(pred));tf=model.features(canonical_from_log(target));base=tf[:1].expand_as(tf)
        scale=model.scale[:,:10]
        err=float((((pf[1:,:10]-tf[1:,:10])/scale)**2).mean().sqrt());per=float((((base[1:,:10]-tf[1:,:10])/scale)**2).mean().sqrt())
        tq=z['heat_ledgers'][:,:4].sum(1)
        score.update(feature_RMSE=err,persistence_feature_RMSE=per,error_over_persistence=err/per,
            final_total_heat_relative_error=float(abs(Q[-1].sum()-tq[-1])/abs(tq[-1])),minimum_accumulated_heat_cell=float(Q.min()))
    np.savez_compressed(out/(label+'.npz'),time=times[:len(saved)],state_y=y.transpose(0,2,1),heat_cells=Q)
    save(out/(label+'_history.json'),history);save(out/(label+'_metrics.json'),score)
    print(json.dumps(dict(hybrid=label,**score)),flush=True);return score,saved

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);cfg=json.loads((ROOT/'HYBRID_PROTOCOL.json').read_text())
    model=train(cfg);model.eval();rows=load_rows('test')
    scores=[dict(seed_group=e['seed_group'],kappa=e['kappa'],**operator_score(model,z)) for e,z in rows];save(ROOT/'HYBRID_OPERATOR_RESULTS.json',scores)
    out=ROOT/'hybrid_rollouts';out.mkdir(exist_ok=False);results=[];controls=[]
    sys.path.insert(0,str(ROOT/'dependencies'));import native_fixture
    with tempfile.TemporaryDirectory(prefix='ashebo_hybrid_long_') as tmp:
        fixture=native_fixture.prepare_native(args.release,tmp);case=native_fixture.make_case(128,fixture,heat_kappa=0)
        for e,z in rows:
            if dict(seed=e['seed_group'],kappa=e['kappa']) not in cfg['rollout_cases']:continue
            label=f"seed{e['seed_group']}_k{e['kappa']}";result,state=integrate(model,case,z,cfg['integrator'],cfg['integrator']['max_step'],out,label)
            results.append(dict(seed_group=e['seed_group'],kappa=e['kappa'],**result));save(ROOT/'HYBRID_ROLLOUT_RESULTS.json',results)
            if dict(seed=e['seed_group'],kappa=e['kappa'])==cfg['time_control']:
                finer,fstate=integrate(model,case,z,cfg['integrator'],cfg['integrator']['control_max_step'],out,label+'_half_step')
                difference=float(np.max(abs(state-fstate)/np.maximum(1,abs(fstate)))) if result['completed'] and finer['completed'] else None
                controls.append(dict(seed_group=e['seed_group'],kappa=e['kappa'],both_complete=result['completed'] and finer['completed'],max_scaled_state_difference=difference))
                save(ROOT/'HYBRID_TIME_CONTROLS.json',controls)
    print(json.dumps(dict(hybrid_complete=True,operator_cases=len(scores),rollouts=len(results),completed=sum(r['completed'] for r in results))),flush=True)

if __name__=='__main__':main()
