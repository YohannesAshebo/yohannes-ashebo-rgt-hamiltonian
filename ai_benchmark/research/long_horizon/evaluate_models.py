#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Score frozen candidates on untouched initial-state groups and free rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
from train_models import ROOT,load_model,load_rows,relative,save

def pointwise(model,z):
    y=torch.tensor(z['state_y'].transpose(0,2,1));con,unit,H,g=model.split(y,False)
    con=con.detach().numpy().transpose(0,2,1);unit=unit.detach().numpy().transpose(0,2,1)
    true=z['rhs_conservative_y'];tu=np.stack([z['rhs_dissipative_unit_y'][...,2],z['rhs_dissipative_unit_y'][...,3],z['heat_cells_unit']],-1)
    k=float(z['kappa']);pred=con.copy();pred[...,2:4]+=k*unit[...,:2]
    total=true+k*z['rhs_dissipative_unit_y'];gy=z['grad_H_x'].copy();gy[...,:2]*=np.exp(z['state_y'][...,:2])
    power=(gy*pred).sum((1,2))+k*unit[...,2].sum(1)
    heat=k*unit[...,2];refheat=k*z['heat_cells_unit'].sum(1)
    charge=(np.exp(z['state_y'][...,:2])*pred[...,:2]).sum((1,2))
    normalized=(pred-total)/model.con_scale.detach().numpy()
    result=dict(conservative_relative_L2=relative(con,true),conservative_per_channel_relative_L2=[relative(con[...,i],true[...,i]) for i in range(8)],
        normalized_full_RHS_RMSE=float(np.sqrt((normalized**2).mean())),
        unit_phase_rate_relative_L2=relative(unit[...,:2],tu[...,:2]),unit_heat_cells_relative_L2=relative(unit[...,2],tu[...,2]),
        unit_total_heat_relative_L2=relative(unit[...,2].sum(1),tu[...,2].sum(1)),
        native_H_plus_predicted_heat_power_max_abs=float(np.max(abs(power))),
        native_H_power_residual_over_mean_reference_heat=float(np.max(abs(power))/max(np.mean(abs(refheat)),1e-30)) if k else None,
        global_content_rate_max_abs=float(np.max(abs(charge))),negative_heat_fraction=float(np.mean(heat < -1e-14)),
        minimum_heat_power_cell=float(heat.min()),learned_H_relative_L2=relative(H.detach().numpy(),z['H']))
    if g is not None:
        conx=con.copy();conx[...,:2]*=np.exp(z['state_y'][...,:2]);conx[...,2:4]+=k*unit[...,:2]
        own=(g.detach().numpy().transpose(0,2,1)*conx).sum((1,2))+heat.sum(1)
        result['learned_H_power_balance_max_abs']=float(np.max(abs(own)))
    return result

def batched_rollout(model,rows,cfg,step,out,label):
    if (out/(label+'_INTEGRATION.json')).exists() and (out/(label+'.npz')).exists():
        result=json.loads((out/(label+'_INTEGRATION.json')).read_text())
        with np.load(out/(label+'.npz')) as z:return result,z['state_y'].transpose(0,1,3,2),z['heat_cells'],z['time']
    initial=np.stack([z['state_y'][0].T for _,z in rows]);b,_,n=initial.shape
    y0=torch.tensor(initial);y=y0.clone();heat=torch.zeros(b,n);k=torch.tensor([float(z['kappa']) for _,z in rows])
    active=torch.ones(b,dtype=torch.bool);failed=[None]*b;saved=[y.numpy().copy()];savedheat=[heat.numpy().copy()];times=[0.]
    target_times=rows[0][1]['time'];next_save=1;steps=int(round(float(target_times[-1])/step));began=time.perf_counter();call_count=0
    last_success=np.zeros(b);last_states=initial.copy();last_heats=np.zeros((b,n));progress=[]
    def fail(mask,t,reason):
        for i in torch.nonzero(mask,as_tuple=False).flatten().tolist():
            if failed[i] is None:failed[i]=dict(reason=reason,stage_time=float(t),last_completed_step_time=float(last_success[i]))
        active[mask]=False
    def rhs(v,t):
        nonlocal call_count
        valid=torch.isfinite(v).all((1,2)) & (v[:,:2].abs().amax((1,2))<600) & (v.abs().amax((1,2))<cfg['max_scaled_chart_deviation_from_initial'])
        fail(active & ~valid,t,'Nonfinite or declared chart magnitude budget exceeded')
        safe=torch.where(active[:,None,None],v,y0)
        dy,dq,_=model.vector(safe,k);call_count+=1
        good=torch.isfinite(dy).all((1,2)) & torch.isfinite(dq).all(1)
        fail(active & ~good,t,'Nonfinite learned vector field')
        return torch.where(active[:,None,None],dy,torch.zeros_like(dy)),torch.where(active[:,None],dq,torch.zeros_like(dq))
    for j in range(steps):
        t=j*step;previous_y=y.clone();previous_heat=heat.clone()
        a,qa=rhs(y,t);bb,qb=rhs(y+.5*step*a,t+.5*step);c,qc=rhs(y+.5*step*bb,t+.5*step);d,qd=rhs(y+step*c,t+step)
        yn=y+step*(a+2*bb+2*c+d)/6;qn=heat+step*(qa+2*qb+2*qc+qd)/6
        good=torch.isfinite(yn).all((1,2)) & torch.isfinite(qn).all(1)
        fail(active & ~good,t+step,'Nonfinite RK4 accepted state')
        y=torch.where(active[:,None,None],yn,previous_y);heat=torch.where(active[:,None],qn,previous_heat)
        now=time.perf_counter()
        for i in torch.nonzero(active,as_tuple=False).flatten().tolist():last_success[i]=t+step
        last_states[active.numpy()]=y[active].numpy();last_heats[active.numpy()]=heat[active].numpy()
        if next_save<len(target_times) and t+step>=target_times[next_save]-1e-9:
            sy=y.numpy().copy();sq=heat.numpy().copy();sy[~active.numpy()]=np.nan;sq[~active.numpy()]=np.nan
            saved.append(sy);savedheat.append(sq);times.append(float(target_times[next_save]));next_save+=1
            progress.append(dict(time=times[-1],active_cases=int(active.sum()),seconds=now-began))
            if next_save%5==0:print(json.dumps(dict(model=model.kind,label=label,**progress[-1])),flush=True)
        if now-began>cfg['max_wall_seconds_per_run'] or call_count>cfg['max_rhs_evaluations']:
            fail(active,t+step,'Declared evaluation time or RHS budget reached');break
        if not active.any():break
    state=np.stack(saved).transpose(1,0,2,3);q=np.stack(savedheat).transpose(1,0,2);timearr=np.asarray(times)
    np.savez_compressed(out/(label+'.npz'),time=timearr,state_y=state.transpose(0,1,3,2),heat_cells=q,last_state_y=last_states.transpose(0,2,1),last_heat_cells=last_heats,last_completed_time=last_success)
    result=dict(step=step,seconds=time.perf_counter()-began,RHS_batch_calls=call_count,completed=[bool(a) and next_save==len(target_times) for a in active],failures=failed,last_completed_time=last_success.tolist(),progress=progress,
        integration='RK4 in log chart without correction; failed cases use inactive placeholders solely to finish other batch members.')
    save(out/(label+'_INTEGRATION.json'),result)
    return result,state,q,timearr

def feature_errors(model,pred,truth):
    from dependencies.radial_generic import canonical_from_log
    # Compare all models in one gauge-invariant intensive feature chart.
    from dependencies.radial_generic import RadialFeatures
    f=RadialFeatures(model.volume,model.radius,True)
    with torch.no_grad():
        p=f(canonical_from_log(torch.tensor(pred))).numpy();t=f(canonical_from_log(torch.tensor(truth))).numpy()
    err=(p[:,:10]-t[:,:10])/model.inv_scale.detach().numpy()[:,:10]
    return float(np.sqrt((err**2).mean()))

def score_rollout(model,rows,integration,states,heat,times,case):
    results=[]
    for i,(entry,z) in enumerate(rows):
        valid=np.isfinite(states[i]).all((1,2));pred=states[i,valid];q=heat[i,valid];t=times[valid];target=z['state_y'][:len(times)].transpose(0,2,1)[valid]
        n=len(z['reference_volume']);htrue=np.array([float(case.energy(s.ravel().astype(np.longdouble))) for s in pred])
        own=[]
        for begin in range(0,len(pred),24):own.extend(model.split(torch.tensor(pred[begin:begin+24]),False)[2].detach().numpy().tolist())
        own=np.asarray(own);inventory=np.exp(pred[:,:2]).sum((1,2));h0=float(z['H'][0]);c0=float(z['total_inventory'][0]);qsum=q.sum(1)
        error=feature_errors(model,pred[1:],target[1:]) if len(pred)>1 else None
        persistence=feature_errors(model,np.repeat(target[:1],max(0,len(target)-1),axis=0),target[1:]) if len(pred)>1 else None
        # An unstable neural state can leave the native Hamiltonian's finite
        # evaluation domain. Record that explicitly; do not encode inf in JSON.
        native_finite=bool(np.isfinite(htrue).all());own_finite=bool(np.isfinite(own).all())
        max_true=float(np.max(abs(htrue+qsum-h0))/max(1,abs(h0))) if native_finite else None
        max_own=float(np.max(abs(own+qsum-own[0]))/max(1,abs(own[0]))) if own_finite else None
        refq=z['heat_ledgers'][:len(times),:4].sum(1)[valid]
        row=dict(kind=model.kind,seed_group=entry['seed_group'],kappa=entry['kappa'],completed=integration['completed'][i],failure=integration['failures'][i],
            last_completed_step_time=integration['last_completed_time'][i],last_scored_time=float(t[-1]),
            feature_RMSE=error,persistence_feature_RMSE=persistence,error_over_persistence=error/persistence if error is not None and persistence else None,
            native_H_plus_predicted_heat_relative_drift=max_true,native_H_finite_on_saved_states=native_finite,
            learned_H_plus_predicted_heat_relative_drift=max_own,learned_H_finite_on_saved_states=own_finite,
            total_content_relative_drift=float(np.max(abs(inventory-c0))/c0),
            final_scored_total_heat_absolute_error=float(abs(qsum[-1]-refq[-1])),
            final_scored_total_heat_relative_error=float(abs(qsum[-1]-refq[-1])/abs(refq[-1])) if abs(refq[-1])>0 else None,
            minimum_accumulated_heat_cell=float(q.min()))
        results.append(row)
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());freeze=json.loads((ROOT/'MODEL_FREEZE.json').read_text())
    assert freeze['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()
    assert freeze['normalizers_sha256']==hashlib.sha256((ROOT/'NORMALIZERS.npz').read_bytes()).hexdigest()
    rows=load_rows('test');out=ROOT/'neural_rollouts';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'dependencies'));import native_fixture
    operators=[];rollouts=[];controls=[]
    with tempfile.TemporaryDirectory(prefix='ashebo_long_evaluate_') as tmp:
        fixture=native_fixture.prepare_native(args.release,tmp);native=native_fixture.make_case(protocol['reference']['cells'],fixture,heat_kappa=0)
        for row in freeze['models']:
            model=load_model(row,protocol)
            cases=[dict(seed_group=e['seed_group'],kappa=e['kappa'],**pointwise(model,z)) for e,z in rows]
            operators.append(dict(kind=row['kind'],cases=cases));save(ROOT/'OPERATOR_TEST_RESULTS.json',dict(models=operators,model_freeze=freeze))
            print(json.dumps(dict(operator_scoring_complete=row['kind'],cases=len(cases))),flush=True)
            result,state,heat,times=batched_rollout(model,rows,protocol['rollout'],protocol['rollout']['maximum_step'],out,row['kind'])
            rollouts.extend(score_rollout(model,rows,result,state,heat,times,native));save(ROOT/'ROLLOUT_TEST_RESULTS.json',dict(cases=rollouts))
            # Predeclared step control: a held-out kappa100 case and kappa500 case.
            chosen=[i for i,(e,z) in enumerate(rows) if e['seed_group']==12 and e['kappa'] in [100,500]]
            cr,cs,cq,ct=batched_rollout(model,[rows[i] for i in chosen],protocol['rollout'],protocol['rollout']['control_step'],out,row['kind']+'_half_step')
            for j,i in enumerate(chosen):
                complete=result['completed'][i] and cr['completed'][j]
                difference=float(np.max(abs(state[i]-cs[j])/np.maximum(1,abs(cs[j])))) if complete and state.shape[1]==cs.shape[1] else None
                controls.append(dict(kind=row['kind'],seed_group=rows[i][0]['seed_group'],kappa=rows[i][0]['kappa'],both_complete=complete,max_scaled_state_difference=difference,
                    interpretation='A time-refinement indicator for this neural vector field; no claim of exact discrete energy conservation.'))
            save(ROOT/'NEURAL_TIME_CONTROLS.json',controls)
    print(json.dumps(dict(scoring_complete=True,operator_cases=sum(len(a['cases']) for a in operators),rollouts=len(rollouts),completed=sum(r['completed'] for r in rollouts))),flush=True)

if __name__=='__main__':main()
