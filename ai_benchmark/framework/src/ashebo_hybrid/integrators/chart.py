"""Float64 chart integrators; continuous identities are audited after stepping."""
from dataclasses import dataclass,field
import time
import numpy as np

@dataclass
class Trajectory:
    time: np.ndarray
    states: np.ndarray
    heat: np.ndarray
    completed: bool
    failure: dict | None
    kappa: float
    integrator: object=field(repr=False)
    requested_times: np.ndarray=field(repr=False)
    stats: dict=field(default_factory=dict)

def _setup(model,z0,t_span,times,dt):
    initial=np.asarray(z0,float);encoded=model.chart.encode(initial)
    if times is None:
        start,stop=map(float,t_span)
        if not np.isfinite(start+stop) or stop<=start:raise ValueError("Invalid time span")
        times=np.r_[np.arange(start,stop,dt),stop]
    times=np.asarray(times,float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError("Times must be finite and strictly increasing")
    state=np.concatenate([encoded,np.zeros(encoded.shape[:-1]+(1,))],axis=-1)
    return initial,state,times

def _result(model,saved,heat,times,requested,completed,failure,kappa,integrator,stats):
    return Trajectory(np.asarray(times),np.asarray(saved),np.asarray(heat),completed,failure,float(kappa),integrator,np.asarray(requested),stats)

class ChartRK4:
    def __init__(self,dt=.005,max_steps=200000,max_seconds=60.):
        if not np.isfinite(dt) or dt<=0:raise ValueError("dt must be positive")
        self.dt=float(dt);self.max_steps=max_steps;self.max_seconds=max_seconds
    def refined(self):return ChartRK4(self.dt/2,self.max_steps*2,self.max_seconds)
    def rollout(self,model,z0,t_span=(0.,1.),kappa=1.,times=None):
        initial,state,requested=_setup(model,z0,t_span,times,self.dt);saved=[initial.copy()];heat=[np.zeros(initial.shape[:-1])];record=[float(requested[0])]
        t=float(requested[0]);steps=0;largest=0.;began=time.perf_counter();failure=None
        try:
            for target in requested[1:]:
                while t<target:
                    if steps>=self.max_steps or time.perf_counter()-began>self.max_seconds:raise TimeoutError("Declared integration computation budget reached")
                    h=min(self.dt,float(target-t))
                    if t+h==t:raise FloatingPointError("Time step cannot advance floating-point clock")
                    a=model.chart_rhs(state,kappa);b=model.chart_rhs(state+.5*h*a,kappa);c=model.chart_rhs(state+.5*h*b,kappa);d=model.chart_rhs(state+h*c,kappa)
                    updated=state+h*(a+2*b+2*c+d)/6
                    if not np.isfinite(updated).all():raise FloatingPointError("Nonfinite RK4 state")
                    model.chart.decode(updated[...,:-1]);state=updated;t=float(t+h);steps+=1;largest=max(largest,h)
                saved.append(model.chart.decode(state[...,:-1]));heat.append(state[...,-1].copy());record.append(float(target))
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,TimeoutError) as error:failure=dict(type=type(error).__name__,message=str(error),last_accepted_time=t)
        complete=failure is None and len(record)==len(requested)
        return _result(model,saved,heat,record,requested,complete,failure,kappa,self,dict(steps=steps,largest_step=largest,step_cap=self.dt,seconds=time.perf_counter()-began))

class AdaptiveDOP853:
    def __init__(self,max_step=.01,rtol=1e-9,atol=1e-11,max_rhs=200000,max_seconds=60.):
        if not np.isfinite(max_step) or max_step<=0 or rtol<=0 or atol<=0:raise ValueError("Positive step/tolerances are required")
        self.max_step=float(max_step);self.rtol=rtol;self.atol=atol;self.max_rhs=max_rhs;self.max_seconds=max_seconds
    def refined(self):return AdaptiveDOP853(self.max_step/2,self.rtol,self.atol,self.max_rhs*2,self.max_seconds)
    def rollout(self,model,z0,t_span=(0.,1.),kappa=1.,times=None):
        try:from scipy.integrate import DOP853
        except ImportError as error:raise ImportError("Install ashebo-hybrid[adaptive] for DOP853") from error
        initial,state,requested=_setup(model,z0,t_span,times,self.max_step);shape=state.shape;saved=[initial.copy()];heat=[np.zeros(initial.shape[:-1])];record=[float(requested[0])]
        began=time.perf_counter();calls=0;largest=0.;steps=0;failure=None;solver=None;next_save=1
        def rhs(t,flat):
            nonlocal calls
            calls+=1
            if calls>self.max_rhs or time.perf_counter()-began>self.max_seconds:raise TimeoutError("Declared adaptive computation budget reached")
            return model.chart_rhs(flat.reshape(shape),kappa).ravel()
        try:
            solver=DOP853(rhs,float(requested[0]),state.ravel(),float(requested[-1]),rtol=self.rtol,atol=self.atol,max_step=self.max_step)
            while solver.status=="running":
                old=solver.t;message=solver.step()
                if solver.status=="failed":raise RuntimeError(message)
                largest=max(largest,float(solver.t-old));steps+=1
                if next_save<len(requested) and requested[next_save]<=solver.t:
                    dense=solver.dense_output()
                    while next_save<len(requested) and requested[next_save]<=solver.t:
                        row=dense(requested[next_save]).reshape(shape);saved.append(model.chart.decode(row[...,:-1]));heat.append(row[...,-1].copy());record.append(float(requested[next_save]));next_save+=1
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,TimeoutError) as error:failure=dict(type=type(error).__name__,message=str(error),last_accepted_time=float(solver.t) if solver else float(requested[0]))
        complete=failure is None and len(record)==len(requested)
        return _result(model,saved,heat,record,requested,complete,failure,kappa,self,dict(steps=steps,RHS_calls=calls,largest_step=largest,step_cap=self.max_step,seconds=time.perf_counter()-began))
