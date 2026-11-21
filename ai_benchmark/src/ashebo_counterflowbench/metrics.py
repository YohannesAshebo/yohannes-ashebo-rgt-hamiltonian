"""Grade against the native Hamiltonian labels, including unsuccessful predictions."""
import numpy as np

def _relative(a,b):
    scale=max(float(np.max(abs(a),initial=0)),float(np.max(abs(b),initial=0)),1e-300)
    den=np.linalg.norm(b/scale)
    return float(np.linalg.norm(a/scale-b/scale)/den) if den else None

def _array(value,shape,name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape:raise ValueError(f"{name}: expected {shape}, received {a.shape}")
    if not np.isfinite(a).all():raise ValueError(f"{name} contains NaN or infinity; report this as a failed prediction")
    return a

def reference_audit(case):
    H=case["H"];Q=case.accumulated_heat;C=case["total_inventory"];rhs=case.rhs_y;heat=case.heat_power
    power=(case.gradient_y*rhs).sum((1,2))+heat.sum(1)
    charge=(np.exp(case.state_y[...,:2])*rhs[...,:2]).sum((1,2))
    return dict(energy_plus_heat_relative_drift=float(np.max(abs(H+Q-H[0]))/max(1,abs(H[0]))),
        total_content_relative_drift=float(np.max(abs(C-C[0]))/abs(C[0])),
        instantaneous_power_residual_max_abs=float(np.max(abs(power))),global_content_rate_max_abs=float(np.max(abs(charge))),
        minimum_heat_cell_power=float(np.min(heat)),negative_heat_fraction=float(np.mean(heat<0)))

def grade_operator(case,rhs_y,heat_cells):
    """Score full rates in log-content chart and per-cell heat power (not density)."""
    rhs=_array(rhs_y,case.state_y.shape,"rhs_y");q=_array(heat_cells,case.heat_power.shape,"heat_cells")
    true=case.rhs_y;power=(case.gradient_y*rhs).sum((1,2))+q.sum(1)
    charge=(np.exp(case.state_y[...,:2])*rhs[...,:2]).sum((1,2))
    result=dict(case=case.id,split=case.split,rhs_relative_L2=_relative(rhs,true),
        rhs_per_channel_relative_L2=[_relative(rhs[...,i],true[...,i]) for i in range(8)],
        heat_cells_relative_L2=_relative(q,case.heat_power),total_heat_rate_relative_L2=_relative(q.sum(1),case.heat_power.sum(1)),
        phase_relaxation_residual_relative_L2=_relative((rhs-case["rhs_conservative_y"])[...,2:4],case.kappa*case["rhs_dissipative_unit_y"][...,2:4]),
        native_H_power_residual_max_abs=float(np.max(abs(power))),global_content_rate_max_abs=float(np.max(abs(charge))),
        minimum_heat_cell_power=float(q.min()),negative_heat_fraction=float(np.mean(q<0)),
        maximum_phase_sum_rate_error=float(np.max(abs((rhs-true)[...,2:4].sum(-1)))))
    if "complete_flux" in case.arrays and "gamma" in case.arrays:
        flux=case["complete_flux"];gamma=case["gamma"]
        expected=flux[:,:-1]-flux[:,1:]+np.stack([-gamma,gamma],axis=-1)
        observed=np.exp(case.state_y[...,:2])*rhs[...,:2]
        result["local_reference_content_balance_max_abs"]=float(np.max(abs(observed-expected)))
    return result

def grade_rollout(case,state_y,heat_cells,native_H):
    """Grade saved trajectories using H evaluated on the PREDICTED states.

    native_H must be an evaluator callable(state_y_at_time)->scalar; passing
    reference H in place of recomputing H on predictions is not a valid audit.
    Heat input here is accumulated per-cell heat, with a zero initial ledger.
    """
    y=_array(state_y,case.state_y.shape,"state_y");Q=_array(heat_cells,case.heat_power.shape,"heat_cells")
    if not np.allclose(y[0],case.state_y[0],rtol=0,atol=1e-12):raise ValueError("Rollout must start from the reference initial state")
    if np.max(abs(Q[0]))>1e-12:raise ValueError("Initial accumulated heat must be zero")
    if not callable(native_H):raise TypeError("native_H must evaluate predicted states")
    H=np.asarray([native_H(row) for row in y],float)
    if H.shape!=(len(case.time),) or not np.isfinite(H).all():raise ValueError("Native H is not finite on this predicted trajectory")
    N=np.exp(y[...,:2]);C=N.sum((1,2));H0=float(case["H"][0]);C0=float(case["total_inventory"][0]);qsum=Q.sum(1)
    target=case.accumulated_heat
    return dict(case=case.id,state_chart_relative_L2=_relative(y,case.state_y),
        native_H_plus_heat_relative_drift=float(np.max(abs(H+qsum-H0))/max(1,abs(H0))),
        total_content_relative_drift=float(np.max(abs(C-C0))/abs(C0)),
        final_total_heat_absolute_error=float(abs(qsum[-1]-target[-1])),
        final_total_heat_relative_error=float(abs(qsum[-1]-target[-1])/abs(target[-1])) if target[-1] else None,
        maximum_cell_heat_decrease=float(max(0,-np.min(np.diff(Q,axis=0)))),minimum_accumulated_heat_cell=float(Q.min()))
