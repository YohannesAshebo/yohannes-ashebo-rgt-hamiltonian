"""Evaluate energy on predictions; report finite-step defects and refinement."""
import numpy as np

def gradient_check(backbone,z,method="complex_step",epsilon=1e-30):
    z=np.asarray(z,float);declared=np.asarray(backbone.gradient(z),float)
    if method=="complex_step":
        numeric=np.empty_like(z)
        for i in range(len(z)):
            perturbed=z.astype(complex);perturbed[i]+=1j*epsilon;numeric[i]=np.imag(backbone.energy(perturbed))/epsilon
    elif method=="autograd":
        import torch
        if not hasattr(backbone,"torch_energy"):raise ValueError("Autograd checking requires backbone.torch_energy(z)")
        x=torch.tensor(z,dtype=torch.float64,requires_grad=True);numeric=torch.autograd.grad(backbone.torch_energy(x),x)[0].detach().numpy()
    else:raise ValueError("Gradient method must be complex_step or autograd")
    return dict(method=method,max_absolute_error=float(np.max(abs(numeric-declared))),max_scaled_error=float(np.max(abs(numeric-declared)/np.maximum(1,abs(declared)))))

class AuditGates:
    def __init__(self,energy_tolerance=1e-8,invariant_tolerance=1e-8,power_tolerance=1e-10,refinement_tolerance=1e-6,gradient_method=None,gradient_tolerance=1e-8,require_active_refinement=True):
        self.energy_tolerance=energy_tolerance;self.invariant_tolerance=invariant_tolerance;self.power_tolerance=power_tolerance;self.refinement_tolerance=refinement_tolerance
        self.gradient_method=gradient_method;self.gradient_tolerance=gradient_tolerance;self.require_active_refinement=require_active_refinement
    def verify(self,model,trajectory,step_halving=True):
        tr=trajectory;states=tr.states[:,None] if tr.states.ndim==2 else tr.states;heat=tr.heat[:,None] if tr.heat.ndim==1 else tr.heat
        values=[[model.evaluate(z,tr.kappa) for z in batch] for batch in states]
        H=np.array([[v["H"] for v in row] for row in values]);C=np.array([[v["invariants"] for v in row] for row in values]);energy=float(np.max(abs(H+heat-H[:1]-heat[:1])/np.maximum(1,abs(H[:1]))))
        inv=float(np.max(abs(C-C[:1])/np.maximum(1,abs(C[:1])),initial=0))
        power=max(abs(float(v["gradient"]@v["rhs"])+v["heat_rate"]) for row in values for v in row)
        reversible=max(abs(float(v["gradient"]@v["conservative"])) for row in values for v in row)
        charge=max(float(np.max(abs(v["invariant_gradients"]@v["rhs"]),initial=0)) for row in values for v in row)
        minimum_heat=min(v["heat_rate"] for row in values for v in row);heat_decrease=float(max(0,-np.min(np.diff(heat,axis=0)))) if len(heat)>1 else 0.
        gradient=None
        if self.gradient_method:gradient=gradient_check(model.backbone,states[0,0],self.gradient_method)
        report=dict(completed=tr.completed,failure=tr.failure,energy_heat_relative_drift=energy,invariant_scaled_drift=inv,
            instantaneous_power_residual_max_abs=power,conservative_power_residual_max_abs=reversible,instantaneous_invariant_rate_max_abs=charge,
            minimum_heat_rate=minimum_heat,maximum_accumulated_heat_decrease=heat_decrease,gradient_check=gradient,step_halving=None)
        passed=tr.completed and energy<=self.energy_tolerance and inv<=self.invariant_tolerance and max(power,reversible,charge)<=self.power_tolerance and minimum_heat>=0 and heat_decrease<=self.energy_tolerance
        if gradient is not None:passed=passed and gradient["max_scaled_error"]<=self.gradient_tolerance
        if step_halving and tr.completed:
            fine=tr.integrator.refined().rollout(model,tr.states[0],kappa=tr.kappa,times=tr.requested_times)
            complete=fine.completed and fine.states.shape==tr.states.shape
            difference=float(np.max(abs(tr.states-fine.states)/np.maximum(1,abs(fine.states)))) if complete else None
            heat_difference=float(np.max(abs(tr.heat-fine.heat)/np.maximum(1,abs(fine.heat)))) if complete else None
            caps_active=all(x.stats.get("largest_step",0)>=.9*x.stats["step_cap"] for x in (tr,fine))
            qualified=complete and difference<=self.refinement_tolerance and heat_difference<=self.refinement_tolerance and (caps_active or not self.require_active_refinement)
            report["step_halving"]=dict(completed=complete,max_scaled_state_difference=difference,max_scaled_heat_difference=heat_difference,step_caps_active=caps_active,qualified=qualified,
                interpretation="Two time levels give a refinement indicator, not an empirical order or spatial convergence certificate.")
            passed=passed and qualified
        report["passed"]=bool(passed);return report
