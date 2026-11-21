"""Compose exact conservative mechanics with constrained unknown dissipation."""
import numpy as np
from .charts import IdentityChart
from .dissipation import tangent_directions,factor_rates

class HybridSystem:
    def __init__(self,backbone,mobility,directions=None,chart=None,project_invariants=True):
        self.backbone=backbone;self.mobility=mobility;self.directions=directions;self.chart=chart or IdentityChart();self.project_invariants=project_invariants
    def evaluate(self,z,kappa=1.):
        z=np.asarray(z,float)
        if z.ndim!=1 or not np.isfinite(z).all():raise ValueError("Expected a finite physical state vector")
        b=self.backbone;gradient=np.asarray(b.gradient(z),float);conservative=np.asarray(b.conservative_rhs(z),float)
        if gradient.shape!=z.shape or conservative.shape!=z.shape:raise ValueError("Backbone gradient/RHS must match the physical state")
        H=float(b.energy(z));C=np.atleast_1d(np.asarray(b.invariants(z),float));A=np.asarray(b.invariant_gradients(z),float)
        if C.shape!=(len(C),) or A.shape!=(len(C),len(z)):raise ValueError("Supply one gradient row per declared invariant")
        raw=self.directions(z) if callable(self.directions) else self.directions
        if raw is None:raw=np.eye(len(z))
        G=tangent_directions(raw,A) if self.project_invariants else np.asarray(raw,float)
        if kappa==0:gamma=np.zeros(G.shape[1])
        else:gamma=self.mobility.numpy_mobility(z) if hasattr(self.mobility,"numpy_mobility") else self.mobility(z)
        diss,heat,torque=factor_rates(gradient,G,gamma,kappa)
        if not np.isfinite(H) or not np.isfinite(C).all() or not np.isfinite(conservative).all():raise FloatingPointError("Nonfinite deterministic backbone")
        return dict(rhs=conservative+diss,conservative=conservative,dissipative=diss,heat_rate=heat,H=H,gradient=gradient,invariants=C,invariant_gradients=A,directions=G,mobility=np.asarray(gamma,float),torque=torque)
    def chart_rhs(self,state,kappa=1.):
        y=np.asarray(state,float);z=self.chart.decode(y[...,:-1]);batch=z[None] if z.ndim==1 else z
        values=[self.evaluate(row,kappa) for row in batch];dz=np.stack([v["rhs"] for v in values]);heat=np.array([v["heat_rate"] for v in values])
        dy=self.chart.chart_rhs(batch,dz);out=np.concatenate([dy,heat[:,None]],axis=-1)
        return out[0] if y.ndim==1 else out
