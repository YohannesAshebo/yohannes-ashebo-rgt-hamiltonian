"""Selected positive physical coordinates; no change to canonical equations."""
import numpy as np

class PositiveLogChart:
    def __init__(self,positive_indices=(),max_abs_log=600.):
        self.positive_indices=tuple(positive_indices);self.max_abs_log=float(max_abs_log)
        if len(set(self.positive_indices))!=len(self.positive_indices) or any(not isinstance(i,int) or i<0 for i in self.positive_indices):raise ValueError("positive_indices must be unique nonnegative integers")
        if not np.isfinite(max_abs_log) or max_abs_log<=0 or max_abs_log>700:raise ValueError("max_abs_log must be in (0,700]")
    def _array(self,z):
        z=np.asarray(z,dtype=np.float64)
        if z.ndim not in (1,2) or not np.isfinite(z).all():raise ValueError("Expected finite state (d,) or (batch,d)")
        if self.positive_indices and max(self.positive_indices)>=z.shape[-1]:raise ValueError("A positive-coordinate index is outside the state")
        return z
    def encode(self,z):
        z=self._array(z);out=z.copy()
        if self.positive_indices:
            values=out[...,self.positive_indices]
            if np.any(values<=0):raise ValueError("Selected physical contents must be strictly positive")
            out[...,self.positive_indices]=np.log(values)
        return out
    def decode(self,y):
        y=self._array(y);out=y.copy()
        if self.positive_indices:
            values=y[...,self.positive_indices]
            if np.any(abs(values)>self.max_abs_log):raise FloatingPointError("Positive-coordinate chart budget exceeded; no clipping applied")
            out[...,self.positive_indices]=np.exp(values)
        return out
    def chart_rhs(self,z,physical_rhs):
        z=self._array(z);rhs=np.asarray(physical_rhs,float).copy()
        if rhs.shape!=z.shape or not np.isfinite(rhs).all():raise ValueError("Physical RHS layout/values are invalid")
        if self.positive_indices:rhs[...,self.positive_indices]/=z[...,self.positive_indices]
        return rhs

class IdentityChart(PositiveLogChart):
    def __init__(self):super().__init__(())
