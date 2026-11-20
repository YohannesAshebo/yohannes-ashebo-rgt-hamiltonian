"""Project precision wrapper around SciPy DOP853; not a new RK algorithm."""
import numpy as np
from scipy.integrate._ivp.rk import DOP853

class ExtendedDOP853(DOP853):
    """Use SciPy's same tableau/controller with persistent extended stages.

    SciPy's normal argument wrapper casts real arrays to float64. Explicitly
    replace that wrapper and stage storage after the standard initialization.
    Tableau coefficients and time/controller arithmetic retain their supplied
    precision; this makes no claim to a full 64-bit-significand integrator.
    """
    def __init__(self,fun,t0,y0,t_bound,**kw):
        original=np.asarray(y0,dtype=np.longdouble).copy()
        super().__init__(fun,t0,np.asarray(y0,dtype=float),t_bound,**kw)
        self.y=original
        def wrapped(t,y):return np.asarray(fun(t,y),dtype=np.longdouble)
        self._fun=wrapped;self.fun_single=wrapped
        self.f=self.fun(self.t,self.y)
        self.K_extended=np.empty_like(self.K_extended,dtype=np.longdouble)
        self.K=self.K_extended[:self.n_stages+1]
