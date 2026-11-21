"""The physical model is supplied by the caller, independently of neural mobility."""
from abc import ABC,abstractmethod
import numpy as np

class DeterministicBackbone(ABC):
    """A time-independent closed-system energy and its conservative vector field.

    Use physical/canonical coordinates, not an implicit logarithmic chart.
    User-supplied state-dependent Poisson matrices need their own Jacobi check;
    skew symmetry and conservation audits alone do not prove Poisson structure.
    """
    @abstractmethod
    def energy(self,z):...
    @abstractmethod
    def gradient(self,z):...
    def poisson(self,z):raise NotImplementedError("Supply poisson(z) or conservative_rhs(z)")
    def conservative_rhs(self,z):return np.asarray(self.poisson(z))@np.asarray(self.gradient(z))
    def invariants(self,z):return np.empty(0)
    def invariant_gradients(self,z):return np.empty((0,len(z)))
