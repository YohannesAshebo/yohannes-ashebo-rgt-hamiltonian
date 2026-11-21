"""Optional canonical adapter connecting the flagship dataset to Ashebo Hybrid."""
import numpy as np
try:from ashebo_hybrid import DeterministicBackbone,HybridSystem,PositiveLogChart
except ImportError as error:raise ImportError("Install the separate ai_benchmark/framework package for this bridge") from error

class CounterflowBackbone(DeterministicBackbone):
    def __init__(self,native):self.native=native;self.n=native.cells;self.last=None;self.labels=None
    def _labels(self,z):
        z=np.asarray(z,float)
        if z.shape!=(8*self.n,):raise ValueError("Expected flattened cell-major canonical state")
        if self.last is None or not np.array_equal(z,self.last):
            y=z.reshape(self.n,8).copy()
            if np.any(y[:,:2]<=0):raise ValueError("Canonical contents must be positive")
            y[:,:2]=np.log(y[:,:2]);self.labels=self.native.evaluate(y,kappa=0);self.last=z.copy()
        return self.labels
    def energy(self,z):return self._labels(z)["H"]
    def gradient(self,z):return self._labels(z)["grad_H_x"].ravel()
    def conservative_rhs(self,z):
        g=self._labels(z)["grad_H_x"]
        return np.stack([g[:,2],g[:,3],-g[:,0],-g[:,1],g[:,6],g[:,7],-g[:,4],-g[:,5]],-1).ravel()
    def invariants(self,z):return np.array([np.asarray(z).reshape(self.n,8)[:,:2].sum()])
    def invariant_gradients(self,z):
        A=np.zeros((1,self.n,8));A[:,:,:2]=1;return A.reshape(1,-1)

def create_system(native,surrogate=None):
    """Mechanical canonical x is flattened cell-major; global heat is auxiliary.

    Native.evaluate may compute diagnostics, but the neural input contains only
    the state and exact native phase torque, never true mobility or currents.
    """
    from .surrogate import load_pretrained
    model=surrogate or load_pretrained();backbone=CounterflowBackbone(native);n=native.cells
    G=np.zeros((8*n,n));cells=np.arange(n);G[8*cells+2,cells]=1;G[8*cells+3,cells]=-1
    def mobility(z):
        labels=backbone._labels(z);D=labels["grad_H_x"][:,2]-labels["grad_H_x"][:,3]
        return model.predict_mobility(labels["state_y"][None],D[None])[0]
    indices=tuple(int(i) for i in (8*cells[:,None]+np.array([0,1])).ravel())
    return HybridSystem(backbone,mobility,directions=G,chart=PositiveLogChart(indices),project_invariants=False)
