"""A custom damped Hamiltonian oscillator; no RGT source or ML dependency."""
import json
import numpy as np
from ashebo_hybrid import DeterministicBackbone,HybridSystem,ChartRK4,AuditGates

class Oscillator(DeterministicBackbone):
    def energy(self,z):return np.dot(z,z)/2
    def gradient(self,z):return z
    def poisson(self,z):return np.array([[0.,1.],[-1.,0.]])

def main():
    system=HybridSystem(Oscillator(),mobility=lambda z:np.array([.2]),directions=np.array([[0.],[1.]]))
    trajectory=ChartRK4(dt=.005).rollout(system,np.array([1.,0.]),t_span=(0.,2.))
    report=AuditGates(gradient_method="complex_step").verify(system,trajectory)
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
