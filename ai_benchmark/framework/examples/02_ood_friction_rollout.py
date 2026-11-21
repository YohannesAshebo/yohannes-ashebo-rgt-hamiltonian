"""Supervised positive mobility and an unseen kappa using known factorization."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ashebo_hybrid import DeterministicBackbone,HybridSystem,AdaptiveDOP853,AuditGates,freeze_artifacts,verify_freeze
from ashebo_hybrid.nn import DissipativeMobilityNet

class Oscillator(DeterministicBackbone):
    def energy(self,z):return np.dot(z,z)/2
    def gradient(self,z):return z
    def poisson(self,z):return np.array([[0.,1.],[-1.,0.]])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(20261120);rng=np.random.default_rng(20261120)
    training=torch.tensor(rng.uniform(-2,2,(256,2)));validation=torch.tensor(rng.uniform(-2,2,(64,2)))
    target=lambda z:.1+.05*z[:,0:1]**2/(1+z[:,0:1]**2)
    model=DissipativeMobilityNet(2,n_directions=1,width=16,layers=2);opt=torch.optim.Adam(model.parameters(),lr=.005);best=float("inf");weights=None
    for step in range(600):
        opt.zero_grad(set_to_none=True);loss=((model(training)-target(training))**2).mean();loss.backward();opt.step()
        if step%20==0:
            with torch.no_grad():score=float(((model(validation)-target(validation))**2).mean())
            if score<best:best=score;weights=copy.deepcopy(model.state_dict())
    model.load_state_dict(weights);model.eval();checkpoint=args.output/"mobility.pt";torch.save(model.state_dict(),checkpoint)
    protocol=dict(seed=20261120,task="learn unit-kappa mobility",training_states=256,validation_states=64,heldout_kappa=500,
        known_kappa_factorization=True,scope="An API demonstration on a synthetic oscillator, not learned zero-shot recovery of kappa dependence")
    freeze_artifacts({"weights":checkpoint},protocol,args.output/"freeze.json")
    system=HybridSystem(Oscillator(),model,directions=np.array([[0.],[1.]]));times=np.linspace(0,2,21)
    trajectory=AdaptiveDOP853(max_step=.01,rtol=1e-10,atol=1e-12).rollout(system,np.array([1.,0.]),kappa=500,times=times)
    report=AuditGates(gradient_method="complex_step").verify(system,trajectory);report["freeze"]=verify_freeze(args.output/"freeze.json");report["kappa"]=500
    np.savez_compressed(args.output/"trajectory.npz",time=trajectory.time,state=trajectory.states,heat=trajectory.heat)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    if not report["passed"] or not report["freeze"]["verified"]:raise SystemExit(1)

if __name__=="__main__":main()
