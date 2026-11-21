"""Optional PyTorch model: exact-H-assisted positive phase/heat response."""
import hashlib
import json
import numpy as np
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:
    raise ImportError("Install ashebo-counterflowbench[torch] to use the neural model") from error
from .data import RESOURCES
from ._geometry import RadialFeatures,DCTBackbone,canonical_from_log

class PositiveMobility(nn.Module):
    """Architecture and state-dict keys match the frozen validated checkpoint."""
    def __init__(self,norm,cfg):
        super().__init__()
        for key,value in norm.items():self.register_buffer(key,torch.tensor(value,dtype=torch.float64))
        self.features=RadialFeatures(norm["volume"],norm["radius"],True)
        self.net=DCTBackbone(12,1,width=cfg["width"],modes=cfg["modes"],layers=cfg["layers"])
    def forward(self,y,D):
        feature=torch.cat([self.features(canonical_from_log(y)),D[:,None]],1)
        c=self.c_base*F.softplus(self.net((feature-self.mean)/self.scale)[:,0])
        return torch.stack([-c*D,c*D,c*D*D],1),c

class PretrainedSurrogate:
    def __init__(self,model):self.model=model
    def _predict(self,state_y,torque):
        """Return (time,cell,3) unit-kappa phase_C, phase_R and heat power.

        Inputs use the dataset's (time,cell,8) log-content chart. Torque must
        be supplied by the exact native Hamiltonian, or its supplied labels.
        """
        y=np.asarray(state_y,dtype=np.float64);D=np.asarray(torque,dtype=np.float64)
        if y.ndim!=3 or y.shape[1:]!=(128,8) or D.shape!=y.shape[:2]:raise ValueError("Expected state_y (T,128,8), torque (T,128)")
        if not np.isfinite(y).all() or not np.isfinite(D).all():raise ValueError("Inputs must be finite")
        device=next(self.model.parameters()).device
        with torch.no_grad():
            unit,c=self.model(torch.tensor(y.transpose(0,2,1).copy(),device=device),torch.tensor(D.copy(),device=device))
        prediction=unit.cpu().numpy().transpose(0,2,1)
        if not np.isfinite(prediction).all():raise FloatingPointError("Nonfinite neural prediction")
        return prediction,c.cpu().numpy()
    def predict_unit(self,state_y,torque):return self._predict(state_y,torque)[0]
    def predict_mobility(self,state_y,torque):return self._predict(state_y,torque)[1]
    def predict_operator(self,case):
        """Known-H task: native conservative rates and torque are supplied."""
        unit=self.predict_unit(case.state_y,case.torque);rhs=case["rhs_conservative_y"].copy()
        rhs[...,2:4]+=case.kappa*unit[...,:2]
        return dict(rhs_y=rhs,heat_cells=case.kappa*unit[...,2],unit_phase_heat=unit)

def load_pretrained(device="cpu"):
    folder=RESOURCES/"pretrained";freeze=json.loads((folder/"HYBRID_FREEZE.json").read_text())
    checks={"native_H_hybrid.pt":"model_sha256","HYBRID_NORMALIZERS.npz":"normalizers_sha256","HYBRID_PROTOCOL.json":"protocol_sha256"}
    for name,key in checks.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=freeze[key]:raise ValueError(f"Frozen model integrity check failed: {name}")
    with np.load(folder/"HYBRID_NORMALIZERS.npz",allow_pickle=False) as z:norm={k:z[k] for k in z.files}
    cfg=json.loads((folder/"HYBRID_PROTOCOL.json").read_text())
    # Module construction consumes random numbers; preserve the caller's RNG.
    with torch.random.fork_rng(devices=[]),torch.device("cpu"):model=PositiveMobility(norm,cfg["network"]).double()
    model.load_state_dict(torch.load(folder/"native_H_hybrid.pt",map_location="cpu",weights_only=True));model.to(device);model.eval()
    return PretrainedSurrogate(model)
