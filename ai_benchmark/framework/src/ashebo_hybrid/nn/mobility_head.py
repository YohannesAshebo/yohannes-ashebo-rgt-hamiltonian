"""Learn nonnegative channel coefficients, then factor the PSD mechanical mobility."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .backbones import DCTBackbone

class DissipativeMobilityNet(nn.Module):
    def __init__(self,state_dim,n_directions=None,width=16,modes=16,layers=2,backend="mlp",feature_map=None,feature_dim=None,feature_shape=None,local_mobility=False,gate=None):
        super().__init__();self.state_dim=state_dim;self.n_directions=n_directions or state_dim;self.backend=backend;self.feature_map=feature_map;self.feature_dim=feature_dim or state_dim;self.feature_shape=feature_shape;self.local_mobility=local_mobility;self.gate=gate
        if state_dim<1 or self.n_directions<1 or width<1 or layers<1:raise ValueError("Positive dimensions are required")
        if backend=="mlp":
            blocks=[nn.Linear(self.feature_dim,width),nn.GELU()]
            for _ in range(layers-1):blocks.extend([nn.Linear(width,width),nn.GELU()])
            blocks.append(nn.Linear(width,self.n_directions));self.net=nn.Sequential(*blocks)
        elif backend=="dct":
            if not feature_shape or len(feature_shape)!=2 or np.prod(feature_shape)!=self.feature_dim:raise ValueError("DCT requires feature_shape=(channels,cells) matching feature_dim")
            if local_mobility and self.n_directions%feature_shape[1]:raise ValueError("Local DCT coefficients require rank divisible by cell count")
            outputs=self.n_directions//feature_shape[1] if local_mobility else self.n_directions
            self.net=DCTBackbone(feature_shape[0],outputs,width,modes,layers)
        else:raise ValueError("backend must be mlp or dct")
        self.double()
    def forward(self,z):
        if z.ndim!=2 or z.shape[-1]!=self.state_dim:raise ValueError("Expected (batch,state_dim) physical states")
        feature=self.feature_map(z) if self.feature_map is not None else z
        if self.backend=="dct":
            value=self.net(feature.reshape(z.shape[0],*self.feature_shape));raw=value.reshape(z.shape[0],-1) if self.local_mobility else value.mean(-1)
        else:raw=self.net(feature)
        gamma=F.softplus(raw)
        if self.gate is not None:
            gate=self.gate(z)
            if torch.any(gate<0) or not torch.isfinite(gate).all():raise ValueError("Mobility gate must be finite and nonnegative")
            gamma=gamma*gate
        return gamma
    def numpy_mobility(self,z):
        parameter=next(self.parameters())
        with torch.no_grad():value=self(torch.tensor(np.asarray(z)[None],device=parameter.device,dtype=parameter.dtype))[0]
        return value.cpu().numpy()
