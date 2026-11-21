import torch
from torch import nn

class PhaseDifferenceFeatures(nn.Module):
    """Drop common phase; retain non-phase fields and periodic relative phases."""
    def __init__(self,state_dim,phase_indices):
        super().__init__();self.phase_indices=tuple(phase_indices)
        if len(self.phase_indices)<2 or len(set(self.phase_indices))!=len(self.phase_indices) or min(self.phase_indices)<0 or max(self.phase_indices)>=state_dim:raise ValueError("Invalid phase indices")
        self.other_indices=tuple(i for i in range(state_dim) if i not in self.phase_indices);self.output_dim=len(self.other_indices)+2*(len(self.phase_indices)-1)
    def forward(self,z):
        delta=z[...,self.phase_indices[1:]]-z[...,self.phase_indices[0:1]]
        return torch.cat([z[...,self.other_indices],delta.sin(),delta.cos()],dim=-1)
