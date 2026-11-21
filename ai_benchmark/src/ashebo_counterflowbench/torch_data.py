"""PyTorch datasets that preserve whole-initial-state train/test grouping."""
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError as error:
    raise ImportError("Install ashebo-counterflowbench[torch] for PyTorch datasets") from error
from .data import load_split

class OperatorDataset(Dataset):
    """One labeled spatial snapshot per item; channel-first float64 tensors."""
    def __init__(self,split="train"):
        self.split=split;self.cases=load_split(split)
        self.indices=[(i,j) for i,c in enumerate(self.cases) for j in range(len(c.time))]
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i,j=self.indices[index];c=self.cases[i]
        y=c.state_y[j];x=y.copy();x[:,:2]=np.exp(x[:,:2]);rhs=c["rhs_conservative_y"][j]+c.kappa*c["rhs_dissipative_unit_y"][j]
        return dict(state_y=torch.tensor(y.T.copy()),state_x=torch.tensor(x.T.copy()),
            rhs_y=torch.tensor(rhs.T.copy()),conservative_rhs_y=torch.tensor(c["rhs_conservative_y"][j].T.copy()),
            unit_dissipative_rhs_y=torch.tensor(c["rhs_dissipative_unit_y"][j].T.copy()),
            grad_H_x=torch.tensor(c["grad_H_x"][j].T.copy()),heat_power=torch.tensor(c.kappa*c["heat_cells_unit"][j].copy()),
            unit_heat_power=torch.tensor(c["heat_cells_unit"][j].copy()),
            kappa=torch.tensor(c.kappa,dtype=torch.float64),time=torch.tensor(float(c.time[j]),dtype=torch.float64),
            reference_volume=torch.tensor(c["reference_volume"].copy()),radius=torch.tensor(c["r"].copy()),seed_group=c.seed_group,case_id=c.id)

class WindowDataset(Dataset):
    """Consecutive reference frames; windows never cross trajectory boundaries."""
    def __init__(self,split="train",frames=11):
        if not isinstance(frames,int) or frames<2 or frames>21:raise ValueError("frames must be an integer in [2,21]")
        self.cases=load_split(split);self.frames=frames
        self.indices=[(i,j) for i,c in enumerate(self.cases) for j in range(len(c.time)-frames+1)]
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i,j=self.indices[index];c=self.cases[i];sl=slice(j,j+self.frames)
        return dict(state_y=torch.tensor(c.state_y[sl].transpose(0,2,1).copy()),time=torch.tensor(c.time[sl].copy()),
            accumulated_heat=torch.tensor(c.accumulated_heat[sl].copy()),kappa=torch.tensor(c.kappa,dtype=torch.float64),seed_group=c.seed_group,case_id=c.id)
