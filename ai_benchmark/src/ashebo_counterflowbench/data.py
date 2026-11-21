"""Versioned, read-only reference trajectories and seed-group splits."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import numpy as np

RESOURCES=Path(__file__).resolve().parent/"resources"
STATE_ORDER=("log_N_C","log_N_R","theta_C","theta_R","alpha","beta","P_alpha","P_beta")
Q_INDICES=(0,1,4,5)
P_INDICES=(2,3,6,7)

def dataset_index():
    """Return a fresh index; editing it cannot change the packaged split."""
    return json.loads((RESOURCES/"dataset/index.json").read_text())

def case_ids(split=None):
    if split not in (None,"train","validation","test"):raise ValueError("split must be train, validation, or test")
    return tuple(c["id"] for c in dataset_index()["cases"] if split is None or c["split"]==split)

@dataclass(frozen=True)
class Trajectory:
    """Array layout (time, cell, field); extensive canonical contents are derived."""
    id: str
    split: str
    arrays: Mapping[str,np.ndarray]

    def __getitem__(self,key):return self.arrays[key]
    def __reduce__(self):return (_restore_trajectory,(self.id,self.split,dict(self.arrays)))
    @property
    def kappa(self):return float(self.arrays["kappa"])
    @property
    def seed_group(self):return int(self.arrays["seed_group"])
    @property
    def time(self):return self.arrays["time"]
    @property
    def state_y(self):return self.arrays["state_y"]
    @property
    def state_x(self):
        x=self.state_y.copy();x[...,:2]=np.exp(x[...,:2]);return x
    @property
    def gradient_y(self):
        g=self.arrays["grad_H_x"].copy();g[...,:2]*=np.exp(self.state_y[...,:2]);return g
    @property
    def rhs_y(self):return self.arrays["rhs_conservative_y"]+self.kappa*self.arrays["rhs_dissipative_unit_y"]
    @property
    def heat_power(self):return self.kappa*self.arrays["heat_cells_unit"]
    @property
    def accumulated_heat(self):return self.arrays["heat_ledgers"][:,:4].sum(axis=1)
    @property
    def torque(self):return self.arrays["grad_H_x"][...,2]-self.arrays["grad_H_x"][...,3]

def _restore_trajectory(identifier,split,arrays):
    for value in arrays.values():value.flags.writeable=False
    return Trajectory(identifier,split,MappingProxyType(arrays))

def load_case(case_id="seed12_k100",verify=True):
    """Load a named, hash-checked case without implicit reshuffling or casting."""
    row=next((r for r in dataset_index()["cases"] if r["id"]==case_id),None)
    if row is None:raise KeyError(f"Unknown benchmark case {case_id!r}")
    path=RESOURCES/"dataset"/row["file"]
    if verify and hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError(f"Dataset checksum failed: {case_id}")
    with np.load(path,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    t,n,c=arrays["state_y"].shape
    if (n,c)!=(128,8) or len(arrays["time"])!=t:raise ValueError("Unexpected state layout")
    if arrays["grad_H_x"].shape!=(t,n,c) or arrays["heat_cells_unit"].shape!=(t,n):raise ValueError("Unexpected label layout")
    if not np.all(np.diff(arrays["time"])>0):raise ValueError("Reference times must strictly increase")
    for value in arrays.values():
        if not np.isfinite(value).all():raise ValueError("Reference contains nonfinite values")
        value.flags.writeable=False
    return Trajectory(row["id"],row["split"],MappingProxyType(arrays))

def load_split(split):return tuple(load_case(i) for i in case_ids(split))

def verify_dataset():
    index=dataset_index();groups={s:set() for s in ("train","validation","test")};counts={s:0 for s in groups};snapshots={s:0 for s in groups}
    for row in index["cases"]:
        case=load_case(row["id"]);groups[case.split].add(case.seed_group);counts[case.split]+=1;snapshots[case.split]+=len(case.time)
    if any(groups[a]&groups[b] for a,b in [("train","validation"),("train","test"),("validation","test")]):raise ValueError("An initial-state group leaks between splits")
    return dict(verified=True,trajectories=len(index["cases"]),counts=counts,saved_states=snapshots,groups={s:sorted(g) for s,g in groups.items()})

def verify_solver(solver_directory):
    """Compare all 94 pre-existing public files; AI additions are outside that set."""
    root=Path(solver_directory).resolve();frozen=json.loads((RESOURCES/"BASELINE_SOURCES.json").read_text());changed=[];missing=[]
    for name,expected in frozen["files"].items():
        path=root/name
        if not path.is_file():missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:changed.append(name)
    return dict(verified=not changed and not missing,baseline_commit=frozen["base_commit"],files_checked=len(frozen["files"]),changed=changed,missing=missing)
