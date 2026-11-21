"""Explicit common-phase invariance without assuming a spatial mesh."""
import numpy as np

def common_phase_features(z,phase_indices):
    z=np.asarray(z);indices=tuple(phase_indices)
    if len(indices)<2 or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=z.shape[-1]:raise ValueError("Supply at least two valid distinct phase indices")
    others=[i for i in range(z.shape[-1]) if i not in indices];relative=z[...,indices[1:]]-z[...,indices[0:1]]
    return np.concatenate([z[...,others],np.sin(relative),np.cos(relative)],axis=-1)
