#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Create static scientific figures from completed audit results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/"results/mechanics_audit.json").read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
for r in data["rechecks"]:
    if r["status"]!="completed_checkpoint_recheck":continue
    n=r["cells"];label=f"N={n}"
    sweep=r["centered_gradient_epsilon_sweep"]
    axes[0].loglog([x["epsilon"] for x in sweep],[x["max_absolute_error"] for x in sweep],"o-",label=label)
    hs=r["heat_direction_epsilon_sweep"]
    axes[1].loglog([x["epsilon"] for x in hs],[x["absolute_balance_residual"] for x in hs],"o-",label=label)
    profile=np.genfromtxt(root/f"results/cell_heat_N{n}.csv",delimiter=",",names=True)
    density=profile["heat_power"]/profile["proper_volume"]
    axes[2].semilogy(profile["coordinate_radius"],np.where(density>0,density,np.nan),label=label,linewidth=1)
axes[0].set(title="Centered gradient check",xlabel="Perturbation epsilon",ylabel="Maximum sampled absolute error")
axes[1].set(title="Heat-only directional derivative",xlabel="Perturbation epsilon",ylabel="Absolute power-balance residual")
axes[2].set(title="Archived s=90 heat profiles",xlabel="Coordinate radius (native units)",ylabel="Heat power / proper volume (native)",xlim=(0,4),ylim=(1e-18,1e2))
for ax in axes:
    ax.grid(True,which="major",alpha=.22);ax.legend(frameon=False)
fig.suptitle("Ashebo Hamiltonian audit mechanics — unchanged archived states",fontsize=13)
fig.savefig(root/"results/audit_diagnostics.png",dpi=180)
fig.savefig(root/"results/audit_diagnostics.pdf")
