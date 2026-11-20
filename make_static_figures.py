#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Plot the manufactured-profile scaling and conserved radial comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/"results/static_scaling"
d=json.loads((root/"static_scaling.json").read_text())
fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
names={"native_grid_preparation":"Native grid preparation",
       "continuum_native_caps":"Fixed continuum native caps",
       "boundary_regular_cap_control":"Boundary-regular cap control"}
for f in d["families"]:
    rows=f["rows"]
    ax[0].plot([r["dx"] for r in rows],[r["heat_power"] for r in rows],"o-",label=names[f["family"]])
    fits=f["richardson_triads"]
    ax[1].plot([t["grids"][-1] for t in fits],[t["p"] for t in fits],"o-",label=names[f["family"]])
ax[0].set(title="Manufactured static heat power",xlabel="Cell width (native units)",ylabel="Integrated heat power (native)")
ax[0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
ax[0].legend(fontsize=8,frameon=False)
ax[1].axhline(1,color="black",linestyle=":",linewidth=1,label="p=1")
ax[1].axhline(2,color="gray",linestyle="--",linewidth=1,label="p=2")
ax[1].set(title="Three-grid observed order",xlabel="Finest grid in each triple",ylabel="Conditional order p",ylim=(.8,2.1))
a=np.genfromtxt(root/"saved_s90_coarsened_difference.csv",delimiter=",",names=True)
dr=8/256
ax[2].plot(a["r"],a["power_256"]/dr,label="N=256",linewidth=1.3)
ax[2].plot(a["r"],a["power_512_binned"]/dr,label="N=512, binned to parent cells",linewidth=1.1)
core=d["saved_checkpoint_radial_comparison"]["core_band"]
ax[2].axvspan(core["reference_coordinate_min"],core["reference_coordinate_max"],color="gray",alpha=.17,label="Core-boundary comparison band")
ax[2].set(title="Archived s=90 radial heat",xlabel="Reference coordinate radius",ylabel="dQ/ds per reference dr",xlim=(0,3.5))
ax[2].legend(fontsize=8,frameon=False)
for a in ax:a.grid(True,alpha=.2)
fig.suptitle("Ashebo RGT: static operator checks and the historical heat jump",fontsize=13)
fig.savefig(root/"static_scaling_and_radial.png",dpi=180)
fig.savefig(root/"static_scaling_and_radial.pdf")
