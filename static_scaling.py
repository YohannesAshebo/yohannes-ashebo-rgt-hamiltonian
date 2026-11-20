#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Zero-integration manufactured-profile scaling and saved-state radial diagnosis.

No evolved N=1024 or N=2048 state is created. All static profiles are explicitly
manufactured. No Hamiltonian term, kappa factor, or archived field state changes.
Three cap/preparation families distinguish per-grid native preparation from
fixed continuum parameter fields and a boundary-regular coefficient control.
"""
from dataclasses import replace
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from analyze_convergence import richardson
from audit_mechanics import scalar_energy

ROOT=Path(__file__).resolve().parent
NATIVE=ROOT/"native"
for path in (NATIVE,NATIVE/"study",NATIVE/"speed_order",NATIVE/"speed_sweep"):
    sys.path.append(str(path))

from run import prepare
from autonomous import CurrentExchange
from autonomous_heat import fields
from strength_match.match_model import build as build_matched

LD=np.longdouble
GRIDS=(128,256,512,1024,2048)
OMEGA=LD("0.04080087121508541")
J0=LD("4.60963676251496e-16")
KAPPA=LD(100)


def sigmoid(z):
    e=np.exp(-abs(z))
    return np.where(z>=0,1/(1+e),e/(1+e))


def smooth_test_fields(r,R=8):
    """The user's test profile, without silently altering its boundary values."""
    x=r/R
    nc=LD(".80")*np.exp(-(r/LD("1.4"))**2)+LD(".02")
    nr=1-LD(".45")*np.exp(-(r/LD("1.8"))**2)
    tc=LD(".35")*np.cos(LD(np.pi)*x)*np.exp(-(r/LD("2.5"))**2)
    tr=-LD(".20")*(1-np.sin(LD(np.pi)*x)**2)
    return nc,tc,nr,tr


def original_continuum_phase_derivative(r):
    # Native launch phase, including its C2 taper ending at coordinate r=4.
    t=np.clip((LD(4)-r)/LD(".75"),0,1)
    taper=t**3*(10-15*t+6*t*t)
    return LD(".15")*r/np.sqrt(r*r+LD(".25")**2)*taper


def fixed_continuum_caps(model,family):
    faces=np.asarray(model.faces[1:-1],dtype=LD)
    p0=LD(model.p.eta)*LD(".15")/np.sqrt(1+LD(".25")**2)
    if family=="continuum_native_caps":
        derivative=original_continuum_phase_derivative(faces)
    elif family=="boundary_regular_cap_control":
        # A separate prescribed, smooth parameter field. Unlike the native
        # compact r=4 taper, this has no internal zero-current cutoff.
        x=faces/LD(model.p.domain_radius)
        derivative=LD(".15")*faces/np.sqrt(faces*faces+LD(".25")**2)*(1-x*x)**2
    else:
        raise ValueError(family)
    common=2*LD(model.p.K)/LD(model.p.eta)*derivative/sigmoid(LD(model.p.eta)*derivative/p0)
    # The balanced native launch's bias=1.5 gives continuum factors 1.2 and .8.
    model.p0=p0
    model.vmax=np.stack([LD("1.2")*common,LD(".8")*common])


def disable_old_exchange(model):
    for target in (model,model.original,model.base):
        target.p=replace(target.p,Omega_p=0.)


def evaluate_model(model,state,omega=OMEGA):
    law=CurrentExchange(model,omega,decreasing=True)
    law.j0=J0
    disable_old_exchange(model)
    d,H,g=fields(model,law,state)
    paired=np.maximum(0,np.minimum(-d["flux"][0]/d["A"],d["flux"][1]/d["A"]))
    jbar=(np.r_[0,paired]+np.r_[paired,0])/2
    D=g[0]-g[1]
    cell_heat=KAPPA*jbar*D**2/d["N"].sum(axis=0)
    # Verify the actual heat-only phase rates against the independent energy
    # derivatives, without differentiating the nonsmooth heat gate.
    extra=law.evaluate(state)
    gradient=model.unpack(model.evaluate(state)["gradient"])+extra["gradient"]
    phase_rate=KAPPA*jbar*D/d["N"].sum(axis=0)
    identity=np.sum(gradient[2]*(-phase_rate)+gradient[3]*phase_rate)+cell_heat.sum()
    assert np.max(abs(gradient[2:4]-g)) < LD("1e-10")
    assert abs(identity)<LD("1e-10")*max(1,abs(cell_heat.sum()))
    assert abs(scalar_energy(model,law,state)-H)<LD("1e-11")*max(1,abs(H))
    return d,H,jbar,D,cell_heat,float(abs(identity))


def _evaluate_with_metadata(N,R_outer,g_exch,kappa,n_C,theta_C,n_R,theta_R,
                            family="continuum_native_caps"):
    """Requested hook: use true extensive contents and the native operator.

    The historical coefficient field is prepared by the native launch rule.
    Fixed-family modes replace ONLY prescribed cap/p0 input values in this
    manufactured-state evaluator, never the Hamiltonian functional.
    """
    if R_outer!=8 or kappa!=100:
        raise ValueError("This declared study fixes outer radius=8 and kappa=100.")
    model,_,_,_,_,_=prepare(N,1.5)
    if family!="native_grid_preparation":
        fixed_continuum_caps(model,family)
    z=np.zeros(N,dtype=LD)
    state=model.pack(np.log(np.asarray(n_C,dtype=LD)*model.V0),
                     np.log(np.asarray(n_R,dtype=LD)*model.V0),
                     theta_C,theta_R,z,z,z,z)
    d,H,jbar,D,q,residual=evaluate_model(model,state,LD(g_exch))
    assert np.allclose(d["V"],model.V0,rtol=0,atol=0)
    return H,q.sum(),q,jbar,D,d["V"],model, residual


def evaluate_audit_operators(N,R_outer,g_exch,kappa,n_C,theta_C,n_R,theta_R,
                             family="continuum_native_caps"):
    """The user's six-output hook, using fixed continuum caps by default."""
    return _evaluate_with_metadata(N,R_outer,g_exch,kappa,n_C,theta_C,n_R,theta_R,family)[:6]


def static_family(family,output):
    rows=[]
    for n in GRIDS:
        dr=LD(8)/n;r=(np.arange(n,dtype=LD)+LD(".5"))*dr
        nc,tc,nr,tr=smooth_test_fields(r)
        H,Q,q,j,D,V,model,residual=_evaluate_with_metadata(n,8,OMEGA,100,nc,tc,nr,tr,family)
        row={"cells":n,"time":None,"time_integrated":False,"state_kind":"manufactured_smooth_profile",
             "outer_radius":8,"dx":float(dr),"active_exchange_strength":float(OMEGA),
             "kappa":100,"j0":float(J0),"p0":float(model.p0),"H":float(H),
             "heat_power":float(Q),"max_abs_jbar":float(np.max(abs(j))),
             "max_abs_D_over_V":float(np.max(abs(D/V))),"heat_identity_abs_error":residual}
        rows.append(row)
        np.savetxt(output/f"{family}_N{n}.csv",np.column_stack([r,V,q,j,D]).astype(float),delimiter=",",
                   header="r,V,heat_power,jbar,D",comments="")
    ratios=[]
    for a,b in zip(rows,rows[1:]):
        ratio=b["heat_power"]/a["heat_power"]
        ratios.append({"coarse":a["cells"],"fine":b["cells"],"ratio":ratio,"alpha":math.log2(ratio)})
    triads=[]
    for i in range(len(rows)-2):
        triads.append(richardson([x["cells"] for x in rows[i:i+3]],[x["heat_power"] for x in rows[i:i+3]]))
    return {"family":family,"rows":rows,"doubling_ratios":ratios,"richardson_triads":triads,
            "continuum_limit_proved":False,"historical_s90_dynamics_explained":False}


def saved_checkpoint(n,output,fixed_omega=None):
    model,_,_,_=build_matched(n,"matched_quadratic")
    omega=LD(model.p.Omega_p) if fixed_omega is None else fixed_omega
    state=np.load(NATIVE/f"data/weak_N{n}_s90.npz")["state"].astype(LD)
    d,H,j,D,q,residual=evaluate_model(model,state,omega)
    rates=model.unpack(d["rhs"])
    d["chi_dot"]=(d["N"][0]*rates[0]-d["N"][1]*rates[1])/(d["V"]*model.p.n_star)-(rates[4]+2*rates[5])*d["chi"]
    core=model.core(state,d)
    record={"cells":n,"time":90,"state_kind":"archived_evolved_checkpoint",
            "active_exchange_strength":float(omega),"heat_power":float(q.sum()),"core":core,
            "heat_identity_abs_error":residual,"fixed_omega_readout_only":fixed_omega is not None}
    if fixed_omega is None:
        np.savetxt(output/f"saved_s90_N{n}.csv",np.column_stack([model.r,d["V"],q,j,D]).astype(float),delimiter=",",
                   header="r,V,heat_power,jbar,D",comments="")
    return model,q,record


def radial_comparison(output):
    m256,q256,r256=saved_checkpoint(256,output)
    m512,q512,r512=saved_checkpoint(512,output)
    assert m256.dr==2*m512.dr
    assert np.max(abs(m256.faces[::1]-m512.faces[::2]))<LD("1e-15")
    fine_binned=q512.reshape(256,2).sum(axis=1)
    difference=fine_binned-q256
    total=float(q512.sum()-q256.sum())
    assert abs(difference.sum()-(q512.sum()-q256.sum()))<LD("1e-17")
    top=[]
    for i in np.argsort(abs(difference))[::-1][:10]:
        top.append({"coarse_cell":int(i),"r":float(m256.r[i]),
                    "coarse_cell_power":float(q256[i]),"fine_binned_cell_power":float(fine_binned[i]),
                    "dQ_dr_256":float(q256[i]/m256.dr),"dQ_dr_512_binned":float(fine_binned[i]/m256.dr),
                    "power_difference":float(difference[i]),"fraction_of_net_jump":float(difference[i]/total)})
    regions=[]
    for lo,hi in ((0,1.75),(1.75,2.5),(2.5,3),(3,8)):
        mask=(m256.r>=lo)&(m256.r<hi)
        a=float(q256[mask].sum());b=float(fine_binned[mask].sum())
        regions.append({"r_min":lo,"r_max":hi,"power_256":a,"power_512":b,
                        "difference":b-a,"fraction_of_net_jump":(b-a)/total})
    roots=[r["core"].get("coordinate_radius") for r in (r256,r512)]
    band=None
    if all(x is not None for x in roots):
        lo=max(0,min(roots)-.25);hi=min(8,max(roots)+.25)
        mask=(m256.r>=lo)&(m256.r<=hi)
        band={"reference_coordinate_min":lo,"reference_coordinate_max":hi,
              "selection":"whole coarse parent cells with centers in the union of the two core +/-0.25 bands",
              "power_256":float(q256[mask].sum()),"power_512":float(fine_binned[mask].sum()),
              "fraction_of_net_jump":float(difference[mask].sum()/total)}
    # This is a readout sensitivity calculation on the SAME old states, not
    # a re-evolution with the new coefficient and not a replacement baseline.
    common=OMEGA
    _,_,fixed256=saved_checkpoint(256,output,common)
    _,_,fixed512=saved_checkpoint(512,output,common)
    np.savetxt(output/"saved_s90_coarsened_difference.csv",
               np.column_stack([m256.r,q256,fine_binned,difference,difference/m256.dr]).astype(float),delimiter=",",
               header="r,power_256,power_512_binned,difference,difference_per_dr",comments="")
    return {"original_records":[r256,r512],"ratio":float(q512.sum()/q256.sum()),"net_jump":total,
            "coordinate":"common reference-coordinate shells, not aligned areal-radius cells",
            "top_shells":top,"regions":regions,"core_band":band,
            "common_omega_readout_records":[fixed256,fixed512],
            "common_omega_readout_ratio":fixed512["heat_power"]/fixed256["heat_power"],
            "high_wavenumber_wave_mechanism_proved":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"results/static_scaling")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    families=[static_family(name,args.output) for name in
              ("native_grid_preparation","continuum_native_caps","boundary_regular_cap_control")]
    radial=radial_comparison(args.output)
    boundary_derivative=LD(".35")*(2*LD(8)/LD("2.5")**2)*np.exp(-(LD(8)/LD("2.5"))**2)
    result={"test_kind":"zero_integration_manufactured_profile_operator_study", "time_integration_performed":False,
            "grids":list(GRIDS),"active_exchange_strength":float(OMEGA),"kappa":100,
            "operator_jbar":"arithmetic mean of paired complete face current PER PROPER AREA, not shell-integrated current",
            "native_canonical_sign":"N_dot = +H_theta/eta; theta_dot = -H_N/eta",
            "provided_theta_C_outer_derivative":float(boundary_derivative),
            "boundary_note":"The supplied theta_C has a small nonzero derivative at r=8. Vanishing phase derivative would not alone impose zero directional current; the cap/boundary law matters.",
            "families":families,"saved_checkpoint_radial_comparison":radial,
            "runtime_seconds":time.perf_counter()-start,
            "no_kappa_grid_rescaling_applied":True,"original_source_and_states_changed":False}
    (args.output/"static_scaling.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for family in families:
        print(family["family"],[(r["cells"],r["heat_power"]) for r in family["rows"]])
        print("p",[t["p"] for t in family["richardson_triads"]])
    print("saved radial regions",radial["regions"])
    print("common core band",radial["core_band"])
    print("runtime_seconds",result["runtime_seconds"])


if __name__=="__main__":main()
