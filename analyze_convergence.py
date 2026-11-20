#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Guarded Richardson arithmetic; verification residuals are not spatial observables.

Three values supply an empirical order only under a shared asymptotic error law.
This program does not identify a physical mechanism from a third value.
It also does not infer an optimal finite-difference step from grid-level errors.
"""
import argparse
import json
import math
from pathlib import Path
import sys

METRICS={
    "active_exchange_strength":"grid_dependent_calibration",
    "kappa_100_heat_power":"physical_model_observable_in_native_units",
    "full_gradient_max_abs_error":"verification_residual",
    "compiled_rhs_relative_error":"verification_residual",
}


def richardson(ns,qs):
    if len(ns)!=3 or ns[1]!=2*ns[0] or ns[2]!=2*ns[1]:
        return {"status":"not_three_dyadic_grids","p":None,"q_infinity":None}
    if not all(math.isfinite(x) for x in qs):
        raise ValueError("Nonfinite observable")
    d1,d2=qs[1]-qs[0],qs[2]-qs[1]
    noise=64*sys.float_info.epsilon*max(abs(x) for x in qs)
    result={"grids":ns,"values":qs,"d_coarse":d1,"d_fine":d2,"p":None,"q_infinity":None}
    if abs(d1)<=noise or abs(d2)<=noise:
        return {**result,"status":"zero_or_roundoff_sized_difference"}
    if (d1>0)!=(d2>0):
        return {**result,"status":"nonmonotonic_not_single_signed_error_law"}
    p=math.log(abs(d1/d2))/math.log(2)
    result["p"]=p
    if p<=0:
        return {**result,"status":"noncontracting_differences_no_finite_limit_estimate"}
    denominator=math.expm1(p*math.log(2))
    result["q_infinity"]=qs[2]+d2/denominator
    result["status"]="conditional_single_term_error_estimate"
    result["asymptotic_regime_proved"]=False
    result["mechanism_identified"]=False
    return result


def analyze(records,assume_comparable=False):
    by_group={}
    for r in records:
        if r.get("status") not in (None,"completed_checkpoint_recheck"):
            continue
        group=(r["time"],r.get("outer_radius"))
        grid=by_group.setdefault(group,{})
        n=int(r["cells"])
        if n in grid:
            raise ValueError(f"Duplicate grid/time/domain record: {group}, N={n}")
        grid[n]=r
    output=[]
    for (time,radius),grid in sorted(by_group.items(),key=lambda kv:(kv[0][0],str(kv[0][1]))):
        cells=sorted(grid)
        omegas=[grid[n].get("active_exchange_strength") for n in cells]
        frozen=all(v is not None for v in omegas) and len(set(omegas))==1
        entry={"time":time,"outer_radius":radius,"grids":cells,
               "fixed_exchange_coefficient":frozen,"comparison_protocol_verified":False,
               "comparability_user_asserted":assume_comparable,"metrics":{}}
        for metric,kind in METRICS.items():
            available=[n for n in cells if metric in grid[n]]
            values=[float(grid[n][metric]) for n in available]
            if not all(math.isfinite(x) for x in values):raise ValueError(metric)
            row={"kind":kind,"grids":available,"values":values,"triplets":[]}
            if len(values)==2:
                row.update(first_difference=values[1]-values[0],
                           ratio=None if values[0]==0 else values[1]/values[0])
            if kind=="verification_residual":
                row.update(status="not_a_spatial_truncation_error_estimate",p=None,q_infinity=None,
                           reason="Residuals depend on epsilon, arithmetic, normalization and test method; sweep epsilon separately.")
            elif len(available)<3:
                row.update(status="insufficient_grids_for_unknown_order",p=None,q_infinity=None)
            else:
                for i in range(len(available)-2):
                    fit=richardson(available[i:i+3],values[i:i+3])
                    fit["accepted_as_continuum_prediction"]=bool(
                        assume_comparable and frozen and kind!="grid_dependent_calibration"
                        and fit["status"]=="conditional_single_term_error_estimate")
                    row["triplets"].append(fit)
            if kind=="grid_dependent_calibration":
                row["interpretation"]="Calibration-sequence diagnostic, not solution convergence at a frozen Hamiltonian parameter."
            entry["metrics"][metric]=row
        if not frozen:
            entry["warning"]="The exchange coefficient varies across grids. A third value alone will not isolate spatial truncation error at a fixed coefficient."
        output.append(entry)
    return {"groups":output,"fd_step_optimality_inferred_from_grid_data":False,
            "requirements":["matched continuum initial data and preparation", "fixed continuum parameter and cap rules",
                            "controlled temporal error", "consistent observable and boundary treatment",
                            "evidence of the asymptotic refinement regime"]}


def self_test():
    ns=[256,512,1024]
    fit=richardson(ns,[3+2/(n*n) for n in ns])
    assert abs(fit["p"]-2)<1e-12 and abs(fit["q_infinity"]-3)<1e-12
    assert richardson(ns,[1.,2.,1.5])["p"] is None
    assert richardson(ns,[1.,2.,4.])["q_infinity"] is None
    assert richardson(ns,[1.,1.,1.])["p"] is None
    duplicate=[dict(cells=256,time=90,outer_radius=8),dict(cells=256,time=90,outer_radius=8)]
    try:analyze(duplicate)
    except ValueError:pass
    else:raise AssertionError("Duplicate samples were silently overwritten")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--assume-comparable",action="store_true")
    args=p.parse_args();self_test()
    data=json.loads(args.input.read_text())
    records=data["rechecks"] if isinstance(data,dict) else data
    result=analyze(records,args.assume_comparable)
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:args.output.write_text(text)
    else:print(text,end="")


if __name__=="__main__":main()
