#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Reconstructed review audit; this is NOT the original full_hamiltonian_audit generator.

Reproduce the native 256/512 s=90 states without evolving or changing them.
Check the energy against a separately written scalar formula, every coordinate
gradient by complex step in a holomorphic extension of the REAL log-content
chart, centered-difference epsilon sweeps, heat-power chain rule, and flux signs.
The native functions themselves do not accept a valid complex-step probe.

Requires Python 3, numpy, scipy, and a C compiler. All compilation/extraction
takes place in a temporary directory. The archived inputs are never changed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np

BLOCKS = ("log_N_C", "log_N_R", "theta_C", "theta_R", "alpha", "beta", "P_alpha", "P_beta")
LD = np.longdouble


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid_extension(z):
    """Two locally holomorphic, algebraically equivalent stable expressions."""
    z = np.asarray(z)
    out = np.empty_like(z)
    positive = np.real(z) >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    expz = np.exp(z[~positive])
    out[~positive] = expz / (1 + expz)
    return out


def softplus_extension(z):
    z = np.asarray(z)
    out = np.empty_like(z)
    positive = np.real(z) >= 0
    out[positive] = z[positive] + np.log1p(np.exp(-z[positive]))
    out[~positive] = np.log1p(np.exp(z[~positive]))
    return out


def energy_components(m, law, state):
    """Independent scalar energy in real-chart coordinates, plus its analytic extension.

    No abs, conjugation, real casting of state, max/min heat gate, or omitted
    weight derivatives enter this scalar formula. The real-coordinate identity
    for |psi_C-psi_R|^2 is used before analytic continuation. Base current signs
    are fixed by the native directional law, so |F_C^0|+|F_R^0| is written as
    the positive sum of their prescribed magnitudes.
    """
    y = np.asarray(state)
    q = y.reshape(8, m.N)
    p = m.p
    inventory = np.exp(q[:2])
    a, b, pa, pb = q[4:]
    ell = a + 2 * b
    volume = m.V0 * np.exp(ell)
    density = inventory / volume
    chi = (density[0] - density[1]) / p.n_star
    # This is the archived floating-point coefficient, not a refitted value.
    cs = m.cs
    potential = (p.g / 2 * (density.sum(axis=0) - p.n_background) ** 2
                 + p.lam / 4 * chi ** 4 - p.J * chi + 3 * p.J * cs / 4
                 + p.k_volume / 2 * (ell - p.zeta * chi) ** 2
                 + p.k_shear / 2 * (a - b) ** 2)
    area = m.A0 * np.exp(b[:-1] + b[1:])
    length = m.dr * np.exp((a[:-1] + a[1:]) / 2)
    sign = np.array([-1, 1])[:, None]
    u = sign * p.eta * np.diff(q[2:4], axis=1) / (law.p0 * length)
    mobility = sigmoid_extension(-(chi[:-1] + chi[1:]) / (2 * cs))
    donors = np.stack([density[0, 1:], density[1, :-1]])
    phase_energy = law.vmax * law.p0 * softplus_extension(u)
    current = mobility * np.sum(law.vmax * sigmoid_extension(u) * donors, axis=0)
    shell_current = (np.r_[0, current] + np.r_[current, 0]) / 2
    weight = law.j0 / (shell_current + law.j0)
    overlap = ((np.sqrt(inventory[0]) - np.sqrt(inventory[1])) ** 2
               + 4 * np.sqrt(inventory[0] * inventory[1]) * np.sin((q[3] - q[2]) / 2) ** 2)
    return {
        "metric_kinetic": np.sum(pa ** 2 / (2 * p.I * m.V0) + pb ** 2 / (4 * p.I * m.V0)),
        "local_potential": np.sum(volume * potential),
        "metric_gradient": (p.k_geometry / 2 * np.sum(m.original.edge_weight
            * (np.diff(a) ** 2 + 2 * np.diff(b) ** 2))
            + 2 * p.k_geometry * np.sum(m.V0 * (a - b) ** 2 / m.r ** 2)),
        "amplitude_gradient": p.K / 2 * np.sum(area / length
            * np.sum(np.diff(np.sqrt(density), axis=1) ** 2, axis=0)),
        "directional_motion": np.sum(mobility * area * length * np.sum(donors * phase_energy, axis=0)),
        "retention": p.Omega_d * p.n_star * np.sum(m.V0 * chi ** 2),
        "legacy_exchange": p.Omega * np.sum(overlap),
        "current_exchange": law.omega * np.sum(weight * (chi / cs) ** 2 * overlap),
    }


def scalar_energy(m, law, state):
    return sum(energy_components(m, law, state).values())


def centered_sweep(m, law, state, gradient):
    rng = np.random.default_rng(20261120)
    n = m.N
    probes = []
    # Random block directions plus single-coordinate probes at BOTH domain
    # ends and near the central core. No ghost cells are added or clamped.
    for block, label in enumerate(BLOCKS):
        v = np.zeros_like(state)
        v[block*n:(block+1)*n] = rng.standard_normal(n) / np.sqrt(n)
        probes.append((label + ":block_direction", v))
        for cell in sorted(set([0, 1, n // 3, n - 2, n - 1])):
            v = np.zeros_like(state)
            v[block*n+cell] = 1
            probes.append((f"{label}:cell_{cell}", v))
    rows = []
    for step in ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7", "1e-7", "3e-8", "1e-8", "1e-9"):
        eps = LD(step)
        errors = []
        for label, direction in probes:
            fd = (scalar_energy(m, law, state + eps*direction)
                  - scalar_energy(m, law, state - eps*direction)) / (2*eps)
            exact = gradient @ direction
            errors.append((float(abs(fd-exact)), label, float(abs(fd-exact)/max(1,abs(fd),abs(exact)))))
        worst = max(errors)
        rows.append({"epsilon": float(eps), "max_absolute_error": worst[0],
                     "worst_probe": worst[1], "max_scaled_error": max(x[2] for x in errors)})
    return rows


def complex_gradient(m, law, state, gradient):
    eps = LD("1e-30")
    probe = np.asarray(state, dtype=np.clongdouble).copy()
    computed = np.empty_like(state)
    for i in range(len(state)):
        probe[i] += 1j*eps
        computed[i] = np.imag(scalar_energy(m, law, probe)) / eps
        probe[i] = state[i]
    error = abs(computed-gradient)
    at = int(np.argmax(error))
    per_block = {label: float(error[i*m.N:(i+1)*m.N].max()) for i,label in enumerate(BLOCKS)}
    return {"epsilon": float(eps), "coordinates_checked": len(state),
            "max_absolute_error": float(error[at]),
            "worst_coordinate": {"block": BLOCKS[at//m.N], "cell": at % m.N},
            "max_scaled_error": float(np.max(error/np.maximum(1,abs(gradient)))),
            "per_block_max_abs_error": per_block}, computed


def audit_one(native, output, n):
    from autonomous import CurrentExchange
    from autonomous_heat import fields, reference_heat, HeatRHS
    from compiled_current import CompiledCurrent
    from continue_run import build, FastRHS
    from refill import set_omega
    ref = json.loads((native / "data/reference.json").read_text())
    state_path = native / "data" / f"weak_N{n}_s90.npz"
    missing = []
    if str(n) not in ref:
        missing.append(f"data/reference.json[{n!r}]: archived exchange strength and current scale")
    if not state_path.exists():
        missing.append(f"data/weak_N{n}_s90.npz: evolved s=90 state")
    if not (native / "pressure_diagnostic" / f"g_0.00_N{n}_s5.npz").exists():
        missing.append(f"pressure_diagnostic/g_0.00_N{n}_s5.npz: strength-matching state")
    if missing:
        return {"cells": n, "time": 90, "status": "not_run_missing_matched_inputs", "missing": missing,
                "richardson_eligible": False, "interpolation_is_not_a_new_evolution": True}
    m, _, _, matching = build(n, "matched_quadratic")
    y = np.load(state_path)["state"].astype(LD)
    law = CurrentExchange(m, m.p.Omega_p, decreasing=True)
    law.j0 = ref[str(n)]["j0"]
    fast = FastRHS(m)
    set_omega(m, fast, 0.)
    compiled = CompiledCurrent(fast, law)
    heated = HeatRHS(fast, law, 100.)
    base = m.evaluate(y)
    extra = law.evaluate(y)
    gradient = (m.unpack(base["gradient"]) + extra["gradient"]).ravel()
    d, native_energy, _ = fields(m, law, y)
    rates = m.unpack(d["rhs"])
    contents = d["N"]
    phase_gradient = gradient.reshape(8,n)[2:4]
    transport = np.zeros_like(contents)
    transport[:, :-1] -= d["flux"]
    transport[:, 1:] += d["flux"]
    source = np.stack([-d["Gamma"], d["Gamma"]])
    d["chi_dot"] = ((contents[0]*rates[0]-contents[1]*rates[1])/(d["V"]*m.p.n_star)
                    -(rates[4]+2*rates[5])*d["chi"])
    core = m.core(y, d)
    hp = reference_heat(m, d, phase_gradient, core.get("coordinate_radius"), 100.)
    h = hp[:8*n]
    power = hp[-5:-1].sum()
    # Independent heat expression, using all transport fluxes including the
    # derivative of the current-dependent exchange weight.
    face_pair = np.maximum(0,np.minimum(-d["flux"][0]/d["A"],d["flux"][1]/d["A"]))
    mean_pair = (np.r_[0,face_pair]+np.r_[face_pair,0])/2
    D = phase_gradient[0]-phase_gradient[1]
    b = mean_pair/contents.sum(axis=0)
    cell_power = 100*b*D**2
    # Finite difference along the heat-only vector; the full conservative
    # vector is checked separately, avoiding subtraction of two large rates.
    heat_fd = []
    for step in ("1e-2","3e-3","1e-3","3e-4","1e-4","3e-5","1e-5","1e-6","1e-7","1e-8","1e-9"):
        eps=LD(step)
        slope=(scalar_energy(m,law,y+eps*h)-scalar_energy(m,law,y-eps*h))/(2*eps)
        heat_fd.append({"epsilon":float(eps),"energy_slope":float(slope),
                        "absolute_balance_residual":float(abs(slope+power))})
    cs, cs_values = complex_gradient(m,law,y,gradient)
    fd = centered_sweep(m,law,y,gradient)
    full_heat = heated(90.,np.r_[np.asarray(y,dtype=float),np.zeros(5)])
    full_heat_power = full_heat[-5:-1].sum()
    tail_flux = np.cumsum((contents*compiled(90.,y).reshape(8,n)[:2]-source)[:,::-1],axis=1)[:,::-1][:,1:]
    sign=np.array([-1,1])[:,None]
    flux_checks={}
    for mode,label in enumerate(("C","R")):
        oriented=sign[mode]*d["flux"][mode]
        wrong=oriented < -LD("1e-12")*max(1,np.max(abs(d["flux"])))
        k=int(np.argmin(oriented))
        flux_checks[label]={"wrong_sign_faces_above_tolerance":int(wrong.sum()),
                            "example_radius":float(m.faces[k+1]),
                            "base_flux":float(base["flux"][mode,k]),
                            "exchange_flux":float(extra["extra_flux"][mode,k]),
                            "total_flux":float(d["flux"][mode,k])}
    # Equality at nearby states checks that the scalar rewrite is the same
    # real energy, rather than merely matching at this single checkpoint.
    rng=np.random.default_rng(401)
    agreement=[]
    for amp in (LD(0),LD("1e-6"),LD("-1e-6")):
        yy=y+amp*rng.standard_normal(len(y))/np.sqrt(len(y))
        direct=m.evaluate(yy)["H"]+law.evaluate(yy,False)["energy"]
        agreement.append(float(abs(scalar_energy(m,law,yy)-direct)))
    metrics={
        "energy_abs_error":float(abs(scalar_energy(m,law,y)-native_energy)),
        "local_continuity_max_abs_error":float(np.max(abs(contents*rates[:2]-transport-source))),
        "total_content_rate_abs_error":float(abs(np.sum(contents*rates[:2]))),
        "canonical_power_abs_error":float(abs(gradient@d["rhs"])),
        "moving_core_balance_abs_error":core["moving_signed_budget_error"],
        "compiled_rhs_relative_error":compiled.compare(y),
        "kappa_100_heat_power":float(power),
        "heat_power_cell_sum_abs_error":float(abs(cell_power.sum()-power)),
        "heated_power_balance_abs_error":float(abs(gradient@h+power)),
        "complex_step_heat_balance_abs_error":float(abs(cs_values@h+power)),
        "C_vs_Python_heat_power_abs_error":float(abs(full_heat_power-power)),
        "flux_from_compiled_tail_max_abs_error":float(np.max(abs(tail_flux-d["flux"]))),
    }
    result={"cells":n,"time":90,"outer_radius":float(m.p.domain_radius),
            "dx":float(m.dr),"status":"completed_checkpoint_recheck",
            "active_exchange_strength":float(law.omega),"kappa_100_heat_power":float(power),
            "checkpoint_sha256":digest(state_path),"coordinate_chart":list(BLOCKS),
            "gradient_normalization":"raw partial derivatives dH/dy; no cell-volume division",
            "native_state_unchanged":True,"independent_real_energy_agreement":agreement,
            "model_parameters":{k:float(v) for k,v in asdict(m.p).items()},
            "current_scale_j0":float(law.j0),"p0":float(law.p0),"calibration":matching,
            "metrics":metrics,"complex_step_rewritten_real_chart":cs,
            "centered_gradient_epsilon_sweep":fd,"heat_direction_epsilon_sweep":heat_fd,
            "flux_sign_checks":flux_checks,"core":core}
    profile=np.column_stack([m.r,d["V"],contents[0],contents[1],D,b,cell_power])
    np.savetxt(output/f"cell_heat_N{n}.csv",profile.astype(float),delimiter=",",
               header="coordinate_radius,proper_volume,N_C,N_R,H_theta_C_minus_H_theta_R,b,heat_power",comments="")
    print(json.dumps({"cells":n,"heat_power":float(power),"complex_step_gradient_error":cs["max_absolute_error"],
                      "smallest_sampled_FD_error":min(x["max_absolute_error"] for x in fd),
                      "heat_identity_error":metrics["heated_power_balance_abs_error"]}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent
    sources=parser.add_mutually_exclusive_group()
    sources.add_argument("--archive",type=Path,help="Audit an archived native source snapshot.")
    sources.add_argument("--native-dir",type=Path,help="Audit a visible native source tree (default: native/).")
    parser.add_argument("--input-audit",type=Path,default=base/"inputs/full_hamiltonian_audit.json")
    parser.add_argument("--output",type=Path,default=base/"results")
    parser.add_argument("--grids",type=int,nargs="+",default=[256,512,1024])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.archive is None and args.native_dir is None:
        args.native_dir=base/"native"
    if args.native_dir is not None and not args.native_dir.is_dir():
        parser.error("The native source directory does not exist.")
    input_paths=[args.input_audit]+([args.archive] if args.archive is not None else [])
    original_hashes={str(p.name):digest(p) for p in input_paths}
    def source_hashes(directory):
        return {str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
                and p.suffix not in (".so",".pyc",".dylib",".dll")}
    native_hashes=source_hashes(args.native_dir) if args.native_dir is not None else None
    with tempfile.TemporaryDirectory(prefix="ashebo_native_audit_") as temporary:
        native=Path(temporary)
        if args.native_dir is not None:
            shutil.copytree(args.native_dir,native,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__","*.so","*.pyc","*.dylib","*.dll"))
        else:
            with ZipFile(args.archive) as z:
                for name in z.namelist():
                    if not (native/name).resolve().is_relative_to(native.resolve()):
                        raise ValueError("Unsafe archive entry")
                z.extractall(native)
        if native_hashes is None:
            native_hashes=source_hashes(native)
        for path in (native,native/"study",native/"speed_order",native/"speed_sweep"):
            sys.path.append(str(path))
        for stem in ("radial_rhs","current_rhs","autonomous_heat_rhs"):
            subprocess.run(["cc","-O3","-fPIC","-shared",str(native/(stem+".c")),
                            "-o",str(native/(stem+".so")),"-lm"],check=True,capture_output=True)
        results=[]
        for n in args.grids:
            results.append(audit_one(native,args.output,n))
            document={"audit_kind":"reconstructed_review_audit_not_original_generator",
                      "source_hashes":original_hashes,"longdouble_epsilon":float(np.finfo(LD).eps),
                      "native_source_mode":"source_tree" if args.native_dir is not None else "archive",
                      "native_file_hashes":native_hashes,
                      "input_audit_unchanged":json.loads(args.input_audit.read_text()),"rechecks":results,
                      "physical_units_calibrated":False}
            (args.output/"mechanics_audit.json").write_text(json.dumps(document,indent=2,allow_nan=False)+"\n")
    assert original_hashes=={str(p.name):digest(p) for p in input_paths}
    if args.native_dir is not None:
        assert native_hashes==source_hashes(args.native_dir)


if __name__=="__main__":
    main()
