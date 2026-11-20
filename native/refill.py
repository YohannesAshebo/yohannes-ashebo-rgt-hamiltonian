"""Test explicit exchange feedback measured at the moving equality surface."""
import argparse
import json
from dataclasses import replace

import numpy as np

from common import write
from continue_run import HERE, PARAMETER_NAMES, FastRHS, build, compare_rhs, integrate

START, STOP = 90., 110.
CASES = ("original", "fixed_reduction", "mean_reduction", "feedback")


def shell_fraction(model, radius):
    lo, hi = model.faces[:-1], model.faces[1:]
    return (np.clip(radius, lo, hi)**3-lo**3)/(hi**3-lo**3)


def edge_balance(model, state, data, original_omega):
    core = model.core(state, data)
    if not core["exists"] or core["coordinate_radius"] < 4*model.dr:
        return None
    weights = shell_fraction(model, core["coordinate_radius"])
    left_to_right = data["flux"] @ (weights[:-1]-weights[1:])
    theta = model.unpack(state)[2:4]
    gamma_original = (2*original_omega/model.p.eta * (data["chi"]/model.cs)**2
                      * np.sqrt(data["N"][0]*data["N"][1]) * np.sin(theta[1]-theta[0]))
    return dict(core=core, weights=weights, inward_c=-left_to_right[0],
                outward_r=left_to_right[1], restoration_original=2*(weights @ gamma_original),
                restoration=2*(weights @ data["Gamma"]))


def controller(balance, previous, threshold=.5):
    if balance is None:
        return previous
    restoration = balance["restoration_original"]
    if restoration <= 0:
        return 1.
    flux = balance["inward_c"]+balance["outward_r"]
    return float(np.clip(threshold*flux/restoration, 0., 1.))


def set_omega(model, fast, omega):
    for target in (model, model.original, model.base):
        target.p = replace(target.p, Omega_p=omega)
    fast.arrays[0][PARAMETER_NAMES.index("Omega_p")] = omega


def run(cells, case, step, threshold):
    model, _, _, _ = build(cells, "matched_quadratic")
    assert model.p.Omega == 0.
    path = HERE/"data"
    state = np.load(path/f"weak_N{cells}_s90.npz")["state"]
    original = model.p.Omega_p
    native0 = model.evaluate(state)
    energy0, content0 = native0["H"], native0["N"].sum()
    fast = FastRHS(model)
    preflight = [compare_rhs(model, fast, (state,))]
    fixed_factor = controller(edge_balance(model, state, native0, original), 1., threshold)
    if case == "mean_reduction":
        archived = json.loads((HERE/"reference"/f"feedback_N{cells}.json").read_text())
        fixed_factor = float(np.mean([decision["factor"] for decision in archived["decisions"]]))
    fixed = np.stack([shell_fraction(model, radius) for radius in (1., 2.3125)])
    work, factor = 0., 1.
    rows, decisions = [], []

    def sample(s, y):
        d = model.evaluate(y)
        balance = edge_balance(model, y, d, original)
        contents = fixed @ d["N"].T
        row = dict(time=s, factor=factor, energy=float(d["H"]), external_work=float(work),
                   energy_error=float(abs(d["H"]-energy0-work)/max(1, abs(energy0))),
                   content_error=float(abs(d["N"].sum()-content0)/content0),
                   fixed_contents_CR=contents.astype(float).tolist(),
                   radius=None, B=None)
        if balance is not None:
            c, weights = balance["core"], balance["weights"]
            mode_contents = d["N"] @ weights
            incoming, outgoing = balance["inward_c"], balance["outward_r"]
            restoration = balance["restoration"]
            sweep = c["signed_rate"]-incoming-outgoing+restoration
            row.update(radius=c["areal_radius"], coordinate_radius=c["coordinate_radius"],
                       B=c["signed_content"], core_contents_CR=mode_contents.astype(float).tolist(),
                       incoming_c=float(incoming), outgoing_r=float(outgoing),
                       restoration=float(restoration), sweep=float(sweep),
                       dB_ds=c["signed_rate"])
        rows.append(row)

    for index in range(round((STOP-START)/step)):
        begin = round(START+index*step, 12)
        d = model.evaluate(state)
        balance = edge_balance(model, state, d, original)
        if case in ("fixed_reduction", "mean_reduction"):
            new_factor = fixed_factor
        elif case == "feedback":
            new_factor = controller(balance, factor, threshold)
        else:
            new_factor = 1.
        before = d["H"]
        after = before
        old_factor = factor
        if new_factor != factor:
            set_omega(model, fast, original*new_factor)
            after = model.hamiltonian(state)
            work += after-before
        factor = new_factor
        decisions.append(dict(time=begin, factor=factor, previous_factor=old_factor, work=float(work),
            energy_before=float(before), energy_after=float(after), step_work=float(after-before),
            coordinate_radius=None if balance is None else balance["core"]["coordinate_radius"]))
        if index == 0:
            sample(begin, state)
        if begin % 5 == 0:
            preflight.append(compare_rhs(model, fast, (state,)))
        state = integrate(fast, state.astype(float), begin, round(begin+step, 12),
                          sample, cadence=.05)
        end = round(begin+step, 12)
        if end % 5 == 0:
            print(case, cells, end, rows[-1]["B"], rows[-1]["radius"], factor, flush=True)
    preflight.append(compare_rhs(model, fast, (state,)))
    tag = f"{case}_N{cells}_dt{step:g}_q{threshold:g}"
    write(path/f"{tag}.json", dict(case=case, cells=cells, original_omega=original,
        update_interval=step, threshold=threshold, initial_energy=float(energy0),
        first_exchange_factor=decisions[0]["factor"], rhs_error=max(preflight), rows=rows, decisions=decisions))
    np.savez_compressed(path/f"{tag}_s110.npz", state=state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", type=int, choices=(256,512))
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--step", type=float, choices=(.1, .25, .5, 1.), default=.25)
    parser.add_argument("--threshold", type=float, choices=(.25, .5, .75), default=.5)
    args = parser.parse_args()
    run(args.cells, args.case, args.step, args.threshold)
