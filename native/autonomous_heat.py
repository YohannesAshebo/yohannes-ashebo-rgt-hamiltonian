"""Run the archived RGT autonomous geometry with an energy-audited heat sink."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np

from autonomous import CurrentExchange
from compiled_current import CompiledCurrent
from continue_run import FastRHS, build, integrate
from refill import set_omega

HERE = Path(__file__).resolve().parent
REGIONS = ("inner", "core_shell", "outer_shell", "far")
RTOL = 2.5e-12


class HeatRHS:
    def __init__(self, fast, law, kappa):
        self.n, self.omega, self.j0, self.kappa = fast.n, law.omega, law.j0, kappa
        self.work = np.empty(15 * self.n)
        self.inputs = (*fast.pointers[:-1], self.work.ctypes.data)
        self.library = ct.CDLL(str(HERE / "autonomous_heat_rhs.so"))
        self.call = self.library.heat_rhs
        self.call.argtypes = [ct.c_int] + [ct.c_void_p] * 8 + [ct.c_double] * 3 + [ct.c_void_p]
        self.call.restype = None

    def __call__(self, _, state):
        state = np.ascontiguousarray(state, dtype=float)
        out = np.empty(8 * self.n + 5)
        self.call(self.n, state.ctypes.data, *self.inputs, self.omega,
                  self.j0, self.kappa, out.ctypes.data)
        return out


def fields(model, law, state):
    data = model.evaluate(state)
    extra = law.evaluate(state)
    energy = data["H"] + extra["energy"]
    data["flux"] += extra["extra_flux"]
    data["Gamma"] += extra["Gamma"]
    data["rhs"] = law.add_rhs(data["rhs"], state)
    contents = data["N"]
    gradient = model.p.eta * contents * model.unpack(data["rhs"])[:2]
    return data, energy, gradient


def reference_heat(model, data, gradient, core, kappa):
    n = model.N
    jc, jr = data["flux"] / data["A"]
    overlap = np.maximum(0, np.minimum(-jc, jr))
    paired = (np.r_[0, overlap] + np.r_[overlap, 0]) / 2
    phase_rate = kappa * paired * (gradient[0] - gradient[1]) / data["N"].sum(axis=0)
    flow = np.zeros((8, n))
    flow[2], flow[3] = -phase_rate, phase_rate
    q = phase_rate * (gradient[0] - gradient[1])
    tags = np.searchsorted([1.75, 2.5, 3.], model.r, side="right")
    sums = [q[tags == i].sum() for i in range(4)]
    sums.append(q[abs(model.r - core) <= .25].sum() if core is not None else 0.)
    return np.r_[flow.ravel(), sums]


def speeds(model, data, radius):
    if radius is None:
        return [None, None]
    donors = np.stack([data["density"][0, 1:], data["density"][1, :-1]])
    velocity = data["flux"] / (data["A"] * donors)
    return [float(np.interp(radius, model.faces[1:-1].astype(float), u.astype(float))) for u in velocity]


def run(cells, kappa, stop, rtol, output):
    start = 90.
    archive = json.loads((HERE / "data/reference.json").read_text())[str(cells)]
    model, _, _, _ = build(cells, "matched_quadratic")
    state = np.load(HERE / f"data/weak_N{cells}_s90.npz")["state"].astype(float)
    assert abs(float(model.p.Omega_p) - archive["original_omega"]) < 1e-14
    law = CurrentExchange(model, model.p.Omega_p, decreasing=True)
    law.j0 = archive["j0"]
    base = FastRHS(model)
    set_omega(model, base, 0.)
    compiled = CompiledCurrent(base, law)
    heat = HeatRHS(base, law, kappa)
    data, initial_energy, _ = fields(model, law, state)
    initial_content = data["N"].sum()
    assert abs(initial_energy - archive["initial_energy"]) < 1e-10
    report = dict(cells=cells, kappa=kappa, start=start, stop=stop, tolerance=rtol,
                  initial_energy=float(initial_energy), initial_content=float(initial_content),
                  regions=REGIONS, rows=[], archive_comparison=[])
    checks = []

    def sample(s, y):
        state = y[:8*cells]
        data, energy, gradient = fields(model, law, state)
        all_rate = heat(s, y)
        heat_power = all_rate[-5:]
        budget = np.zeros(5) if kappa == 0 else y[-5:]
        rates = model.unpack(data["rhs"])
        data["chi_dot"] = ((data["N"][0]*rates[0]-data["N"][1]*rates[1])
                           / (data["V"]*model.p.n_star) - (rates[4]+2*rates[5])*data["chi"])
        c = model.core(state, data)
        radius = c.get("coordinate_radius")
        if radius is not None and radius < 4*model.dr:
            radius = None
        edge_speed = speeds(model, data, radius)
        far_speed = speeds(model, data, 3.5)
        row = dict(s=s, B=c.get("signed_content") if radius is not None else None,
                   R=c.get("areal_radius") if radius is not None else None,
                   core_coordinate_radius=radius, chi_center=float(data["chi"][0]),
                   u_edge=edge_speed, u_far=far_speed,
                   H=float(energy), heat=budget.tolist(), power=heat_power.tolist(),
                   energy_error=float(abs(energy + budget[:4].sum()-initial_energy)/abs(initial_energy)),
                   content_error=float(abs(data["N"].sum()-initial_content)/initial_content))
        report["rows"].append(row)
        if s % 10 == 0 or s == stop:
            native = reference_heat(model, data, gradient, radius, kappa)
            absolute_heat_error = float(np.max(abs(native[-5:] - heat_power)))
            # Subtract the pointwise roundoff of the conservative C and Python paths.
            conservative = compiled(s, state)
            delta_phase = (all_rate[:8*cells] - conservative).reshape(8, cells)[2:4]
            source_power = float(np.sum(gradient * delta_phase))
            checks.append(dict(s=s, absolute_heat_error=absolute_heat_error,
                               power_balance_error=abs(source_power + float(heat_power[:4].sum())),
                               base_rhs_error=compiled.compare(state)))
            assert absolute_heat_error < 1e-8, checks[-1]
            assert checks[-1]["power_balance_error"] < 1e-8, checks[-1]
            if kappa == 0:
                prior = next((r for r in archive["rows"] if r["s"] == s), None)
                if prior and radius is not None:
                    report["archive_comparison"].append(dict(s=s, delta_B=row["B"]-prior["B"],
                                                             delta_R=row["R"]-prior["radius"]))
            report["checks"] = checks
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, allow_nan=False)+"\n")
            temporary.replace(output)
            print(cells, kappa, s, row["B"], row["R"], budget[:4].sum(), row["energy_error"], flush=True)

    if kappa == 0:
        # Reproduce the archived Python early leg and compiled continuation.
        def early_rhs(s, y):
            return law.add_rhs(base(s, y), y)

        sample(start, state)
        state = integrate(early_rhs, state, start, 110, sample, cadence=.05, rtol=rtol)
        if stop > 110:
            integrate(compiled, state, 110, stop, sample, cadence=.1, rtol=rtol)
    else:
        y = np.r_[state, np.zeros(5)]
        sample(start, y)
        integrate(heat, y, start, stop, sample, cadence=.1, rtol=rtol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=int, choices=(256, 512))
    parser.add_argument("kappa", type=float)
    parser.add_argument("stop", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rtol", type=float, default=RTOL)
    args = parser.parse_args()
    if args.kappa < 0 or args.stop < 110:
        parser.error("Use a nonnegative coupling and a stop time of at least 110.")
    run(args.cells, args.kappa, args.stop, args.rtol, args.output)
