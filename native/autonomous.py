"""Test a time-independent local current weight for phase exchange."""
import argparse
import json

import numpy as np
from scipy.optimize import brentq

from common import write
from continue_run import HERE, FastRHS, build, compare_rhs, integrate
from refill import set_omega, shell_fraction


def sigmoid(z):
    u = np.exp(-abs(z))
    return np.where(z >= 0., 1., u)/(1.+u)


class CurrentExchange:
    """Autonomous exchange Hamiltonian and derivatives in log-content coordinates."""

    def __init__(self, model, original_omega, decreasing=False):
        self.model = model
        self.omega = original_omega
        self.n = model.N
        self.sign = np.array([-1., 1.])[:, None]
        self.j0 = 1.
        self.decreasing = decreasing
        self.p0 = float(model.p0) if decreasing else model.p0
        self.vmax = np.asarray(model.vmax, dtype=float)

    def evaluate(self, state, gradient=True):
        m, n, p = self.model, self.n, self.model.p
        q = np.asarray(state).reshape(8, n)
        N = np.exp(q[:2])
        volume = m.V0*np.exp(q[4]+2*q[5])
        density = N/volume
        chi = (density[0]-density[1])/p.n_star
        length = m.dr*np.exp((q[4, :-1]+q[4, 1:])/2)
        z = self.sign*p.eta*np.diff(q[2:4], axis=1)/(self.p0*length)
        mob = sigmoid(-(chi[:-1]+chi[1:])/(2*m.cs))
        sig = sigmoid(z)
        speed = self.vmax*sig
        donor = np.stack([density[0, 1:], density[1, :-1]])
        pieces = speed*donor
        magnitude = pieces.sum(axis=0)
        face_j = mob*magnitude
        j = (np.r_[0., face_j]+np.r_[face_j, 0.])/2
        factor = self.j0/(j+self.j0) if self.decreasing else j/(j+self.j0)
        delta = q[3]-q[2]
        root = np.sqrt(N[0]*N[1])
        overlap = (np.sqrt(N[0])-np.sqrt(N[1]))**2+4*root*np.sin(delta/2)**2
        g = self.omega*(chi/m.cs)**2
        raw_energy = g*overlap
        energy = raw_energy @ factor
        if not gradient:
            return dict(energy=energy, j=j, raw_energy=raw_energy, factors=factor)

        result = np.zeros_like(q)
        # Direct derivatives of the exchange energy at fixed weight.
        density_term = 2*self.omega*chi*overlap/(m.cs**2*volume*p.n_star)
        mu = np.stack((density_term, -density_term))+g*(1-root*np.cos(delta)/N)
        result[:2] = N*factor*mu
        gamma = 2*g*root*np.sin(delta)*factor/p.eta
        result[2:4] = self.sign*p.eta*gamma
        result[4] = -2*factor*raw_energy
        result[5] = -4*factor*raw_energy

        # Chain rule from cell weight through adjacent face currents.
        weight_grad = raw_energy*self.j0/(j+self.j0)**2
        if self.decreasing:
            weight_grad *= -1
        face_grad = (weight_grad[:-1]+weight_grad[1:])/2
        chi_grad = -mob*(1-mob)*magnitude/(2*m.cs)
        pressure_grad = mob*(pieces*(1-sig)*z).sum(axis=0)/2
        theta_grad = face_grad*mob*pieces*(1-sig)*self.sign*p.eta/(length*self.p0)
        result[2:4, :-1] -= theta_grad
        result[2:4, 1:] += theta_grad
        for side, endpoint, donating_mode in ((0, slice(None, -1), 1), (1, slice(1, None), 0)):
            result[:2, endpoint] -= (self.sign*face_grad*chi_grad*density[:, endpoint]/p.n_star)
            donor_grad = face_grad*mob*pieces[donating_mode]
            result[donating_mode, endpoint] += donor_grad
            volume_grad = -face_grad*chi_grad*chi[endpoint]-donor_grad
            result[4, endpoint] += volume_grad-face_grad*pressure_grad
            result[5, endpoint] += 2*volume_grad
        return dict(energy=energy, j=j, raw_energy=raw_energy, factors=factor,
                    gradient=result, contents=N, Gamma=gamma, extra_flux=theta_grad/p.eta)

    def calibrate(self, initial, target):
        data = self.evaluate(initial, gradient=False)
        j, raw = data["j"], data["raw_energy"]
        def residual(log_j0):
            j0 = np.exp(log_j0)
            factor = j0/(j+j0) if self.decreasing else j/(j+j0)
            return float(raw @ factor/raw.sum())-target
        self.j0 = float(np.exp(brentq(residual, -45., 20.)))

    def add_rhs(self, out, y):
        n, eta = self.n, self.model.p.eta
        d = self.evaluate(y)
        rhs, grad = out.reshape(8, n), d["gradient"]
        rhs[:2] += grad[2:4]/(eta*d["contents"])
        rhs[2:4] -= grad[:2]/(eta*d["contents"])
        rhs[6:] -= grad[4:6].astype(rhs.dtype, copy=False)
        return rhs.ravel()


def finite_difference(law, state):
    rng = np.random.default_rng(419)
    y = np.asarray(state, dtype=np.longdouble).ravel()
    exact = law.evaluate(y)["gradient"].ravel()
    worst = 0.
    for group in range(8):
        v = np.zeros_like(y)
        sl = slice(group*law.n, (group+1)*law.n)
        v[sl] = rng.standard_normal(law.n)/np.sqrt(law.n)
        e = np.longdouble("1e-7")
        finite = (law.evaluate(y+e*v, False)["energy"]-law.evaluate(y-e*v, False)["energy"])/(2*e)
        worst = max(worst, float(abs(finite-exact@v)))
    assert worst < 1e-8, worst
    return worst


def run(cells, decreasing, j0_override=None):
    m, _, _, _ = build(cells, "matched_quadratic")
    data_dir = HERE/"data"
    y = np.load(data_dir/f"weak_N{cells}_s90.npz")["state"]
    prior = json.loads((HERE/"reference"/f"mean_reduction_N{cells}.json").read_text())
    target = prior["decisions"][0]["factor"]
    assert m.p.Omega == 0.
    law = CurrentExchange(m, m.p.Omega_p, decreasing)
    base = FastRHS(m)
    set_omega(m, base, 0.)
    if j0_override is None:
        law.calibrate(y, target)
    else:
        law.j0 = j0_override
    difference = [finite_difference(law, y)]
    base_error = compare_rhs(m, base, (y,))
    first = m.evaluate(y)
    h0 = first["H"]+law.evaluate(y, False)["energy"]
    n0 = first["N"].sum()
    fixed = shell_fraction(m, 1.)
    rows = []

    def rhs(s, state):
        return law.add_rhs(base(s, state), state)

    def sample(s, state):
        d = m.evaluate(state)
        extra = law.evaluate(state)
        d["flux"] += extra["extra_flux"]
        d["Gamma"] += extra["Gamma"]
        d["rhs"] = law.add_rhs(d["rhs"], state)
        rate = m.unpack(d["rhs"])
        d["chi_dot"] = ((d["N"][0]*rate[0]-d["N"][1]*rate[1])/(d["V"]*m.p.n_star)
                       -(rate[4]+2*rate[5])*d["chi"])
        c = m.core(state, d)
        rows.append(dict(time=s, B=c.get("signed_content"), radius=c.get("areal_radius"),
                         C_r1=float(d["N"][0]@fixed), R_r1=float(d["N"][1]@fixed),
                         energy=float(d["H"]+extra["energy"]),
                         energy_error=float(abs(d["H"]+extra["energy"]-h0)/abs(h0)),
                         content_error=float(abs(d["N"].sum()-n0)/n0),
                         exchange_fraction=float(extra["energy"]/extra["raw_energy"].sum()),
                         min_factor=float(extra["factors"].min()),
                         max_factor=float(extra["factors"].max())))
        if s % 5 == 0:
            print(cells, s, rows[-1], flush=True)

    sample(90., y)
    end = integrate(rhs, y.astype(float), 90., 110., sample, cadence=.05)
    difference.append(finite_difference(law, end))
    tag = f"autonomous_decreasing_N{cells}" if decreasing else f"autonomous_N{cells}"
    if j0_override is not None:
        tag += "_shared"
    write(data_dir/f"{tag}.json", dict(cells=cells, j0=law.j0, decreasing=decreasing,
        initial_energy=float(h0), target_initial_exchange_fraction=target,
        initial_exchange_fraction=rows[0]["exchange_fraction"],
        gradient_error=max(difference), base_rhs_error=base_error, rows=rows))
    np.savez_compressed(data_dir/f"{tag}_s110.npz", state=end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", type=int, choices=(256,512))
    parser.add_argument("--decreasing", action="store_true")
    parser.add_argument("--j0", type=float)
    args = parser.parse_args()
    if args.j0 is not None and args.j0 <= 0:
        parser.error("--j0 must be positive")
    run(args.cells, args.decreasing, args.j0)
