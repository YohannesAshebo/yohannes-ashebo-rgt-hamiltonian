"""Replace only the weighted exchange energy and all its derivatives."""
import numpy as np

from common import DirectionalRadial, LD
from run import prepare

WEIGHTS = ('quadratic', 'absolute', 'saturating', 'unweighted')


class WeightModel(DirectionalRadial):
    def __init__(self, base, weight):
        self.__dict__.update(vars(base))
        self.weight = weight

    def evaluate(self, state):
        d = super().evaluate(state)
        if self.weight == 'quadratic':
            return d
        p = self.p
        chi, volume, n = d['chi'], d['V'], d['N']
        q = self.unpack(state)
        relative_phase = q[3] - q[2]
        x = chi / self.cs
        old = x*x
        old_derivative = 2*x / self.cs
        if self.weight == 'absolute':
            w, dw = abs(x), np.sign(x)/self.cs
        elif self.weight == 'saturating':
            w, dw = chi*chi/(1+old), 2*chi/(1+old)**2
        elif self.weight == 'unweighted':
            w, dw = np.ones_like(x), np.zeros_like(x)
        elif self.weight == 'smooth_absolute':
            epsilon = LD('1e-3')
            norm = np.hypot(x, epsilon)
            w, dw = old/(norm+epsilon), x/(self.cs*norm)
        else:
            raise ValueError(self.weight)
        signed = np.array([1,-1])[:, None]
        root = np.sqrt(n[0]*n[1])
        cos = np.cos(relative_phase)
        overlap = (np.sqrt(n[0])-np.sqrt(n[1]))**2 + 4*root*np.sin(relative_phase/2)**2
        difference, derivative = w-old, dw-old_derivative
        mu = p.Omega_p*(signed * derivative * overlap / (volume*p.n_star)
                         + difference*(1-root*cos/n))
        geom = -p.Omega_p*derivative*chi*overlap
        extra_gamma = 2*p.Omega_p/p.eta * difference*root*np.sin(relative_phase)
        gamma = 2/p.eta * (p.Omega + p.Omega_p*w)*root*np.sin(relative_phase)
        gradient = self.unpack(d['gradient'])
        rhs = self.unpack(d['rhs'])
        gradient[:2] += n*mu
        gradient[2:4] -= signed*p.eta*extra_gamma
        gradient[4] += geom
        gradient[5] += 2*geom
        rhs[:2] -= signed*extra_gamma/n
        rhs[2:4] -= mu/p.eta
        rhs[6] -= geom
        rhs[7] -= 2*geom
        d['H'] += p.Omega_p*np.dot(difference, overlap)
        d['energies']['weighted_phase_exchange'] = p.Omega_p*np.dot(w, overlap)
        d['Gamma'] = gamma
        d['chi_dot'] -= 2*extra_gamma/(volume*p.n_star)
        d['balance'] = rhs[:2]*n-d['transport']+signed*gamma
        return d


def build(cells, weight):
    base, state, _, reference, _, _ = prepare(cells, 1.50)
    return WeightModel(base, weight), state, reference
