"""Strength matching on a frozen, resolved common reference core."""
import numpy as np

from common import ROOT
from weight_sweep.weight_model import build as original_build


def match(cells):
    model, y0, _ = original_build(cells, 'quadratic')
    initial = model.evaluate(y0)
    y5 = np.load(ROOT / 'pressure_diagnostic' / f'g_0.00_N{cells}_s5.npz')['state']
    data = model.evaluate(y5)
    core = model.core(y5, data)
    radius = core['coordinate_radius']
    lo, hi = model.faces[:-1], model.faces[1:]
    weight = (np.clip(radius, lo, hi)**3-lo**3)/(hi**3-lo**3)
    x2 = (data['chi']/model.cs)**2
    volume = weight*data['V']
    quadratic = np.dot(volume, x2)
    saturating = np.dot(volume, data['chi']**2/(1+x2))
    initial_x2 = (initial['chi']/model.cs)**2
    return dict(reference_time=5., core_radius=radius, S_quadratic=float(quadratic),
                S_saturating=float(saturating), ratio=float(saturating/quadratic),
                initial_max_abs_chi=float(np.max(abs(initial['chi']))),
                initial_core_resolved=model.core(y0, initial)['exists'],
                initial_whole_domain_S_quadratic=float(np.dot(initial['V'],initial_x2)),
                initial_whole_domain_S_saturating=float(np.dot(initial['V'],initial['chi']**2/(1+initial_x2))))


def build(cells, case):
    match_info = match(cells)
    ratio = match_info['ratio']
    weight, factor = {'matched_quadratic': ('quadratic',ratio),
                      'matched_saturating': ('saturating',1/ratio),
                      'long_saturating': ('saturating',1.)}[case]
    model, state, reference = original_build(cells, weight)
    from dataclasses import replace
    for target in (model, model.original, model.base):
        target.p = replace(target.p, Omega_p=target.p.Omega_p*factor)
    match_info.update(weight=weight, omega_p=float(model.p.Omega_p))
    return model, state, reference, match_info
