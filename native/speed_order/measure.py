"""Measure native speed ordering and signed-content budget on the moving core."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ExtendedDOP853, ROOT
from directional_radial import sigmoid
from force_balance import build

OUT = Path(__file__).resolve().parent


def observe(model, time, state):
    data = model.evaluate(state)
    core = model.core(state, data)
    row = {'time': float(time), **core}
    if not core['exists']:
        return row
    radius = core['coordinate_radius']
    faces = model.faces[1:-1]
    fields = model.unpack(state)
    rates = model.unpack(data['rhs'])
    sign = np.array([-1, 1])[:, None]
    speed = abs(data['speed'])
    momentum = model.p.eta * np.diff(fields[2:4], axis=1) / data['L']
    momentum_rate = (model.p.eta * np.diff(rates[2:4], axis=1) / data['L']
                     - momentum * (rates[4, :-1] + rates[4, 1:]) / 2)
    z = sign * momentum / model.p0
    mobility_rate = -(1 - data['mobility']) * (data['chi_dot'][:-1] + data['chi_dot'][1:]) / (2 * model.cs)
    speed_rate = speed * (mobility_rate + (1 - sigmoid(z)) * sign * momentum_rate / model.p0)
    metric = np.exp((fields[4, :-1] + fields[4, 1:]) / 2)
    acceleration = speed_rate + data['speed'] * np.gradient(speed, model.dr, axis=1) / metric
    donors = np.stack([data['density'][0, 1:], data['density'][1, :-1]])
    currents = data['flux'] * sign
    interpolation_grid = faces.astype(float)

    def sample(array, coordinate):
        return [float(np.interp(coordinate, interpolation_grid, f.astype(float))) for f in array]

    row['probes'] = []
    for fraction in (.5, .9, 1.):
        coordinate = fraction * radius
        if not faces[0] <= coordinate <= faces[-1]:
            continue
        v = sample(speed, coordinate)
        rho = sample(donors, coordinate)
        current = sample(currents, coordinate)
        row['probes'].append(dict(fraction=fraction, speed=v, donor_density=rho, current=current,
                                  speed_ratio=v[0]/v[1], donor_density_ratio=rho[0]/rho[1],
                                  material_speed_rate=sample(acceleration, coordinate)))

    lo, hi = model.faces[:-1], model.faces[1:]
    upper = np.minimum(hi, radius)
    weights = np.where(upper > lo, (upper**3 - lo**3) / (hi**3 - lo**3), 0)
    moving = (radius > lo) & (radius < hi)
    weight_rate = np.where(moving, 3 * radius**2 / (hi**3 - lo**3) * core['coordinate_radius_rate'], 0)
    influx, outflux = -(currents * np.diff(weights)).sum(axis=1)
    exchange = -2 * np.sum(weights * data['Gamma'])
    sweep = np.sum(weight_rate * (data['N'][0] - data['N'][1]))
    balance = influx + outflux + exchange + sweep
    row['budget'] = dict(inward_convergence=float(influx), outward_divergence=float(outflux),
                         signed_exchange=float(exchange), sweep=float(sweep),
                         reconstructed_rate=float(balance), error=float(abs(balance - core['signed_rate'])))
    return row


def run(name):
    archive = np.load(ROOT / 'results' / name / 'states.npz')
    cells = archive['states'].shape[1] // 8
    model, _ = build(cells, .5, 0 if name.startswith('phase0') else None)
    rows = [observe(model, t, y) for t, y in zip(archive['time'], archive['states'])]
    solver = ExtendedDOP853(model.rhs, 3., archive['states'][-1], 6.,
                            rtol=2.5e-12, atol=2e-14, max_step=.005)
    for time in np.arange(61, 121) / 20:
        solver.t_bound = float(time)
        solver.status = 'running'
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise RuntimeError(message)
        rows.append(observe(model, time, solver.y))
    (OUT / f'{name}.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    run(parser.parse_args().run)
