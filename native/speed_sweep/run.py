"""Energy-matched Hamiltonian bias, without prescribing evolved speeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ExtendedDOP853, LD
from force_balance import build
from measure import observe

OUT = Path(__file__).resolve().parent


def prepare(cells, bias):
    model, initial = build(cells, .5, 0)
    caps = model.vmax.copy()
    baseline = model.evaluate(initial)
    target = baseline['energies']['directional_motion']
    scale = np.sqrt(LD(str(bias)))
    model.vmax = caps * np.array([scale, 1 / scale])[:, None]
    energy = model.evaluate(initial)['energies']['directional_motion']
    common_factor = target / energy
    model.vmax *= common_factor
    adjusted = model.evaluate(initial)
    energy_match = abs(adjusted['H'] - baseline['H']) / max(1, abs(baseline['H']))
    assert energy_match < 1e-14
    return model, initial, baseline, adjusted, float(common_factor), float(energy_match)


def run(cells, bias):
    model, initial, baseline, adjusted, factor, energy_match = prepare(cells, bias)
    faces = model.faces[1:-1].astype(float)
    total_initial = adjusted['N'].sum()
    energy_initial = adjusted['H']
    launch_speeds = [np.interp(1, faces, abs(v).astype(float)) for v in adjusted['speed']]
    sign = np.array([-1, 1])

    def sample(t, state):
        row = observe(model, t, state)
        data = model.evaluate(state)
        row['energy_error'] = float(abs(data['H'] - energy_initial) / max(1, abs(energy_initial)))
        row['content_error'] = float(abs(data['N'].sum() - total_initial) / total_initial)
        if row['exists']:
            radius = row['coordinate_radius']
            for key, reference in [('current_ratio', adjusted), ('unbiased_current_ratio', baseline)]:
                ref = np.array([np.interp(radius, faces, f.astype(float)) for f in reference['flux']]) * sign
                current = row['probes'][-1]['current']
                row[key] = [float(c / f) if f > 0 else None for c, f in zip(current, ref)]
        return row

    rows = [sample(0, initial)]
    solver = ExtendedDOP853(model.rhs, 0., initial, 6., rtol=2.5e-12, atol=2e-14, max_step=.005)
    for time in np.arange(1, 121) / 20:
        solver.t_bound = float(time)
        solver.status = 'running'
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise RuntimeError(message)
        rows.append(sample(time, solver.y))
    result = dict(cells=cells, bias=bias, common_factor=factor, energy_match_error=energy_match,
                  initial_speed_ratio_at_r1=float(launch_speeds[0] / launch_speeds[1]),
                  rows=rows)
    name = f'bias_{bias:.2f}_N{cells}'
    (OUT / f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / f'{name}_restart.npz', state=solver.y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cells', type=int)
    parser.add_argument('bias', type=float)
    args = parser.parse_args()
    run(args.cells, args.bias)
