"""Continue the archived weak-quadratic trajectory with a verified fast RHS."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
from scipy.integrate import DOP853

from common import ExtendedDOP853, write
from measure import observe
from strength_match.match_model import build

HERE = Path(__file__).resolve().parent
PARAMETER_NAMES = ('eta', 'K', 'Omega', 'g', 'lam', 'J', 'k_volume', 'k_shear',
                   'zeta', 'I', 'k_geometry', 'n_background', 'n_star', 'Omega_d', 'Omega_p')


class FastRHS:
    def __init__(self, model):
        self.lib = ct.CDLL(str(HERE/'radial_rhs.so'))
        self.call = self.lib.quadratic_rhs
        self.call.argtypes = [ct.c_int] + [ct.c_void_p]*9
        self.call.restype = None
        self.n = model.N
        p = [getattr(model.p, name) for name in PARAMETER_NAMES]
        arrays = (p+[model.dr, model.cs, model.p0], model.V0, model.A0,
                  model.original.edge_weight, model.r, model.vmax, np.empty(6*self.n))
        self.arrays = tuple(np.ascontiguousarray(a, dtype=float) for a in arrays)
        self.pointers = tuple(a.ctypes.data for a in self.arrays)

    def __call__(self, _, y):
        y = np.ascontiguousarray(y, dtype=float)
        out = np.empty(8*self.n)
        self.call(self.n, y.ctypes.data, *self.pointers, out.ctypes.data)
        return out


def compare_rhs(model, fast, states):
    errors = []
    for y in states:
        native = model.rhs(0., y)
        compiled = fast(0., y)
        errors.append(float(np.max(abs(compiled-native))/max(1., np.max(abs(native)))))
    assert max(errors) < 1e-11, errors
    return max(errors)


def integrate(rhs, y, begin, end, sample, cadence=.1, native=False, rtol=2.5e-12):
    solver_type = ExtendedDOP853 if native else DOP853
    solver = solver_type(rhs, float(begin), y, float(end), rtol=rtol, atol=.008*rtol, max_step=.005)
    stride = round(1/cadence)
    for tick in range(round(begin*stride)+1, round(end*stride)+1):
        time = tick / stride
        solver.t_bound = time
        solver.status = 'running'
        while solver.status == 'running':
            message = solver.step()
            if solver.status == 'failed':
                raise RuntimeError(f's={solver.t:g}: {message}')
        sample(time, solver.y)
    return solver.y


def run(cells, stop, begin):
    m, initial, _, matching = build(cells, 'matched_quadratic')
    fast = FastRHS(m)
    raw = HERE/'omega_sweep'
    if begin == 32:
        checkpoint = raw/f'long_matched_N{cells}_s32.npz'
    else:
        checkpoint = HERE/'data'/f'weak_N{cells}_s{begin:g}.npz'
    restart = np.load(checkpoint)['state']
    launch_h = m.evaluate(initial)
    anchor = (json.loads((raw/f'long_matched_N{cells}.json').read_text())['rows'][-1]
              if begin == 32 else json.loads((HERE/'data'/f'weak_N{cells}_to{begin:g}.json').read_text())['rows'][-1])
    native_anchor = observe(m, begin, restart)
    assert anchor['exists'] == native_anchor['exists']
    if anchor['exists']:
        for key in ('signed_content','areal_radius'):
            assert abs(anchor[key]-native_anchor[key]) < 1e-12
    y8 = np.load(HERE/'strength_match'/f'matched_quadratic_N{cells}_s8.npz')['state']
    preflight = compare_rhs(m, fast, (initial, y8, restart))
    print(cells, 'RHS preflight', preflight, 'from', begin, 'to', stop, flush=True)
    rows, rhs_errors = [], [preflight]
    path = HERE/'data'/f'weak_N{cells}_to{stop:g}.json'

    def sample(time, state):
        d = m.evaluate(state)
        row = observe(m, time, state)
        row.update(central_chi=float(d['chi'][0]),
                   global_signed=float((d['N'][0]-d['N'][1]).sum()),
                   fixed_B_r1=float((d['N'][0]-d['N'][1])[m.r<1.].sum()),
                   energy_error=float(abs(d['H']-launch_h['H'])/max(1, abs(launch_h['H']))),
                   content_error=float(abs(d['N'].sum()-launch_h['N'].sum())/launch_h['N'].sum()))
        rows.append(row)
        if time % 5 == 0 or time == stop:
            rhs_errors.append(compare_rhs(m, fast, (state,)))
        if time % 10 == 0 or time == stop:
            write(path, dict(cells=cells, begin=begin, stop=time, target=stop, matching=matching,
                             fast_rhs_max_error=max(rhs_errors), rows=rows))
            np.savez_compressed(HERE/'data'/f'weak_N{cells}_s{time:g}.npz', state=state)
            print(cells, time, 'R', row.get('areal_radius'), 'B', row.get('signed_content'),
                  'central', row['central_chi'], 'H error', row['energy_error'], flush=True)

    sample(float(begin), restart)
    integrate(fast, np.asarray(restart, dtype=float), begin, stop, sample,
              cadence=.1 if stop <= 100 else .25)


def compare_short_trajectory(cells):
    m, initial, _, _ = build(cells, 'matched_quadratic')
    fast = FastRHS(m)
    start = np.load(HERE/'omega_sweep'/f'long_matched_N{cells}_s32.npz')['state']
    results = [integrate(rhs, y, 32, 33, lambda *_: None, native=native)
               for rhs, y, native in ((m.rhs, start, True), (fast, start.astype(float), False))]
    a, b = [observe(m, 33, y) for y in results]
    assert a['exists'] and b['exists']
    result = dict(cells=cells, state_max_abs_difference=float(np.max(abs(results[0]-results[1]))),
                  differences={k: abs(a[k]-b[k]) for k in ('signed_content','areal_radius')})
    assert max(result['differences'].values()) < 1e-7
    write(HERE/'data'/f'verification_N{cells}.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cells', type=int, choices=(256, 512))
    parser.add_argument('--begin', type=int, default=32)
    parser.add_argument('--stop', type=int, default=100)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        compare_short_trajectory(args.cells)
    else:
        run(args.cells, args.stop, args.begin)
