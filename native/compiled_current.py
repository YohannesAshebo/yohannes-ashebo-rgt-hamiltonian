"""Float64 C fast path, verified against the full Python chain rule."""
import ctypes as ct

import numpy as np

from continue_run import HERE


class CompiledCurrent:
    def __init__(self, base, law):
        self.base = base
        self.law = law
        self.lib = ct.CDLL(str(HERE/"current_rhs.so"))
        self.call = self.lib.add_current_rhs
        self.call.argtypes = [ct.c_int] + [ct.c_void_p]*3 + [ct.c_double]*2 + [ct.c_void_p]*2
        self.call.restype = None
        self.work = np.empty(14*base.n)
        # The base scratch holds contents, volume, and chi after the base call.
        base.pointers = (*base.pointers[:-1], self.work.ctypes.data)

    def __call__(self, time, state):
        state = np.ascontiguousarray(state, dtype=float)
        out = self.base(time, state)
        self.call(self.base.n, state.ctypes.data, self.base.pointers[0],
                  self.base.pointers[-2], self.law.omega, self.law.j0,
                  self.work.ctypes.data, out.ctypes.data)
        return out

    def compare(self, state):
        y = np.asarray(state, dtype=float)
        slow = self.law.add_rhs(self.base(0., y), y)
        fast = self(0., y)
        discrepancy = float(np.max(abs(slow-fast))/max(1., np.max(abs(slow))))
        assert discrepancy < 1.e-11, discrepancy
        return discrepancy
