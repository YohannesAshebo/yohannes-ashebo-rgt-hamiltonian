"""Read-only access to the published native solver, compiled in a temporary copy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from .data import RESOURCES,Trajectory
from types import MappingProxyType

_ACTIVE=False

def default_solver_directory():
    candidate=Path(__file__).resolve().parents[3]
    if (candidate/"native").is_dir() and (candidate/"static_scaling.py").is_file():return candidate
    raise ValueError("Pass solver_directory pointing to a checkout of yohannes-ashebo-rgt-hamiltonian")

class NativeHamiltonian:
    """Use one context per process; temporary imports/kernels are cleaned up.

    No files are generated in the source checkout. The radial coefficients,
    caps, initial preparation and canonical equations match the benchmark.
    Native geometry and number-phase fields are all retained.
    """
    def __init__(self,solver_directory=None,cells=128):
        self.directory=Path(solver_directory).resolve() if solver_directory is not None else default_solver_directory()
        if not isinstance(cells,int) or cells<2:raise ValueError("cells must be an integer >=2")
        self.cells=cells;self.case=None;self._temporary=None
    def __enter__(self):
        global _ACTIVE
        if _ACTIVE:raise RuntimeError("Native import contexts cannot overlap; use one per process")
        frozen=json.loads((RESOURCES/"BASELINE_SOURCES.json").read_text())
        for name,expected in frozen["files"].items():
            if not (name.startswith("native/") or name in ("static_scaling.py","audit_mechanics.py")):continue
            path=self.directory/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f"Native source differs from the validated baseline: {name}")
        self._oldpath=sys.path.copy();self._oldmodules=set(sys.modules);self._oldbytecode=sys.dont_write_bytecode
        native_names={path.stem for path in (self.directory/"native").rglob("*.py")}
        self._collisions={name:sys.modules.pop(name) for name in native_names if name in sys.modules}
        self._temporary=tempfile.TemporaryDirectory(prefix="counterflowbench_native_");_ACTIVE=True
        try:
            sys.dont_write_bytecode=True
            from . import _native_fixture
            self.fixture=_native_fixture.prepare_native(self.directory,self._temporary.name)
            self.case=_native_fixture.make_case(self.cells,self.fixture,heat_kappa=0)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*exc):
        global _ACTIVE
        self.case=None
        if self._temporary is not None:
            tmp=Path(self._temporary.name).resolve()
            for name,module in list(sys.modules.items()):
                file=getattr(module,"__file__",None)
                if file and Path(file).is_relative_to(tmp):sys.modules.pop(name,None)
            sys.modules.update(self._collisions)
            sys.path[:]=self._oldpath;sys.dont_write_bytecode=self._oldbytecode
            self._temporary.cleanup();self._temporary=None
        _ACTIVE=False
    def _require(self):
        if self.case is None:raise RuntimeError("Use NativeHamiltonian inside a with context")
    def _state(self,state_y):
        self._require();y=np.asarray(state_y,dtype=np.longdouble)
        if y.shape!=(self.cells,8) or not np.isfinite(y).all():raise ValueError(f"Expected finite state ({self.cells},8)")
        return y.T.ravel()
    def energy(self,state_y):
        self._require();return float(self.case.energy(self._state(state_y)))
    def conservative_rhs(self,state_y):
        y=self._state(state_y)
        return self.case.compiled(0,np.asarray(y,float)).reshape(8,self.cells).T.copy()
    def hybrid_rollout(self,initial_y,times,kappa,surrogate,max_step=.01,rtol=1e-9,atol=1e-11):
        """Couple a learned response to unchanged native conservative mechanics.

        This is an inference integrator, not a differentiable C-kernel wrapper.
        The surrogate receives only state and exact phase torque at each RHS
        evaluation; no native current, true mobility or future state is supplied.
        """
        self._require();self._state(initial_y);times=np.asarray(times,float);n=self.cells
        if times.ndim!=1 or len(times)<2 or times[0]!=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError("times must start at zero and strictly increase")
        if not np.isfinite(kappa) or kappa<0 or max_step<=0:raise ValueError("Invalid kappa or maximum step")
        from scipy.integrate import solve_ivp
        initial=np.r_[np.asarray(initial_y,float).T.ravel(),np.zeros(n)]
        def rhs(t,state):
            y=state[:8*n].reshape(8,n).T;con=self.conservative_rhs(y)
            D=np.exp(y[:,0])*con[:,0]-np.exp(y[:,1])*con[:,1]
            unit=surrogate.predict_unit(y[None],D[None])[0] if kappa else np.zeros((n,3))
            con[:,2:4]+=kappa*unit[:,:2]
            value=np.r_[con.T.ravel(),kappa*unit[:,2]]
            if not np.isfinite(value).all():raise FloatingPointError("Nonfinite hybrid prediction")
            return value
        sol=solve_ivp(rhs,(0,float(times[-1])),initial,t_eval=times,method="DOP853",rtol=rtol,atol=atol,max_step=max_step)
        if not sol.success:raise RuntimeError(sol.message)
        return dict(time=sol.t,state_y=sol.y[:8*n].T.reshape(len(times),8,n).transpose(0,2,1),heat_cells=sol.y[8*n:].T)
    def evaluate(self,state_y,kappa=100.):
        if not np.isfinite(kappa) or kappa<0:raise ValueError("kappa must be finite and >=0")
        y=self._state(state_y);n=self.cells;m=self.case.model;base=m.evaluate(y);extra=self.case.law.evaluate(y)
        N=np.exp(y[:2*n].reshape(2,n));g=m.unpack(base["gradient"])+extra["gradient"]
        gx=g.copy();gx[:2]/=N;con=self.case.law.add_rhs(base["rhs"].copy(),y).reshape(8,n)
        flux=base["flux"]+extra["extra_flux"];jf=np.maximum(0,np.minimum(-flux[0]/base["A"],flux[1]/base["A"]))
        j=(np.r_[0,jf]+np.r_[jf,0])/2;c=j/N.sum(axis=0);D=g[2]-g[3]
        unit=np.zeros((8,n),dtype=np.longdouble);unit[2]=-c*D;unit[3]=c*D
        complete_flux=np.zeros((n+1,2));complete_flux[1:-1]=np.asarray(flux.T,float)
        values=dict(state_y=np.asarray(state_y,float),grad_H_x=np.asarray(gx.T,float),rhs_conservative_y=np.asarray(con.T,float),
            rhs_dissipative_unit_y=np.asarray(unit.T,float),heat_cells_unit=np.asarray(c*D*D,float),unit_mobility=np.asarray(c,float),
            H=np.asarray(base["H"]+extra["energy"],float),total_inventory=np.asarray(N.sum(),float),complete_flux=complete_flux,
            gamma=np.asarray(base["Gamma"]+extra["Gamma"],float))
        if not all(np.isfinite(v).all() for v in values.values()):raise ValueError("Native evaluation is nonfinite on this state")
        values["rhs_y"]=values["rhs_conservative_y"]+kappa*values["rhs_dissipative_unit_y"];values["heat_cells"]=kappa*values["heat_cells_unit"]
        return values
    def generate(self,seed_group=0,kappa=100.,stop=10.,frames=21,output=None,max_step=.005):
        """Generate one new reference using the original native C RHS.

        Outputs use native model units. New choices of grid/horizon are not
        automatically covered by the shipped N=128 validation or checkpoint.
        """
        self._require()
        if output is not None:
            path=Path(output)
            if path.suffix!=".npz":raise ValueError("Reference output must end with .npz")
            if path.exists():raise FileExistsError(f"Refusing to replace {path}")
        if seed_group<0 or not isinstance(seed_group,int):raise ValueError("seed_group must be a nonnegative integer")
        if not np.isfinite(kappa) or kappa<0 or not np.isfinite(stop) or stop<=0 or not isinstance(frames,int) or frames<2 or max_step<=0:raise ValueError("Invalid reference generation settings")
        try:from scipy.integrate import solve_ivp
        except ImportError as error:raise ImportError("Install ashebo-counterflowbench[native] to integrate references") from error
        from ._native_fixture import make_case
        case=make_case(self.cells,self.fixture,heat_kappa=kappa);rng=np.random.default_rng(20261120+seed_group)
        coefficients=rng.normal(size=(8,3));r=np.asarray(case.model.r,float);x=r/8.;basis=np.stack([np.cos((i+1)*np.pi*x) for i in range(3)])
        perturb=coefficients@basis/np.sqrt(3);q=case.initial[:8*self.cells].reshape(8,self.cells).copy()
        q[:2]+=.01*perturb[:2];q[2:4]+=.01*perturb[2:4];q[4:6]+=.001*x*x*perturb[4:6];q[6:8]+=.001*case.model.V0*x*x*perturb[6:8]
        initial=np.r_[q.ravel(),np.zeros(5)];times=np.linspace(0,stop,frames)
        solution=solve_ivp(case.rhs,(0,stop),initial,t_eval=times,method="DOP853",rtol=2.5e-12,atol=2e-14,max_step=max_step)
        if not solution.success:raise RuntimeError(solution.message)
        states=solution.y[:8*self.cells].T.reshape(frames,8,self.cells).transpose(0,2,1);stack={}
        for y in states:
            label=self.evaluate(y,kappa)
            for name,value in label.items():
                if name not in ("rhs_y","heat_cells"):stack.setdefault(name,[]).append(value)
        values={name:np.asarray(v) for name,v in stack.items()};values.update(time=times,heat_ledgers=solution.y[8*self.cells:].T,
            reference_volume=np.asarray(case.model.V0,float),r=r,kappa=np.asarray(kappa),seed_group=np.asarray(seed_group),perturbation_coefficients=coefficients)
        trajectory=Trajectory(f"seed{seed_group:02d}_k{kappa:g}","generated",MappingProxyType(values))
        if output is not None:
            path=Path(output)
            if path.exists():raise FileExistsError(f"Refusing to replace {path}")
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values)
        return trajectory
