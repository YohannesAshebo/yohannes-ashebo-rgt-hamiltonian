from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine'));sys.path.insert(0,str(ROOT/'integrator'))
from directional_radial import DirectionalRadial,BASE,LD
from _extended_dop853 import ExtendedDOP853
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def fixed(m,y,d=None):
    d=m.evaluate(y) if d is None else d;q=y.reshape(8,m.N);f=d['rhs'].reshape(8,m.N)
    edge=int(np.argmin(abs(m.faces[1:-1]-1)));mask=m.r<1
    I=-d['flux'][0,edge];O=d['flux'][1,edge];X=np.sum(d['Gamma'][mask])
    C=np.sum(d['N'][0,mask]);R=np.sum(d['N'][1,mask]);L=d['L'][edge]
    p=m.p.eta*np.diff(q[2:4],axis=1)[:,edge]/L
    pdot=m.p.eta*np.diff(f[2:4],axis=1)[:,edge]/L-p*(f[4,edge]+f[4,edge+1])/2
    return dict(C=float(C),R=float(R),M=float(C+R),S=float(C-R),I=float(I),O=float(O),X=float(X),
        C_rate=float(I-X),R_rate=float(X-O),S_rate=float(I+O-2*X),M_rate=float(I-O),
        central_chi=float(d['chi'][0]),minimum_chi=float(np.min(d['chi'])),maximum_chi=float(np.max(d['chi'])),
        momentum_drive=pdot.astype(float).tolist(),geometry_acceleration_max=float(max(np.max(abs(f[6]/(m.p.I*m.V0))),np.max(abs(f[7]/(2*m.p.I*m.V0))))),
        geometry_rate_max=float(np.max(abs(f[4:6]))),relative_phase_rate_max=float(np.max(abs(f[3]-f[2]))),
        local_balance_error=float(np.max(abs(d['balance']))),
        fixed_budget_error=float(max(abs(np.sum(d['N'][0,mask]*f[0,mask])-I+X),abs(np.sum(d['N'][1,mask]*f[1,mask])-X+O))),
        core=m.core(y,d))
