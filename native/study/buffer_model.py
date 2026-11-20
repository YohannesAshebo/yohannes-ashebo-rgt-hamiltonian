"""Conservative outer-flow preparation; no damping or new bulk terms."""
import numpy as np
from common import LD
from domain_reference import model as raw_model
from directional_radial import sigmoid

def taper(x):
    # Stable form near the zero-current endpoint, with C2 joins.
    z=np.clip(1-np.asarray(x,dtype=LD),0,1)
    return z**3*(10-15*z+6*z*z)

def phase_profile(radius,cells,width=.75,order=64,taper_end=None):
    r=(np.arange(cells,dtype=LD)+LD('.5'))*LD(radius/cells)
    end=LD(radius if taper_end is None else taper_end)
    if end>radius:raise ValueError('Taper endpoint exceeds the computational domain')
    start=end-LD(width);core=LD('.15')*(np.sqrt(r*r+LD('.25')**2)-LD('.25'))
    outer=r>start
    if not np.any(outer):return core
    nodes,weights=np.polynomial.legendre.leggauss(order);nodes=np.asarray(nodes,dtype=LD);weights=np.asarray(weights,dtype=LD)
    distance=np.minimum(r[outer],end)-start;s=start+distance[:,None]*(nodes[None,:]+1)/2
    derivative=LD('.15')*s/np.sqrt(s*s+LD('.25')**2)*taper((s-start)/LD(width))
    base=LD('.15')*(np.sqrt(start*start+LD('.25')**2)-LD('.25'))
    core[outer]=base+distance/2*np.sum(derivative*weights,axis=1)
    return core

def model(radius,cells,buffered=True,width=.75,taper_end=None):
    m,y=raw_model(radius,cells)
    if not buffered:return m,y
    end=radius if taper_end is None else taper_end
    if end-width<=3.125:raise ValueError('Buffer must leave all measurement-region neighbors unchanged')
    q=m.unpack(y).copy();phase=phase_profile(radius,cells,width,taper_end=end)
    q[2]=-phase;q[3]=phase+LD(np.pi)/2;y=q.ravel()
    native=m.original.terms_and_gradient(m.to_original(y))
    pC=m.p.eta*np.diff(q[2])/m.dr;pR=m.p.eta*np.diff(q[3])/m.dr
    j=int(np.argmin(abs(m.faces[1:-1]-1)));m.p0=abs(pC[j])
    rho=np.exp(q[:2])/m.V0;chi=(rho[0]-rho[1])/m.p.n_star;mob=sigmoid(-(chi[:-1]+chi[1:])/(2*m.cs))
    # Exactly the existing calibration rule, now applied to the tapered phases.
    m.vmax=np.stack([abs(native['flux'][0])/(m.A0*rho[0,1:]*mob*sigmoid(-pC/m.p0)),abs(native['flux'][1])/(m.A0*rho[1,:-1]*mob*sigmoid(pR/m.p0))])
    return m,y
