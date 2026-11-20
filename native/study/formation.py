"""Unseeded equal-density initial data; native equations are unchanged."""
import numpy as np
from common import LD,DirectionalRadial
from directional_radial import sigmoid
from energy import decompose,quantile_radius,region_energy,powers

def initial(m,seed=0):
    r=np.asarray(m.r,dtype=LD);V0=np.asarray(m.V0,dtype=LD)
    shape=np.exp(-r**2/LD('.25')**2);shape-=np.sum(shape*V0)/np.sum(V0)
    chi=LD(seed)*shape
    phase=LD('.15')*(np.sqrt(r*r+LD('.25')**2)-LD('.25'))
    z=np.zeros(m.N,dtype=LD)
    return m.pack(np.log(V0*(1+chi)/2),np.log(V0*(1-chi)/2),-phase,phase+LD(np.pi)/2,z,z,z,z)

def calibrated(n):
    m=DirectionalRadial(n);y=initial(m);q=m.unpack(y);native=m.original.terms_and_gradient(m.to_original(y))
    pC=m.p.eta*np.diff(q[2])/m.dr;pR=m.p.eta*np.diff(q[3])/m.dr
    j=int(np.argmin(abs(m.faces[1:-1]-1)));m.p0=abs(pC[j])
    rho=np.exp(q[:2])/m.V0;chi=(rho[0]-rho[1])/m.p.n_star
    mob=sigmoid(-(chi[:-1]+chi[1:])/(2*m.cs))
    m.vmax=np.stack([abs(native['flux'][0])/(m.A0*rho[0,1:]*mob*sigmoid(-pC/m.p0)),abs(native['flux'][1])/(m.A0*rho[1,:-1]*mob*sigmoid(pR/m.p0))])
    return m,y,native['flux']

def contrast(m,d):
    shell=(m.r>=2)&(m.r<3);ref=np.sum(d['cell'][shell])/np.sum(m.V0[shell])
    excess=d['cell']-ref*m.V0
    return excess,ref

def candidate(m,d,H0):
    excess,ref=contrast(m,d);density=excess/m.V0;result=dict(exists=False,coordinate_radius=None,areal_radius=None,volume_radius=None,energy=None,reason='central energy contrast not positive')
    if density[0]<=0:return result
    crossings=np.flatnonzero((density[:-1]>0)&(density[1:]<=0))
    if not len(crossings):result['reason']='no first zero within domain';return result
    j=int(crossings[0]);f=density[j]/(density[j]-density[j+1]);rb=m.r[j]+f*m.dr
    if rb>=1:result['reason']='first zero outside r<1';return result
    lo=m.faces[:-1];hi=m.faces[1:];upper=np.minimum(hi,rb);frac=np.where(upper>lo,(upper**3-lo**3)/(hi**3-lo**3),0)
    E=np.sum(frac*excess)
    if E<=LD('1e-6')*abs(H0):result['reason']='connected excess below fixed threshold';return result
    return dict(exists=True,coordinate_radius=float(rb),volume_radius=float((3*np.sum(frac*d['V'])/(4*np.pi))**(LD(1)/3)),energy=float(E),areal_radius=None,reason='central positive connected energy excess, threshold passed')

def readout(m,t,y,H0):
    d=decompose(m,y);excess,ref=contrast(m,d);pos=np.maximum(excess,0);neg=np.minimum(excess,0);mask=m.r<1
    core=candidate(m,d,H0);power=powers(m,d);j=int(np.argmin(abs(m.faces[1:-1]-1)))
    if core['exists']:
        rb=LD(core['coordinate_radius']);k=int(np.searchsorted(m.r,rb)-1);k=max(0,min(k,m.N-2));f=(rb-m.r[k])/m.dr
        beta=m.unpack(y)[5];core['areal_radius']=float(rb*np.exp((1-f)*beta[k]+f*beta[k+1]))
    r50,rp50=quantile_radius(m,pos,d['V'],.5);r90,rp90=quantile_radius(m,pos,d['V'],.9)
    out=dict(time=float(t),H=float(d['H']),Hcore=float(region_energy(m,d)),shell_reference=float(ref),signed_core_excess=float(np.sum(excess[mask])),
        central_energy_contrast=float(excess[0]/m.V0[0]),positive_excess=float(np.sum(pos)),negative_excess=float(np.sum(neg)),signed_global_excess=float(np.sum(excess)),
        positive_core_fraction=None if pos.sum()==0 else float(pos[mask].sum()/pos.sum()),R50=r50,R90=r90,proper_R50=rp50,proper_R90=rp90,
        central_chi=float(d['chi'][0]),minimum_chi=float(np.min(d['chi'])),maximum_chi=float(np.max(d['chi'])),C_content_in=float(-d['content_flux'][0,j]),R_content_out=float(d['content_flux'][1,j]),
        core_boundary_power=float(power.sum()),boundary_power_components=power.astype(float).tolist(),candidate=core)
    return out,d,excess
