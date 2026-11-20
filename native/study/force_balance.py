"""Exact native phase-gradient momentum and log-current decomposition."""
from dataclasses import replace
import numpy as np
from common import LD
from buffer_model import model
from energy import decompose
from moving_budget import Ports
from directional_radial import sigmoid

MU_NAMES=['common_compression','quartic_bias','volume_coupling','retention_mass',
          'weighted_phase_exchange','legacy_exchange','C_amplitude','R_amplitude',
          'C_motion','R_motion','geometry_gradient']
FORCE_NAMES=MU_NAMES+['radial_metric_strain']
CURRENT_NAMES=FORCE_NAMES+['mobility_change','face_area_change','donor_density_change']

def build(cells,omega_p=.5,phase_offset=None):
    m,y=model(8,cells,True,.75,4)
    # Initial data, p0 and speed caps are preserved. Omega_p contributes no
    # phase-gradient flux and its initial chi^2 energy is zero analytically.
    m.p=replace(m.p,Omega_p=omega_p)
    m.original.p=replace(m.original.p,Omega_p=omega_p)
    m.base.p=replace(m.base.p,Omega_p=omega_p)
    if phase_offset is not None:
        q=m.unpack(y).copy()
        q[3]+=LD(phase_offset)-LD(np.pi)/2
        y=q.ravel()
    return m,y

def mu_channels(m,y):
    q=m.unpack(y);p=m.p;ed=decompose(m,y)
    nc,nr=ed['N']/ed['V'];chi=ed['chi'];ell=q[4]+2*q[5]
    signs=np.array([1,-1],dtype=LD)[:,None]
    root=np.sqrt(ed['N'][0]*ed['N'][1]);cos=np.cos(q[3]-q[2])
    Y=ed['N'][0]+ed['N'][1]-2*root*cos
    ratio=np.stack([np.sqrt(ed['N'][1]/ed['N'][0]),np.sqrt(ed['N'][0]/ed['N'][1])])
    mu=np.zeros((len(MU_NAMES),2,m.N),dtype=LD)
    mu[0]=p.g*(nc+nr-p.n_background)
    mu[1]=signs*(p.lam*chi**3-p.J)/p.n_star
    mu[2]=signs*(-p.k_volume*p.zeta*(ell-p.zeta*chi))/p.n_star
    mu[3]=signs*2*p.Omega_d*chi*m.V0/ed['V']
    mu[4]=signs*2*p.Omega_p*chi*Y/(m.cs**2*ed['V']*p.n_star)+p.Omega_p*(chi/m.cs)**2*(1-ratio*cos)
    mu[5]=p.Omega*(1-ratio*cos)
    for k in range(5):
        mu[6+k,:,:-1]+=ed['ge'][k,0,:2]
        mu[6+k,:,1:]+=ed['ge'][k,1,:2]
    return mu,ed

class ForceBalance:
    def __init__(self,m):
        self.m=m;self.face=int(np.argmin(abs(m.faces[1:-1]-1)))
        self.ids,self.centers,self.view=Ports(m).view((self.face,))
        self.center=int(self.centers[0]);self.sigma=np.array([-1,1],dtype=LD)
        self.force_size=2*len(FORCE_NAMES);self.current_size=2*len(CURRENT_NAMES)

    def evaluate(self,y,native=None):
        m=self.m;d=m.evaluate(y) if native is None else native;q=m.unpack(y);j=self.face;k=self.center
        mu,ed=mu_channels(self.view,q[:,self.ids].ravel())
        L=d['L'][j];p=m.p.eta*(q[2:4,j+1]-q[2:4,j])/L
        z=self.sigma*p/m.p0
        direct_mu=-m.p.eta*d['rhs'].reshape(8,m.N)[2:4,j:j+2]
        reconstruction=float(np.max(abs(mu.sum(axis=0)[:,k:k+2]-direct_mu)))
        lam_dot=(d['rhs'].reshape(8,m.N)[4,j]+d['rhs'].reshape(8,m.N)[4,j+1])/2
        force=-self.sigma[None,:]*(mu[:,:,k+1]-mu[:,:,k])/(m.p0*L)
        force=np.concatenate([force,(-z*lam_dot)[None,:]],axis=0)
        current=np.zeros((len(CURRENT_NAMES),2),dtype=LD)
        current[:len(FORCE_NAMES)]=(1-sigmoid(z))[None,:]*force
        qdot=d['rhs'].reshape(8,m.N);ell_dot=qdot[4]+2*qdot[5]
        current[-3]=-(1-d['mobility'][j])*(d['chi_dot'][j]+d['chi_dot'][j+1])/(2*m.cs)
        current[-2]=qdot[5,j]+qdot[5,j+1]
        current[-1]=np.array([qdot[0,j+1]-ell_dot[j+1],qdot[1,j]-ell_dot[j]],dtype=LD)
        return dict(force=force,current=current,z=z,p=p,flux=d['flux'][:,j],
                    speed=d['speed'][:,j],force_reconstruction=reconstruction,
                    mu_channels=mu[:,:,k:k+2],native=d)

    def ledgers(self,y,native=None):
        d=self.evaluate(y,native)
        return np.r_[d['force'].ravel(),d['current'].ravel()],d

    def unpack_ledgers(self,L):
        return L[:self.force_size].reshape(len(FORCE_NAMES),2),L[self.force_size:].reshape(len(CURRENT_NAMES),2)
