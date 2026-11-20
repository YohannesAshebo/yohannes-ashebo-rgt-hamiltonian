"""v0.5 structural diagnostic: same spherical discretization and BC as v0.4.

Canonical c,r are integrated-cell amplitudes. Omega is the legacy exchange;
Omega_d,Omega_p are the new channels. Signed zeta is the v0.4 convention:
U_volume=kv/2*(alpha+2*beta-zeta*chi)^2. Thus B uses zeta=-0.2,
equivalent to the specification's plus-form zeta=+0.2.

No extra drive, source, clipping, density floor or state renormalization.
The inherited 1e-14 threshold only guards divisions in velocity READOUTS;
it is never applied to the Hamiltonian or evolved state.
"""
from dataclasses import dataclass
import numpy as np
from candidate_v04 import Model as ModelV04, Parameters as ParametersV04

@dataclass(frozen=True)
class Parameters(ParametersV04):
    Omega_d: float=0.
    Omega_p: float=0.

class Model(ModelV04):
    def __init__(self,cells,parameters=Parameters()):
        super().__init__(cells,parameters)

    def additions(self,y,d):
        p=self.p
        c,r,*_=self.unpack(y)
        w=d['V']/self.V0
        chi=(d['nc']-d['nr'])/p.n_star
        chi_star=(p.J/p.lam)**(1/3)
        Y=abs(c-r)**2
        f=chi**2/chi_star**2
        Ed=self.V0*p.Omega_d*p.n_star*chi**2
        Ep=p.Omega_p*f*Y
        return w,chi,chi_star,Y,f,Ed,Ep

    def terms_and_gradient(self,y):
        d=super().terms_and_gradient(y)
        p=self.p;c,r,*_=self.unpack(y)
        w,chi,cs,Y,f,Ed,Ep=self.additions(y,d)
        diagonal=2*p.Omega_d*chi/w+2*p.Omega_p*chi*Y/(cs**2*d['V']*p.n_star)
        d['gc']+=diagonal*c+p.Omega_p*f*(c-r)
        d['gr']-=diagonal*r
        d['gr']+=p.Omega_p*f*(r-c)
        # Both additions are OUTSIDE w U and depend on w through chi.
        # At fixed canonical modes: delta_Q E=-2 E tr(delta Q).
        d['ga']-=2*(Ed+Ep)
        d['gb']-=4*(Ed+Ep)
        d['energies']['retention_mass']=np.sum(Ed)
        d['energies']['weighted_phase_exchange']=np.sum(Ep)
        d['H']=sum(d['energies'].values())
        d['retention_mass_cell']=Ed;d['phase_exchange_cell']=Ep
        return d

    def local_rates(self,y,d=None):
        if d is None:d=self.terms_and_gradient(y)
        p=self.p;c,r,*_=self.unpack(y)
        divergence=np.zeros((2,self.N))
        divergence[:,:-1]+=d['flux'];divergence[:,1:]-=d['flux']
        chi=(d['nc']-d['nr'])/p.n_star
        cs=(p.J/p.lam)**(1/3)
        overlap=np.imag(c.conj()*r)
        gamma_legacy=2*p.Omega/p.eta*overlap
        gamma_p=2*p.Omega_p/p.eta*(chi/cs)**2*overlap
        gamma_d=np.zeros(self.N) # phase-independent Ed has no direct conversion
        ell_rate=d['a_rate']+2*d['b_rate']
        t_tr=-(divergence[0]-divergence[1])/(d['V']*p.n_star)
        t_legacy=-2*gamma_legacy/(d['V']*p.n_star)
        t_p=-2*gamma_p/(d['V']*p.n_star)
        t_d=np.zeros(self.N)
        t_vol=-ell_rate*chi
        components=np.array([t_tr,t_legacy+t_p+t_d,t_vol])
        B=np.sum(np.maximum(components,0),axis=0)
        D=np.sum(np.maximum(-components,0),axis=0)
        return dict(divergence=divergence,Gamma=gamma_legacy+gamma_p,
            ell_rate=ell_rate,chi=chi,chi_rate=np.sum(components,axis=0),
            formation=B,restoration=D,activity=B+D,components=components,
            t_tr=t_tr,t_ex_legacy=t_legacy,t_ex_mass=t_d,t_ex_phase=t_p,t_vol=t_vol)

    def noether(self,y):
        ans=super().noether(y)
        d=self.terms_and_gradient(y)
        extra=(d['retention_mass_cell']+d['phase_exchange_cell'])/self.V0
        ans['energy']+=extra
        ans['Mrr']-=extra;ans['Mtt']-=extra
        ans['pressure']=(ans['Mrr']+2*ans['Mtt']-ans['Arr'])/3
        ans['tau_r']=ans['Arr']+ans['pressure']-ans['Mrr']
        ans['tau_t']=ans['pressure']-ans['Mtt']
        return ans
