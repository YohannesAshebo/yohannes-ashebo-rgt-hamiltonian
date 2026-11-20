"""Explicit spherical Hamiltonian trial, not a unique or validated RGT law.

The two complex modes are an added motion closure. Native amplitudes are
proper content-density square roots; geometry is an independent spatial metric.
All dynamics and process-record rates come from one declared Hamiltonian.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Parameters:
    eta: float=1.
    K: float=.05
    Omega: float=.5
    g: float=1.
    lam: float=.4
    J: float=.01
    k_volume: float=.3
    k_shear: float=.2
    zeta: float=.2
    I: float=2.
    k_geometry: float=.02
    n_background: float=1.
    n_star: float=1.
    domain_radius: float=4.


class Model:
    def __init__(self,cells,parameters=Parameters()):
        self.N=cells
        self.p=parameters
        self.dr=parameters.domain_radius/cells
        self.faces=np.arange(cells+1)*self.dr
        self.r=(np.arange(cells)+.5)*self.dr
        self.V0=4*np.pi/3*np.diff(self.faces**3)
        self.edge_weight=4*np.pi*self.faces[1:-1]**2/self.dr

    def unpack(self,y):
        a=np.asarray(y).reshape(10,self.N)
        return a[0]+1j*a[1],a[2]+1j*a[3],a[4],a[5],a[6],a[7],a[8],a[9]

    def pack(self,c,r,alpha,beta,Pa,Pb,Cf,Cr):
        return np.concatenate([c.real,c.imag,r.real,r.imag,alpha,beta,Pa,Pb,Cf,Cr])

    def initial(self,verification_geometry=False):
        r=self.r;p=self.p
        n=1+.2*np.exp(-r*r)
        chi=.18*np.exp(-r*r/.9**2)-.05
        alpha=np.zeros(self.N);beta=np.zeros(self.N)
        Pa=np.zeros(self.N);Pb=np.zeros(self.N)
        if verification_geometry:
            alpha=.02*np.exp(-r*r/2)
            beta=alpha+.005*r*r*np.exp(-r*r/2)
            Pa=.01*self.V0*np.exp(-r*r/2)
            Pb=.02*self.V0*np.exp(-r*r/2)
        V=self.V0*np.exp(alpha+2*beta)
        f=r*r/2-r**4/(2*p.domain_radius**2)+r**6/(6*p.domain_radius**4)
        c=np.sqrt(V*(n+p.n_star*chi)/2)*np.exp(-.15j*f)
        out=np.sqrt(V*(n-p.n_star*chi)/2)*np.exp(1j*(.15*f+np.pi/2))
        return self.pack(c,out,alpha,beta,Pa,Pb,np.zeros(self.N),np.zeros(self.N))

    def potential(self,nc,nr,alpha,beta):
        p=self.p
        n=nc+nr;chi=(nc-nr)/p.n_star
        ell=alpha+2*beta;v=ell-p.zeta*chi;d=alpha-beta
        offset=.75*p.J*(p.J/p.lam)**(1/3)
        U=p.g/2*(n-p.n_background)**2+p.lam/4*chi**4-p.J*chi+offset+\
          p.k_volume/2*v*v+p.k_shear/2*d*d
        Un=p.g*(n-p.n_background)
        Uchi=p.lam*chi**3-p.J-p.k_volume*p.zeta*v
        muc=Un+Uchi/p.n_star;mur=Un-Uchi/p.n_star
        Ua=p.k_volume*v+p.k_shear*d
        Ub=2*p.k_volume*v-p.k_shear*d
        pressure=nc*muc+nr*mur-U
        return U,muc,mur,Ua,Ub,pressure

    def terms_and_gradient(self,y):
        p=self.p
        c,out,alpha,beta,Pa,Pb,Cf,Cr=self.unpack(y)
        ell=alpha+2*beta
        V=self.V0*np.exp(ell)
        phi=np.array([c,out])/np.sqrt(V)
        nc,nr=np.abs(phi)**2
        U,muc,mur,Ua,Ub,pressure=self.potential(nc,nr,alpha,beta)
        gc=muc*c+p.Omega*(c-out)
        gr=mur*out+p.Omega*(out-c)
        ga=V*(-pressure+Ua)
        gb=V*(-2*pressure+Ub)
        a_rate=Pa/(p.I*self.V0)
        b_rate=Pb/(2*p.I*self.V0)
        w=self.edge_weight*np.exp(((-alpha+2*beta)[:-1]+(-alpha+2*beta)[1:])/2)
        diff=phi[:,:-1]-phi[:,1:]
        square=np.sum(np.abs(diff)**2,axis=0)
        field_edge=p.K/2*w*square
        flux=p.K/p.eta*w*np.imag(phi[:,:-1].conj()*phi[:,1:])
        for mode,gradient in enumerate((gc,gr)):
            edge=p.K/2*w*diff[mode]
            gradient[:-1]+=edge/np.sqrt(V[:-1])
            gradient[1:]-=edge/np.sqrt(V[1:])
        zi=np.sum(np.real(diff.conj()*phi[:,:-1]),axis=0)
        zj=np.sum(np.real((-diff).conj()*phi[:,1:]),axis=0)
        ga[:-1]+=p.K/2*w*(-.5*square-zi)
        ga[1:]+=p.K/2*w*(-.5*square-zj)
        gb[:-1]+=p.K/2*w*(square-2*zi)
        gb[1:]+=p.K/2*w*(square-2*zj)

        da=alpha[:-1]-alpha[1:];db=beta[:-1]-beta[1:]
        ga_edge=p.k_geometry*self.edge_weight*da
        gb_edge=2*p.k_geometry*self.edge_weight*db
        ga[:-1]+=ga_edge;ga[1:]-=ga_edge
        gb[:-1]+=gb_edge;gb[1:]-=gb_edge
        angular_force=4*p.k_geometry*self.V0*(alpha-beta)/self.r**2
        ga+=angular_force;gb-=angular_force
        energies=dict(
            geometry_kinetic=np.sum(Pa*Pa/(2*p.I*self.V0)+Pb*Pb/(4*p.I*self.V0)),
            common_local=np.sum(V*U),
            reciprocal_exchange=p.Omega*np.sum(np.abs(c-out)**2),
            field_gradient=np.sum(field_edge),
            geometry_gradient=p.k_geometry/2*np.sum(self.edge_weight*(da*da+2*db*db))+\
                              2*p.k_geometry*np.sum(self.V0*(alpha-beta)**2/self.r**2))
        return dict(H=sum(energies.values()),energies=energies,gc=gc,gr=gr,
                    ga=ga,gb=gb,a_rate=a_rate,b_rate=b_rate,V=V,phi=phi,
                    nc=nc,nr=nr,U=U,pressure_local=pressure,flux=flux)

    def hamiltonian(self,y):
        return self.terms_and_gradient(y)['H']

    def real_gradient(self,y):
        d=self.terms_and_gradient(y)
        return self.pack(2*d['gc'],2*d['gr'],d['ga'],d['gb'],
                         d['a_rate'],d['b_rate'],np.zeros(self.N),np.zeros(self.N))

    def local_rates(self,y,d=None):
        if d is None:d=self.terms_and_gradient(y)
        p=self.p
        c,out,alpha,beta,Pa,Pb,Cf,Cr=self.unpack(y)
        divergence=np.zeros((2,self.N))
        divergence[:,:-1]+=d['flux']
        divergence[:,1:]-=d['flux']
        Gamma=2*p.Omega/p.eta*np.imag(c.conj()*out)
        ell_rate=d['a_rate']+2*d['b_rate']
        chi=(d['nc']-d['nr'])/p.n_star
        components=np.array([
            -(divergence[0]-divergence[1])/(d['V']*p.n_star),
            -2*Gamma/(d['V']*p.n_star),
            -ell_rate*chi])
        B=np.sum(np.maximum(components,0),axis=0)
        D=np.sum(np.maximum(-components,0),axis=0)
        return dict(divergence=divergence,Gamma=Gamma,ell_rate=ell_rate,
                    chi=chi,chi_rate=np.sum(components,axis=0),
                    formation=B,restoration=D,activity=B+D,components=components)

    def rhs(self,t,y):
        d=self.terms_and_gradient(y)
        rates=self.local_rates(y,d)
        return self.pack(-1j*d['gc']/self.p.eta,-1j*d['gr']/self.p.eta,
                         d['a_rate'],d['b_rate'],-d['ga'],-d['gb'],
                         rates['formation'],rates['restoration'])

    def D(self,a):
        return np.gradient(a,self.r,axis=-1,edge_order=2)

    def div(self,a):
        return self.D(self.r*self.r*a)/(self.r*self.r)

    def noether(self,y):
        """Continuum Noether readouts evaluated on the numerical fields.

        Densities and fluxes here are per reference volume/area. They are not
        automatically SI mass/stress observations or a spacetime gravity law.
        """
        p=self.p;d=self.terms_and_gradient(y)
        c,out,alpha,beta,Pa,Pb,Cf,Cr=self.unpack(y)
        dc,dr,adot,bdot,*_=self.unpack(self.rhs(0,y))
        ell=alpha+2*beta;A=np.exp(-alpha+2*beta)
        psi=np.array([c,out])/np.sqrt(self.V0)
        psi_rate=np.array([dc,dr])/np.sqrt(self.V0)
        phi=d['phi'];phir=self.D(phi)
        phi_rate=np.array([dc,dr])/np.sqrt(d['V'])-.5*(adot+2*bdot)*phi
        ar=self.D(alpha);br=self.D(beta)
        T=Pa*Pa/(2*p.I*self.V0**2)+Pb*Pb/(4*p.I*self.V0**2)
        field_grad=p.K/2*A*np.sum(np.abs(phir)**2,axis=0)
        exchange=p.Omega*np.abs(psi[0]-psi[1])**2
        geometry_grad=p.k_geometry/2*(ar*ar+2*br*br+4*(alpha-beta)**2/self.r**2)
        potential=field_grad+np.exp(ell)*d['U']+exchange+geometry_grad
        L=-p.eta*np.sum(np.imag(psi.conj()*psi_rate),axis=0)+T-potential
        momentum=p.eta*np.sum(np.imag(psi.conj()*self.D(psi)),axis=0)-\
                 Pa/self.V0*ar-Pb/self.V0*br
        Mrr=p.K*A*np.sum(np.abs(phir)**2,axis=0)+p.k_geometry*(ar*ar+2*br*br)+L
        Mtt=2*p.k_geometry*(alpha-beta)**2/self.r**2+L
        energy=T+potential
        energy_flux=-p.K*A*np.sum(np.real(phir.conj()*phi_rate),axis=0)-\
                    p.k_geometry*(ar*adot+2*br*bdot)
        n=np.abs(phi)**2
        phase_grad=np.divide(np.imag(phi.conj()*phir),n,
                             out=np.zeros_like(n),where=n>1e-14)
        u=p.K/p.eta*np.exp(-alpha)*phase_grad
        Arr=p.K*A*np.sum(n*phase_grad**2,axis=0)
        pressure=(Mrr+2*Mtt-Arr)/3
        tau_r=Arr+pressure-Mrr;tau_t=pressure-Mtt
        areal=self.r*np.exp(beta)
        B_r=self.D(areal);B_rr=self.D(B_r)
        curvature=2*(1-np.exp(-2*alpha)*B_r**2)/areal**2-\
                  4*np.exp(-2*alpha)*(B_rr-ar*B_r)/areal
        return dict(momentum=momentum,Mrr=Mrr,Mtt=Mtt,energy=energy,
                    energy_flux=energy_flux,pressure=pressure,tau_r=tau_r,
                    tau_t=tau_t,Arr=Arr,mode_velocity=u,curvature=curvature)

    def motion(self,y,d=None):
        if d is None:d=self.terms_and_gradient(y)
        _,_,alpha,beta,*_=self.unpack(y)
        area=4*np.pi*self.faces[1:-1]**2*np.exp(beta[:-1]+beta[1:])
        mode_density=np.sqrt(np.array([d['nc'],d['nr']])[:,:-1]*
                             np.array([d['nc'],d['nr']])[:,1:])
        velocity=np.divide(d['flux'],area*mode_density,
                           out=np.zeros_like(mode_density),where=mode_density>1e-14)
        inward=np.sum(np.maximum(-d['flux'],0),axis=0)
        outward=np.sum(np.maximum(d['flux'],0),axis=0)
        nin=np.sum(np.where(d['flux']<0,mode_density,0),axis=0)
        nout=np.sum(np.where(d['flux']>0,mode_density,0),axis=0)
        uin=np.divide(inward,area*nin,out=np.zeros_like(inward),where=nin>1e-14)
        uout=np.divide(outward,area*nout,out=np.zeros_like(outward),where=nout>1e-14)
        total=np.sum(np.abs(d['flux']))
        wrong=np.sum(np.maximum(d['flux'][0],0)+np.maximum(-d['flux'][1],0))
        return dict(mode_velocity=velocity,inward_current=inward/area,
                    outward_current=outward/area,inward_speed=uin,outward_speed=uout,
                    native_direction_mismatch=wrong/total if total>0 else 0.)

    def core(self,y,d=None):
        if d is None:d=self.terms_and_gradient(y)
        chi=(d['nc']-d['nr'])/self.p.n_star
        _,_,alpha,beta,*_=self.unpack(y)
        if chi[0]<=0:
            return dict(exists=False,reason='central retention is not positive',radius=np.nan,coordinate_radius=np.nan,radius_rate=np.nan)
        idx=np.flatnonzero((chi[:-1]>0)&(chi[1:]<=0))
        if not len(idx):
            return dict(exists=False,reason='no finite equality edge inside domain',radius=np.nan,coordinate_radius=np.nan,radius_rate=np.nan)
        i=idx[0];fraction=chi[i]/(chi[i]-chi[i+1])
        rb=self.r[i]+fraction*self.dr
        beta_b=np.interp(rb,self.r,beta)
        rates=self.local_rates(y,d)
        gradient=np.interp(rb,self.r,self.D(chi))
        rb_rate=-np.interp(rb,self.r,rates['chi_rate'])/gradient
        R_rate=np.exp(beta_b)*((1+rb*np.interp(rb,self.r,self.D(beta)))*rb_rate+
                              rb*np.interp(rb,self.r,d['b_rate']))
        return dict(exists=True,reason='connected positive core',
                    coordinate_radius=rb,radius=rb*np.exp(beta_b),radius_rate=R_rate)
