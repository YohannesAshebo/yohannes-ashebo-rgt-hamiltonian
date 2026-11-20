"""Experimental radial C-inward/R-outward Hamiltonian motion law.

Physical source constants/local energies are inherited unchanged. The old
phase-gradient energy is replaced, retaining its amplitude-gradient part.
State coordinates are log contents, unwrapped phases, metric and momenta.
Log contents ensure positive evaluation stages without clipping; they are a
coordinate transformation, not a canonical change to the stated equations.
This candidate is not a derived or validated physical RGT law.
"""
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'radial_original'))
from hamiltonian_v05 import Model as Original,Parameters
LD=np.longdouble
BASE=Parameters(eta=1.,K=.05,Omega=0.,g=1.,lam=.4,J=.01,k_volume=.3,k_shear=.2,zeta=-.2,I=2.,k_geometry=.02,n_background=1.,n_star=1.,domain_radius=4.,Omega_d=.007694891760045136,Omega_p=.5)

def sigmoid(x):
    a=np.exp(-abs(x));return np.where(x>=0,1/(1+a),a/(1+a))

def softplus(x):return np.maximum(x,0)+np.log1p(np.exp(-abs(x)))

class DirectionalRadial:
    def __init__(self,cells,kind='candidate',parameters=BASE):
        if kind not in ['original','no_phase','candidate','constant_mobility']:raise ValueError(kind)
        self.N=cells;self.kind=kind;self.p=replace(parameters,Omega_p=0.) if kind=='no_phase' else parameters
        self.original=Original(cells,self.p)
        self.base=Original(cells,replace(self.p,K=0.)) if kind in ['candidate','constant_mobility'] else self.original
        self.r=self.original.r;self.faces=self.original.faces;self.V0=self.original.V0;self.dr=self.original.dr
        self.A0=4*np.pi*self.faces[1:-1]**2;self.cs=(self.p.J/self.p.lam)**(1/3)
        initial=self.initial();uC,uR,tC,tR,a,b,*_=self.unpack(initial)
        state=self.to_original(initial);d=self.original.terms_and_gradient(state)
        pC=self.p.eta*np.diff(tC)/self.dr;pR=self.p.eta*np.diff(tR)/self.dr
        edge=int(np.argmin(abs(self.faces[1:-1]-1.)));self.p0=abs(pC[edge])
        if not self.p0>0:raise ValueError('Calibration requires nonzero initial reference momentum')
        nc=np.exp(uC)/self.V0;nr=np.exp(uR)/self.V0;chi=(nc-nr)/self.p.n_star
        m=sigmoid(-.5*(chi[:-1]+chi[1:])/self.cs) if kind=='candidate' else np.ones(cells-1,dtype=LD)
        self.vmax=np.stack([abs(d['flux'][0])/(self.A0*nc[1:]*m*sigmoid(-pC/self.p0)),abs(d['flux'][1])/(self.A0*nr[:-1]*m*sigmoid(pR/self.p0))])
        self.initial_source_flux=d['flux'].copy()

    def pack(self,*values):return np.concatenate(values).astype(LD)
    def unpack(self,y):return np.asarray(y,dtype=LD).reshape(8,self.N)

    def initial(self,verification_geometry=False):
        y=self.original.initial(verification_geometry).astype(LD)
        c,r,a,b,pa,pb,*_=self.original.unpack(y)
        return self.pack(np.log(abs(c)**2),np.log(abs(r)**2),np.angle(c),np.angle(r),a,b,pa,pb)

    def to_original(self,y):
        uc,ur,tc,tr,a,b,pa,pb=self.unpack(y);Nc=np.exp(uc);Nr=np.exp(ur)
        if not np.all(np.isfinite(Nc+Nr)) or np.any(Nc<=0) or np.any(Nr<=0):raise FloatingPointError('Log-content chart overflow/underflow; no floor applied')
        c=np.sqrt(Nc)*np.exp(1j*tc);r=np.sqrt(Nr)*np.exp(1j*tr);z=np.zeros(self.N,dtype=LD)
        return self.original.pack(c,r,a,b,pa,pb,z,z)

    def evaluate(self,y):
        p=self.p;uc,ur,tc,tr,a,b,pa,pb=self.unpack(y);Nc=np.exp(uc);Nr=np.exp(ur)
        source=self.to_original(y);d=self.base.terms_and_gradient(source);c,r,*_=self.original.unpack(source)
        gN=np.stack([np.real(d['gc']/c),np.real(d['gr']/r)])
        gT=2*np.stack([np.imag(c.conj()*d['gc']),np.imag(r.conj()*d['gr'])])
        ga=d['ga'].copy();gb=d['gb'].copy();V=d['V'];chi=(d['nc']-d['nr'])/p.n_star
        rho=np.stack([Nc,Nr])/V;amplitude=np.sqrt(rho)
        energies={k:v for k,v in d['energies'].items()};flux=d['flux'].copy();motion_energy=LD(0);amp_energy=LD(0)
        A=self.A0*np.exp(b[:-1]+b[1:]);L=self.dr*np.exp(.5*(a[:-1]+a[1:]));m=np.ones(self.N-1,dtype=LD);speed=None
        if self.kind in ['candidate','constant_mobility']:
            # The K=0 base excludes the entire old matter gradient. Restore
            # only its phase-independent amplitude-gradient energy.
            diff=amplitude[:,:-1]-amplitude[:,1:];w=A/L;square=np.sum(diff**2,axis=0)
            amp_energy=p.K/2*np.sum(w*square)
            edge=p.K/2*w[None,:]*diff
            gN[:,:-1]+=edge/(V[:-1]*amplitude[:,:-1]);gN[:,1:]-=edge/(V[1:]*amplitude[:,1:])
            left=np.sum(diff*amplitude[:,:-1],axis=0);right=np.sum(-diff*amplitude[:,1:],axis=0)
            ga[:-1]+=p.K/2*w*(-.5*square-left);ga[1:]+=p.K/2*w*(-.5*square-right)
            gb[:-1]+=p.K/2*w*(square-2*left);gb[1:]+=p.K/2*w*(square-2*right)
            phase=np.stack([tc,tr]);momentum=p.eta*np.diff(phase,axis=1)/L
            signs=np.array([-1.,1.],dtype=LD)[:,None];z=signs*momentum/self.p0
            eps=self.vmax*self.p0*softplus(z);velocity=signs*self.vmax*sigmoid(z)
            if self.kind=='candidate':m=sigmoid(-.5*(chi[:-1]+chi[1:])/self.cs)
            dm=-m*(1-m)/self.cs if self.kind=='candidate' else np.zeros_like(m)
            donors=np.stack([rho[0,1:],rho[1,:-1]])
            B=A*L*donors;T0=np.sum(B*eps,axis=0);motion_energy=np.sum(m*T0)
            flux=m[None,:]*A[None,:]*donors*velocity;speed=m[None,:]*velocity
            gT[:,:-1]-=p.eta*flux;gT[:,1:]+=p.eta*flux
            gN[0,1:]+=m*A*L/V[1:]*eps[0];gN[1,:-1]+=m*A*L/V[:-1]*eps[1]
            # Full retention-mobility chain rule in both mode densities.
            chain=T0*dm
            for region in [slice(None,-1),slice(1,None)]:
                gN[0,region]+=chain/(2*V[region]*p.n_star);gN[1,region]-=chain/(2*V[region]*p.n_star)
            for j in [0,1]:
                region=slice(None,-1) if j==0 else slice(1,None)
                for mode,donor in [(0,1),(1,0)]:
                    flag=1 if j==donor else 0
                    ga[region]+=m*B[mode]*((.5-flag)*eps[mode]-.5*momentum[mode]*velocity[mode])
                    gb[region]+=m*B[mode]*(1-2*flag)*eps[mode]
                ga[region]-=.5*chain*chi[region];gb[region]-=chain*chi[region]
            energies['field_gradient']=amp_energy;energies['directional_motion']=motion_energy
        H=sum(energies.values());Ndot=gT/p.eta;theta_dot=-gN/p.eta
        rhs=self.pack(Ndot[0]/Nc,Ndot[1]/Nr,theta_dot[0],theta_dot[1],d['a_rate'],d['b_rate'],-ga,-gb)
        gradient=self.pack(Nc*gN[0],Nr*gN[1],gT[0],gT[1],ga,gb,d['a_rate'],d['b_rate'])
        transport=np.zeros((2,self.N),dtype=LD);transport[:,:-1]-=flux;transport[:,1:]+=flux
        overlap=np.sqrt(Nc*Nr)*np.sin(tr-tc)
        Gamma=2/p.eta*(p.Omega+p.Omega_p*(chi/self.cs)**2)*overlap
        source_rate=np.stack([-Gamma,Gamma]);balance=Ndot-transport-source_rate
        ell_dot=d['a_rate']+2*d['b_rate'];chi_dot=(Ndot[0]-Ndot[1])/(V*p.n_star)-ell_dot*chi
        return dict(H=H,energies=energies,rhs=rhs,gradient=gradient,N=np.stack([Nc,Nr]),density=rho,V=V,chi=chi,chi_dot=chi_dot,
            flux=flux,transport=transport,Gamma=Gamma,balance=balance,ell_dot=ell_dot,A=A,L=L,mobility=m,speed=speed)

    def rhs(self,t,y):return self.evaluate(y)['rhs']
    def hamiltonian(self,y):return self.evaluate(y)['H']

    def core(self,y,d=None):
        d=self.evaluate(y) if d is None else d;chi=d['chi'];_,_,_,_,a,b,_,_=self.unpack(y)
        if chi[0]<=0:return dict(exists=False,reason='central chi not positive')
        idx=np.flatnonzero((chi[:-1]>0)&(chi[1:]<=0))
        if not len(idx):return dict(exists=False,reason='no first zero inside domain')
        i=int(idx[0]);f=chi[i]/(chi[i]-chi[i+1]);rb=self.r[i]+f*self.dr
        beta=(1-f)*b[i]+f*b[i+1];bbdot=(1-f)*d['rhs'].reshape(8,self.N)[5,i]+f*d['rhs'].reshape(8,self.N)[5,i+1]
        rb_dot=-((1-f)*d['chi_dot'][i]+f*d['chi_dot'][i+1])/((chi[i+1]-chi[i])/self.dr)
        R=rb*np.exp(beta);Rdot=np.exp(beta)*((1+rb*(b[i+1]-b[i])/self.dr)*rb_dot+rb*bbdot)
        lo=self.faces[:-1];hi=self.faces[1:];upper=np.minimum(hi,rb)
        fraction=np.where(upper>lo,(upper**3-lo**3)/(hi**3-lo**3),0)
        moving=(rb>lo)&(rb<hi);wdot=np.where(moving,3*rb**2/(hi**3-lo**3)*rb_dot,0)
        total=d['N'].sum(axis=0);signed=d['N'][0]-d['N'][1];Ndot=d['rhs'].reshape(8,self.N)[:2]*d['N']
        proper_volume=np.sum(fraction*d['V']);M=np.sum(fraction*total);S=np.sum(fraction*signed)
        change_weights=np.diff(fraction);fixed_transport=np.sum(change_weights*(d['flux'][0]+d['flux'][1]))
        moving_transport=np.sum(wdot*total);direct=np.sum(fraction*Ndot.sum(axis=0))+moving_transport
        signed_direct=np.sum(fraction*(Ndot[0]-Ndot[1]))+np.sum(wdot*signed)
        signed_budget=np.sum(change_weights*(d['flux'][0]-d['flux'][1]))-2*np.sum(fraction*d['Gamma'])+np.sum(wdot*signed)
        return dict(exists=True,coordinate_radius=float(rb),areal_radius=float(R),areal_radius_rate=float(Rdot),volume_radius=float((3*proper_volume/(4*np.pi))**(LD(1)/3)),
            total_content=float(M),signed_content=float(S),coordinate_radius_rate=float(rb_dot),
            moving_total_budget_error=float(abs(direct-fixed_transport-moving_transport)),moving_signed_budget_error=float(abs(signed_direct-signed_budget)),
            total_rate=float(direct),signed_rate=float(signed_direct),boundary_motion_term=float(moving_transport),
            scope='Fractional shell-cell definition with exact discrete weight derivative; not a continuum moving-surface qualification')
