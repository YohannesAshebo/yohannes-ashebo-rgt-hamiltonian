"""Independent Hamiltonian face powers and moving-selection product budgets."""
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
from common import LD
from energy import decompose,bracket

NLEDGER=18
LEDGER_NAMES=['Q_Camp','Q_Ramp','Q_Cmotion','Q_Rmotion','Q_geometry',
              'fixed_Q_Camp','fixed_Q_Ramp','fixed_Q_Cmotion','fixed_Q_Rmotion','fixed_Q_geometry',
              'C_in','R_out','signed_exchange','sweep_absolute_energy','sweep_excess_energy','reference_change','sweep_C','sweep_R']

def face_powers(m,d):
    g=d['gl'];ge=d['ge'];inside=g[:,:-1].copy();outside=g[:,1:].copy()
    inside[:,1:]+=ge[:,1,:,:-1].sum(axis=0);outside[:,:-1]+=ge[:,0,:,1:].sum(axis=0)
    return np.stack([(bracket(inside,ge[k,0],m.p.eta)+bracket(ge[k,1],outside,m.p.eta))/2 for k in range(5)])

def region(m,y,d=None):
    d=decompose(m,y,False) if d is None else d;shell=(m.r>=2)&(m.r<3)
    mu=d['cell'][shell].sum()/m.V0[shell].sum();ex=d['cell']-mu*m.V0;v=ex/m.V0
    w=np.zeros(m.N,dtype=LD);rb=None;edge=None;areal=None;vr=None
    crossing=np.flatnonzero((v[:-1]>0)&(v[1:]<=0))
    if v[0]>0 and len(crossing):
        edge=int(crossing[0]);f=v[edge]/(v[edge]-v[edge+1]);rb=LD(m.r[edge]+f*m.dr)
        lo=m.faces[:-1];hi=m.faces[1:];upper=np.minimum(hi,rb)
        w=np.where(upper>lo,(upper**3-lo**3)/(hi**3-lo**3),0).astype(LD)
        beta=m.unpack(y)[5];areal=rb*np.exp((1-f)*beta[edge]+f*beta[edge+1]);vr=(3*np.sum(w*d['V'])/(4*np.pi))**(LD(1)/3)
    starts=np.sum((v>0)&np.r_[True,v[:-1]<=0])
    return dict(w=w,mu=mu,h=d['cell'],N=d['N'],V=d['V'],radius=rb,root_edge=edge,areal=areal,volume_radius=vr,
        Eabs=np.sum(w*d['cell']),Eex=np.sum(w*ex),C=np.sum(w*d['N'][0]),R=np.sum(w*d['N'][1]),positive_components=int(starts),
        H=d['H'],total_content=d['N'].sum(),central_contrast=v[0],fixed_energy=np.sum(d['cell'][m.r<1]),content_flux=d['content_flux'])

class Ports:
    def __init__(self,m):self.m=m;self.fixed=int(np.flatnonzero((m.r[:-1]<1)&(m.r[1:]>=1))[0])
    @lru_cache(maxsize=512)
    def view(self,faces):
        m=self.m;indices=[];true_edges=[];centers=[]
        for j in faces:
            lo=max(0,min(j-1,m.N-4));start=len(indices);indices.extend(range(lo,lo+4));centers.append(start+j-lo)
            true_edges.extend([lo,lo+1,lo+2,-1])
        true_edges=true_edges[:-1];indices=np.array(indices);edgeidx=np.array(true_edges);valid=edgeidx>=0
        k=len(indices);A0=np.zeros(k-1,dtype=LD);ew=np.zeros(k-1,dtype=LD);caps=np.zeros((2,k-1),dtype=LD)
        A0[valid]=m.A0[edgeidx[valid]];ew[valid]=m.original.edge_weight[edgeidx[valid]];caps[:,valid]=m.vmax[:,edgeidx[valid]]
        original=SimpleNamespace(potential=m.original.potential,edge_weight=ew)
        view=SimpleNamespace(N=k,p=m.p,cs=m.cs,p0=m.p0,r=m.r[indices],V0=m.V0[indices],dr=m.dr,A0=A0,vmax=caps,original=original,
                             unpack=lambda y:np.asarray(y,dtype=LD).reshape(8,k))
        return indices,np.array(centers),view
    def instantaneous(self,y,faces):
        ids,centers,view=self.view(tuple(faces));q=self.m.unpack(y)[:,ids]
        d=decompose(view,q.ravel());return face_powers(view,d)[:,centers],d['content_flux'][:,centers]
    def interval(self,stages,B,dt,wa,wb):
        m=self.m;w=(wa+wb)/2;coeff=w[:-1]-w[1:];selected=np.flatnonzero(coeff!=0)
        faces=np.unique(np.r_[selected,self.fixed]);where={int(j):i for i,j in enumerate(faces)}
        weights=np.array([coeff[j] for j in faces],dtype=LD);fixed=where[self.fixed]
        ans=np.zeros(13,dtype=LD)
        for b,y in zip(B,stages):
            if not b:continue
            p,flux=self.instantaneous(y,tuple(int(j) for j in faces));q=m.unpack(y)
            N=np.exp(q[:2]);V=m.V0*np.exp(q[4]+2*q[5]);chi=(N[0]-N[1])/(V*m.p.n_star)
            Gamma=2/m.p.eta*(m.p.Omega+m.p.Omega_p*(chi/m.cs)**2)*np.sqrt(N[0]*N[1])*np.sin(q[3]-q[2])
            item=np.r_[np.sum(p*weights,axis=1),p[:,fixed],-np.sum(flux[0]*weights),np.sum(flux[1]*weights),np.sum(w*Gamma)]
            ans+=LD(b)*item
        return LD(dt)*ans

def advance(a,b,quad,V0):
    w=(a['w']+b['w'])/2;dw=b['w']-a['w'];h=(a['h']+b['h'])/2;mu=(a['mu']+b['mu'])/2
    n=(a['N']+b['N'])/2
    sweep_h=np.sum(h*dw);sweep_e=np.sum((h-mu*V0)*dw);reference=-(b['mu']-a['mu'])*np.sum(w*V0)
    increments=np.r_[quad,sweep_h,sweep_e,reference,np.sum(n[0]*dw),np.sum(n[1]*dw)]
    predicted=np.array([quad[:5].sum()+sweep_h,quad[:5].sum()+sweep_e+reference,quad[10]-quad[12]+increments[16],quad[12]-quad[11]+increments[17]])
    actual=np.array([b['Eabs']-a['Eabs'],b['Eex']-a['Eex'],b['C']-a['C'],b['R']-a['R']])
    algebra=float(abs(np.sum(w*(b['h']-a['h']))-quad[:5].sum()))
    return increments,actual-predicted,algebra

def summary(m,t,y,r,L,H0,N0,fixed0):
    moving_power=L[:5].sum();nc=L[10]-L[12]+L[16];nr=L[12]-L[11]+L[17]
    residuals=dict(absolute_energy=float(r['Eabs']-moving_power-L[13]),excess_energy=float(r['Eex']-moving_power-L[14]-L[15]),C=float(r['C']-nc),R=float(r['R']-nr),fixed_energy=float(r['fixed_energy']-fixed0-L[5:10].sum()))
    j=int(np.flatnonzero((m.r[:-1]<1)&(m.r[1:]>=1))[0]);I=-r['content_flux'][0,j];O=r['content_flux'][1,j]
    return dict(time=float(t),radius=None if r['radius'] is None else float(r['radius']),areal_radius=None if r['areal'] is None else float(r['areal']),volume_radius=None if r['volume_radius'] is None else float(r['volume_radius']),
        region_exists=r['radius'] is not None,positive_components=r['positive_components'],Eabs=float(r['Eabs']),Eex=float(r['Eex']),C=float(r['C']),R=float(r['R']),reference=float(r['mu']),central_contrast=float(r['central_contrast']),
        H=float(r['H']),fixed_energy=float(r['fixed_energy']),fixed_excess=float(r['fixed_energy']-r['mu']*np.sum(m.V0[m.r<1])),
        Q_energy=float(moving_power),Q_channels=L[:5].astype(float).tolist(),fixed_Q_channels=L[5:10].astype(float).tolist(),sweep_energy=float(L[13]),sweep_excess=float(L[14]),reference_change=float(L[15]),
        C_in=float(L[10]),R_out=float(L[11]),signed_exchange=float(L[12]),sweep_C=float(L[16]),sweep_R=float(L[17]),budget_residuals=residuals,
        scaled_global_energy_error=float(abs(r['H']-H0)/max(1,abs(H0))),relative_global_content_error=float(abs(r['total_content']-N0)/N0),
        fixed_I=float(I),fixed_O=float(O),fixed_I_native_zero=bool(I==0),fixed_O_native_zero=bool(O==0),log10_I=None if I<=0 else float(np.log10(I)),log10_O=None if O<=0 else float(np.log10(O)))
