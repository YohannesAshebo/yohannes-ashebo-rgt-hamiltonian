"""Full native Hamiltonian decomposition and nearest-neighbor energy ports.

No physical equation is changed. Gradients here use canonical contents,
whereas the inherited integrator stores their logarithms.
"""
import numpy as np
from common import LD
from directional_radial import sigmoid,softplus

LOCAL_NAMES=['geometry_kinetic','common_local','reciprocal_exchange',
             'retention_mass','weighted_phase_exchange','geometry_angular']
EDGE_NAMES=['C_amplitude','R_amplitude','C_motion','R_motion','geometry_gradient']

def bracket(f,g,eta):
    return ((f[0]*g[2]-f[2]*g[0]+f[1]*g[3]-f[3]*g[1])/eta
            +f[4]*g[6]-f[6]*g[4]+f[5]*g[7]-f[7]*g[5])

def background(m):
    q=m.initial().reshape(8,m.N).copy();q[4:]=0
    q[0]=np.log(m.V0*(m.p.n_background-.05*m.p.n_star)/2)
    q[1]=np.log(m.V0*(m.p.n_background+.05*m.p.n_star)/2)
    return q.ravel()

def decompose(m,y,gradient=True):
    p=m.p;n=m.N;uc,ur,tc,tr,a,b,pa,pb=m.unpack(y)
    N=np.exp(np.stack([uc,ur]));V=m.V0*np.exp(a+2*b);rho=N/V
    nc,nr=rho;chi=(nc-nr)/p.n_star;cs=m.cs
    U,muc,mur,Ua,Ub,pressure=m.original.potential(nc,nr,a,b)
    delta=tr-tc;cos=np.cos(delta);root=np.sqrt(N[0]*N[1])
    Y=N[0]+N[1]-2*root*cos
    Ed=m.V0*p.Omega_d*p.n_star*chi**2;Ep=p.Omega_p*(chi/cs)**2*Y
    local=np.array([pa**2/(2*p.I*m.V0)+pb**2/(4*p.I*m.V0),V*U,
                    p.Omega*Y,Ed,Ep,2*p.k_geometry*m.V0*(a-b)**2/m.r**2],dtype=LD)
    A=m.A0*np.exp(b[:-1]+b[1:]);L=m.dr*np.exp((a[:-1]+a[1:])/2)
    amp=np.sqrt(rho);diff=amp[:,:-1]-amp[:,1:];w=A/L
    mom=p.eta*np.diff(np.stack([tc,tr]),axis=1)/L
    signs=np.array([-1,1],dtype=LD)[:,None]
    eps=m.vmax*m.p0*softplus(signs*mom/m.p0)
    vel=signs*m.vmax*sigmoid(signs*mom/m.p0)
    mobility=sigmoid(-(chi[:-1]+chi[1:])/(2*cs));dm=-mobility*(1-mobility)/cs
    donors=np.stack([rho[0,1:],rho[1,:-1]]);B=A*L*donors
    edge=np.empty((5,n-1),dtype=LD)
    edge[:2]=p.K/2*w*diff**2;edge[2:4]=mobility*B*eps
    edge[4]=p.k_geometry/2*m.original.edge_weight*(np.diff(a)**2+2*np.diff(b)**2)
    cell=local.sum(axis=0);cell[:-1]+=edge.sum(axis=0)/2;cell[1:]+=edge.sum(axis=0)/2
    out=dict(local=local,edge=edge,cell=cell,H=cell.sum(),V=V,N=N,chi=chi,A=A,L=L,mobility=mobility,
             content_flux=mobility*A*donors*vel)
    if not gradient:return out
    gl=np.zeros((8,n),dtype=LD);ge=np.zeros((5,2,8,n-1),dtype=LD)
    diagonal=2*p.Omega_d*chi/(V/m.V0)+2*p.Omega_p*chi*Y/(cs**2*V*p.n_star)
    coupling=p.Omega+p.Omega_p*(chi/cs)**2
    gl[0]=muc+diagonal+coupling*(1-np.sqrt(N[1]/N[0])*cos)
    gl[1]=mur-diagonal+coupling*(1-np.sqrt(N[0]/N[1])*cos)
    gamma=2/p.eta*coupling*root*np.sin(delta)
    gl[2]=-p.eta*gamma;gl[3]=p.eta*gamma
    angular=4*p.k_geometry*m.V0*(a-b)/m.r**2
    gl[4]=V*(-pressure+Ua)-2*(Ed+Ep)+angular
    gl[5]=V*(-2*pressure+Ub)-4*(Ed+Ep)-angular
    gl[6]=pa/(p.I*m.V0);gl[7]=pb/(2*p.I*m.V0)
    for s in range(2):
        ge[s,0,s]=p.K/2*w*diff[s]/(V[:-1]*amp[s,:-1])
        ge[s,1,s]=-p.K/2*w*diff[s]/(V[1:]*amp[s,1:])
        square=diff[s]**2
        for side,region,factor in [(0,slice(None,-1),diff[s]*amp[s,:-1]),(1,slice(1,None),-diff[s]*amp[s,1:])]:
            ge[s,side,4]=p.K/2*w*(-square/2-factor)
            ge[s,side,5]=p.K/2*w*(square-2*factor)
        k=s+2;donor=1-s;region=slice(1,None) if donor else slice(None,-1)
        ge[k,0,2+s]=-p.eta*out['content_flux'][s]
        ge[k,1,2+s]=p.eta*out['content_flux'][s]
        ge[k,donor,s]+=mobility*A*L/V[region]*eps[s]
        chain=B[s]*eps[s]*dm
        for side,region in [(0,slice(None,-1)),(1,slice(1,None))]:
            ge[k,side,0]+=chain/(2*V[region]*p.n_star)
            ge[k,side,1]-=chain/(2*V[region]*p.n_star)
            flag=int(side==donor)
            ge[k,side,4]+=mobility*B[s]*((LD(.5)-flag)*eps[s]-mom[s]*vel[s]/2)-chain*chi[region]/2
            ge[k,side,5]+=mobility*B[s]*(1-2*flag)*eps[s]-chain*chi[region]
    ge[4,0,4]=p.k_geometry*m.original.edge_weight*(a[:-1]-a[1:]);ge[4,1,4]=-ge[4,0,4]
    ge[4,0,5]=2*p.k_geometry*m.original.edge_weight*(b[:-1]-b[1:]);ge[4,1,5]=-ge[4,0,5]
    full=gl.copy();full[:,:-1]+=ge[:,0].sum(axis=0);full[:,1:]+=ge[:,1].sum(axis=0)
    out.update(gl=gl,ge=ge,gradient=full)
    return out

def region_weights(m,lam=LD(.5)):
    mask=m.r<1;weights=(mask[:-1].astype(LD)+mask[1:])/2
    edge=int(np.flatnonzero(mask[:-1]&~mask[1:])[0]);weights[edge]=lam
    return mask,weights,edge

def region_energy(m,d,lam=LD(.5)):
    mask,w,_=region_weights(m,lam)
    return d['local'][:,mask].sum()+np.sum(d['edge']*w)

def core_gradient(m,d,lam=LD(.5)):
    mask,w,_=region_weights(m,lam);g=d['gl']*mask
    g[:,:-1]+=np.sum(d['ge'][:,0]*w,axis=0)
    g[:,1:]+=np.sum(d['ge'][:,1]*w,axis=0)
    return g

def powers(m,d,lam=LD(.5)):
    _,_,j=region_weights(m,lam);g=d['gl'];ge=d['ge']
    inside=g[:,j].copy();outside=g[:,j+1].copy()
    if j>0:inside+=ge[:,1,:,j-1].sum(axis=0)
    if j+1<m.N-1:outside+=ge[:,0,:,j+1].sum(axis=0)
    return np.array([(1-lam)*bracket(inside,ge[k,0,:,j],m.p.eta)+lam*bracket(ge[k,1,:,j],outside,m.p.eta) for k in range(5)],dtype=LD)

def unit_motion_core(m,y,caps,lam=LD(.5)):
    q=m.unpack(y);V=m.V0*np.exp(q[4]+2*q[5]);rho=np.exp(q[:2])/V
    chi=(rho[0]-rho[1])/m.p.n_star;mobility=sigmoid(-(chi[:-1]+chi[1:])/(2*m.cs))
    A=m.A0*np.exp(q[5,:-1]+q[5,1:]);L=m.dr*np.exp((q[4,:-1]+q[4,1:])/2)
    mom=m.p.eta*np.diff(q[2:4],axis=1)/L
    energy=mobility*A*L*np.stack([rho[0,1:],rho[1,:-1]])*caps*m.p0*softplus(np.array([-1,1])[:,None]*mom/m.p0)
    _,w,_=region_weights(m,lam)
    return np.array([np.sum(energy*w,axis=1),np.sum(energy,axis=1)])

def quantile_radius(m,weight,V,q):
    total=weight.sum()
    if total<=LD('1e-25'):return None,None
    cumulative=np.cumsum(weight);i=int(np.searchsorted(cumulative,q*total));i=min(i,m.N-1)
    prev=cumulative[i-1] if i else LD(0);f=(q*total-prev)/weight[i] if weight[i]>0 else LD(0)
    coordinate=(m.faces[i]**3+f*(m.faces[i+1]**3-m.faces[i]**3))**(LD(1)/3)
    proper=((np.sum(V[:i])+f*V[i])*3/(4*np.pi))**(LD(1)/3)
    return float(coordinate),float(proper)

def localization(m,d,bg):
    mask,_,_=region_weights(m);excess=d['cell']-bg['cell'];pos=np.maximum(excess,0);neg=np.minimum(excess,0)
    shell=(m.r>=3)&(m.r<4);ref=d['cell'][shell].sum()/m.V0[shell].sum()
    excess_shell=d['cell']-ref*m.V0;pos_shell=np.maximum(excess_shell,0)
    result=dict(H=float(d['H']),Hcore=float(region_energy(m,d)),excess_H=float(excess.sum()),
        excess_core=float(excess[mask].sum()),positive_excess=float(pos.sum()),negative_excess=float(neg.sum()),
        positive_excess_core=float(pos[mask].sum()),negative_excess_core=float(neg[mask].sum()),
        core_absolute_fraction=float(d['cell'][mask].sum()/d['H']),
        core_positive_excess_fraction=None if pos.sum()==0 else float(pos[mask].sum()/pos.sum()),
        shell_excess_core=float(excess_shell[mask].sum()),shell_positive_excess=float(pos_shell.sum()),
        central_energy_density=float(d['cell'][0]/d['V'][0]),central_excess_per_reference_volume=float(excess[0]/m.V0[0]))
    for tag,w in [('absolute',d['cell']),('excess',pos),('shell',pos_shell)]:
        for q in [.5,.9]:
            rc,rp=quantile_radius(m,w,d['V'],q)
            result[f'{tag}_R{int(q*100)}']=rc;result[f'{tag}_proper_R{int(q*100)}']=rp
    return result
