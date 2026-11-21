#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Corrected finite-grid DCT / canonical / auxiliary-heat building blocks."""
import math
import torch
from torch import nn
from torch.nn import functional as F

Q=(0,1,4,5)
P=(2,3,6,7)


def dct1d_ortho(x):
    """Orthonormal DCT-II. The transform matrix C need not be symmetric."""
    n=x.shape[-1];k=torch.arange(n,device=x.device,dtype=x.dtype)
    phase=torch.exp(-.5j*math.pi*k/n)
    spectrum=torch.fft.rfft(torch.cat([x,x.flip(-1)],dim=-1),dim=-1)[...,:n]
    scale=torch.full((n,),1/math.sqrt(2*n),device=x.device,dtype=x.dtype)
    scale[0]=1/(2*math.sqrt(n))
    return (spectrum*phase).real*scale


def idct1d_ortho(X):
    """C^T X, the inverse of orthonormal DCT-II (orthonormal DCT-III)."""
    n=X.shape[-1];k=torch.arange(n,device=X.device,dtype=X.dtype)
    amplitude=torch.full((n,),math.sqrt(2*n),device=X.device,dtype=X.dtype)
    amplitude[0]=2*math.sqrt(n)
    half=X*amplitude*torch.exp(.5j*math.pi*k/n)
    half=torch.cat([half,torch.zeros_like(half[...,:1])],dim=-1)
    return torch.fft.irfft(half,n=2*n,dim=-1)[...,:n]


def canonical_from_log(y):
    return torch.cat([y[:,:2].exp(),y[:,2:]],dim=1)


def canonical_gradient_from_log(gradient_y,x):
    return torch.cat([gradient_y[:,:2]/x[:,:2],gradient_y[:,2:]],dim=1)


def phase_flux_source(x,feature_gradient,spacing,number_dissipation=None):
    """Own-model closed-wall balance for invariant phase features and B=I.

    Feature slots 6,7 are padded forward phase derivatives; 8,9 are sin/cos
    of theta_C-theta_R. These are candidate fluxes, not native flux labels.
    """
    phase=x[:,2]-x[:,3]
    local=feature_gradient[:,8]*phase.cos()-feature_gradient[:,9]*phase.sin()
    interior=feature_gradient[:,6:8,:-1]/spacing
    flux=F.pad(interior,(1,1))
    gamma=-local
    if number_dissipation is not None:gamma=gamma-number_dissipation[:,0]
    return flux,gamma


class RadialFeatures(nn.Module):
    """Eleven differentiable features; optional common-phase invariance."""
    def __init__(self,reference_volume,radius,gauge_invariant=True):
        super().__init__();self.gauge_invariant=gauge_invariant
        self.register_buffer('V0',torch.as_tensor(reference_volume,dtype=torch.float64))
        r=torch.as_tensor(radius,dtype=torch.float64)
        self.register_buffer('radius',r/r.max())
        self.dr=float(r[1]-r[0]) if len(r)>1 else 1.

    def forward(self,x):
        if x.shape[1]!=8 or x.shape[-1]!=len(self.V0):raise ValueError('Expected native canonical (batch,8,N) state')
        if torch.any(x[:,:2]<=0):raise ValueError('Canonical contents must remain positive; use the log-chart interface for integration')
        density=torch.log(x[:,:2]/self.V0)
        geometry=x[:,4:6];momenta=x[:,6:8]/self.V0
        dc=F.pad(torch.diff(x[:,2:3],dim=-1)/self.dr,(0,1))
        dr=F.pad(torch.diff(x[:,3:4],dim=-1)/self.dr,(0,1))
        if self.gauge_invariant:
            relative=x[:,2:3]-x[:,3:4]
            phase=torch.cat([dc,dr,relative.sin(),relative.cos()],dim=1)
        else:
            phase=torch.cat([dc,dr,x[:,2:4]],dim=1)
        coord=self.radius.view(1,1,-1).expand(x.shape[0],-1,-1)
        return torch.cat([density,geometry,momenta,phase,coord],dim=1)


class DCTSpectralConv1d(nn.Module):
    """Real cosine feature mixing. It is not itself a Poisson operator."""
    def __init__(self,cin,cout,modes):
        super().__init__();self.modes=modes
        self.weight=nn.Parameter(torch.randn(cin,cout,modes)/(cin*cout))
    def forward(self,x):
        X=dct1d_ortho(x);m=min(self.modes,x.shape[-1])
        active=torch.einsum('bim,iom->bom',X[...,:m],self.weight[...,:m])
        return idct1d_ortho(F.pad(active,(0,x.shape[-1]-m)))


class DCTBackbone(nn.Module):
    def __init__(self,cin,cout,width=16,modes=24,layers=2):
        super().__init__();self.lift=nn.Conv1d(cin,width,1)
        self.spec=nn.ModuleList([DCTSpectralConv1d(width,width,modes) for _ in range(layers)])
        self.local=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(layers)])
        self.head=nn.Sequential(nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,cout,1))
    def forward(self,features):
        h=self.lift(features)
        for spectral,local in zip(self.spec,self.local):h=F.gelu(spectral(h)+local(h))
        return self.head(h)


class DCTEnergyBackbone(nn.Module):
    """Candidate learned energy density reduced to an extensive scalar H."""
    def __init__(self,reference_volume,width=16,modes=24,layers=2):
        super().__init__();self.net=DCTBackbone(11,1,width,modes,layers)
        self.register_buffer('V0',torch.as_tensor(reference_volume,dtype=torch.float64))
    def forward(self,features):return (self.net(features)[:,0]*self.V0).sum(dim=-1)


class RankKFrictionBackbone(nn.Module):
    def __init__(self,k=2,width=16,modes=24,layers=2):
        super().__init__();self.k=k;self.net=DCTBackbone(11,9*k,width,modes,layers)
    def forward(self,features):
        out=self.net(features);b,_,n=out.shape
        return out[:,:8*self.k].reshape(b,self.k,8,n),out[:,8*self.k:]


class CanonicalPoisson(nn.Module):
    """Native pair order; optional constant DCT filters with fixed DC=1.

    Identity mode recovers the native canonical J. Learning other modes
    changes the candidate reversible operator, despite preserving Poisson.
    """
    def __init__(self,modes=24,learn_filter=False):
        super().__init__();self.learn_filter=learn_filter
        self.log_modes=nn.Parameter(torch.zeros(4,modes),requires_grad=learn_filter)
    def apply_B(self,vector4):
        if not self.learn_filter:return vector4
        n=vector4.shape[-1];m=min(n,self.log_modes.shape[-1])
        # Every pair has the SAME zero-mode multiplier. This is necessary
        # for the native combined-content generator under a learned H.
        low=torch.cat([torch.ones((4,1),device=vector4.device,dtype=vector4.dtype),self.log_modes[:,1:m].exp()],dim=-1)
        multiplier=F.pad(low,(0,n-m),value=1.)
        return idct1d_ortho(dct1d_ortho(vector4)*multiplier)
    def forward(self,gradient):
        gq=gradient[:,Q];gp=gradient[:,P]
        qdot=self.apply_B(gp);pdot=-self.apply_B(gq)
        # Native order: N_C,N_R,theta_C,theta_R,alpha,beta,Palpha,Pbeta.
        return torch.stack([qdot[:,0],qdot[:,1],pdot[:,0],pdot[:,1],qdot[:,2],qdot[:,3],pdot[:,2],pdot[:,3]],dim=1)


def project_friction_directions(G,support='phase_only'):
    """Enforce combined-content conservation independently of energy balance."""
    if support=='phase_only':
        phase=(G[:,:,2]-G[:,:,3])/2
        z=torch.zeros_like(phase)
        return torch.stack([z,z,phase,-phase,z,z,z,z],dim=2)
    if support=='content_conserving':
        exchange=(G[:,:,0]-G[:,:,1])/2
        return torch.cat([exchange[:,:,None],-exchange[:,:,None],G[:,:,2:]],dim=2)
    if support=='unrestricted':return G
    raise ValueError(support)


def friction_apply(vu,vs,gradient_H,G,gamma):
    """M=(G,-tau) diag(gamma) (G,-tau)^T; all covectors are canonical."""
    tau=(G*gradient_H[:,None]).sum(dim=2)
    w=(G*vu[:,None]).sum(dim=2)-tau*vs
    rate=gamma*w
    return (rate[:,:,None]*G).sum(dim=1),-(rate*tau).sum(dim=1,keepdim=True)


class RadialPoissonGENERIC8Field(nn.Module):
    """Candidate learned-energy model with explicit native chart and constraints.

    energy_backbone(features) -> scalar H per batch item.
    friction_backbone(features) -> G (B,K,8,N), raw gamma (B,K,N).
    This creates a candidate learned H; it does not modify the native solver.
    """
    def __init__(self,energy_backbone,friction_backbone,reference_volume,radius,modes=24,
                 gauge_invariant=True,learn_poisson_filter=False,friction_support='phase_only'):
        super().__init__();self.energy_net=energy_backbone;self.friction_net=friction_backbone
        self.features=RadialFeatures(reference_volume,radius,gauge_invariant)
        self.poisson=CanonicalPoisson(modes,learn_poisson_filter);self.friction_support=friction_support
    def rhs(self,x,s,create_graph=None,mobility_gate=None):
        if create_graph is None:create_graph=self.training
        # Preserve a previously connected state graph; do not detach at every
        # step, which would invalidate end-to-end rollout differentiation.
        with torch.enable_grad():
            u=x if x.requires_grad else x.clone().requires_grad_(True)
            feat=self.features(u)
            H=self.energy_net(feat)
            if H.shape!=(x.shape[0],):raise ValueError('Energy backbone must return one scalar per batch item')
            grad=torch.autograd.grad(H.sum(),u,create_graph=create_graph)[0]
            G,raw=self.friction_net(feat)
            G=project_friction_directions(G,self.friction_support)
            gamma=F.softplus(raw)
            if mobility_gate is not None:
                if torch.any(mobility_gate<0):raise ValueError('Mobility gate must be nonnegative')
                gamma=gamma*mobility_gate
            # grad S=(0,1). A softplus alone is positive, not an exact switch.
            diss,heat=friction_apply(torch.zeros_like(u),torch.ones_like(s),grad,G,gamma)
            reversible=self.poisson(grad)
        return reversible+diss,heat,H,grad
    def rhs_log_chart(self,y,s,create_graph=None,mobility_gate=None):
        x=canonical_from_log(y)
        dx,ds,H,gradient=self.rhs(x,s,create_graph,mobility_gate)
        dy=torch.cat([dx[:,:2]/x[:,:2],dx[:,2:]],dim=1)
        return dy,ds,H,gradient
