#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Unchanged published kernels and the previous manufactured input recipe."""
from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys
import numpy as np
from scipy.optimize import brentq


def prepare_native(release,temporary):
    native=Path(temporary)/'native'
    shutil.copytree(Path(release)/'native',native,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.dylib','*.dll'))
    sys.path[:0]=[str(native),str(native/'study'),str(native/'speed_order'),str(native/'speed_sweep'),str(release)]
    for stem in ['radial_rhs','current_rhs','autonomous_heat_rhs']:
        subprocess.run(['cc','-O3','-fPIC','-shared',str(native/(stem+'.c')),'-o',str(native/(stem+'.so')),'-lm'],check=True,capture_output=True)
    spec=importlib.util.spec_from_file_location('published_r4_fixture',Path(release)/'static_scaling.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    return fixture


class NativeCase:
    def energy(self,state):
        q=np.asarray(state)[:8*self.n]
        return self.model.evaluate(q)['H']+self.law.evaluate(q,False)['energy']


def make_case(n,fixture,heat_kappa=100.,exchange_on=True):
    # This is the same 8-point shell-integration recipe as the earlier audit's
    # matched_profiles.make_current. Only separately labeled controls disable
    # existing coefficients; no native energy/gradient/RHS code is edited.
    from run import prepare
    from autonomous import CurrentExchange
    from autonomous_heat import HeatRHS
    from compiled_current import CompiledCurrent
    from continue_run import FastRHS
    model,*_=prepare(n,1.5)
    fixture.fixed_continuum_caps(model,'continuum_native_caps')
    nodes,weights=np.polynomial.legendre.leggauss(8)
    lo=np.asarray(model.faces[:-1],dtype=np.longdouble)
    hi=np.asarray(model.faces[1:],dtype=np.longdouble)
    r=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*nodes[None,:]/2
    nc,_,nr,_=fixture.smooth_test_fields(r,8)
    measure=4*np.longdouble(np.pi)*r*r*weights[None,:]*(hi-lo)[:,None]/2
    Nc=np.sum(nc*measure,axis=1);Nr=np.sum(nr*measure,axis=1)
    _,tc,_,tr=fixture.smooth_test_fields(np.asarray(model.r,dtype=np.longdouble),8)
    zero=np.zeros(n,dtype=np.longdouble)
    state=model.pack(np.log(Nc),np.log(Nr),tc,tr,zero,zero,zero,zero)
    fixture.disable_old_exchange(model)
    case=NativeCase();case.n=n;case.model=model
    case.initial=np.r_[np.asarray(state,float),np.zeros(5)]
    case.law=CurrentExchange(model,fixture.OMEGA if exchange_on else 0.,decreasing=True)
    case.law.j0=fixture.J0
    case.fast=FastRHS(model);case.compiled=CompiledCurrent(case.fast,case.law)
    case.rhs=HeatRHS(case.fast,case.law,heat_kappa)
    return case


def source_diagnostics(case):
    m=case.model;n=case.n;y=case.initial[:8*n]
    base=m.evaluate(y);exchange=case.law.evaluate(y)
    base_gradient=m.unpack(base['gradient'])
    direct=np.stack([2*exchange['factors']*exchange['raw_energy'],4*exchange['factors']*exchange['raw_energy']])
    total=-exchange['gradient'][4:6]
    force_base=-base_gradient[4:6]/m.V0
    force_direct=direct/m.V0
    force_chain=(total-direct)/m.V0
    c=float(np.sqrt(m.p.k_geometry/m.p.I))
    return dict(radius=np.asarray(m.r,float),volume=np.asarray(m.V0,float),
        weight=np.asarray(exchange['factors'],float),current=np.asarray(exchange['j'],float),
        force_base=np.asarray(force_base,float),force_direct=np.asarray(force_direct,float),
        force_chain=np.asarray(force_chain,float),
        principal_geometry_speed=c,k_geometry=float(m.p.k_geometry),inertia=float(m.p.I))


def transition_scale(case,fixture):
    """Continuous-face-current proxy on the declared smooth initial fields.

    This estimates an input length scale, not the exact finite-grid cell
    average, an observed physical length, or a full coupled characteristic.
    """
    m=case.model;p0=float(m.p0);eta=float(m.p.eta);cs=float(m.cs)
    def sigmoid(z):
        e=np.exp(-abs(z));return 1/(1+e) if z>=0 else e/(1+e)
    def current(radius):
        r=np.longdouble(radius);R=np.longdouble(8)
        nc,_,nr,_=fixture.smooth_test_fields(r,R)
        derivative=fixture.original_continuum_phase_derivative(r)
        common=2*m.p.K/m.p.eta*derivative/sigmoid(float(eta*derivative/p0))
        x=np.pi*r/R;envelope=np.exp(-(r/np.longdouble('2.5'))**2)
        dtc=np.longdouble('.35')*envelope*(-np.pi/R*np.sin(x)-2*r/np.longdouble('2.5')**2*np.cos(x))
        dtr=np.longdouble('.20')*np.pi/R*np.sin(2*x)
        mob=sigmoid(-float((nc-nr)/m.p.n_star)/cs)
        return float(mob*common*(1.2*nc*sigmoid(-eta*float(dtc)/p0)+.8*nr*sigmoid(eta*float(dtr)/p0)))
    j0=float(case.law.j0);rows=[]
    for weight in [.1,.5,.9]:
        target=j0*(1/weight-1)
        delta=brentq(lambda d:np.log(current(4-d)/target),1e-10,.01,xtol=1e-14)
        rows.append(dict(weight=weight,radius=4-delta,distance_inside_cutoff=delta,current=target))
    return dict(scope='Continuous-face-current proxy on the initial manufactured fields; fixed native coefficients.',
        cap_taper_regular_at_cutoff='C2 quintic taper; derivative coefficient vanishes cubically approaching r=4',
        j0=j0,rows=rows,width_10_to_90=rows[2]['radius']-rows[0]['radius'])


def reference_volumes(n):
    edges=np.arange(n+1,dtype=float)*8/n
    return 4*np.pi/3*np.diff(edges**3)


def profiles(states,n,parent=None):
    q=states[:,:8*n].reshape(len(states),8,n)
    V0=reference_volumes(n);contents=np.exp(q[:,:2]);V=V0[None,:]*np.exp(q[:,4]+2*q[:,5])
    scalar={'alpha':q[:,4],'beta':q[:,5],'relative_phase':q[:,2]-q[:,3]}
    momenta={'P_alpha_per_reference_volume':q[:,6],'P_beta_per_reference_volume':q[:,7]}
    if parent is None:
        density=[contents[:,i]/V for i in [0,1]];w=V0
        momenta={k:v/V0[None,:] for k,v in momenta.items()}
    else:
        if n!=2*parent:raise ValueError('Only dyadic coarsening is defined')
        def sums(a):return a.reshape(len(states),parent,2).sum(axis=-1)
        density=[sums(contents[:,i])/sums(V) for i in [0,1]]
        w=reference_volumes(parent)
        scalar={k:sums(v*V0[None,:])/w[None,:] for k,v in scalar.items()}
        momenta={k:sums(v)/w[None,:] for k,v in momenta.items()}
    return {'n_C':density[0],'n_R':density[1],**scalar,**momenta},w


def norm(a,b,w):
    return float(np.sqrt(np.sum(w*((a-b)/np.maximum(1.,abs(b)))**2)/np.sum(w)))
