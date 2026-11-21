"""Low-rank PSD mobility, tangent constraints and auxiliary heat bookkeeping."""
import numpy as np

def tangent_directions(raw,gradients):
    G=np.asarray(raw,float);A=np.asarray(gradients,float)
    if G.ndim!=2 or not np.isfinite(G).all():raise ValueError("Dissipative directions must be a finite (dimension,rank) matrix")
    if A.shape==(0,G.shape[0]):return G
    if A.ndim!=2 or A.shape[1]!=G.shape[0] or not np.isfinite(A).all():raise ValueError("Invariant gradients must have shape (invariants,dimension)")
    # SVD avoids squaring the conditioning as A A^T would do.
    _,s,Vh=np.linalg.svd(A,full_matrices=False)
    if not len(s) or s[0]==0:return G
    rank=int(np.sum(s>np.finfo(float).eps*max(A.shape)*s[0]));rows=Vh[:rank]
    return G-rows.T@(rows@G)

def factor_rates(gradient,G,gamma,kappa=1.):
    g=np.asarray(gradient,float);G=np.asarray(G,float);gamma=np.asarray(gamma,float)
    if g.ndim!=1 or G.ndim!=2 or G.shape[0]!=len(g) or gamma.shape!=(G.shape[1],):raise ValueError("Gradient, directions and mobility have incompatible shapes")
    if not np.isfinite(g).all() or not np.isfinite(G).all() or not np.isfinite(gamma).all() or np.any(gamma<0):raise ValueError("Mobility must be finite and nonnegative")
    if not np.isfinite(kappa) or kappa<0:raise ValueError("kappa must be finite and nonnegative")
    torque=G.T@g;weighted=kappa*gamma*torque
    return -G@weighted,float(np.dot(torque,weighted)),torque

def extended_metric_action(v_z,v_Q,gradient,G,gamma):
    """M = sum gamma_k (G_k,-tau_k)(G_k,-tau_k)^T, tau=G^T grad H.

    This operator is symmetric PSD and annihilates (grad H,1) algebraically.
    The heat variable is formal extensive storage, not physical entropy.
    """
    G=np.asarray(G,float);gamma=np.asarray(gamma,float);tau=G.T@np.asarray(gradient,float)
    if np.any(gamma<0):raise ValueError("Negative mobility")
    w=G.T@np.asarray(v_z,float)-tau*v_Q
    return G@(gamma*w),float(-np.dot(tau,gamma*w))
