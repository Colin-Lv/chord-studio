"""Exact marginals for the installed madmom CRF's latent initial-state convention.

These are model-internal probabilities, not calibrated real-world confidence.
The initial sentinel is summed out before the first observed frame.
"""
import numpy as np
from scipy.special import logsumexp


def marginals(emissions, initial, transition, final):
    emissions=np.asarray(emissions,dtype=np.float64)
    initial=np.asarray(initial,dtype=np.float64)
    transition=np.asarray(transition,dtype=np.float64)
    final=np.asarray(final,dtype=np.float64)
    n,k=emissions.shape
    if n==0 or initial.shape!=(k,) or final.shape!=(k,) or transition.shape!=(k,k):
        raise ValueError('Invalid CRF shape')
    if not all(np.isfinite(x).all() for x in [emissions,initial,transition,final]):
        raise ValueError('Non-finite CRF potential')
    forward=np.empty_like(emissions);backward=np.empty_like(emissions)
    forward[0]=emissions[0]+logsumexp(initial[:,None]+transition,axis=0)
    for t in range(1,n):
        forward[t]=emissions[t]+logsumexp(forward[t-1,:,None]+transition,axis=0)
    backward[-1]=final
    for t in range(n-2,-1,-1):
        backward[t]=logsumexp(transition+emissions[t+1][None,:]+backward[t+1][None,:],axis=1)
    log_z=logsumexp(forward[-1]+final)
    probability=np.exp(forward+backward-log_z)
    if not np.allclose(probability.sum(axis=1),1,atol=1e-8):
        raise ValueError('CRF marginal normalization failed')
    return probability,float(log_z)


def maximum_path(emissions, initial, transition, final):
    n,k=emissions.shape
    previous=np.asarray(initial).copy();back=np.zeros((n,k),dtype=int)
    for t in range(n):
        scores=previous[:,None]+transition
        back[t]=scores.argmax(axis=0)
        previous=emissions[t]+scores.max(axis=0)
    path=np.zeros(n,dtype=int);path[-1]=(previous+final).argmax()
    for t in range(n-2,-1,-1):path[t]=back[t+1,path[t+1]]
    return path
