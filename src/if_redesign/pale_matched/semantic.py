"""Exact Stage196 PCA26 functions; fit on legal source patients only."""
import numpy as np

def semantic_state(hidden,mask):
    _,_,vt=np.linalg.svd(hidden-hidden.mean(0),full_matrices=False)
    basis=vt[:26].T;std=np.maximum((hidden@basis).std(0),.005)
    return dict(basis=basis,std=std)

def semantic_features(hidden,mask,state):
    x=np.clip(hidden@state['basis']/state['std'],-16,16)
    return np.where(mask.any(1)[:,None],x,0.)
