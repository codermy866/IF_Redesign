"""Label-free source-centering and gray reanchoring of frozen frame contrasts.

The caller must fit basis, scale and fit_centering on training sources only.
This function applies fixed state; it never estimates state on input rows.
"""
import numpy as np

def repair_project(raw,valid,state):
    basis=state['basis'];scale=state['scale'];mu=state['fit_centering']@basis
    neutral=np.clip(-mu/scale,-3,3)
    out=np.zeros((*valid.shape,8),dtype=np.float64);clipped=np.zeros_like(out,dtype=bool)
    for start in range(0,len(raw),8):
        z=(np.asarray(raw[start:start+8],dtype=np.float64)@basis-mu)/scale
        clipped[start:start+8]=abs(z)>=3
        out[start:start+8]=(np.clip(z,-3,3)-neutral)/6
    return np.where(valid[...,None],out,0.),clipped
