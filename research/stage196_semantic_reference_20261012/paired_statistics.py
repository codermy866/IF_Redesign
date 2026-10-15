"""Paired patient resampling; center variation remains a separate uncertainty."""
import numpy as np
from scipy.stats import beta


def paired_bootstrap(y,groups,scores,repeats=3000,seed=196):
    """Class/center-stratified paired bootstrap, conditional on fixed fitted models."""
    rng=np.random.default_rng(seed);counts=np.zeros((repeats,len(y)),dtype=np.int32)
    for c in np.unique(groups):
        for label in (0,1):
            at=np.flatnonzero((groups==c)&(y==label))
            if len(at):counts[:,at]=rng.multinomial(len(at),np.full(len(at),1/len(at)),size=repeats)
    result={}
    for name,z in scores.items():
        order=np.argsort(z,kind='stable');sorted_z=z[order]
        starts=np.r_[0,np.flatnonzero(np.diff(sorted_z)!=0)+1]
        pos=np.add.reduceat(counts[:,order]*(y[order]==1),starts,axis=1)
        neg=np.add.reduceat(counts[:,order]*(y[order]==0),starts,axis=1)
        numerator=(pos*(neg.cumsum(1)-.5*neg)).sum(1)
        result[name]=numerator/(pos.sum(1)*neg.sum(1))
    return result


def binomial_interval(events,total,alpha=.05):
    if not total:return (None,None)
    return (0. if not events else float(beta.ppf(alpha/2,events,total-events+1)),
            1. if events==total else float(beta.ppf(1-alpha/2,events+1,total-events)))


def source_threshold(y,z,target):
    positive=np.sort(np.asarray(z)[np.asarray(y)==1])
    if not len(positive):raise ValueError('No source positives for threshold')
    return float(positive[int(np.floor((1-target)*len(positive)))])
