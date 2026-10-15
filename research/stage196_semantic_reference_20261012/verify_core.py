"""Numerical checks use synthetic data only; no patient performance claims."""
from pathlib import Path
import json
import numpy as np
import core


def main():
    rng=np.random.default_rng(196);n=72
    d=rng.normal(size=(n,20));mask=rng.random((n,20))>.2
    y=np.tile(np.repeat([0,1],12),3);groups=np.repeat(['a','b','c'],24)
    base=2*(2*y-1).astype(float);errs={}
    for kind in core.ENCODINGS:
        state=core.source_state(d,mask,kind)
        assert np.array_equal(core.transform(np.zeros_like(d),mask,state),np.zeros((n,26)))
        assert np.array_equal(core.transform(d,np.zeros_like(mask),state),np.zeros((n,26)))
        # Column permutations within modality cannot affect distribution coordinates.
        perm=np.r_[rng.permutation(12),12+rng.permutation(8)]
        np.testing.assert_allclose(core.transform(d[:,perm],mask[:,perm],state),core.transform(d,mask,state),atol=1e-12)
    state=core.source_state(d,mask,'LAPLACE');x=core.transform(d,mask,state)
    deletion=core.fixed_deletion_design(d,mask,x,state,.1)
    w=rng.normal(0,.8,26)
    assert core.objective(w,x,base,y,groups,.03,'CFV_RISK')[0] > core.objective(w,x,base,y,groups,.03,'NONE')[0]+.001
    for constraint in ('NONE','CFV_RISK','SIMPLE_CONSISTENCY'):
        fun=lambda v:core.objective(v,x,base,y,groups,.03,constraint,deletion)
        analytic=fun(w)[1];numeric=[]
        for j in range(26):
            step=np.zeros(26);step[j]=1e-6
            numeric.append((fun(w+step)[0]-fun(w-step)[0])/2e-6)
        error=float(np.max(np.abs(analytic-numeric)));assert error<1e-6
        errs[constraint]=error
        head=core.fit(d,mask,base,y,groups,'LAPLACE',constraint,.03,.1)
        assert np.array_equal(core.predict(head,np.zeros_like(d),mask,base),base)
    report=dict(state='PASS',dimensions=26,constraint_gradient_errors=errs,
                zero_anchor=True,permutation_invariance=True,synthetic_only=True,
                fixed_deletion_is_approximation=True)
    (Path(__file__).parent/'CORE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
