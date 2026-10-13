import numpy as np
import torch
from if_redesign.rime import RIMEHead,coordinate_scale,prepare_inputs

def predict(head,inputs):
    with torch.no_grad():return head(**{k:v for k,v in inputs.items() if k!='indices'})

def test_nonzero_head_is_replication_and_permutation_invariant():
    rng=np.random.default_rng(43)
    h=rng.normal(scale=.3,size=(2,3,4,8));valid=np.ones(h.shape[:-1],bool)
    context=rng.normal(size=(2,34));anchor=np.array([.3,-.2]);sigma=.9
    scale=coordinate_scale(h,valid,[0])
    head=RIMEHead(34).double()
    with torch.no_grad():
        for p in head.parameters():p.copy_(torch.as_tensor(rng.normal(scale=.1,size=p.shape)))
    base=prepare_inputs(h,valid,context,anchor,sigma,fitted_scale=scale)
    expected=predict(head,base)
    for copies in [1,2,4,8]:
        changed=np.concatenate([h,np.repeat(h[:,:,:1],copies,axis=2)],axis=2)
        mask=np.ones(changed.shape[:-1],bool)
        inputs=prepare_inputs(changed,mask,context,anchor,sigma,fitted_scale=scale)
        torch.testing.assert_close(predict(head,inputs),expected,rtol=1e-12,atol=1e-12)
    inputs=prepare_inputs(h[:,:,::-1].copy(),valid,context,anchor,sigma,fitted_scale=scale)
    torch.testing.assert_close(predict(head,inputs),expected,rtol=1e-12,atol=1e-12)

def test_coordinate_scale_ignores_heldout_changes_and_empty_sites_fallback():
    rng=np.random.default_rng(29);h=rng.normal(size=(4,2,3,8))*.2
    valid=np.ones(h.shape[:-1],bool)
    a=coordinate_scale(h,valid,[0,1]);changed=h.copy();changed[2:]*=10000
    np.testing.assert_array_equal(a,coordinate_scale(changed,valid,[0,1]))
    mask=np.zeros_like(valid)
    inputs=prepare_inputs(h,mask,np.zeros((4,26)),np.arange(4)*.1,.9,fitted_scale=a)
    head=RIMEHead(26).double()
    with torch.no_grad():head.beta.fill_(1.)
    torch.testing.assert_close(predict(head,inputs),inputs['anchor'])
