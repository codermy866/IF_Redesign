import numpy as np
import torch
from if_redesign.rime.measure import gaussian_kernel,reference_measure,measure
from if_redesign.rime.readout import coordinates,RIMEHead

def test_analytic_two_atoms_and_copy_invariance():
    h=np.array([[0.,0.],[.5,0.]])
    q=np.array([.1,.3]);tau=.3;sigma=.7
    a,w,d=reference_measure(h,q,sigma,tau)
    rho=gaussian_kernel(h,sigma)[0,1]
    assert abs(w[0]-np.clip(.5+tau*(q[0]-q[1])/(2*(1-rho)),0,1))<1e-6
    for copies in [1,2,4,8]:
        aa,ww,dd=reference_measure(np.concatenate([h,np.repeat(h[:1],copies,axis=0)]),np.r_[q,np.repeat(q[0],copies)],sigma,tau)
        np.testing.assert_allclose(ww,w,atol=1e-7)
        assert dd['kkt_residual']<1e-6

def test_solver_matches_reference_and_implicit_gradient():
    rng=np.random.default_rng(17)
    h=rng.normal(size=(6,8));q=rng.normal(size=6)*.1
    atoms,w,_=reference_measure(h,q,2.,.3)
    order=[np.flatnonzero(np.all(h==x,axis=1))[0] for x in atoms]
    k=torch.tensor(gaussian_kernel(atoms,2.),dtype=torch.double)[None]
    qt=torch.tensor(q[order],dtype=torch.double)[None].requires_grad_(True)
    valid=torch.ones_like(qt,dtype=torch.bool)
    actual=measure(k,qt,valid,.3)
    np.testing.assert_allclose(actual.detach().numpy()[0],w,atol=1e-6)
    assert torch.autograd.gradcheck(lambda x:measure(k,x,valid,.3),(qt,),atol=1e-5)

def test_boundary_and_permutation():
    k=torch.eye(4,dtype=torch.double)[None];q=torch.tensor([[2.,-.5,0.,1.]],dtype=torch.double)
    valid=torch.ones_like(q,dtype=torch.bool)
    w=measure(k,q,valid,1.)
    assert float(w.min())>=-1e-9 and abs(float(w.sum())-1)<1e-9
    perm=[3,0,2,1]
    wp=measure(k[:,perm][:,:,perm],q[:,perm],valid[:,perm],1.)
    torch.testing.assert_close(w[:,perm],wp)

def test_gray_missing_and_56_coordinates():
    h=torch.zeros((2,12,10,8),dtype=torch.double)
    assert coordinates(h).shape[-1]==56 and coordinates(h,'moment').abs().max()==0
    head=RIMEHead(26).double();head.beta.data.fill_(1.)
    valid=torch.zeros((2,12,10),dtype=torch.bool)
    k=torch.eye(10,dtype=torch.double).expand(2,12,10,10)
    anchor=torch.tensor([-.4,.6],dtype=torch.double)
    torch.testing.assert_close(head(h,k,valid,torch.zeros((2,26),dtype=torch.double),anchor,torch.ones(56,dtype=torch.double)),anchor)
