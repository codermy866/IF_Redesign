import runtime
from runtime import save,HERE
import numpy as np,torch
from scipy.special import expit
from objectives import penalty,anchor_kl,dro_update,objective
from data import repair_project
from training import fit
from if_redesign.rime.training import train_head
from if_redesign.rime.readout import RIMEHead

def main():
    torch.set_num_threads(1)
    a=torch.tensor([1.,-1.,2.,-2.],dtype=torch.double,requires_grad=True)
    z=torch.tensor([-1.,1.,2.,-2.],dtype=torch.double,requires_grad=True)
    y=torch.tensor([1.,0.,1.,0.],dtype=torch.double);g=torch.tensor([0,0,1,1])
    loss=penalty(z,a,y,g,'A5_ARDR',dict(gamma=3.,epsilon=0.));loss.backward()
    assert a.grad is None and z.grad[:2].abs().min()>0 and torch.equal(z.grad[2:],torch.zeros_like(z.grad[2:]))
    for method in ['A2_ANCHOR_KL','A4_GLOBAL_RISK','A5_ARDR']:
        assert float(penalty(a.detach(),a.detach(),y,g,method,dict(gamma=1.,epsilon=0.)))<1e-12
    kl=anchor_kl(torch.tensor([1000.,-1000.]),torch.tensor([-1000.,1000.]));assert torch.isfinite(kl) and kl>0
    aa=torch.tensor([.2,-.5],dtype=torch.double);zz=torch.tensor([.8,-.9],dtype=torch.double)
    p=torch.sigmoid(aa);q=torch.sigmoid(zz)
    expected=(p*torch.log(p/q)+(1-p)*torch.log((1-p)/(1-q))).mean()
    torch.testing.assert_close(anchor_kl(aa,zz),expected,atol=1e-14,rtol=0)
    weights=torch.tensor([.5,.5],dtype=torch.double);risks=torch.tensor([.2,1.],dtype=torch.double)
    result=dro_update(weights,risks,.01);reference=weights*torch.exp(.01*risks);reference/=reference.sum();torch.testing.assert_close(result,reference)
    rng=np.random.default_rng(17);raw=rng.normal(size=(8,2,4,16));vm=np.ones(raw.shape[:-1],bool)
    state=dict(basis=np.eye(16)[:,:8],scale=np.ones(8),fit_centering=np.arange(16)*10.)
    neutral,_=repair_project(np.zeros_like(raw),vm,state);assert np.array_equal(neutral,np.zeros_like(neutral))
    h,_=repair_project(raw,vm,state);assert abs(h).max()<=1
    h=torch.tensor(rng.normal(size=(8,2,4,8))*.1,dtype=torch.double)
    inp=dict(h=h,k=torch.zeros(8,1,1,1,dtype=torch.double),valid=torch.ones(8,2,4,dtype=torch.bool),b=torch.tensor(rng.normal(size=(8,34)),dtype=torch.double),anchor=torch.zeros(8,dtype=torch.double),coordinate_scale=torch.ones(56,dtype=torch.double))
    yy=np.tile([0,1],4);groups=np.repeat(['a','b'],4)
    legacy,old=train_head(inp|dict(indices=np.arange(8)),None,yy,groups,'mlp','laplace',.3,.003,17,fixed_epochs=2)
    new,md=fit(inp,None,yy,groups,None,.003,17,'A0_PALE',{},fixed_epochs=2)
    for name,value in legacy.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],atol=1e-12,rtol=1e-12)
    state0=None
    for method,candidate in [('A0_PALE',{}),('A1_STRONG_L2',{'lambda_multiplier':2.}),('A2_ANCHOR_KL',{'gamma':1.}),('A3_GROUP_DRO',{'eta':.01}),('A4_GLOBAL_RISK',{'gamma':1.,'epsilon':0.}),('A5_ARDR',{'gamma':1.,'epsilon':0.})]:
        model,meta=fit(inp,inp,yy,groups,yy,.003,17,method,candidate)
        assert torch.isfinite(model(**inp)).all()
        assert set(model.state_dict())==set(legacy.state_dict()) and meta['parameters']==old['parameters']
    save(HERE/'PREFLIGHT_GATE.json',dict(state='PASS',anchor_stopped_gradient=True,centre_specific_active_gradient=True,
        extreme_KL_finite=True,KL_matches_probability_definition=True,DRO_matches_reference_update=True,
        source_only_repair_gray_zero=True,forward_architecture_capacity_unchanged=True,A0_legacy_objective_parity=True,
        all_six_objectives_finite=True,clinical_performance_test=False))
    print('ARDR mathematical/gradient/initialization/reference and source-repair preflight PASS')
if __name__=='__main__':main()
