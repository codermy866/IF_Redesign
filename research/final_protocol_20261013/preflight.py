"""Numerical/gradient/capacity controls; these are not clinical results."""
import runtime
from runtime import HERE,ROOT,save
import importlib.util,time
import numpy as np,torch
from if_redesign.rime.training import train_head
from if_redesign.rime.readout import RIMEHead
from train_control import train_control
from scorers import ScoreControl,site_auxiliary

def fixture():
    torch.manual_seed(191);h=torch.randn(16,2,4,8,dtype=torch.double)*.15
    delta=h[...,None,:]-h[...,None,:,:]
    k=torch.exp(-delta.square().sum(-1)/(.35**2*2))
    values=dict(h=h,k=k,valid=torch.ones(16,2,4,dtype=torch.bool),b=torch.randn(16,34,dtype=torch.double),
        anchor=torch.randn(16,dtype=torch.double)*.2,coordinate_scale=torch.ones(56,dtype=torch.double))
    def subset(indices):
        return {key:(value if key=='coordinate_scale' else value[indices]) for key,value in values.items()}|dict(indices=np.asarray(indices))
    return subset(np.arange(8)),subset(np.arange(8,16)),np.tile([0,1],8),np.repeat(['sourceA','sourceB'],8)

def main():
    torch.set_num_threads(1);ti,vi,y,g=fixture();start=time.monotonic()
    spec=importlib.util.spec_from_file_location('if_redesign.rime.original_private_training',ROOT/'src/if_redesign/rime/training.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    results=[]
    for pool in ['attention','measure']:
        for read in ['moment','laplace']:
            old,om=legacy.train_head(ti,vi,y,g,pool,read,.3,.003,17,max_epochs=8,patience=3)
            new,nm=train_head(ti,vi,y,g,pool,read,.3,.003,17,max_epochs=8,patience=3,selector_warmup_epochs=0)
            err=max(float((old.state_dict()[k]-v).abs().max()) for k,v in new.state_dict().items())
            assert err==0 and om['best_epoch']==nm['best_epoch']
            results.append(dict(pool=pool,readout=read,legacy_parameter_max_error=err,best_epoch=nm['best_epoch'],effective_parameters=nm['parameters']))
    new,md=train_head(ti,vi,y,g,'measure','laplace',.3,.003,17,max_epochs=8,patience=3)
    ctrl,cm=train_control(ti,vi,y,g,.3,.003,17,max_epochs=8,patience=3)
    assert max(float((new.state_dict()[k]-v).abs().max()) for k,v in ctrl.state_dict().items())==0
    head=RIMEHead(34).double();z=head(**{k:v for k,v in ti.items() if k!='indices'})
    z.sum().backward();gradients={name:float(getattr(head,name).grad.abs().max()) for name in ['a','U','V','g']}
    assert all(v==0 for v in gradients.values()) and float(head.beta.grad.abs().max())>0
    with torch.no_grad():head.beta.fill_(.02)
    head.zero_grad();head(**{k:v for k,v in ti.items() if k!='indices'}).sum().backward()
    after={name:float(getattr(head,name).grad.abs().max()) for name in ['a','U','V','g']}
    assert all(v>0 for v in after.values())
    capacities=[]
    for mode in ['conditional','geometry','local','local_capacity_matched']:
        torch.manual_seed(17);m=ScoreControl(34,.3,'measure','laplace',mode).double()
        capacities.append(dict(mode=mode,effective_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),allocated_parameters=sum(p.numel() for p in m.parameters())))
        zz=m(**{k:v for k,v in ti.items() if k!='indices'});assert torch.isfinite(zz).all()
        if mode in ['local','local_capacity_matched','geometry']:
            assert torch.equal(m.scores(ti['h'],ti['b']),m.scores(ti['h'],ti['b']+3))
        if mode=='local_capacity_matched':
            reference=capacities[0]
            assert capacities[-1]['effective_parameters']==reference['effective_parameters']
            assert capacities[-1]['allocated_parameters']==reference['allocated_parameters']
    # Constant site advantages cannot change simplex QP frame weights.
    from if_redesign.rime.measure import measure
    q=head.scores(ti['h'],ti['b']).detach();valid=ti['valid'];k=ti['k']
    assert torch.allclose(measure(k,q,valid,.3),measure(k,q+.2,valid,.3),atol=1e-10,rtol=0)
    aux=site_auxiliary(head,ti,torch.full((8,2),.05,dtype=torch.double),.1);head.zero_grad();aux.backward()
    assert float(head.a.grad.abs().max())>0
    save(HERE/'PREFLIGHT_GATE.json',dict(state='PASS',clinical_evaluation=False,legacy_parity=results,
        control_joint_parameter_parity=True,beta_zero_prediction_gradient=gradients,beta_nonzero_prediction_gradient=after,
        B2_capacity=capacities,constant_site_reward_cancels_on_simplex=True,site_auxiliary_has_selector_gradient=True,
        seconds=time.monotonic()-start))

if __name__=='__main__':main()
