import runtime
from runtime import HERE,PREVIOUS,config,save,sha
import json,numpy as np,torch,joblib
from numpy.polynomial.legendre import legval
from heads import make_head,summarize_sites,zero_readout,readout_dim
from training import fit
from data import prepare,inputs,source_scale

def main():
    torch.set_num_threads(1);torch.manual_seed(17);cfg=config()
    x=torch.linspace(-1,1,17,dtype=torch.double).reshape(-1,1)
    expected=np.column_stack([legval(x.numpy().ravel(),[0]*d+[1])-legval(0,[0]*d+[1]) for d in range(1,8)])
    np.testing.assert_allclose(zero_readout(x,'moment').numpy(),expected,atol=1e-14)
    h=torch.randn(8,2,4,8,dtype=torch.double)*.1;valid=torch.ones(8,2,4,dtype=torch.bool)
    ti=dict(h=h,valid=valid,k=torch.zeros(8,1,1,1,dtype=torch.double),b=torch.randn(8,34,dtype=torch.double),anchor=torch.randn(8,dtype=torch.double),coordinate_scale=torch.ones(56,dtype=torch.double))
    y=np.tile([0,1],4);g=np.repeat(['a','b'],4)
    records=[]
    for name,spec in cfg['variants'].items():
        inp=ti|dict(coordinate_scale=torch.ones(readout_dim(spec['readout']),dtype=torch.double))
        head=make_head(34,spec).double()
        with torch.no_grad():head.beta.fill_(.1)
        z=head(**inp);assert torch.isfinite(z).all()
        empty=inp|dict(valid=torch.zeros_like(valid))
        target=ti['anchor'] if spec['anchor'] else torch.zeros_like(ti['anchor'])
        torch.testing.assert_close(head(**empty),target,atol=0,rtol=0)
        zero=inp|dict(h=torch.zeros_like(h))
        torch.testing.assert_close(head(**zero),target,atol=1e-12,rtol=0)
        torch.testing.assert_close(head(**(inp|dict(h=h[:,:,torch.tensor([3,1,0,2])]))),z,atol=1e-12,rtol=0)
        z.sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
        model,md=fit(inp,None,y,g,None,.003,17,spec,fixed_epochs=2)
        assert torch.isfinite(model(**inp)).all()
        records.append(dict(method=name,dimensions=readout_dim(spec['readout']),parameters=md['parameters'],finite_gradient=True))
    # Same mean, different distribution: order must have an observable effect.
    hh=torch.zeros(1,1,2,8,dtype=torch.double);hh[0,0,:,0]=torch.tensor([-.8,.8])
    vm=torch.ones(1,1,2,dtype=torch.bool)
    before=summarize_sites(hh,vm,cfg['variants']['P4_NO_MLP_LAPLACE'],None,None)
    after=summarize_sites(hh,vm,cfg['variants']['P5_MLP_MEAN_LAPLACE'],None,None)
    assert (before-after).abs().max()>.1
    # Full pipeline P3 checkpoint forward equality, all outer folds and seeds.
    errors=[]
    for outer in cfg['centers']:
        part=prepare(outer,'outer');assert np.array_equal(source_scale(part,cfg['variants']['P3_PALE']),part['coordinate_scale'])
        old=PREVIOUS/'private/runs'/outer
        with np.load(old/'A0_PALE.npz') as f:
            for si,seed in enumerate(cfg['seeds']):
                ck=torch.load(old/f'A0_PALE_seed{seed}_OUTER.pt',map_location='cpu',weights_only=False)
                model=make_head(part['anchor']['b'].shape[1],cfg['variants']['P3_PALE']).double();model.load_state_dict(ck['state']);model.eval()
                with torch.no_grad():z=model(**inputs(part,f['test'],'cpu')).numpy()
                error=float(abs(z-f['seed_logits'][si]).max());assert error<1e-10;errors.append(error)
    # Original A0 source objective and fixed-epoch parameters must agree.
    oldmodule=__import__('importlib.util',fromlist=['util'])
    s=oldmodule.spec_from_file_location('previous_objectives',PREVIOUS/'objectives.py');obj=oldmodule.module_from_spec(s);s.loader.exec_module(obj)
    s=oldmodule.spec_from_file_location('previous_training',PREVIOUS/'training.py');oldtrain=oldmodule.module_from_spec(s)
    import sys;sys.modules['objectives']=obj;s.loader.exec_module(oldtrain)
    oldhead,_=oldtrain.fit(ti,None,y,g,None,.003,17,'A0_PALE',{},fixed_epochs=2)
    newhead,_=fit(ti,None,y,g,None,.003,17,cfg['variants']['P3_PALE'],fixed_epochs=2)
    for key,value in oldhead.state_dict().items():torch.testing.assert_close(value,newhead.state_dict()[key],atol=1e-12,rtol=1e-12)
    save(HERE/'PREFLIGHT_GATE.json',dict(state='PASS',variants=records,legendre_reference=True,mask_and_zero_reference=True,
        source_only_projection_assertions=True,order_counterexample=True,P3_all15checkpoints_max_error=max(errors),P3_legacy_fixed_epoch_objective_parity=True,synthetic_tests_not_clinical_results=True))
    print('All variant mathematical/gradient checks and exact P3 reuse gate PASS',flush=True)
if __name__=='__main__':main()
