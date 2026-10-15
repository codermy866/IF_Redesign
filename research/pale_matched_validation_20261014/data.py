import runtime
from runtime import HERE,PREVIOUS,config,save
import numpy as np,torch,joblib
from rime_dataset import load_patients

def prepare(outer,held):
    part=joblib.load(PREVIOUS/'private/fold_features'/outer/(held+'.joblib'))
    rows=load_patients();g=np.array([r['center'] for r in rows])
    tr=np.flatnonzero(g!=outer)
    if held!='outer':tr=tr[g[tr]!=held]
    assert np.array_equal(part['train'],tr)
    assert np.array_equal(part['anchor']['train'],tr)
    assert np.array_equal(part['projection']['training_patients'],tr)
    assert part['repair']['version']=='FR1'
    return part

def inputs(part,at,device,scale=None):
    at=np.asarray(at);tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    return dict(h=tensor(part['h'][at]),valid=torch.as_tensor(part['valid'][at],dtype=torch.bool,device=device),
        k=tensor(np.zeros((len(at),1,1,1))),b=tensor(part['anchor']['b'][at]),
        anchor=tensor(part['anchor']['anchor'][at]).detach(),
        coordinate_scale=tensor(part['coordinate_scale'] if scale is None else scale))

def source_scale(part,spec):
    from heads import summarize_sites,zero_readout,readout_dim
    if spec['readout']=='laplace':return part['coordinate_scale']
    tr=part['train'];h=torch.as_tensor(part['h'][tr],dtype=torch.double)
    valid=torch.as_tensor(part['valid'][tr],dtype=torch.bool)
    # Label-free reference transformation with adaptation replaced by identity.
    # Trainable readouts use a fixed unit scale; no post-fit or target scaling.
    if spec['readout']=='learned':return np.ones(readout_dim(spec['readout']))
    with torch.no_grad():
        if spec['readout']=='std':
            v=summarize_sites(h,valid,spec,None,None).numpy()
            vm=valid.any(-1).numpy();weight=vm/vm.sum(-1).clip(1)[:,None]/len(tr)
            mean=(v*weight[...,None]).sum((0,1));var=((v-mean)**2*weight[...,None]).sum((0,1))
        else:
            kind=spec['readout']
            if kind=='lse':v=torch.stack([h]+[(s*h).clamp_min(0) for s in [1.,-1.] for _ in [.5,2.,8.]],-1).flatten(-2).numpy()
            else:v=zero_readout(h,kind).numpy()
            vm=valid.numpy();count=vm.sum(-1);sites=(count>0).sum(-1)
            weight=vm/count.clip(1)[...,None]/sites.clip(1)[:,None,None]/len(tr)
            mean=(v*weight[...,None]).sum((0,1,2));var=((v-mean)**2*weight[...,None]).sum((0,1,2))
    return np.maximum(np.sqrt(var),.005)
