"""A common, source-only centering/gray-reanchoring correction for every arm."""
import runtime
from runtime import HERE,BASE,OLD,save,config,sha
import json,joblib,numpy as np,torch
from rime_dataset import load_patients
from if_redesign.rime.readout import coordinates

def repair_project(raw,valid,state):
    basis=state['basis'];scale=state['scale'];mu=state['fit_centering']@basis
    neutral=np.clip(-mu/scale,-3,3)
    out=np.zeros((*valid.shape,8),dtype=np.float64);clipped=np.zeros_like(out,dtype=bool)
    for start in range(0,len(raw),8):
        z=(np.asarray(raw[start:start+8],dtype=np.float64)@basis-mu)/scale
        clipped[start:start+8]=abs(z)>=3
        out[start:start+8]=(np.clip(z,-3,3)-neutral)/6
    return np.where(valid[...,None],out,0.),clipped

def prepare(outer,partname):
    path=HERE/'private/fold_features'/outer/(partname+'.joblib')
    if path.exists():return joblib.load(path)
    oldpath=OLD/'private/fold_features'/outer/(partname+'.joblib');part=joblib.load(oldpath)
    rows=load_patients();g=np.array([r['center'] for r in rows]);tr=np.flatnonzero(g!=outer)
    if partname!='outer':tr=tr[g[tr]!=partname]
    assert np.array_equal(part['train'],tr) and np.array_equal(part['anchor']['train'],tr)
    assert np.array_equal(part['projection']['training_patients'],tr)
    raw=np.load(OLD/'private/frame_hidden.npy',mmap_mode='r');valid=part['valid']
    h,clipped=repair_project(raw,valid,part['projection']);values=h[tr][valid[tr]]
    source_clipped=float(clipped[tr][valid[tr]].mean());nonconstant=int((values.std(0)>config()['projection_gate']['std_nonconstant_threshold']).sum())
    gate=config()['projection_gate']
    if source_clipped>gate['maximum_source_clip_fraction'] or nonconstant<gate['minimum_nonconstant_dimensions']:
        raise RuntimeError('Shared source preprocessing repair failed its fixed gate; no performance comparisons')
    part['h']=h;part['repair']=dict(version='FR1',source=tr,original_part_sha256=sha(oldpath),source_clip_fraction=source_clipped,
        nonconstant_dimensions=nonconstant,old_clip_fraction=float((abs(joblib.load(oldpath)['h'][tr][valid[tr]])>=1-1e-12).mean()),
        neutral_exactly_zero=True,patient_anchor_unchanged=True,no_labels_used=True)
    with torch.no_grad():phi=coordinates(torch.tensor(h[tr],dtype=torch.double),'laplace').numpy()
    vm=valid[tr];count=vm.sum(-1);sites=(count>0).sum(-1)
    weight=vm/count.clip(1)[...,None]/sites.clip(1)[:,None,None]/len(tr)
    mean=(phi*weight[...,None]).sum((0,1,2));part['coordinate_scale']=np.maximum(np.sqrt(((phi-mean)**2*weight[...,None]).sum((0,1,2))),.005)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(part,path)
    save(path.with_suffix('.audit.json'),part['repair'])
    return part

def inputs(part,at,device):
    at=np.asarray(at);tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    return dict(h=tensor(part['h'][at]),valid=torch.as_tensor(part['valid'][at],dtype=torch.bool,device=device),
        k=tensor(np.zeros((len(at),1,1,1))),b=tensor(part['anchor']['b'][at]),anchor=tensor(part['anchor']['anchor'][at]).detach(),
        coordinate_scale=tensor(part['coordinate_scale']))
