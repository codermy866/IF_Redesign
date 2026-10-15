"""Full patient objective with bounded-memory, stochastic-forward exact replay."""
import runtime
from runtime import HERE,PREVIOUS,BASE,save
import copy,time,json,numpy as np,torch
from sklearn.metrics import roc_auc_score
from recent_models import make

def cfg():return json.loads((HERE/'RECENT_CONFIG.json').read_text())
def objective(z,y,g):
    values=[];rank=z.new_zeros(())
    centres=torch.unique(g)
    for c in centres:
        zz=z[g==c];yy=y[g==c];values.append(torch.nn.functional.binary_cross_entropy_with_logits(zz,yy))
        pos=zz[yy==1];neg=zz[yy==0]
        if len(pos) and len(neg):rank+=torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean()/len(centres)
    return torch.stack(values).mean()+.1*rank

def rngstate(device):return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
def restore_rng(state,device):
    torch.set_rng_state(state[0].cpu())
    if state[1] is not None:torch.cuda.set_rng_state(state[1].cpu(),device)

def gradient_epoch(head,bags,anchor,y,g,order,epoch):
    device=anchor.device;N=len(bags);states={};values=torch.zeros_like(anchor)
    head.train()
    with torch.no_grad():
        for i in order:
            states[i]=(rngstate(device),head.buffer_state())
            values[i],_=head.one(bags[i],anchor[i],y[i],epoch)
    after=head.buffer_state();after_rng=rngstate(device)
    leaf=values.detach().requires_grad_(True);loss=objective(leaf,y,g);dz=torch.autograd.grad(loss,leaf)[0]
    centres=torch.unique(g);weights=torch.stack([1/(len(centres)*(g==g[i]).sum()) for i in range(N)]).to(y)
    extras=0.
    for i in order:
        state,buffers=states[i];restore_rng(state,device);head.restore_buffers(buffers)
        z,extra=head.one(bags[i],anchor[i],y[i],epoch)
        if not torch.isfinite(z) or not torch.isfinite(extra):raise RuntimeError('Nonfinite author adapter source objective')
        if not torch.allclose(z.detach(),values[i],atol=1e-5,rtol=1e-5):raise RuntimeError('Author stochastic replay mismatch')
        (z*dz[i]+weights[i]*extra).backward();extras+=float((weights[i]*extra).detach())
    head.restore_buffers(after);restore_rng(after_rng,device)
    return float(loss.detach())+extras

def predict(head,bags,anchor):
    head.eval()
    with torch.no_grad():return np.array([float(head.one(bag,anchor[i])[0]) for i,bag in enumerate(bags)])

def fit(kind,train,validation,train_anchor,validation_anchor,y,g,y_val,weight_decay,seed,fixed_epochs=None,resume_path=None):
    c=cfg();torch.manual_seed(seed);np.random.seed(seed);start=time.monotonic();device=train_anchor.device
    head=make(kind,train[0].shape[-1]).to(device)
    params=[p for p in head.parameters() if p.requires_grad]
    opt=torch.optim.SGD(params,lr=c['lr'][kind],momentum=.9,weight_decay=weight_decay) if kind=='DGR' else torch.optim.AdamW(params,lr=c['lr'][kind],weight_decay=weight_decay)
    yy=torch.as_tensor(y,dtype=train_anchor.dtype,device=device);_,encoded=np.unique(g,return_inverse=True);groups=torch.tensor(encoded,device=device)
    limit=c['max_epochs'] if fixed_epochs is None else fixed_epochs;best=None;bad=0;curve=[];first_epoch=0;prior_seconds=0.
    if resume_path is not None and resume_path.exists():
        ck=torch.load(resume_path,map_location=device,weights_only=False)
        assert ck['kind']==kind and ck['seed']==seed and ck['weight_decay']==weight_decay
        head.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);best=ck['best'];bad=ck['bad'];curve=ck['curve'];first_epoch=ck['epoch']+1;prior_seconds=ck['seconds'];restore_rng(ck['rng'],device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    epoch=first_epoch-1
    for epoch in range(first_epoch,limit+1):
        if time.time()>c['deadline_unix']:raise TimeoutError('Original campaign deadline reached')
        trainloss=None
        if epoch:
            opt.zero_grad();order=np.random.default_rng(seed+epoch).permutation(len(train))
            trainloss=gradient_epoch(head,train,train_anchor,yy,groups,order,epoch)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if validation is not None:
            z=predict(head,validation,validation_anchor);p=1/(1+np.exp(-np.clip(z,-700,700)))
            nll=float(np.mean(np.logaddexp(0,z)-y_val*z));brier=float(np.mean((p-y_val)**2));auc=float(roc_auc_score(y_val,z));score=(-nll,-brier,auc,-epoch)
            curve.append(dict(epoch=epoch,NLL=nll,Brier=brier,AUROC=auc,training_loss=trainloss))
            if best is None or score>best[0]:best=(score,copy.deepcopy(head.state_dict()),epoch);bad=0
            else:bad+=1
            if bad>=c['patience']:break
        elif epoch==limit:best=(None,copy.deepcopy(head.state_dict()),epoch)
        if resume_path is not None and epoch%5==0:
            temp=resume_path.with_suffix('.tmp')
            torch.save(dict(kind=kind,seed=seed,weight_decay=weight_decay,epoch=epoch,state=head.state_dict(),optimizer=opt.state_dict(),best=best,bad=bad,curve=curve,rng=rngstate(device),seconds=prior_seconds+time.monotonic()-start),temp);temp.replace(resume_path)
            save(resume_path.with_suffix('.progress.json'),dict(epoch=epoch,kind=kind,seed=seed,selected_epoch=best[2],time=time.time(),source_only=True))
    head.load_state_dict(best[1]);head.eval()
    return head,dict(kind=kind,seed=seed,weight_decay=weight_decay,best_epoch=best[2],epochs_run=epoch,curve=curve,
        cap_reached_still_improving=bool(validation is not None and epoch==limit and best[2]>=limit-5),
        parameters=sum(p.numel() for p in params),fit_seconds=prior_seconds+time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        gradient_rule='full-source centre BCE and full within-centre rank; exact two-pass dropout and buffer replay',source_only=True)
