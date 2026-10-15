import runtime
from runtime import config
import copy,time,numpy as np,torch
from sklearn.metrics import roc_auc_score
from heads import make_head

def fit(train,validation,y_train,g_train,y_val,lam,seed,spec,fixed_epochs=None):
    cfg=config();start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    head=make_head(train['b'].shape[-1],spec).to(train['h'])
    optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=cfg['lr'])
    yy=torch.as_tensor(y_train,dtype=train['h'].dtype,device=train['h'].device)
    _,encoded=np.unique(g_train,return_inverse=True);groups=torch.as_tensor(encoded,device=yy.device)
    best=None;bad=0;curve=[];limit=cfg['max_epochs'] if fixed_epochs is None else fixed_epochs
    if train['h'].device.type=='cuda':torch.cuda.reset_peak_memory_stats(train['h'].device)
    for epoch in range(limit+1):
        info=None
        if epoch:
            head.train();optimizer.zero_grad();z=head(**train)
            risk=[];rank=z.new_zeros(())
            for c in torch.unique(groups):
                zz=z[groups==c];y=yy[groups==c]
                risk.append(torch.nn.functional.binary_cross_entropy_with_logits(zz,y))
                pos=zz[y==1];neg=zz[y==0]
                if len(pos) and len(neg):rank+=torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean()/len(torch.unique(groups))
            classification=torch.stack(risk).mean() if spec['weighting']=='centre' else torch.nn.functional.binary_cross_entropy_with_logits(z,yy)
            regularizer=lam*sum(p.square().sum() for p in head.parameters() if p.requires_grad)
            loss=classification+spec['ranking']*rank+regularizer
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite source objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            info=dict(classification=float(classification.detach()),ranking=float(rank.detach()),regularizer=float(regularizer.detach()))
        head.eval()
        with torch.no_grad():
            if validation is not None:
                v=head(**validation).cpu().numpy();p=1/(1+np.exp(-np.clip(v,-700,700)))
                nll=float(np.mean(np.logaddexp(0,v)-y_val*v));brier=float(np.mean((p-y_val)**2));auc=float(roc_auc_score(y_val,v));score=(-nll,-brier,auc,-epoch)
            else:score=None
        if validation is not None:
            curve.append(dict(epoch=epoch,NLL=nll,Brier=brier,AUROC=auc,loss_terms=info))
            if best is None or score>best[0]:best=(score,copy.deepcopy(head.state_dict()),epoch);bad=0
            else:bad+=1
            if bad>=cfg['patience']:break
        elif epoch==limit:best=(None,copy.deepcopy(head.state_dict()),epoch)
    head.load_state_dict(best[1]);head.eval()
    with torch.no_grad():delta=head(**train)-train['anchor']
    return head,dict(seed=seed,lambda_=lam,best_epoch=best[2],epochs_run=epoch,curve=curve,
        fit_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        allocated_parameters=sum(p.numel() for p in head.parameters()),source_only=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(train['h'].device) if train['h'].device.type=='cuda' else 0,
        mean_abs_source_delta=float(delta.abs().mean()),source_nonzero_fraction=float((delta.abs()>1e-4).double().mean()))
