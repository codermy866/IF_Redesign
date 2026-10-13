"""Only scorer mode, selector warmup, or prespecified auxiliary supervision vary."""
import copy,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from if_redesign.rime.training import loss
from scorers import ScoreControl,site_auxiliary

def train_control(train,validation,y,groups,tau,lam,seed,*,mode='conditional',warmup=0,
                  fixed_epochs=None,max_epochs=300,patience=30,auxiliary=None,aux_weight=0.,auxiliary_mode='value'):
    start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    head=ScoreControl(train['b'].shape[-1],tau,'measure','laplace',mode).to(train['h'])
    selector=[p for p in (getattr(head,'a',None),getattr(head,'U',None),getattr(head,'V',None),head.g) if p is not None and p.requires_grad]
    if mode=='local_capacity_matched':selector.extend(head.local_scorer.parameters());selector.append(head.local_skip)
    optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.003)
    if warmup:
        for p in selector:p.requires_grad_(False)
    at=train['indices'];yy=torch.as_tensor(y[at],device=train['h'].device,dtype=train['h'].dtype)
    gg=groups[at];best=None;bad=0;curve=[]
    for epoch in range((fixed_epochs if fixed_epochs is not None else max_epochs)+1):
        if epoch==warmup+1 and warmup:
            for p in selector:p.requires_grad_(True)
        train_loss=aux_loss=None
        if epoch:
            head.train();optimizer.zero_grad()
            z=head(**{k:v for k,v in train.items() if k!='indices'})
            objective=loss(z,yy,gg,head,lam)
            if auxiliary is not None:
                if auxiliary_mode=='value':aux_loss=site_auxiliary(head,train,auxiliary,aux_weight)
                elif auxiliary_mode=='patient':
                    q=head.scores(train['h'],train['b']);mask=train['valid']
                    mean=(q*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
                    aux_loss=aux_weight*torch.nn.functional.binary_cross_entropy_with_logits(train['anchor']+mean,yy)
                else:raise ValueError(auxiliary_mode)
                objective=objective+aux_loss
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite control loss')
            train_loss=float(objective.detach());aux_loss=None if aux_loss is None else float(aux_loss.detach())
            objective.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        if validation is not None:
            head.eval()
            with torch.no_grad():zz=head(**{k:v for k,v in validation.items() if k!='indices'}).cpu().numpy()
            vy=y[validation['indices']]
            score=(float(roc_auc_score(vy,zz)),-float(np.mean(np.logaddexp(0,zz)-vy*zz)),-epoch)
            curve.append(dict(epoch=epoch,AUROC=score[0],NLL=-score[1],train_objective=train_loss,auxiliary_objective=aux_loss))
            if best is None or score>best[0]:best=(score,copy.deepcopy(head.state_dict()),epoch);bad=0
            elif epoch>warmup:bad+=1
            if epoch>warmup and bad>=patience:break
        elif fixed_epochs is not None and epoch==fixed_epochs:best=(None,copy.deepcopy(head.state_dict()),epoch)
    head.load_state_dict(best[1]);head.eval()
    for p in selector:p.requires_grad_(True)
    return head,dict(best_epoch=best[2],epochs_run=epoch,curve=curve,fit_seconds=time.monotonic()-start,
        parameters=sum(p.numel() for group in optimizer.param_groups for p in group['params']),
        allocated_parameters=sum(p.numel() for p in head.parameters()),seed=seed,tau=tau,lambda_=lam,
        pooling='measure',readout='laplace',source_only=True,selector_warmup_epochs=warmup,score_mode=mode,
        auxiliary_weight=aux_weight,auxiliary_mode=auxiliary_mode,site_supervision_identifies_frame_value=False)
