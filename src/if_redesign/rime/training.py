"""Patient-level centre-balanced training; outer test labels never enter fit."""
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .readout import RIMEHead

def loss(z,y,groups,model,lam):
    value=z.new_zeros(())
    centers=np.unique(groups)
    for center in centers:
        at=torch.as_tensor(groups==center,device=z.device)
        zz=z[at];yy=y[at]
        value=value+torch.nn.functional.binary_cross_entropy_with_logits(zz,yy)/len(centers)
        pos=zz[yy==1];neg=zz[yy==0]
        if len(pos) and len(neg):value=value+.1*torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean()/len(centers)
    return value+lam*sum((p*p).sum() for p in model.parameters() if p.requires_grad)

def train_head(train,validation,y,groups,pooling,readout,tau,lam,seed,
               max_epochs=300,patience=30,fixed_epochs=None,caprmil_factory=None,
               selector_warmup_epochs=0):
    torch.manual_seed(seed);np.random.seed(seed)
    head=RIMEHead(train['b'].shape[-1],tau,pooling,readout).to(device=train['h'].device,dtype=train['h'].dtype)
    if pooling=='pca_refit' and 'initial_semantic_weight' in train:
        # Regularize effective semantic coefficients, not just a residual delta.
        with torch.no_grad():head.pca_weights.copy_(train['initial_semantic_weight'])
    if pooling=='caprmil':
        if caprmil_factory is None:
            raise ValueError("CAPR-MIL control requires a caller-supplied, separately licensed block factory")
        head.caprmil=caprmil_factory().to(device=train['h'].device,dtype=train['h'].dtype)
    if selector_warmup_epochs < 0 or not isinstance(selector_warmup_epochs,int):
        raise ValueError('selector_warmup_epochs must be a nonnegative integer')
    # Construct all trainable parameter groups before freezing the selector.
    # Phase 1 fits the patient residual against geometry-only/kernel or uniform
    # attention; Phase 2 releases conditional scores and site attention.
    # Zero reproduces the legacy joint-training regime exactly.
    selector=tuple(p for p in (head.a,head.U,head.V,head.g) if p.requires_grad)
    optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.003)
    if selector_warmup_epochs:
        for p in selector:p.requires_grad_(False)
    indices=train['indices'];yy=torch.as_tensor(y[indices],device=train['h'].device,dtype=train['h'].dtype)
    gg=groups[indices];best=None;bad=0;curve=[]
    # The beta=0 anchor is a real selection candidate, including epoch zero.
    for epoch in range((fixed_epochs if fixed_epochs is not None else max_epochs)+1):
        if epoch and selector_warmup_epochs and epoch==selector_warmup_epochs+1:
            for p in selector:p.requires_grad_(True)
        if epoch:
            head.train();optimizer.zero_grad();z=head(**{k:v for k,v in train.items() if k!='indices'})
            objective=loss(z,yy,gg,head,lam)
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite patient objective')
            objective.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        if validation is not None:
            head.eval()
            with torch.no_grad():zz=head(**{k:v for k,v in validation.items() if k!='indices'}).cpu().numpy()
            vy=y[validation['indices']]
            score=(float(roc_auc_score(vy,zz)),-float(np.mean(np.logaddexp(0,zz)-vy*zz)),-epoch)
            curve.append(dict(epoch=epoch,AUROC=score[0],NLL=-score[1]))
            if best is None or score>best[0]:best=(score,copy.deepcopy(head.state_dict()),epoch);bad=0
            elif epoch > selector_warmup_epochs:bad+=1
            # Do not terminate before the source-defined selector phase has run.
            if epoch > selector_warmup_epochs and bad>=patience:break
        elif fixed_epochs is not None and epoch==fixed_epochs:
            best=(None,copy.deepcopy(head.state_dict()),epoch)
    head.load_state_dict(best[1]);head.eval()
    for p in selector:p.requires_grad_(True)
    return head,dict(best_epoch=best[2],epochs_run=epoch,curve=curve,
        parameters=sum(p.numel() for group in optimizer.param_groups for p in group['params']),allocated_parameters=sum(p.numel() for p in head.parameters()),seed=seed,tau=tau,lambda_=lam,
        pooling=pooling,readout=readout,source_only=True,
        selector_warmup_epochs=selector_warmup_epochs)
