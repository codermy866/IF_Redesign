import runtime
from runtime import config
import copy,time,numpy as np,torch
from sklearn.metrics import roc_auc_score
from if_redesign.rime.readout import RIMEHead
from objectives import objective,centre_risks

def fit(train,validation,y_train,g_train,y_val,lam,seed,method,candidate,fixed_epochs=None):
    cfg=config();start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
    head=RIMEHead(train['b'].shape[-1],.3,'mlp','laplace').to(train['h'])
    optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.003)
    yy=torch.as_tensor(y_train,dtype=train['h'].dtype,device=train['h'].device)
    _,encoded=np.unique(g_train,return_inverse=True);groups=torch.as_tensor(encoded,device=yy.device)
    C=len(np.unique(encoded));weights=torch.ones(C,dtype=yy.dtype,device=yy.device)/C
    best=None;bad=0;curve=[];limit=cfg['max_epochs'] if fixed_epochs is None else fixed_epochs
    for epoch in range(limit+1):
        info=None
        if epoch:
            head.train();optimizer.zero_grad();z=head(**train)
            value,weights,terms=objective(z,train['anchor'],yy,groups,head,lam,method,candidate,weights)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite source training objective')
            value.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            info={k:float(v.detach()) for k,v in terms.items()}
        head.eval()
        with torch.no_grad():
            if validation is not None:
                v=head(**validation).cpu().numpy();p=1/(1+np.exp(-np.clip(v,-700,700)))
                nll=float(np.mean(np.logaddexp(0,v)-y_val*v));brier=float(np.mean((p-y_val)**2));auc=float(roc_auc_score(y_val,v))
                score=(-nll,-brier,auc,-epoch)
            else:score=None
        if validation is not None:
            curve.append(dict(epoch=epoch,NLL=nll,Brier=brier,AUROC=auc,loss_terms=info))
            if best is None or score>best[0]:best=(score,copy.deepcopy(head.state_dict()),epoch);bad=0
            else:bad+=1
            if bad>=cfg['patience']:break
        elif epoch==limit:best=(None,copy.deepcopy(head.state_dict()),epoch)
    head.load_state_dict(best[1]);head.eval()
    with torch.no_grad():
        z=head(**train);risk,_=centre_risks(z,yy,groups);anchor,_=centre_risks(train['anchor'].detach(),yy,groups)
        delta=z-train['anchor'];risk_delta=(risk-anchor).cpu().numpy();eps=candidate.get('epsilon',0.)
    md=dict(method=method,candidate=candidate,lambda_=lam,seed=seed,best_epoch=best[2],epochs_run=epoch,curve=curve,
        fit_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),allocated_parameters=sum(p.numel() for p in head.parameters()),
        training_centres=sorted(set(g_train)),training_anchor_risk=anchor.cpu().numpy(),training_model_risk=risk.cpu().numpy(),
        training_risk_delta=risk_delta,constraint_satisfaction_fraction=float((risk_delta<=eps+1e-8).mean()),
        source_mean_abs_delta=float(delta.abs().mean()),source_nonzero_delta_fraction=float((delta.abs()>1e-4).double().mean()),
        dro_weights=weights.cpu().numpy(),source_only=True,selection='NLL then Brier then AUC, all six arms')
    return head,md
