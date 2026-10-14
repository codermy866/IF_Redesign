"""No new forward architecture: six patient-loss comparisons on one PALE head."""
import torch

def centre_risks(z,y,groups):
    losses=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
    return torch.stack([losses[groups==c].mean() for c in torch.unique(groups)]),losses

def anchor_kl(a,z):
    a=a.detach();p=torch.sigmoid(a)
    return (p*(a-z)+torch.nn.functional.softplus(z)-torch.nn.functional.softplus(a)).clamp_min(0).mean()

def penalty(z,a,y,groups,method,candidate):
    a=a.detach()
    if method in ['A0_PALE','A1_STRONG_L2','A3_GROUP_DRO']:return z.new_zeros(())
    if method=='A2_ANCHOR_KL':return candidate['gamma']*anchor_kl(a,z)
    if method=='A4_GLOBAL_RISK':
        delta=torch.nn.functional.binary_cross_entropy_with_logits(z,y)-torch.nn.functional.binary_cross_entropy_with_logits(a,y)
        return candidate['gamma']*torch.relu(delta-candidate['epsilon']).square()
    if method=='A5_ARDR':
        current,_=centre_risks(z,y,groups);reference,_=centre_risks(a,y,groups)
        return candidate['gamma']*torch.relu(current-reference-candidate['epsilon']).square().mean()
    raise ValueError(method)

def dro_update(weights,risks,eta):
    # Exponentiated-gradient group weighting (Sagawa et al. reference), stable
    # log-space form; no extra size adjustment, no held-out risk in weights.
    return torch.softmax(torch.log(weights.clamp_min(1e-300))+eta*risks.detach(),0)

def objective(z,a,y,groups,head,lam,method,candidate,dro_weights):
    risks,_=centre_risks(z,y,groups);ranking=z.new_zeros(())
    for c in torch.unique(groups):
        zz=z[groups==c];yy=y[groups==c];pos=zz[yy==1];neg=zz[yy==0]
        if len(pos) and len(neg):ranking+=torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean()/len(risks)
    if method=='A3_GROUP_DRO':
        dro_weights=dro_update(dro_weights,risks,candidate['eta']);classification=(dro_weights*risks).sum()
    else:classification=risks.mean()
    multiplier=candidate.get('lambda_multiplier',1.)
    regularizer=lam*multiplier*sum(p.square().sum() for p in head.parameters() if p.requires_grad)
    extra=penalty(z,a,y,groups,method,candidate)
    return classification+.1*ranking+regularizer+extra,dro_weights,dict(classification=classification,ranking=ranking,regularizer=regularizer,extra=extra)
