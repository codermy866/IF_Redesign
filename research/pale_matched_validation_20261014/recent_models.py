"""Pinned authors' MIL blocks with explicit patient-anchor adaptation."""
import runtime
from runtime import HERE
import sys,importlib,torch
from torch import nn
from types import SimpleNamespace

class AuthorHead(nn.Module):
    def __init__(self,kind,dim):
        super().__init__();self.kind=kind
        folder={'RRT':'RRT-MIL','WiKG':'WiKG','ACMIL':'ACMIL','DGR':'DGR-MIL'}[kind]
        sys.path.insert(0,str(HERE/'vendor'/folder))
        if kind=='RRT':
            from modules.rrt import RRTMIL
            self.model=RRTMIL(input_dim=dim,mlp_dim=512,n_classes=2,region_num=4,n_layers=2,n_heads=8,epeg=True,epeg_k=15,crmsa_k=3)
        elif kind=='WiKG':
            from model import WiKG
            self.model=WiKG(dim_in=dim,dim_hidden=512,topk=6,n_classes=2,agg_type='bi-interaction',dropout=.3,pool='attn')
            # Author file enables global anomaly debugging on import; that is
            # an execution diagnostic, not part of the forward/learning rule.
            torch.autograd.set_detect_anomaly(False)
        elif kind=='ACMIL':
            from architecture.transformer import ACMIL_GA
            self.model=ACMIL_GA(SimpleNamespace(D_feat=dim,D_inner=512,n_class=2,n_token=5),n_token=5,n_masked_patch=10,mask_drop=.6)
        else:
            from models.dgrmil import DGRMIL
            self.model=DGRMIL(in_features=dim,num_classes=2,L=512,n_lesion=5,dropout_node=.4,dropout_patch=.1)
    def buffer_state(self):
        if self.kind!='DGR':return {}
        return {name:getattr(self.model,name).detach().clone() for name in ['normalcenter','postivecenter']}
    def restore_buffers(self,state):
        if self.kind=='DGR':
            with torch.no_grad():
                for name,value in state.items():getattr(self.model,name).copy_(value)
    def one(self,features,anchor,label=None,epoch=0):
        x=features.unsqueeze(0);extra=anchor.new_zeros(())
        if self.kind=='WiKG':self.model.topk=min(6,len(features))
        if self.kind=='ACMIL':
            # Keep MBA and stochastic masking; never mask the whole small bag.
            self.model.n_masked_patch=min(10,max(0,len(features)-1))
            branches,slide,attn=self.model(x)
            r=slide.reshape(-1)[1]-slide.reshape(-1)[0]
            if self.training:
                branch=branches[...,1]-branches[...,0]+anchor
                extra=torch.nn.functional.binary_cross_entropy_with_logits(branch,label.expand_as(branch))
                distributions=attn.softmax(-1)[0]
                pairs=[torch.nn.functional.cosine_similarity(distributions[i],distributions[j],dim=0) for i in range(5) for j in range(i+1,5)]
                extra=extra+torch.stack(pairs).mean()
        elif self.kind=='DGR':
            result=self.model(x,bag_mode='abnormal' if self.training and bool(label.item()) else 'normal')
            logits=result[0].reshape(-1);r=logits[1]-logits[0]
            if self.training and epoch>=10:
                positive,negative,lesion=result[3],result[4],result[5]
                normalized=torch.nn.functional.normalize(lesion.squeeze(0),dim=-1).double()
                gram=normalized@normalized.T+1e-10*torch.eye(5,device=x.device,dtype=torch.double)
                diversity=-torch.logdet(gram)
                triplet=torch.nn.functional.triplet_margin_with_distance_loss(lesion.squeeze(0),positive.repeat(5,1).detach(),negative.repeat(5,1).detach(),distance_function=lambda a,b:1-torch.nn.functional.cosine_similarity(a,b),margin=1.)
                extra=.1*(diversity.to(x.dtype)+triplet)
        else:
            logits=self.model(x).reshape(-1);r=logits[1]-logits[0]
        return anchor+r,extra

class SimpleBagHead(nn.Module):
    def __init__(self,kind,dim):
        super().__init__();self.kind=kind
        self.encoder=nn.Sequential(nn.Linear(dim,56),nn.Tanh())
        if kind=='ABMIL':self.attention=nn.Sequential(nn.Linear(56,16),nn.Tanh(),nn.Linear(16,1))
        self.predictor=nn.Sequential(nn.Linear(56,16),nn.Tanh(),nn.Linear(16,1)) if kind=='DeepSets' else nn.Linear(56,1)
    def buffer_state(self):return {}
    def restore_buffers(self,state):pass
    def one(self,features,anchor,label=None,epoch=0):
        h=self.encoder(features)
        if self.kind=='ABMIL':h=(self.attention(h).softmax(0)*h).sum(0)
        else:h=h.mean(0)
        return anchor+self.predictor(h).squeeze(),anchor.new_zeros(())

def make(kind,dim):return SimpleBagHead(kind,dim) if kind in ['DeepSets','ABMIL'] else AuthorHead(kind,dim)
