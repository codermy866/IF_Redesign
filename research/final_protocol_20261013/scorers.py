"""A single score-conditioning axis; no new pooling or representation."""
import math
import torch
from torch import nn
from if_redesign.rime.readout import RIMEHead

class ScoreControl(RIMEHead):
    def __init__(self,context_dim,tau,pooling,readout,mode='conditional'):
        super().__init__(context_dim,tau,pooling,readout)
        self.score_mode=mode
        if mode=='geometry':
            self.pooling='geometry'
            for p in (self.a,self.U,self.V):p.requires_grad_(False)
        elif mode=='local':
            for p in (self.U,self.V):p.requires_grad_(False)
        elif mode=='local_capacity_matched':
            # Exactly the same effective AND allocated scorer budget. A
            # bias-free hidden layer uses 9*w coefficients; the remainder
            # (0..8) supplies genuinely trainable direct local feature paths.
            target=24+2*context_dim
            self.local_width,remainder=divmod(target,9)
            del self.a,self.U,self.V
            self.local_scorer=nn.Sequential(nn.Linear(8,self.local_width,bias=False),nn.Tanh(),nn.Linear(self.local_width,1,bias=False))
            self.local_skip=nn.Parameter(torch.zeros(remainder))
            with torch.no_grad():
                self.local_scorer[0].weight.mul_(.01)
                self.local_scorer[2].weight.zero_()
        elif mode not in ['conditional','context_shuffled']:raise ValueError(mode)

    def scores(self,h,b):
        if self.score_mode=='geometry':return torch.zeros_like(h[...,0])
        if self.score_mode=='local':return torch.tanh(h@self.a)
        if self.score_mode=='local_capacity_matched':
            return torch.tanh(self.local_scorer(h).squeeze(-1)+h[...,:len(self.local_skip)]@self.local_skip)
        return super().scores(h,b)

def site_auxiliary(head,inputs,targets,coefficient):
    """Site supervision is weak: it does not identify within-site frame value.

    Repeating a scalar advantage as every frame's QP reward is constant on the
    simplex. Instead supervise the shared scorer's site mean; its relative
    frame scores continue to be trained by the patient decision objective.
    """
    q=head.scores(inputs['h'],inputs['b'])
    valid=inputs['valid'];sites=valid.any(-1)
    means=(q*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    per_patient=(((means-targets)**2)*sites).sum(-1)/sites.sum(-1).clamp_min(1)
    return coefficient*per_patient.mean()
