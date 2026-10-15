"""Frozen released/bare Qwen decisions with complete-prefix view replacement.

This is the completed semantic/evidence reference extractor, distinct from
MedicalQwen task adaptation. The returned branch axis is [released, bare].
"""
import numpy as np
from .native import NativeJEV

class InterventionJEV(NativeJEV):
    def __init__(self,bundle_dir,device=0):
        super().__init__(bundle_dir,device)
        # Prefixes, not just target names, are checked against released tensors below.
        from peft import get_peft_model_state_dict
        if any('visual' in name or 'vision' in name for name in get_peft_model_state_dict(self.model,save_embedding_layers=False)):
            raise RuntimeError('Vision adapter prevents frozen vision-feature reuse')
        self.vision_cache=True;self.preflight_done=False
        from PIL import Image
        self.gray_inputs=self.inputs([Image.new('RGB',(224,224),(128,128,128)) for _ in range(20)])
        self.gray_features=self.encode(self.gray_inputs)
        self.null=self.forward(self.gray_inputs,self.gray_features,hidden=True)

    def inputs(self,images):
        prompt='[kind] noul\n[state] Views 1-12 are cervical OCT; views 13-20 are colposcopy. Uniform gray views contain no visual evidence.\n'
        prompt+=''.join(f'View {i+1}: <|vision_start|><|image_pad|><|vision_end|>\n' for i in range(20))
        prompt+='[question] Do these cervical images show features suggestive of CIN2 or worse?\n[options]\nfalse\ntrue\n[decision]:'
        inputs=self.processor(text=[prompt],images=images,return_tensors='pt',add_special_tokens=False)
        result={k:v.to(self.device) for k,v in inputs.items()}
        grid=result['image_grid_thw']; t=self.torch
        if grid.shape!=(20,3) or not t.equal(grid,t.tensor([[1,14,14]]*20,device=grid.device,dtype=grid.dtype)):
            raise RuntimeError('Fixed image geometry violated')
        if result['input_ids'].shape[1]>4096:raise RuntimeError('Unexpected token expansion')
        return result

    def encode(self,inputs):
        with self.torch.inference_mode():
            return tuple(self.model.get_base_model().get_image_features(
                inputs['pixel_values'],inputs['image_grid_thw']).pooler_output)

    def forward(self,inputs,features=None,hidden=False):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from contextlib import nullcontext
        kwargs=dict(inputs)
        if features is not None:
            kwargs.pop('pixel_values')
            kwargs['mm_encoder_outputs']={'image':BaseModelOutputWithPooling(pooler_output=features)}
        qs=[];hs=[]
        with self.torch.inference_mode():
            for bare in (False,True):
                with self.model.disable_adapter() if bare else nullcontext():
                    out=self.model(**kwargs,use_cache=False,output_hidden_states=hidden,logits_to_keep=1)
                    logits=out.logits[0,-1,self.ids].float().cpu().numpy()
                    qs.append(float(logits[1]-logits[0]) if bare else float((logits[1]-logits[0]+self.bias[1]-self.bias[0])/self.T))
                    if hidden:hs.append(out.hidden_states[-1][0,-1].float().cpu().numpy())
                    del out
        result=dict(q=np.asarray(qs))
        if hidden:result['hidden']=np.stack(hs)
        if not all(np.isfinite(v).all() for v in result.values()):raise RuntimeError('Nonfinite inference')
        return result

    def replaced(self,inputs,features,slots,hidden=False,cached=True):
        if cached:
            changed=list(features)
            for slot in slots:changed[slot]=self.gray_features[slot]
            return self.forward(inputs,tuple(changed),hidden)
        changed=dict(inputs); pixels=inputs['pixel_values'].clone()
        offsets=self.torch.cat((self.torch.zeros(1,device=self.device,dtype=self.torch.long),
                               inputs['image_grid_thw'].prod(1).cumsum(0)))
        for slot in slots:
            lo,hi=int(offsets[slot]),int(offsets[slot+1]);pixels[lo:hi]=self.gray_inputs['pixel_values'][lo:hi]
        changed['pixel_values']=pixels
        return self.forward(changed,None,hidden)

    def parity(self,inputs,features):
        cases=[[],[0],list(range(12)),list(range(12,20)),list(range(20))];rows=[]
        for slots in cases:
            direct=self.replaced(inputs,features,slots,cached=False)['q']
            cached=self.replaced(inputs,features,slots,cached=True)['q']
            rows.append(dict(neutralized_slots=slots,direct=direct.tolist(),cached=cached.tolist(),
                             maximum_logit_difference=float(np.max(np.abs(direct-cached)))))
        self.vision_cache=all(row['maximum_logit_difference']<=.001 for row in rows)
        if not self.vision_cache:self.null=self.forward(self.gray_inputs,None,hidden=True)
        self.parity_report=dict(
             state='PASS_CACHED_VISION' if self.vision_cache else 'PASS_DIRECT_FALLBACK',
             parity=rows,tolerance=.001,hidden_dimension=5120,adapter_tensors=self.adapter_tensor_count,
             label_free=True,vision_adapter_absent=True,language_recomputed_for_every_intervention=True,
             loaded_adapter_equality_verified=True,official_vllm_parity_verified=False,
             peak_memory_GiB=self.torch.cuda.max_memory_allocated()/1024**3)
        self.preflight_done=True

    def extract(self,images,mask):
        inputs=self.inputs(images);features=self.encode(inputs)
        if not self.preflight_done:self.parity(inputs,features)
        full=self.forward(inputs,features if self.vision_cache else None,hidden=True)
        delta=np.zeros((2,20),float)
        for slot in np.flatnonzero(mask):
            delta[:,slot]=full['q']-self.replaced(inputs,features,[int(slot)],cached=self.vision_cache)['q']
        only_k=self.replaced(inputs,features,list(range(12)),cached=self.vision_cache)['q']
        only_o=self.replaced(inputs,features,list(range(12,20)),cached=self.vision_cache)['q']
        interaction=full['q']-only_o-only_k+self.null['q']
        if not mask[:12].any() or not mask[12:].any():interaction[:]=0
        return dict(delta=delta,mask=mask,A=full['q']-self.null['q'],I=interaction,q=full['q'],
                    hidden=full['hidden']-self.null['hidden'],only_o=only_o,only_k=only_k)

def stack_interventions(rows, clinical_logits, branch=0):
    """Convert label-free rows to input arrays for semantic/evidence readouts."""
    clinical_logits=np.asarray(clinical_logits,dtype=float)
    if branch not in (0,1) or not rows or clinical_logits.shape!=(len(rows),):
        raise ValueError("Expected branch 0/1 and one clinical logit per row")
    result={name:np.stack([row[name][branch] for row in rows])
            for name in ("hidden","delta","A","I")}
    result["mask"]=np.stack([row["mask"] for row in rows]).astype(bool)
    result["clinical"]=clinical_logits
    return result
