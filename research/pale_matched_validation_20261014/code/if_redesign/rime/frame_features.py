"""Label-free independent-frame hidden contrasts on the frozen released JEV.

The canonical 20-image encoder batch preserves BF16 inference geometry.
Weights are externally supplied; construction needs the jev27b extra and CUDA.
"""
import numpy as np
from PIL import Image
from ..qwen27b.native import NativeJEV
from ..qwen27b.adapter import raw_oct

class FrozenFrameJEV(NativeJEV):
    def __init__(self,bundle,device=0):
        super().__init__(bundle,device)
        from ..qwen27b.adapter import MedicalQwen
        self.template=MedicalQwen.inputs(self,[Image.new('RGB',(224,224),(128,128,128)) for _ in range(20)],'reread')
        with self.torch.inference_mode():
            self.gray_features=tuple(self.model.get_base_model().get_image_features(self.template['pixel_values'],self.template['image_grid_thw']).pooler_output)
        self.null=self.forward(self.gray_features)
    def forward(self,features,direct=None):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        kwargs={k:v for k,v in self.template.items() if k!='pixel_values'}
        if direct is None:kwargs['mm_encoder_outputs']={'image':BaseModelOutputWithPooling(pooler_output=features)}
        else:kwargs['pixel_values']=direct
        with self.torch.inference_mode():
            out=self.model(**kwargs,use_cache=False,output_hidden_states=True,logits_to_keep=1)
            h=out.hidden_states[-1][0,-1].float().cpu().numpy()
            logits=out.logits[0,-1,self.ids].float().cpu().numpy()
        if not np.isfinite(h).all():raise RuntimeError('Nonfinite hidden state')
        return h,logits
    def extract(self,path,parity=False):
        image=raw_oct(path)
        pixel=self.processor.image_processor(images=[image],return_tensors='pt')
        grid=pixel['image_grid_thw'].to(self.device)
        assert grid.tolist()==[[1,14,14]]
        values=pixel['pixel_values'].to(self.device)
        # Keep the canonical 20-image encoder batch. Single-image encoder batches
        # can round differently in BF16, even though the vision encoder has no
        # cross-image attention. Do not silently loosen the parity requirement.
        direct=self.template['pixel_values'].clone();direct[:196]=values
        with self.torch.inference_mode():
            feat=tuple(self.model.get_base_model().get_image_features(direct,self.template['image_grid_thw']).pooler_output)
        h,q=self.forward(feat);difference=h-self.null[0]
        report=None
        if parity:
            dh,dq=self.forward(None,direct)
            report=dict(hidden_max_error=float(np.max(abs(dh-h))),logit_max_error=float(np.max(abs(dq-q))))
            if report['hidden_max_error']>1e-5 or report['logit_max_error']>1e-5:raise RuntimeError('Single-frame cached-vision parity failed')
        return difference,q-self.null[1],report
