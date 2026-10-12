"""Source-only task adaptation on pinned Qwen 27B, retaining native vision.

Only the released language-attention LoRA q/v tensors in layers 48-63
are adapted. Original backbone files and published adapter files are immutable.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,re,hashlib
from contextlib import nullcontext
import numpy as np
from PIL import Image
from .native import NativeJEV

def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class MedicalQwen(NativeJEV):
    def __init__(self,bundle_dir,device=0):
        super().__init__(bundle_dir,device)
        t=self.torch;t.cuda.set_per_process_memory_fraction(.90,device)
        self.model.requires_grad_(False)
        self.parameters={}
        for name,param in self.model.named_parameters():
            match=re.search(r'language_model\.layers\.(\d+)\.',name)
            if match and int(match[1])>=48 and any(x in name for x in ('.q_proj.lora_A.','.q_proj.lora_B.','.v_proj.lora_A.','.v_proj.lora_B.')):
                self.parameters[name]=param
        if len(self.parameters)!=16 or sum(p.numel() for p in self.parameters.values())!=3014656:
            raise RuntimeError('Pinned released language-LoRA scope differs')
        self.initial={name:param.detach().clone() for name,param in self.parameters.items()}
        self.gray=[Image.new('RGB',(224,224),(128,128,128)) for _ in range(20)]
        self.gray_features=self.encode(self.inputs(self.gray,'diagnosis'))
        self.nulls={}
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def reset(self):
        with self.torch.no_grad():
            for name,param in self.parameters.items():param.copy_(self.initial[name])
        self.nulls={}

    def enable_training(self,enabled):
        self.model.train(enabled)
        for param in self.parameters.values():param.requires_grad_(enabled)
        active={name for name,param in self.model.named_parameters() if param.requires_grad}
        if active!=(set(self.parameters) if enabled else set()):raise RuntimeError('Trainable parameter scope drift')

    def inputs(self,images,task):
        prompt='[kind] noul\n[state] Views 1-12 are cervical OCT; views 13-20 are colposcopy. Uniform gray views contain no visual evidence.\n'
        prompt+=''.join(f'View {i+1}: <|vision_start|><|image_pad|><|vision_end|>\n' for i in range(20))
        question=('Do these cervical images show features suggestive of CIN2 or worse?' if task=='diagnosis' else
                  'Do these OCT frames show a structural abnormality on image reread? This is an imaging finding, not a CIN2+ pathology label.')
        prompt+='[question] '+question+'\n[options]\nfalse\ntrue\n[decision]:'
        inputs=self.processor(text=[prompt],images=images,return_tensors='pt',add_special_tokens=False)
        inputs={k:v.to(self.device) for k,v in inputs.items()};grid=inputs['image_grid_thw'];t=self.torch
        if grid.shape!=(20,3) or not t.equal(grid,t.tensor([[1,14,14]]*20,device=grid.device,dtype=grid.dtype)):
            raise RuntimeError('Fixed image-token geometry changed')
        if inputs['input_ids'].shape[1]>4096:raise RuntimeError('Prompt expansion')
        return inputs

    def encode(self,inputs):
        with self.torch.no_grad():
            return tuple(v.detach() for v in self.model.get_base_model().get_image_features(
                         inputs['pixel_values'],inputs['image_grid_thw']).pooler_output)

    def forward(self,inputs,features,hidden=False,grad=False):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        kwargs={k:v for k,v in inputs.items() if k!='pixel_values'}
        kwargs['mm_encoder_outputs']={'image':BaseModelOutputWithPooling(pooler_output=features)}
        with nullcontext() if grad else self.torch.no_grad():
            out=self.model(**kwargs,use_cache=False,output_hidden_states=hidden,logits_to_keep=1)
            logits=out.logits[0,-1,self.ids].float()+self.torch.tensor(self.bias,device=self.device,dtype=self.torch.float32)
            result=dict(logits=logits)
            if hidden:result['hidden']=out.hidden_states[-1][0,-1].float()
        return result

    def prepared(self,images,task):
        inputs=self.inputs(images,task)
        return inputs,self.encode(inputs)

    def null(self,task,hidden=False):
        key=(task,hidden)
        if key not in self.nulls:
            row=self.forward(self.inputs(self.gray,task),self.gray_features,hidden)
            self.nulls[key]={k:v.detach().cpu().numpy() for k,v in row.items()}
        return self.nulls[key]

    def proximal(self):
        return sum((param.float()-self.initial[name].float()).square().sum() for name,param in self.parameters.items())/sum(p.numel() for p in self.parameters.values())

    def save_adapter(self,path):
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        save_file({name:param.detach().float().cpu().contiguous() for name,param in self.parameters.items()},str(tmp));tmp.replace(path)

    def load_adapter(self,path):
        from safetensors.torch import load_file
        tensors=load_file(str(path),device='cpu')
        if set(tensors)!=set(self.parameters):raise RuntimeError('Medical adapter parameter keys mismatch')
        with self.torch.no_grad():
            for name,param in self.parameters.items():param.copy_(tensors[name].to(device=param.device,dtype=param.dtype))
        self.nulls={}

def raw_oct(path):
    # Match the old single-frame field of view, keeping aspect ratio before crop.
    with Image.open(path) as image:
        image=image.convert('RGB');w,h=image.size
        size=(int(256*w/h),256) if w>=h else (256,int(256*h/w))
        image=image.resize(size,Image.Resampling.BILINEAR)
        x=int(round((image.width-224)/2));y=int(round((image.height-224)/2))
        return image.crop((x,y,x+224,y+224))

def images(item,task='diagnosis',site=None,phase=0,colpo_only=False):
    out=[Image.new('RGB',(224,224),(128,128,128)) for _ in range(20)]
    if task=='reread':
        if site is None:raise ValueError('Reread task needs a scanner position')
        indices=((0,3,6,9),(1,2,5,8))[int(phase)%2]
        for j,index in enumerate(indices):out[j]=raw_oct(item['oct_sites'][str(site)][index])
    else:
        if not colpo_only:
            for site in range(1,13):out[site-1]=raw_oct(item['oct_sites'][str(site)][int(phase)%10])
        for j,path in enumerate(item['colpo_paths'][:8]):
            with Image.open(path) as im:out[12+j]=im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    return out
