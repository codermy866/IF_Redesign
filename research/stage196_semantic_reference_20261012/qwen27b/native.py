"""Load an externally supplied, pinned JEV 27B bundle without network inference."""
from pathlib import Path
import json
import numpy as np

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RELEASED_BUNDLE = "autotrust/JEV-27B-VL"
BUNDLE_REVISION = "fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc"


def read(path):
    return json.loads(Path(path).read_text())


class NativeJEV:
    def __init__(self,bundle_dir,device=0):
        bundle_dir=Path(bundle_dir).expanduser().resolve(strict=True)
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        self.torch=torch;torch.set_num_threads(2);torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.48,device)
        self.device=f'cuda:{device}';self.processor=AutoProcessor.from_pretrained(bundle_dir,local_files_only=True)
        # Bound the actual processor resolution, including its internal resize.
        self.processor.image_processor.size={'shortest_edge':224*224,'longest_edge':224*224}
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        raw=AutoModelForImageTextToText.from_pretrained(bundle_dir,local_files_only=True,
              quantization_config=config,dtype=torch.bfloat16,device_map={'':device},attn_implementation='eager')
        self.model=PeftModel.from_pretrained(raw,bundle_dir/'adapter_vllm',is_trainable=False).eval()
        released=load_file(str(bundle_dir/'adapter_vllm/adapter_model.safetensors'),device='cpu')
        loaded=get_peft_model_state_dict(self.model,save_embedding_layers=False)
        if set(released)!=set(loaded):raise RuntimeError('Released and loaded decision adapter keys differ')
        for name,expected in released.items():
            actual=loaded[name]
            if not torch.equal(actual,expected.to(device=actual.device,dtype=actual.dtype)):
                raise RuntimeError('Released decision adapter tensor differs: '+name)
        self.adapter_tensor_count=len(released);del released,loaded
        self.head=read(bundle_dir/'adapter_vllm/decision_head.json');self.calibration=read(bundle_dir/'calibration.json')
        lo,hi=self.head['slots']['ranges']['noul'];self.ids=self.head['verbalizer_ids'][lo:hi]
        assert len(self.ids)==2 and self.head['slots']['verbalizers'][lo:hi]==['false','true']
        self.bias=np.array(self.head['bias'][lo:hi]);self.T=self.calibration['per_kind']['noul']
        tok=self.processor.tokenizer
        assert [tok.encode(s,add_special_tokens=False) for s in ['false','true']]==[[i] for i in self.ids]
        self.warmup=True
