"""Convert official LLaVA-Med Mistral keys to installed Transformers 4.57.

Preserves learned tensors exactly. The added image-token row is zero and is
replaced by vision embeddings in the forward pass; suppress it in generation.
"""
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig,
                          MistralConfig, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor)

ROOT = Path(__file__).resolve().parents[1]


def convert_key(key):
    if key.startswith("model.vision_tower.vision_tower."):
        return key.replace("model.vision_tower.vision_tower.", "model.vision_tower.", 1)
    for layer, name in ((0,"linear_1"),(2,"linear_2")):
        prefix = f"model.mm_projector.{layer}."
        if key.startswith(prefix):
            return key.replace(prefix, f"model.multi_modal_projector.{name}.", 1)
    if key.startswith("model."):
        return key.replace("model.", "model.language_model.", 1)
    if key == "lm_head.weight":
        return key
    raise ValueError(f"Unknown checkpoint key: {key}")


def main():
    source = ROOT / "models/pretrained/llava_med"
    destination = ROOT / "models/pretrained/llava_med_hf"
    if (destination / "conversion_complete.json").exists():
        return
    spec = json.loads((source / "config.json").read_text())
    assert spec["mm_vision_tower"] == "openai/clip-vit-large-patch14-336"
    assert spec["mm_projector_type"] == "mlp2x_gelu" and spec["mm_vision_select_layer"] == -2
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = MistralConfig(**{k:v for k,v in spec.items() if k not in ("model_type", "architectures", "vocab_size")},
                         vocab_size=len(tokenizer))
    vision = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                             num_attention_heads=16, image_size=336, patch_size=14)
    config = LlavaConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), image_token_index=image_id,
                         vision_feature_layer=-2, vision_feature_select_strategy="default", image_seq_length=576)
    with init_empty_weights():
        model = LlavaForConditionalGeneration(config)
    assigned = set()
    for shard in sorted(source.glob("model-*.safetensors")):
        tensors = load_file(shard)
        mapped = {}
        for key, value in tensors.items():
            new_key = convert_key(key)
            if new_key in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
                assert value.shape[0] == spec["vocab_size"]
                value = torch.cat([value, value.new_zeros(len(tokenizer)-value.shape[0], value.shape[1])], 0)
            if new_key in assigned:
                raise ValueError("Duplicate key after conversion")
            assigned.add(new_key)
            mapped[new_key] = value
        result = model.load_state_dict(mapped, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        print(json.dumps(dict(shard=shard.name, mapped=len(mapped))), flush=True)
    missing = [name for name,p in model.named_parameters() if p.is_meta]
    if missing:
        raise ValueError(f"Missing converted weights: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    model.generation_config.suppress_tokens = [image_id]
    model.save_pretrained(destination, max_shard_size="4GB")
    processor = LlavaProcessor(tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(size={"shortest_edge":336}, crop_size={"height":336,"width":336}),
        patch_size=14, vision_feature_select_strategy="default", num_additional_image_tokens=1)
    processor.save_pretrained(destination)
    (destination / "conversion_complete.json").write_text(json.dumps(dict(source=str(source),
        mapped_keys=len(assigned), learned_tensors_preserved=True, image_token_id=image_id,
        note="Architecture/key conversion; native-vs-converted numerical parity still needs separate validation"), indent=2))


if __name__ == "__main__":
    main()
