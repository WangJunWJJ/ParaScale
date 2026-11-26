# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Exercise optional product dependencies without downloads or CUDA."""

import argparse
import importlib.metadata
import json


def verify(product: str) -> dict:
    """Check real lazy imports and a finite forward/backward, not package metadata alone."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    names = ["torch", "torchvision", "transformers"]
    if product == "training":
        from peft import LoraConfig, get_peft_model

        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        ))
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))
        tokens = torch.tensor([[1, 2, 3, 4]])
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not torch.isfinite(loss) or not grads or not all(
            grad is not None and torch.isfinite(grad).all() for grad in grads
        ):
            raise RuntimeError("training profile failed LoRA backward")
        names.append("peft")
    elif product == "inference":
        from diffusers import StableDiffusionPipeline, UNet2DModel

        if not callable(StableDiffusionPipeline):
            raise RuntimeError("StableDiffusionPipeline is unavailable")
        model = UNet2DModel(
            sample_size=8, in_channels=3, out_channels=3,
            layers_per_block=1, block_out_channels=(8,), norm_num_groups=4,
            down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
        ).eval()
        with torch.no_grad():
            result = model(torch.randn(1, 3, 8, 8), timestep=1).sample
        if result.shape != (1, 3, 8, 8) or not torch.isfinite(result).all():
            raise RuntimeError("inference profile failed diffusion forward")
        names.append("diffusers")
    else:
        raise ValueError(f"unknown product: {product}")
    return {"product": product, "dependency_smoke": "passed", "quality_acceptance": False,
            "versions": {name: importlib.metadata.version(name) for name in names}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["training", "inference"], required=True)
    print(json.dumps(verify(parser.parse_args().product), indent=2))
