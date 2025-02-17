import os
import math
import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


FP8_GROUP_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    )
    # Data params
    parser.add_argument(
        "--sequence_length", 
        default=8, 
        type=int, 
        help="Calibration sequence length."
    )
    args = parser.parse_args()
    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def dequantize_weight_from_fp8(W, s):
    g = FP8_GROUP_SIZE
    # Dequantize weight
    d_out, d_in = W.shape
    # Pad weight if needed
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    W = F.pad(W, (0, pad_in, 0, pad_out))
    d_out_pad, d_in_pad = W.shape

    W = W.view(d_out_pad // g, g, d_in_pad // g, g) 
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    W = (W * s).view(d_out_pad, d_in_pad)

    # Remove padding
    W = W[:d_out, :d_in]
    return W


def load_param_shard(weight_dir: str, weight_path: str) -> dict[str, torch.Tensor]:
    param_shard = {}
    with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
        param_shard_keys = f.keys()
        for k in param_shard_keys:
            param_shard[k] = f.get_tensor(k)
    return param_shard


def dequantize_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    state_dict_keys = list(state_dict.keys())
    # Dequantize
    for k in state_dict_keys:
        if k.endswith("scale_inv"):
            layer_name, _ = k.rsplit(".", 1)

            W = state_dict[f"{layer_name}.weight"].to(torch.bfloat16) 
            s = state_dict[f"{layer_name}.weight_scale_inv"].to(torch.bfloat16) 

            state_dict[f"{layer_name}.weight"] = dequantize_weight_from_fp8(W, s)
            del state_dict[f"{layer_name}.weight_scale_inv"]


def main():
    args = parse_args()
    device = "cuda"

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        ).eval()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Generate random input
    inputs = torch.randint(0, model.vocab_size, size=(1, args.sequence_length), device=device)

    # Load initial weight shard
    weight_dir = args.model_name_or_path
    current_shard_id = 1
    weight_path = f"model-{current_shard_id:05}-of-000163.safetensors"

    param_buffer = load_param_shard(weight_dir, weight_path)
        
    # Load input embeddings on GPU
    model.model.embed_tokens.to_empty(device=device)
    with torch.no_grad():
        model.model.embed_tokens.data = param_buffer["model.embed_tokens.weight"]
        inputs = model.model.embed_tokens(inputs)
    # Offload embeddings back to meta
    model.model.embed_tokens.to(device="meta")
    del param_buffer["model.embed_tokens.weight"]

    for block_idx, block in tqdm(
        enumerate(model.model.layers), 
        desc="Processing transformer blocks",
        total=len(model.model.layers)
    ):
        prefix = f"model.layers.{block_idx}"
        block_keys_with_prefix = set(f"{prefix}.{k}" for k in block.state_dict())
        while not is_subset(block_keys_with_prefix, set(param_buffer.keys())):
            current_shard_id += 1
            weight_path = f"model-{current_shard_id:05}-of-000163.safetensors"
            param_buffer.update(load_param_shard(weight_dir, weight_path))
        # Select weights corresponding to chosen block and dequantize them
        block_state_dict = {k[len(prefix)+1:]: v for k, v in param_buffer.items() if k.startswith(prefix)}
        dequantize_state_dict(block_state_dict)
        # Put block onto GPU
        block.to_empty(device=device)
        block.load_state_dict(block_state_dict)

        with torch.no_grad():
            inputs = block(inputs)[0]

        # Offload block
        block.to(device="meta")
        for k in block_keys_with_prefix:
            del param_buffer[k]
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
