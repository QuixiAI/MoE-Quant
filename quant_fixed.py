#!/usr/bin/env python3
"""
Fixed quantization script that saves directly to HuggingFace-compatible format
instead of creating 70,000 individual files like a psychopath
"""

# Copy the imports and setup from quant.py
import os
import gc
import re
import argparse

from tqdm import tqdm
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Import the sane saving logic
from save_directly import QuantizedModelSaver

# Rest of imports from original
from src import dist_utils, data_utils, model_utils, quant_utils, loading_utils, gptq, awq

# This is a template - you'll need to copy the main logic from quant.py
# but replace the insane file saving with:

def save_quantized_weights(args, model, model_config, quantized_layers):
    """Save quantized weights in HuggingFace-compatible format"""
    
    if not args.save_dir or not dist_utils.is_main():
        return
        
    print("Saving quantized model in HuggingFace format...")
    
    # Initialize the saver
    saver = QuantizedModelSaver(
        save_dir=args.save_dir,
        model_config=model_config,
        max_shard_size="10GB"  # Adjust as needed
    )
    
    # Add quantized layers
    for layer_name, layer_data in tqdm(quantized_layers.items(), desc="Saving layers"):
        if "qweight" in layer_data:
            # Quantized layer
            saver.add_quantized_layer(
                layer_name=layer_name,
                qweight=layer_data["qweight"],
                scale=layer_data["scale"],
                zero=layer_data["zero"],
                awq_scale=layer_data.get("awq_scale", None)
            )
        else:
            # Unquantized layer (embeddings, etc)
            saver.add_unquantized_layer(
                layer_name=layer_name,
                weight=layer_data["weight"]
            )
    
    # Save metadata
    metadata = {
        "quantization_method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        "symmetric": args.sym,
        "quantize_only_experts": args.quantize_only_experts,
    }
    
    # Finalize and create index
    saver.finalize(metadata)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.save_dir)
    
    print(f"✓ Model saved to {args.save_dir} in HuggingFace format")
    print(f"✓ Ready for use with vLLM or transformers")