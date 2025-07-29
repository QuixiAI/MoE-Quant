"""
Direct safetensors saving for quantized models - replaces the insane individual file approach
"""

import os
import json
import torch
from safetensors.torch import save_file
from typing import Dict, Any
import gc


class QuantizedModelSaver:
    """Accumulates quantized weights and saves them in HuggingFace-compatible format"""
    
    def __init__(self, save_dir: str, model_config: Any, max_shard_size: str = "10GB"):
        self.save_dir = save_dir
        self.model_config = model_config
        self.max_shard_size = max_shard_size
        
        # Parse max shard size to bytes
        size_map = {"GB": 1024**3, "MB": 1024**2}
        for unit, multiplier in size_map.items():
            if max_shard_size.endswith(unit):
                self.max_shard_size_bytes = int(max_shard_size[:-len(unit)]) * multiplier
                break
        
        # State dict accumulator
        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_count = 0
        self.weight_map = {}  # Maps weight name to shard filename
        
        os.makedirs(save_dir, exist_ok=True)
        
    def add_quantized_layer(self, layer_name: str, qweight: torch.Tensor, 
                           scale: torch.Tensor, zero: torch.Tensor, 
                           awq_scale: torch.Tensor = None):
        """Add a quantized layer to the current shard"""
        
        # Calculate size of this layer's tensors
        layer_size = 0
        tensors_to_add = {}
        
        # Add quantized weight (packed as int32)
        if qweight.dtype == torch.uint8:
            # Pack uint8 to int32 for storage efficiency
            from compressed_tensors.compressors import pack_to_int32
            packed_weight = pack_to_int32(qweight.cpu())
            tensors_to_add[f"{layer_name}.weight_packed"] = packed_weight
            layer_size += packed_weight.nbytes
        else:
            tensors_to_add[f"{layer_name}.weight"] = qweight.cpu()
            layer_size += qweight.nbytes
            
        # Add scales and zeros
        tensors_to_add[f"{layer_name}.scale"] = scale.cpu()
        tensors_to_add[f"{layer_name}.zero"] = zero.cpu()
        layer_size += scale.nbytes + zero.nbytes
        
        # Add AWQ scale if present
        if awq_scale is not None:
            tensors_to_add[f"{layer_name}.awq_scale"] = awq_scale.cpu()
            layer_size += awq_scale.nbytes
        
        # Check if we need to start a new shard
        if self.current_shard_size + layer_size > self.max_shard_size_bytes and self.current_shard:
            self._save_current_shard()
            
        # Add to current shard
        self.current_shard.update(tensors_to_add)
        self.current_shard_size += layer_size
        
        # Update weight map
        shard_filename = f"model-{self.shard_count+1:05d}.safetensors"
        for key in tensors_to_add:
            self.weight_map[key] = shard_filename
            
    def add_unquantized_layer(self, layer_name: str, weight: torch.Tensor):
        """Add an unquantized layer (e.g., embeddings, layernorm)"""
        layer_size = weight.nbytes
        
        if self.current_shard_size + layer_size > self.max_shard_size_bytes and self.current_shard:
            self._save_current_shard()
            
        self.current_shard[f"{layer_name}.weight"] = weight.cpu()
        self.current_shard_size += layer_size
        
        shard_filename = f"model-{self.shard_count+1:05d}.safetensors"
        self.weight_map[f"{layer_name}.weight"] = shard_filename
        
    def _save_current_shard(self):
        """Save the current shard to disk"""
        if not self.current_shard:
            return
            
        self.shard_count += 1
        shard_filename = f"model-{self.shard_count:05d}.safetensors"
        shard_path = os.path.join(self.save_dir, shard_filename)
        
        print(f"Saving shard {shard_filename} with {len(self.current_shard)} tensors...")
        save_file(self.current_shard, shard_path)
        
        # Clear current shard
        self.current_shard = {}
        self.current_shard_size = 0
        gc.collect()
        
    def finalize(self, metadata: Dict[str, Any] = None):
        """Save final shard and create index file"""
        # Save any remaining tensors
        if self.current_shard:
            self._save_current_shard()
            
        # Create index file
        index = {
            "metadata": {
                "total_size": sum(os.path.getsize(os.path.join(self.save_dir, f)) 
                                for f in os.listdir(self.save_dir) 
                                if f.endswith('.safetensors')),
                "format": "safetensors",
            },
            "weight_map": self.weight_map
        }
        
        if metadata:
            index["metadata"].update(metadata)
            
        index_path = os.path.join(self.save_dir, "model.safetensors.index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
            
        # Save config
        self.model_config.save_pretrained(self.save_dir)
        
        print(f"Model saved to {self.save_dir} with {self.shard_count} shards")