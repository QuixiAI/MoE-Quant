# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoE-Quant is a repository for GPTQ quantization of DeepSeekV3/DeepSeekR1 model family (671B parameters). The codebase provides optimized quantization algorithms specifically designed for massive mixture-of-experts models.

## Key Commands

### Model Quantization

#### GPTQ Quantization (Original)
```bash
torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS --master_port 29501 quant.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name_or_path $DATASET \
    --num_calibration_samples 512 \
    --max_sequence_length 4096 \
    --bits 4 \
    --group_size 128 \
    --rel_damp 0.1 \
    --sym \
    --offload_activations \
    --quantization_order activation \
    --quantization_scale mse \
    --quantize_only_experts \
    --tie_gptq_handles \
    --dtype bfloat16 \
    --save_dir <SAVE_DIR>
```

#### AWQ Quantization
```bash
torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS --master_port 29501 quant.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name_or_path $DATASET \
    --method awq \
    --num_calibration_samples 256 \
    --max_sequence_length 4096 \
    --bits 4 \
    --group_size 128 \
    --sym \
    --duo_scaling \
    --awq_grid_size 20 \
    --offload_activations \
    --quantize_only_experts \
    --dtype bfloat16 \
    --save_dir <SAVE_DIR>
```

### Model Packing
```bash
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```

### Applying AWQ Scales (AWQ only)
For AWQ-quantized models, you need to apply the smoothing scales to LayerNorm layers:
```bash
python apply_awq_scales.py \
    --model_path $MODEL_PATH \
    --quantized_path $QUANTIZED_MODEL_PATH \
    --output_path $MODEL_PATH-awq-scaled
```

## Architecture Overview

### Core Components

1. **Main Scripts**
   - `quant.py`: Main quantization script that orchestrates GPTQ/AWQ algorithms across distributed GPUs
   - `pack_quantized_model.py`: Converts quantized weights into compressed_tensors format for HuggingFace/vLLM inference

2. **Source Modules (`src/`)**
   - `gptq.py`: Core GPTQ algorithm implementation with optimized Triton kernels
   - `gptq_loop.py`: Low-level GPTQ quantization loops with Triton acceleration
   - `awq.py`: AWQ algorithm implementation following GPTQ's exact architectural pattern
   - `awq_loop.py`: Low-level AWQ quantization loops with Triton acceleration
   - `dist_utils.py`: Distributed computing utilities for expert/data parallelism
   - `model_utils.py`: Model architecture handling and expert routing utilities
   - `loading_utils.py`: Efficient model loading with memory mapping
   - `quant_utils.py`: Quantization primitives and compressed tensor handling
   - `data_utils.py`: Calibration dataset loading and preprocessing
   - `linalg_utils.py`: Linear algebra utilities for Hessian computation

### Key Optimizations

1. **Expert Parallelism**: MLP experts are sharded across devices to fit Hessians/activations in VRAM
2. **Data Parallelism**: Calibration data is split uniformly across processes
3. **Fast Triton Kernels**: ~10x speedup over PyTorch implementation for GPTQ
4. **Tied Handles**: Reuses Hessians (GPTQ) or activation statistics (AWQ) for gate and up projections to reduce memory
5. **Memory Efficiency**:
   - **GPTQ**: Incremental Hessian updates, distributed synchronization
   - **AWQ**: Running statistics instead of storing all activations, offloading support
6. **AWQ Optimizations**:
   - Computes running mean instead of storing all activations (O(features) vs O(samples × features))
   - Caches only small activation sample (256 tokens) for grid search
   - Supports offloading to CPU for memory-constrained scenarios
   - Distributed synchronization of statistics across GPUs
   - Triton kernel support for faster scale computation (experimental)

### Quantization Strategy

- Supports 4-bit symmetric quantization with configurable group sizes
- Can quantize only experts (non-shared layers) for better accuracy
- Two quantization methods available:
  - **GPTQ**: Uses Hessian-based optimization for weight quantization
  - **AWQ**: Uses activation-weighted scaling to protect salient channels
- Uses activation order and MSE scale for optimal results (GPTQ)
- Uses duo scaling and grid search for optimal smoothing factors (AWQ)
- Outputs compressed_tensors format compatible with HuggingFace transformers and vLLM

**Important AWQ Note**: AWQ quantizes weights with smoothing scales (W' = W * s). The inverse scales (s^-1) need to be applied to the previous layer (typically LayerNorm) weights for correct inference. This is handled automatically in AutoAWQ but needs manual handling in this implementation.

## Important Parameters

### Common Parameters
- `--model_name_or_path`: Must be exact path to model weights (e.g., `$HF_HOME/hub/models/models--deepseek-ai--DeepSeek-V3-0324/snapshots/commit_hash/`)
- `--dataset_name_or_path`: Options are `open-thoughts`, `open-platypus`, `fineweb-edu`
- `--quantize_only_experts`: Recommended for better accuracy with slight memory overhead
- `--bits`: Quantization bitwidth (currently only 4-bit supported)
- `--group_size`: Quantization group size (default: 128)

### GPTQ-Specific Parameters
- `--quantization_order`: Use `activation` for best results
- `--quantization_scale`: Use `mse` for best results
- `--tie_gptq_handles`: Reuse Hessians for gate/up projections
- `--rel_damp`: Relative damping factor (default: 0.01)

### AWQ-Specific Parameters
- `--method awq`: Select AWQ quantization method
- `--duo_scaling`: Enable duo scaling (recommended)
- `--awq_grid_size`: Grid search size for optimal scales (default: 20)