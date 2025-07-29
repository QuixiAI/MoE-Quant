# Fork of MoE-Quant adding AWQ and Kimi K2 support

```
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py     --model_name_or_path /home/hotaisle/models/Kimi-K2-Instruct-BF16     --dataset_name_or_path open-platypus     --method awq     --num_calibration_samples 256     --max_sequence_length 4096     --bits 4     --group_size 128     --sym     --duo_scaling     --awq_grid_size 20     --offload_activations     --quantize_only_experts     --dtype bfloat16     --save_dir /home/hotaisle/models/Kimi-K2-Instruct-AWQ
```

## MoE-Quant
---

This repository provides code for [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978) quantization of [DeepSeekV3](https://huggingface.co/deepseek-ai/DeepSeek-V3)/[DeepSeekR1](https://huggingface.co/deepseek-ai/DeepSeek-R1) model family.

### News 🔥

- [2025/06] Quantized DeepSeek-R1-0528 model is on 🤗 hub. 

### Features

In order to quantize large model (671B parameters) with the `GPTQ` algorithm in reasonable time we introduce several optimizations:

1) **Fast `triton` kernel for `GPTQ`**: 
Since one has to quantize a lot (really a lot - ~45k) of linear layers, a faster `GPTQ` procedure is critical optimization. The provided `triton` implementation allows one to achieve ~10x relative to default `torch` implementation.
2) **Expert parallelism**: We shard MLP experts across all devices to fit Hessians into VRAM, required for `GPTQ` calibration. Each process stores only a fraction of expert layers and corresponding Hessians.
3) **Data parallelism**:  To accelerate forward propagation we split calibration data uniformly across processes.

**The total runtime of the algorithm to quantize DeepSeek-V3/R1 is 2 hours on a server with `8xH100` (for 512 calibration sequences of length 4096).**  

Currently we support conversion of `GPTQ`-quantized model into the [compressed_tensors](https://github.com/neuralmagic/compressed-tensors) format supported in HuggingFace transformers and vLLM. 

At the moment only 4-bit symmetric quantization with different quantization group sizes is supported.
We plan to implement other bit widths and quantization formats (`AutoGPQ`) in the future. 


### GPTQ-quantized models on 🤗

---
#### DeepSeek-R1

| Models | Experts Quantized | Attention blocks quantized | Size (Gb) |
| ------ |  --------- | --------- | --------- |
| [ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g](https://huggingface.co/ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g) | ✅  | ✅  | 325 GB |
| [ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts](https://huggingface.co/ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts)| ✅ | ❌ | 346 GB |

These models easily fit onto single 8x `A100/H100` node with context long enough for most of the applications of interest, including reasoning chains.

**Evaluation results on OpenLLM Leaderboard V1 tasks** 

|                                              | Recovery (%) | Average Score | ARC-Challenge<br>acc_norm, 25-shot | GSM8k<br>exact_match, 5-shot | HellaSwag<br>acc_norm, 10-shot | MMLU<br>acc, 5-shot | TruthfulQA<br>mc2, 0-shot | WinoGrande<br>acc, 5-shot |
| :------------------------------------------: | :----------: | :-----------: | :--------------------------------: | :--------------------------: | :----------------------------: | :-----------------: | :-----------------------: | :-----------------------: |
| deepseek/DeepSeek-R1                         | 100.00       | 81.04         | 72.53                              | 95.91                        | 89.30                          | 87.22               | 59.28                     | 82.00                     |
| cognitivecomputations/DeepSeek-R1-AWQ        | 100.07       | 81.10         | 73.12                              | 95.15                        | 89.07                          | 86.86               | 60.09                     | 82.32                     |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g         | 99.86        | 80.93         | 72.70                              | 95.68                        | 89.25                          | 86.83               | 58.77                     | 82.32                     |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts | 100.30       | 81.28         | 72.53                              | 95.68                        | 89.36                          | 86.99               | 59.77                     | 83.35                     |

**Evaluation results on reasoning tasks (AIME-24, GPQA-Diamond, MATH-500)** 

|                                              | Recovery (%) | Average Score | AIME 2024<br>pass@1 | MATH-500<br>pass@1 | GPQA Diamond<br>pass@1 |
| -------------------------------------------- | :----------: | :-----------: | :-----------------: | :----------------: | :--------------------: |
| deepseek/DeepSeek-R1                         | 100.00       | 82.99         | 78.33               | 97.24              | 73.38                  |
| cognitivecomputations/DeepSeek-R1-AWQ        | 94.29        | 78.25         | 70.67               | 93.64              | 70.46                  |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g         | 96.52        | 80.10         | 72.96               | 97.09              | 70.26                  |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts | **98.81**        | 82.00         | 77.00               | 97.08              | 71.92                  |

---
#### DeepSeek-R1-0528

| Models | Experts Quantized | Attention blocks quantized | Size (Gb) |
| ------ |  --------- | --------- | --------- |
| [ISTA-DASLab/DeepSeek-R1-0528-GPTQ-4b-128g-experts](https://huggingface.co/ISTA-DASLab/DeepSeek-R1-0528-GPTQ-4b-128g-experts)| ✅ | ❌ | 346 GB |

**Evaluation results on reasoning tasks (AIME-24, GPQA-Diamond, MATH-500)** 

|                                             | Recovery (%) | Average Score | AIME 2024<br>pass@1 | MATH-500<br>pass@1 | GPQA Diamond<br>pass@1 |
| ------------------------------------------- | :----------: | :-----------: | :-----------------: | :----------------: | :--------------------: |
| deepseek/DeepSeek-R1-0528                   | 100.00       | 88.61         | 88.66               | 97.52              | 79.65                  |
| ISTA-DASLab/DeepSeek-R1-0528-GPTQ-4b-128g-experts | 99.82   | 88.45         | 87.33               | 97.40              | 80.61                  |

### CPU-Only Version

For systems without GPUs, we provide a CPU-only implementation that supports both GPTQ and AWQ quantization methods. See [CPU_QUANTIZATION.md](CPU_QUANTIZATION.md) for detailed instructions.

**Key features:**
- Pure PyTorch implementation without GPU/CUDA dependencies
- Multi-threaded execution using 80% of CPU cores by default
- Compatible output format with GPU version
- Supports both GPTQ and AWQ quantization methods

**Quick start:**
```shell
python quant_cpu.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name_or_path $DATASET \
    --method gptq \
    --num_calibration_samples 128 \
    --max_sequence_length 2048 \
    --bits 4 \
    --group_size 128 \
    --cpu_ratio 0.8 \
    --save_dir <SAVE_DIR>
```

### Usage

**Model quantization**

**GPTQ:**
```shell
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
    --quantization_order $QUANTIZATION_ORDER \
    --quantization_scale $QUANTIZATION_SCALE \
    --quantize_only_experts \
    --tie_gptq_handles \
    --dtype bfloat16 \
    --save_dir <SAVE_DIR>
```

**AWQ:**
```shell
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

Above:
* `--model_name_or_path` - **exact path** to model weights, say (`$HF_HOME/hub/models/models--deepseek-ai--DeepSeek-V3-0324/snapshots/commit_hash/`)
* `--dataset_name_or_path` - dataset used for calibration. We provide 3 choices `open-thoughts`, `open-platypus`, `fineweb-edu`
* `--method` - quantization method: `gptq` (default) or `awq`
* `--num_calibration_samples` - number of calibration samples
* `--max_sequence_length` - maximal length of calibration samples (samples longer are capped to this value)
* `--quantization_order` - (GPTQ only) `default` or `activation`, we recommend using the latter for best results
* `--quantization_scale` - (GPTQ only) `absmax` or `mse`, we recommend using the latter for best results
* `--duo_scaling` - (AWQ only) use duo scaling for better quantization quality
* `--awq_grid_size` - (AWQ only) grid search size for finding optimal scales
* `--quantize_only_experts` - quantize only *non-shared* experts. Yields potentially better accuracy at the cost of slightly higher memory overhead.
* `--tie_gptq_handles` - (GPTQ only) reuse the same Hessian for `up` and `gate` projections to reduce memory overhead on quantization
* `--save_dir` - directory to save the model

The scripts above produces a directory with quantization metadata for each quantized layer, i.e `quantized_weight`, `scale`, and `zero`.

**Model packing**

To convert the model into `compressed_tensors` format run `pack_quantized_model.py` script

```shell
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```

Above:
* `--model_name_or_path` - **exact path** to model weights
* `--quantized_model_path` - path to quantized weights (output of `quant.py`)
* `--packed_model_path` - path to model in `compressed_tensors` format ready for inference in HF and vLLM.

### Environment

This code was tested with the following versions of libraries:
* `torch                             2.5.1` 
* `transformers                      4.50.0`
* `vllm                              0.8.2`

### Performance benchmarking
We follow the standard vLLM performance benchmarking with ShareGPT dataset and observe the following metrics (lower is better):

|                                              | Time to First Token<br>Median TTFT (ms) ↓ | Time per Output Token<br>Median TPOT (ms) ↓ | Inter-token Latency<br>Median ITL (ms) ↓ |
| -------------------------------------------- | :-------------------------------------: | :---------------------------------------: | :------------------------------------: |
| cognitivecomputations/DeepSeek-R1-AWQ        | 1585.45                                 | 55.41                                     | 43.06                                  |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts | 1344.68                                 | 41.49                                     | 36.33                                  |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g         | 815.19                                  | 44.65                                     | 37.88                                  |

GPTQ models are faster across all metrics than AWQ models because GPTQ uses less bits-per-parameter than AWQ. More specifically, AWQ has to use smaller group-size of 64 (vs 128 in GPTQ) to preserve accuracy, and zero-points due to asymmetric quantization. 

### Contributors

Denis Kuznedelev (Yandex), Eldar Kurtić (Red Hat AI & ISTA), Jiale Chen (ISTA), Michael Goin (Red Hat AI), Elias Frantar (ISTA), Dan Alistarh (Red Hat AI & ISTA).

### Citation

```
@article{gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```
