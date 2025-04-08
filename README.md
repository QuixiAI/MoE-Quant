## MoE-Quant
---

This repository provides code for `GPTQ` quantization of [DeepSeekV3](https://huggingface.co/deepseek-ai/DeepSeek-V3)/[DeepSeekR1](https://huggingface.co/deepseek-ai/DeepSeek-R1) model family.

### Features

In order to quantize large model (671B parameters) with the `GPTQ` algorithm in reasonable time we introduce several optimizations:

1) **Fast `triton` kernel for `GPTQ`**: 
Since one has to quantize a lot (really a lot - ~45k) of linear layers, a faster `GPTQ` procedure is critical optimization. The provided `triton` implementation allows one to achieve ~10x relative to default `torch` implementation.
2) **Expert parallelism**: We shard MLP experts across all devices to fit Hessians into VRAM, required for `GPTQ` calibration. Each process stores only a fraction of expert layers and corresponding Hessians.
3) **Data parallelism**:  To accelerate forward propagation we split calibration data uniformly across processes.

**The total runtime of the algorithm to quantize DeepSeek-V3/R1 is 2 hours on a server with `8xH100` (for 512 calibration sequences of length 4096).**  

Currently we support conversion of `GPTQ`-quantized model into the [compressed_tensors](https://github.com/neuralmagic/compressed-tensors) format supported in HuggingFace transformers and vLLM. 

At the moment only 4-bit symmetric quantization with different quantization group sizes is supported.
We plan to implement other bit widths and quantization formats (`AWQ`, `AutoGPQ`) in the future. 


### GPTQ-quantized models on 🤗

| Models | Experts Quantized | Attention blocks quantized | Size (Gb) |
| ------ |  --------- | --------- | --------- |
| [ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g](https://huggingface.co/ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g) | ✅  | ✅  | 325 GB |
| [ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts](https://huggingface.co/ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts)| ✅ | ❌ | 346 GB |

These models easily fit onto single 8x `A100/H100` node with context long enough for most of the applications of interest, including reasoning chains.

**Evaluation results on reasoning tasks (AIME-24, GPQA-Diamond, MATH-500)** 

|      Model                                   | AIME-2024 pass@1 | MATH-500 pass@1 | GPQA-Diamond pass@1 | Average | Recovery |
|-----------------------------------------|------------------|-----------------|---------------------|---------|----------|
| deepseek-ai/DeepSeek-R1                       |     78.34       |       97.24     |        73.383       |  82.99  |   100.00    |
| cognitivecomputations/DeepSeek-R1-AWQ |     70.67      |       93.64     |        70.456       |  78.25  |  94.29   |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g            |  72.96    |    97.08  |        70.26        |  80.10   |  96.52   |
| ISTA-DASLab/DeepSeek-R1-GPTQ-4b-128g-experts            |  77.00         |    97.08       |        71.92       |   82.00    |  98.81   |


### Usage

**Model quantization**

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

Above:
* `--model_name_or_path` - **exact path** to model weights, say (`$HF_HOME/hub/models/models--deepseek-ai--DeepSeek-V3-0324/snapshots/commit_hash/`)
* `--dataset_name_or_path` - dataset used for calibration. We provide 3 choices `open-thoughts`, `open-platypus`, `fineweb-edu`
* `--num_calibration_samples` - number of calibration samples
* `--max_sequence_length` - maximal length of calibration samples (samples longer are capped to this value)
* `--quantization_order` - `default` or `activation`, we recommend using the latter for best results
* `--quantization_scale` - `absmax` or `mse`, we recommend using the latter for best results
* `--quantize_only_experts` - quantize only *non-shared* experts. Yields potentially better accuracy at the cost of slightly higher memory overhead.
* `--tie_gptq_handles` - reuse the same Hessian for `up` and `gate` projections to reduce memory overhead on quantization
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

### Enviroment

This code was tested with the following versions of libraries:
* `torch                             2.5.1` 
* `transformers                      4.50.0`
* `vllm                              0.8.2`
