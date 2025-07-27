import os
import gc
import re
import argparse

from tqdm import tqdm
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    import wandb
    wandb_enabled = True
except:
    wandb_enabled = False


from src import dist_utils, data_utils, model_utils, quant_utils, loading_utils, gptq, awq


ROUTED_EXPERTS_REGEX = ".*mlp.experts.\d+.(down|gate|up)_proj$"
TIED_FFN_GROUPS = ("gate_proj", "up_proj")


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
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to calibration dataset",
    )
    parser.add_argument("--num_calibration_samples", default=128, type=int, help="Number of samples for calibration.")
    parser.add_argument("--max_sequence_length", default=8192, type=int, help="Calibration sequence length.")
    # Quantization method
    parser.add_argument(
        "--method",
        type=str,
        default="gptq",
        choices=["gptq", "awq"],
        help="Quantization method to use",
    )
    # Quantization params
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4],
        help="Quantization bitwidth.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument("--sym", action="store_true", help="Whether to use symmetric quantization")
    # GPTQ-specific params
    parser.add_argument("--rel_damp", type=float, default=1e-2, help="GPTQ relative damping")
    parser.add_argument("--block_size", type=int, default=128, help="GPTQ block size")
    parser.add_argument("--quantization_scale", type=str, default="absmax", choices=["absmax", "mse"])
    parser.add_argument("--quantization_order", type=str, default="default", choices=["default", "activation"])
    parser.add_argument("--tie_gptq_handles", action="store_true", help="whether to reuse hessian between gate and up projections.")
    # AWQ-specific params
    parser.add_argument("--duo_scaling", action="store_true", default=True, help="AWQ duo scaling")
    parser.add_argument("--awq_grid_size", type=int, default=20, help="AWQ grid search size")
    # Common params
    parser.add_argument(
        "--quantize_only_experts",
        default=False,
        action="store_true",
        help="Whether to quantize only routed (non-shared) experts.",
    )
    # Save params
    parser.add_argument("--save_dir", type=str, default=None, help="where to save quantized model.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Log to W&B")
    parser.add_argument("--log_error", default=False, action="store_true", help="Whether to log relative L2 error")
    # Misc params
    parser.add_argument("--offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--resume", action="store_true", help="whether to resume quantization from latest checkpoint.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--dtype", default="float16", type=str, choices=["float16", "bfloat16"], help="Torch dtype used."
    )
    args = parser.parse_args()

    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def get_resume_block_idx(save_dir: os.PathLike) -> int:
    resume_block_idx = 0
    if os.path.exists(save_dir):
        for layer_name in os.listdir(save_dir):
            block_idx = int(layer_name.split(".")[2])
            resume_block_idx = max(resume_block_idx, block_idx)
    return resume_block_idx


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    torch.set_grad_enabled(False)
    torch.cuda.set_device(device)
    offload_device = "cpu" if args.offload_activations else None
    dtype = getattr(torch, args.dtype)
    # Init W&B logger
    if args.log_wandb and dist_utils.is_main():
        assert wandb_enabled, "wandb not installed. try `pip install wandb`"
        wandb.init(config=args)

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # Sanity check
    supported_architectures = ["DeepseekV3ForCausalLM", "KimiK2ForCausalLM"]
    assert config.architectures[0] in supported_architectures, f"Only {supported_architectures} are supported!"
    
    # Model type detection
    model_type = config.architectures[0]
    is_deepseek_v3 = model_type == "DeepseekV3ForCausalLM"
    is_kimi_k2 = model_type == "KimiK2ForCausalLM"
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.ep_size = world_size

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=dtype
        ).eval()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Prepare calibration dataset
    calibration_dataset = data_utils.prepare_calibration_dataset(
        args.dataset_name_or_path, tokenizer, args.max_sequence_length, args.num_calibration_samples, args.seed
    )

    # Take slices (if running on multiple workers)
    num_seq_per_rank = len(calibration_dataset) // world_size
    calibration_dataset = calibration_dataset[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    dist_utils.barrier(device_ids=[rank])

    # Load initial weight shard
    weight_dir = args.model_name_or_path
    current_shard_id = 1
    
    # Detect total number of shards from model.safetensors.index.json
    import json
    index_file = os.path.join(weight_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        # Get unique shard filenames
        shard_files = set(index_data['weight_map'].values())
        # Extract the total count from filename pattern
        import re
        for shard_file in shard_files:
            match = re.search(r'model-\d+-of-(\d+)\.safetensors', shard_file)
            if match:
                total_shards = int(match.group(1))
                break
        else:
            total_shards = 163  # Default for DeepSeek V3
    else:
        total_shards = 163  # Default for DeepSeek V3
    
    weight_path = f"model-{current_shard_id:05}-of-{total_shards:06}.safetensors"

    param_buffer = {}
    if dist_utils.is_main():
        param_buffer = loading_utils.load_param_shard(weight_dir, weight_path)
    dist_utils.barrier(device_ids=[rank])

    # Get resume block id
    resume_block_idx = 0
    if args.resume:
        resume_block_idx = get_resume_block_idx(args.save_dir)

    # Prepare input embeddings and position ids
    inputs = []
    position_ids = []
    model.model.embed_tokens.to_empty(device=device)
    if dist_utils.is_main():
        model.model.embed_tokens.weight.data = param_buffer["model.embed_tokens.weight"].to(device=device, dtype=dtype)
    if dist_utils.is_dist_available_and_initialized():
        dist_utils.broadcast_parameters(model.model.embed_tokens)
    for i in range(num_seq_per_rank):
        seq_length = calibration_dataset[i].shape[1]
        inputs.append(model.model.embed_tokens(calibration_dataset[i].to(device)).to(offload_device))
        position_ids.append(torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0))
    # Offload embeddings back to meta
    model.model.embed_tokens.to(device="meta")
    param_buffer.pop("model.embed_tokens.weight", None)

    for block_idx, block in tqdm(
        enumerate(model.model.layers), desc="Processing transformer blocks", total=len(model.model.layers)
    ):
        prefix = f"model.layers.{block_idx}."

        # Collect state dict keys from all processes
        rank_block_keys = [k for k in block.state_dict()]
        if dist_utils.is_main():
            block_keys_with_prefix = [f"{prefix}{k}" for k in rank_block_keys]
            other_ranks_keys = []
            for i in range(1, world_size):
                other_rank_keys = [None for _ in rank_block_keys]
                dist.recv_object_list(other_rank_keys, src=i)
                block_keys_with_prefix.extend([f"{prefix}{k}" for k in other_rank_keys])
                other_ranks_keys.append(other_rank_keys)
            # Make it a set
            block_keys_with_prefix = set(block_keys_with_prefix)
        else:
            block_keys_with_prefix = []
            other_ranks_keys = []
            dist.send_object_list(rank_block_keys, dst=0)

        if dist_utils.is_main():
            can_dequantize = True
            # Select weights corresponding to current block
            block_state_dict = {k[len(prefix) :]: v for k, v in param_buffer.items() if k.startswith(prefix)}
            while not (is_subset(block_keys_with_prefix, set(param_buffer.keys())) and can_dequantize):
                current_shard_id += 1
                weight_path = f"model-{current_shard_id:05}-of-{total_shards:06}.safetensors"
                param_buffer.update(loading_utils.load_param_shard(weight_dir, weight_path))
                # Update weights corresponding to current block
                block_state_dict = {k[len(prefix) :]: v for k, v in param_buffer.items() if k.startswith(prefix)}
                can_dequantize = quant_utils.can_dequantize_from_fp8(block_state_dict)
            # Dequantize weights corresponding to current block
            quant_utils.dequantize_state_dict(block_state_dict, dtype)

        # Put block onto GPU
        block.to_empty(device=device)

        # Simply load block state dict on master and broadcast
        # Handle model-specific layer configurations
        if is_kimi_k2:
            # Kimi K2: only layer 0 is fully dense (no experts)
            is_dense_layer = block_idx == 0
        else:
            # DeepSeek V3 and others: use first_k_dense_replace config
            is_dense_layer = block_idx < model.config.first_k_dense_replace
        
        if is_dense_layer:
            if dist_utils.is_main():
                block.load_state_dict(block_state_dict)
            if dist_utils.is_dist_available_and_initialized():
                dist_utils.broadcast_parameters(block)
        # Send dict with part of expets to target device
        else:
            if dist_utils.is_main():
                # Load state dict on master
                rank_state_dict = {k: block_state_dict[k] for k in rank_block_keys}
                block.load_state_dict(rank_state_dict)
                # Send to other processes
                for i in range(1, world_size):
                    rank_state_dict = {k: block_state_dict[k] for k in other_ranks_keys[i - 1]}
                    for k in rank_state_dict:
                        dist.send(rank_state_dict[k].to(device), dst=i)
            else:
                rank_state_dict = block.state_dict()
                for k in rank_state_dict:
                    dist.recv(rank_state_dict[k], src=0)
                block.load_state_dict(rank_state_dict)
            del rank_state_dict
        # Clear memory before calibration
        torch.cuda.empty_cache()
        gc.collect()

        if block_idx >= resume_block_idx:
            # Standard per-layer quantization (GPTQ or AWQ)
            layers = model_utils.select_layers(model, prefix, ".*", model_utils.LINEAR_LAYERS)
            handles = {}
            hooks = {}

            for layer_name, layer in layers.items():

                def update_handle_hook(name):
                    def _hook(_, inp, out):
                        handles[name].update(inp[0])

                    return _hook

                if args.quantize_only_experts and re.search(ROUTED_EXPERTS_REGEX, layer_name) is None:
                    continue

                if args.method == "gptq":
                    tied_gptq_handle = None
                    if args.tie_gptq_handles and layer_name.endswith("up_proj"):
                        parent_name, _ = layer_name.rsplit(".", 1)
                        tied_layer_name = f"{parent_name}.gate_proj"
                        tied_gptq_handle = handles[tied_layer_name]

                    handles[layer_name] = gptq.GPTQ(
                        layer,
                        args.group_size,
                        args.sym,
                        args.rel_damp,
                        args.block_size,
                        args.quantization_order,
                        args.quantization_scale,
                        is_distributed=re.search(ROUTED_EXPERTS_REGEX, layer_name) is None,
                        tied_gptq_handle=tied_gptq_handle
                    )
                else:  # args.method == "awq"
                    tied_awq_handle = None
                    if args.tie_gptq_handles and layer_name.endswith("up_proj"):  # Reuse same flag for AWQ
                        parent_name, _ = layer_name.rsplit(".", 1)
                        tied_layer_name = f"{parent_name}.gate_proj"
                        tied_awq_handle = handles.get(tied_layer_name)

                    handles[layer_name] = awq.AWQ(
                        layer,
                        args.group_size,
                        args.sym,
                        args.duo_scaling,
                        args.awq_grid_size,
                        is_distributed=re.search(ROUTED_EXPERTS_REGEX, layer_name) is None,
                        tied_awq_handle=tied_awq_handle,
                        offload_device=offload_device
                    )    

                # Only register hook if not tied (for both GPTQ and AWQ)
                if (args.method == "gptq" and tied_gptq_handle is None) or \
                   (args.method == "awq" and tied_awq_handle is None):
                    hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

            # Collect Hessians
            for i in range(num_seq_per_rank):
                block(inputs[i].to(device), position_ids=position_ids[i])

            for _, h in hooks.items():
                h.remove()

            dist_utils.barrier(device_ids=[rank])

            shared_handles = {k: v for k, v in handles.items() if re.search(ROUTED_EXPERTS_REGEX, k) is None}
            expert_handles = {k: v for k, v in handles.items() if k not in shared_handles}

            # Quantized shared handles first
            num_issue_zero_samples = 0
            num_issue_nan_hessian = 0
            num_issue_non_invertible = 0
            for handle_name, handle in shared_handles.items():
                dist_utils.print_on_main(f"Quantizing layer {handle_name}")
                qweight, scale, zero = handle.quantize(args.bits)
                # Construct dequantized weight
                dequantized_weight = quant_utils.dequantize_linear_weight(qweight, scale, zero)
                assert (
                    torch.isfinite(dequantized_weight).all().item()
                ), f"[rank{rank}] {handle_name} weight is broken after quantization."
                # Update issue tracker
                num_issue_zero_samples += handle.issue_zero_samples
                if args.method == "gptq":
                    num_issue_nan_hessian += handle.issue_nan_hessian
                    num_issue_non_invertible += handle.issue_non_invertible
                else:  # AWQ
                    num_issue_nan_hessian += int(handle.issue_nan_activations)

                if args.log_error:
                    if (args.method == "gptq" and handle.has_hessian_issues()) or \
                       (args.method == "awq" and handle.has_activation_issues()):
                        dist_utils.print_on_main(
                            f"An issue occured on {'Hessian' if args.method == 'gptq' else 'activation'} computation. Output error cannot be estimated."
                        )
                    else:
                        if args.method == "gptq":
                            relative_mse = quant_utils.get_relative_mse_error(
                                dequantized_weight.float(), handle.layer.weight.float(), handle.H
                            )
                        else:  # AWQ - compute simple MSE since we don't have Hessian
                            relative_mse = (dequantized_weight.float() - handle.layer.weight.float()).pow(2).mean()
                        dist_utils.print_on_main(f"Relative error: {relative_mse.item():.2e}")
                        if args.log_wandb and dist_utils.is_main():
                            wandb.log({f"relative_error/{handle_name}": relative_mse.item()}, step=0)

                if args.save_dir and dist_utils.is_main():
                    os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
                    save_dict = {"qweight": qweight, "scale": scale, "zero": zero}
                    # Save AWQ smoothing scale if available
                    if args.method == "awq" and hasattr(handle, 'smooth_scale') and handle.smooth_scale is not None:
                        save_dict["awq_smooth_scale"] = handle.smooth_scale
                    torch.save(
                        save_dict,
                        os.path.join(args.save_dir, handle_name, f"quantized_weight.pt"),
                    )
                # Replace original weight by quantized one
                handle.layer.weight.data = dequantized_weight
                # Destroy handle
                handle.reset()

            dist_utils.print_on_main("-" * 10)
            dist_utils.print_on_main(f"{args.method.upper()} calibration issues for shared modules:")
            dist_utils.print_on_main(f"Zero samples: {num_issue_zero_samples}")
            if args.method == "gptq":
                dist_utils.print_on_main(f"Non-invertible: {num_issue_non_invertible}")
                dist_utils.print_on_main(f"NaN Hessian: {num_issue_nan_hessian}")
            else:  # AWQ
                dist_utils.print_on_main(f"NaN activations: {num_issue_nan_hessian}")
            dist_utils.print_on_main("-" * 10)

            # Quantize experts
            num_issue_zero_samples = 0
            num_issue_nan_hessian = 0
            num_issue_non_invertible = 0
            if len(expert_handles) > 0:
                dist_utils.print_on_main(f"Processing experts")

                expert_messages = None
                if dist_utils.is_main():
                    expert_messages = [None for _ in range(world_size)]
                rank_expert_message = ""

                for handle_name, handle in expert_handles.items():
                    rank_expert_message += f"Quantizing layer {handle_name}\n"
                    qweight, scale, zero = handle.quantize(args.bits)
                    # Construct dequantized weight
                    dequantized_weight = quant_utils.dequantize_linear_weight(qweight, scale, zero)
                    assert (
                        torch.isfinite(dequantized_weight).all().item()
                    ), f"[rank{rank}] {handle_name} weight is broken after quantization."
                    # Update issue tracker
                    num_issue_zero_samples += handle.issue_zero_samples
                    if args.method == "gptq":
                        num_issue_nan_hessian += handle.issue_nan_hessian
                        num_issue_non_invertible += handle.issue_non_invertible
                    else:  # AWQ
                        num_issue_nan_hessian += int(handle.issue_nan_activations)

                    rank_expert_message += f"Tokens collected: {handle.tokens_collected}.\n"

                    if args.log_error:
                        if (args.method == "gptq" and handle.has_hessian_issues()) or \
                           (args.method == "awq" and handle.has_activation_issues()):
                            rank_expert_message += f"{'Hessian' if args.method == 'gptq' else 'Activation'} issue. Output error cannot be estimated.\n"
                        else:
                            if args.method == "gptq":
                                relative_mse = quant_utils.get_relative_mse_error(
                                    dequantized_weight.float(), handle.layer.weight.float(), handle.H
                                )
                            else:  # AWQ
                                relative_mse = (dequantized_weight.float() - handle.layer.weight.float()).pow(2).mean()
                            rank_expert_message += f"Relative error: {relative_mse.item():.2e}\n"
                            # TODO send to main process
                            if args.log_wandb and dist_utils.is_main():
                                wandb.log({f"relative_error/{handle_name}": relative_mse.item()}, step=0)

                    if args.save_dir:
                        os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
                        save_dict = {"qweight": qweight, "scale": scale, "zero": zero}
                        # Save AWQ smoothing scale if available
                        if args.method == "awq" and hasattr(handle, 'smooth_scale') and handle.smooth_scale is not None:
                            save_dict["awq_smooth_scale"] = handle.smooth_scale
                        torch.save(
                            save_dict,
                            os.path.join(args.save_dir, handle_name, f"quantized_weight.pt"),
                        )
                    # Replace original weight by quantized one
                    handle.layer.weight.data = dequantized_weight
                    # Destroy handle
                    handle.reset()

                dist_utils.barrier(device_ids=[rank])

                dist.gather_object(rank_expert_message, expert_messages)
                if dist_utils.is_main():
                    for expert_message in expert_messages:
                        dist_utils.print_on_main(expert_message)

                # TODO sync data from other processes
                dist_utils.print_on_main("-" * 10)
                dist_utils.print_on_main(f"{args.method.upper()} calibration issues for expert modules:")
                dist_utils.print_on_main(f"Zero samples: {num_issue_zero_samples}")
                if args.method == "gptq":
                    dist_utils.print_on_main(f"Non-invertible: {num_issue_non_invertible}")
                    dist_utils.print_on_main(f"NaN Hessian: {num_issue_nan_hessian}")
                else:  # AWQ
                    dist_utils.print_on_main(f"NaN activations: {num_issue_nan_hessian}")
                dist_utils.print_on_main("-" * 10)

            del handles
            del shared_handles
            del expert_handles
            del hooks
            torch.cuda.empty_cache()
            gc.collect()
        else:
            dist_utils.print_on_main(f"Block {block_idx} is already quantized. Skipping quantization.")

        # Update activations
        for i in range(num_seq_per_rank):
            inputs[i] = block(inputs[i].to(device), position_ids=position_ids[i])[0].to(offload_device)
            assert torch.isfinite(inputs[i]).all().item(), "NaN of inf encountered."

        # Offload block
        block.to(device="meta")
        for k in block_keys_with_prefix:
            param_buffer.pop(k, None)

        torch.cuda.empty_cache()
        gc.collect()

    # Save quantization metadata
    if args.save_dir:
        torch.save(
            {
                "method": args.method,
                "bits": args.bits,
                "group_size": args.group_size,
                "quantize_only_experts": args.quantize_only_experts,
                "symmetric": args.sym,
            },
            os.path.join(args.save_dir, "metadata.pt")
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
