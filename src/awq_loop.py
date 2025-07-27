from typing import Tuple

import torch
import triton
from triton import language as tl

from src.quant_utils import tl_quantize, tl_dequantize

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


@triton.jit
def awq_scale_kernel(
    w_ptr,
    scale_ptr,
    w_scaled_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply AWQ scale to weights: W_scaled = W * scale"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    w = tl.load(w_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets, mask=mask)
    w_scaled = w * scale
    
    tl.store(w_scaled_ptr + offsets, w_scaled, mask=mask)


def apply_awq_scale_triton(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply AWQ scale to weight matrix using Triton"""
    n_elements = weight.numel()
    w_scaled = torch.empty_like(weight)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    awq_scale_kernel[grid](
        weight.flatten(),
        scale.flatten(),
        w_scaled.flatten(),
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return w_scaled.view(weight.shape)


@triton.jit
def awq_search_scale_kernel(
    x_mean_ptr,
    w_mean_ptr,
    scale_out_ptr,
    ratio,
    duo_scaling: tl.constexpr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute AWQ scale for a given ratio"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x_mean = tl.load(x_mean_ptr + offsets, mask=mask)
    w_mean = tl.load(w_mean_ptr + offsets, mask=mask)
    
    if duo_scaling:
        # s = (x_mean^ratio) / (w_mean^(1-ratio) + eps)
        scale = tl.pow(x_mean, ratio) / (tl.pow(w_mean, 1.0 - ratio) + 1e-4)
    else:
        # s = x_mean^ratio
        scale = tl.pow(x_mean, ratio)
    
    # Clamp scale
    scale = tl.maximum(scale, 1e-4)
    
    tl.store(scale_out_ptr + offsets, scale, mask=mask)


def compute_awq_scale_triton(
    x_mean: torch.Tensor,
    w_mean: torch.Tensor,
    ratio: float,
    duo_scaling: bool = True,
) -> torch.Tensor:
    """Compute AWQ scale for given ratio using Triton"""
    n_elements = x_mean.numel()
    scale = torch.empty_like(x_mean)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    awq_search_scale_kernel[grid](
        x_mean,
        w_mean,
        scale,
        ratio,
        duo_scaling,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    # Normalize scale
    scale_max = scale.max()
    scale_min = scale.min()
    if scale_max > 0 and scale_min > 0:
        scale = scale / (scale_max * scale_min).sqrt()
    
    # Handle inf/nan
    scale[torch.isinf(scale)] = 1.0
    scale[torch.isnan(scale)] = 1.0
    
    return scale


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def awq_loop_graph(
    weight: torch.Tensor,
    x_mean: torch.Tensor,
    w_mean: torch.Tensor,
    scale_out: torch.Tensor,
    qweight_out: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    best_ratio: float,
    duo_scaling: bool,
    dtype: torch.dtype,
    direct: bool = False,
):
    """AWQ quantization loop with CUDA graph support"""
    # Compute best scale
    best_scale = compute_awq_scale_triton(x_mean, w_mean, best_ratio, duo_scaling)
    
    # Apply scale to weight
    w_scaled = apply_awq_scale_triton(weight, best_scale.view(1, -1))
    
    # Quantize scaled weight
    n_elements = w_scaled.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Use existing quantize_error kernel for quantization
    from src.gptq_loop import quantize_error_triton_kernel
    quantize_error_triton_kernel[grid](
        w_scaled.flatten(),
        qweight_out.flatten(),
        torch.empty_like(w_scaled).flatten(),  # error buffer (unused)
        scale.flatten(),
        qzero.flatten(),
        maxq,
        torch.empty(0, dtype=dtype) if dtype is not None else None,
        n_elements,
        BLOCK_SIZE=128,
    )
    
    # Store best scale
    scale_out.copy_(best_scale)
    
    if direct:
        return qweight_out, scale_out
    else:
        return qweight_out


def awq_loop(
    weight: torch.Tensor,
    x_mean: torch.Tensor,
    w_mean: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: torch.Tensor,
    grid_size: int,
    duo_scaling: bool,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AWQ quantization with grid search
    
    Args:
        weight: (R, C) weight tensor to quantize
        x_mean: (C,) per-channel activation means
        w_mean: (C,) per-channel weight importance
        scale: (R, G) quantization scale
        qzero: (R, G) quantization zero point
        maxq: () maximum quantized value
        grid_size: number of grid points to search
        duo_scaling: whether to use duo scaling
        dtype: target dtype
        
    Returns:
        qweight: (R, C) quantized weight
        awq_scale: (C,) optimal AWQ smoothing scale
    """
    device = weight.device
    n_rows, n_cols = weight.shape
    
    # Find best ratio via grid search
    best_error = float('inf')
    best_ratio = 0.5
    
    # Cache a small sample of activations for error computation
    # This is a simplified version - in practice you'd use actual activations
    test_acts = torch.randn(32, n_cols, device=device).abs() * x_mean.view(1, -1)
    org_output = torch.matmul(test_acts, weight.t())
    
    for i in range(grid_size):
        ratio = i / grid_size
        
        # Compute scale for this ratio
        test_scale = compute_awq_scale_triton(x_mean, w_mean, ratio, duo_scaling)
        
        # Apply scale and quantize
        w_scaled = apply_awq_scale_triton(weight, test_scale.view(1, -1))
        
        # Simple quantization for error computation
        qw = torch.round(w_scaled / scale).clamp(0, maxq)
        w_dequant = qw * scale
        
        # Apply inverse scale to activations
        scaled_acts = test_acts / test_scale.view(1, -1)
        
        # Compute error
        quant_output = torch.matmul(scaled_acts, w_dequant.t())
        error = (org_output - quant_output).pow(2).mean().item()
        
        if error < best_error:
            best_error = error
            best_ratio = ratio
    
    # Use CUDA graph for final quantization
    previous_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(weight.device)
    
    if not hasattr(awq_loop, "graph_info"):
        awq_loop.graph_info = {}
        
    graph_key = (n_rows, n_cols, weight.dtype, dtype, duo_scaling, device)
    
    if graph_key not in awq_loop.graph_info:
        graph = torch.cuda.CUDAGraph()
        graph_tensors = {
            "weight": torch.empty_like(weight),
            "x_mean": torch.empty_like(x_mean),
            "w_mean": torch.empty_like(w_mean),
            "scale_out": torch.empty_like(x_mean),
            "qweight_out": torch.empty_like(weight, dtype=torch.uint8),
            "scale": torch.empty_like(scale),
            "qzero": torch.empty_like(qzero),
            "maxq": torch.empty_like(maxq),
        }
        
        # Warmup
        n_warmups = 5
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                awq_loop_graph(
                    **graph_tensors,
                    best_ratio=best_ratio,
                    duo_scaling=duo_scaling,
                    dtype=dtype,
                    direct=True,
                )
        torch.cuda.current_stream().wait_stream(s)
        
        # Record graph
        with torch.cuda.graph(graph):
            awq_loop_graph(
                **graph_tensors,
                best_ratio=best_ratio,
                duo_scaling=duo_scaling,
                dtype=dtype,
                direct=True,
            )
            
        awq_loop.graph_info[graph_key] = {"graph": graph, "tensors": graph_tensors}
    
    graph, graph_tensors = (
        awq_loop.graph_info[graph_key]["graph"],
        awq_loop.graph_info[graph_key]["tensors"],
    )
    
    # Copy inputs
    graph_tensors["weight"].copy_(weight)
    graph_tensors["x_mean"].copy_(x_mean)
    graph_tensors["w_mean"].copy_(w_mean)
    graph_tensors["scale"].copy_(scale)
    graph_tensors["qzero"].copy_(qzero)
    graph_tensors["maxq"].copy_(maxq)
    
    # Replay graph
    graph.replay()
    
    torch.cuda.set_device(previous_device)
    
    return graph_tensors["qweight_out"], graph_tensors["scale_out"]