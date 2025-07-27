from enum import Enum
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils, model_utils, linalg_utils, quant_utils, awq_loop


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class AWQ:
    """
    AWQ (Activation-aware Weight Quantization) implementation following GPTQ's exact architectural pattern.
    """

    def __init__(
        self,
        layer: nn.Module,
        group_size: Optional[int] = None,
        sym: bool = False,
        duo_scaling: bool = True,
        grid_size: int = 20,
        is_distributed: bool = False,
        tied_awq_handle: Optional["AWQ"] = None,
        offload_device: Optional[str] = None,
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = model_utils.get_number_of_rows_and_cols(layer)
        
        # Quantization hyperparameters
        self.sym = sym
        self.group_size = group_size
        
        # AWQ hyperparameters
        self.duo_scaling = duo_scaling
        self.grid_size = grid_size
        
        # Backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        
        # Memory optimization: offload device
        self.offload_device = offload_device or self.W_device
        
        # Activation statistics (memory efficient)
        self.activation_sum = None
        self.activation_count = 0
        self.is_distributed = is_distributed
        self.tied_awq_handle = tied_awq_handle
        self.num_tied_handles = 0
        
        # For tied handles, share statistics
        if tied_awq_handle is not None:
            tied_awq_handle.num_tied_handles += 1
            # Share activation statistics with tied handle
            if tied_awq_handle.activation_sum is not None:
                self.activation_sum = tied_awq_handle.activation_sum
                self.activation_count = tied_awq_handle.activation_count
                
        # Smoothing scale (computed during quantization)
        self.smooth_scale = None
        
        # Flags for tracking issues
        self.issue_zero_samples = False
        self.issue_nan_activations = False
        
        # Flag to track if pre-step is completed
        self.pre_step_completed = False
        
    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "AWQ supports only linear and convolutional layers."
        
    @property
    def tokens_collected(self) -> int:
        """For compatibility with GPTQ interface"""
        return self.activation_count
        
    def has_activation_issues(self) -> bool:
        return any([self.issue_zero_samples, self.issue_nan_activations])
        
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Collect input activations for AWQ calibration.
        Memory efficient: computes running statistics instead of storing all activations.
        
        Args:
            input: batch of layer inputs
        """
        # Input reshaping (same as GPTQ)
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
            
        input = input.float()
        
        # Compute running sum of absolute values (for mean calculation)
        input_abs = input.abs()
        if self.activation_sum is None:
            self.activation_sum = input_abs.sum(dim=0).to(self.offload_device)
        else:
            self.activation_sum += input_abs.sum(dim=0).to(self.offload_device)
            
        self.activation_count += input.shape[0]
        
    def reset(self) -> None:
        """
        Reset the AWQ instance, clearing stored statistics and freeing memory.
        """
        self.W = self.layer.weight
        if self.num_tied_handles == 0:
            self.activation_sum = None
            self.activation_count = 0
        elif self.tied_awq_handle:
            self.tied_awq_handle.num_tied_handles -= 1
            if self.tied_awq_handle.num_tied_handles == 0:
                self.tied_awq_handle.activation_sum = None
                self.tied_awq_handle.activation_count = 0
        self.smooth_scale = None
        self.issue_zero_samples = False
        self.issue_nan_activations = False
        self.pre_step_completed = False
        torch.cuda.empty_cache()
        
    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        Preparatory step - compute activation and weight statistics.
        Following GPTQ's pattern exactly.
        """
        # 1) Activation statistics preparation
        if self.activation_count == 0:
            if self.tied_awq_handle and self.tied_awq_handle.activation_sum is not None:
                self.activation_sum = self.tied_awq_handle.activation_sum
                self.activation_count = self.tied_awq_handle.activation_count
            else:
                # No activations collected
                self.issue_zero_samples = True
                self.activation_sum = torch.ones(self.d_col, device=self.W_device, dtype=torch.float32)
                self.activation_count = 1
                
        # Synchronize activation statistics if distributed
        if self.is_distributed and dist_utils.is_dist_available_and_initialized():
            if self.tied_awq_handle is None or self.tied_awq_handle.activation_sum is None:
                # Sum across all processes
                self.activation_sum = self.activation_sum.to(self.W_device)
                dist.all_reduce(self.activation_sum, op=dist.ReduceOp.SUM)
                activation_count_tensor = torch.tensor(self.activation_count, device=self.W_device)
                dist.all_reduce(activation_count_tensor, op=dist.ReduceOp.SUM)
                self.activation_count = activation_count_tensor.item()
                
        # Check for NaN
        if torch.isnan(self.activation_sum).any().item():
            self.activation_sum = torch.ones(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_nan_activations = True
            
        # 2) Weight preparation
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
            
        # Get pruned channels (following GPTQ pattern)
        self.x_mean = (self.activation_sum / self.activation_count).to(self.W_device)
        pruned_ids = self.x_mean == 0
        self.W[:, pruned_ids] = 0
        
        # Flag pre step as completed
        self.pre_step_completed = True
        
    @torch.no_grad()
    def _quantize(self, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the weight matrix using AWQ algorithm.
        Following GPTQ's _quantize pattern exactly.
        """
        # 1) Define constants
        d_row, d_col, device, dtype = self.d_row, self.d_col, self.W_device, self.W_dtype
        
        # 2) Get quantization group size
        group_size = self.group_size or d_col
        num_groups = d_col // group_size
        
        is_main_awq_process = dist_utils.is_main() or not self.is_distributed
        
        if is_main_awq_process:
            # Get scale, qzero for quantization
            scale, zero, maxq = quant_utils.get_quantization_grid(
                weight=self.W,
                group_size=self.group_size,
                bits=bits,
                symmetric=self.sym,
                dtype=dtype,
                quantization_scale="mse",  # AWQ typically uses MSE
            )
            
            # Compute weight statistics (per-channel)
            W = self.W.clone()
            org_shape = W.shape
            
            # For group quantization
            if self.group_size is not None and self.group_size > 0:
                W = W.view(-1, self.group_size)
            else:
                W = W.view(-1, W.shape[-1])
                
            # Normalize weights by group
            W_abs = W.abs()
            W_max = W_abs.amax(dim=1, keepdim=True).clamp(min=1e-6)
            W_scale = W_abs / W_max
            
            # Reshape back and compute mean
            W_scale = W_scale.view(org_shape)
            w_mean = W_scale.mean(dim=0)
            
            # Use AWQ loop with Triton kernels
            qweight, awq_scale = awq_loop.awq_loop(
                weight=self.W,
                x_mean=self.x_mean,
                w_mean=w_mean,
                scale=scale,
                qzero=zero,
                maxq=maxq,
                grid_size=self.grid_size,
                duo_scaling=self.duo_scaling,
                dtype=dtype,
            )
            
            # Store smoothing scale
            self.smooth_scale = awq_scale
            
            # Convert to uint8 and reshape scales/zeros
            qweight = qweight.contiguous().to(torch.uint8)
            scale = scale[:, ::group_size].to(dtype)
            zero = zero[:, ::group_size].to(dtype)
            
        else:
            # Non-main processes
            qweight = torch.empty(d_row, d_col, device=device, dtype=torch.uint8)
            scale = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            zero = torch.empty(d_row, num_groups, device=device, dtype=dtype)
            
        # Broadcast results if distributed
        if self.is_distributed and dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(qweight, src=0)
            dist.broadcast(scale, src=0)
            dist.broadcast(zero, src=0)
            
        return qweight, scale, zero
        
    def quantize(self, bits: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Main quantization entry point, following GPTQ pattern.
        """
        self.quantization_pre_step()
        return self._quantize(bits)