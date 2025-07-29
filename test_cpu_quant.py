#!/usr/bin/env python3
"""
Test script for CPU-only quantization
"""
import torch
import torch.nn as nn
from src.cpu_versions import gptq_cpu, awq_cpu

def test_gptq_cpu():
    print("Testing GPTQ CPU implementation...")
    
    # Create a simple linear layer
    layer = nn.Linear(128, 64, bias=False)
    layer.weight.data = torch.randn_like(layer.weight)
    
    # Create GPTQ handle
    handle = gptq_cpu.GPTQ(
        layer,
        group_size=32,
        sym=True,
        rel_damp=0.01,
        block_size=128,
        quantization_order="default",
        quantization_scale="absmax"
    )
    
    # Generate some fake input data
    batch_size = 16
    seq_len = 32
    for _ in range(5):
        fake_input = torch.randn(batch_size, seq_len, 128)
        handle.update(fake_input)
    
    # Quantize
    qweight, scale, zero = handle.quantize(bits=4)
    
    print(f"Original weight shape: {layer.weight.shape}")
    print(f"Quantized weight shape: {qweight.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Zero shape: {zero.shape}")
    print(f"Issues - Zero samples: {handle.issue_zero_samples}, NaN Hessian: {handle.issue_nan_hessian}, Non-invertible: {handle.issue_non_invertible}")
    print("GPTQ CPU test passed!\n")

def test_awq_cpu():
    print("Testing AWQ CPU implementation...")
    
    # Create a simple linear layer
    layer = nn.Linear(128, 64, bias=False)
    layer.weight.data = torch.randn_like(layer.weight)
    
    # Create AWQ handle
    handle = awq_cpu.AWQ(
        layer,
        group_size=32,
        sym=True,
        duo_scaling=True,
        grid_size=20
    )
    
    # Generate some fake input data
    batch_size = 16
    seq_len = 32
    for _ in range(5):
        fake_input = torch.randn(batch_size, seq_len, 128)
        handle.update(fake_input)
    
    # Quantize
    qweight, scale, zero = handle.quantize(bits=4)
    
    print(f"Original weight shape: {layer.weight.shape}")
    print(f"Quantized weight shape: {qweight.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Zero shape: {zero.shape}")
    print(f"Smoothing scale shape: {handle.smooth_scale.shape if handle.smooth_scale is not None else 'None'}")
    print(f"Issues - Zero samples: {handle.issue_zero_samples}, NaN activations: {handle.issue_nan_activations}")
    print("AWQ CPU test passed!\n")

def test_tied_handles():
    print("Testing tied handles...")
    
    # Create two linear layers (gate and up projections)
    gate_layer = nn.Linear(128, 64, bias=False)
    up_layer = nn.Linear(128, 64, bias=False)
    
    # Create GPTQ handles with tying
    gate_handle = gptq_cpu.GPTQ(
        gate_layer,
        group_size=32,
        sym=True,
        rel_damp=0.01
    )
    
    up_handle = gptq_cpu.GPTQ(
        up_layer,
        group_size=32,
        sym=True,
        rel_damp=0.01,
        tied_gptq_handle=gate_handle
    )
    
    # Update only gate handle
    batch_size = 16
    seq_len = 32
    for _ in range(5):
        fake_input = torch.randn(batch_size, seq_len, 128)
        gate_handle.update(fake_input)
    
    # Quantize both
    gate_qweight, gate_scale, gate_zero = gate_handle.quantize(bits=4)
    up_qweight, up_scale, up_zero = up_handle.quantize(bits=4)
    
    print(f"Gate handle samples: {gate_handle.num_samples}")
    print(f"Up handle samples: {up_handle.num_samples} (should be same as gate)")
    print(f"Tied handles: {gate_handle.num_tied_handles}")
    print("Tied handles test passed!\n")

if __name__ == "__main__":
    print("Running CPU-only quantization tests...\n")
    test_gptq_cpu()
    test_awq_cpu()
    test_tied_handles()
    print("All tests passed!")