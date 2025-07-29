#!/usr/bin/env python3
"""
Test script for CPU parallelization
"""
import time
import torch
import torch.nn as nn
import multiprocessing
from src.cpu_versions import gptq_cpu, awq_cpu

def test_parallel_performance():
    print("Testing parallel CPU performance...")
    print(f"Total CPU cores available: {multiprocessing.cpu_count()}")
    
    # Test different thread counts
    thread_counts = [1, int(multiprocessing.cpu_count() * 0.5), int(multiprocessing.cpu_count() * 0.8), multiprocessing.cpu_count()]
    
    # Create a larger layer for better parallelization testing
    layer_size = (512, 1024)
    layer = nn.Linear(layer_size[1], layer_size[0], bias=False)
    layer.weight.data = torch.randn_like(layer.weight)
    
    # Set threads once at the beginning
    torch.set_num_threads(1)  # Start with 1 thread
    
    # Generate test data
    batch_size = 32
    seq_len = 64
    num_batches = 10
    test_inputs = [torch.randn(batch_size, seq_len, layer_size[1]) for _ in range(num_batches)]
    
    results = {}
    
    for num_threads in thread_counts:
        print(f"\nTesting with {num_threads} threads...")
        torch.set_num_threads(num_threads)
        # Don't set interop threads as it causes issues
        
        # Test GPTQ
        print("  Testing GPTQ...")
        # Create a fresh layer for each test
        test_layer = nn.Linear(layer_size[1], layer_size[0], bias=False)
        test_layer.weight.data = layer.weight.data.clone()
        
        handle = gptq_cpu.GPTQ(
            test_layer,
            group_size=128,
            sym=True,
            rel_damp=0.01,
            block_size=256,
            quantization_order="default",
            quantization_scale="absmax"
        )
        
        # Time the update phase
        start_time = time.time()
        for inp in test_inputs:
            handle.update(inp)
        update_time = time.time() - start_time
        
        # Time the quantization phase
        start_time = time.time()
        qweight, scale, zero = handle.quantize(bits=4)
        quant_time = time.time() - start_time
        
        gptq_total = update_time + quant_time
        print(f"    Update time: {update_time:.2f}s")
        print(f"    Quantization time: {quant_time:.2f}s")
        print(f"    Total time: {gptq_total:.2f}s")
        
        # Test AWQ
        print("  Testing AWQ...")
        # Create a fresh layer for each test
        test_layer = nn.Linear(layer_size[1], layer_size[0], bias=False)
        test_layer.weight.data = layer.weight.data.clone()
        
        handle = awq_cpu.AWQ(
            test_layer,
            group_size=128,
            sym=True,
            duo_scaling=True,
            grid_size=20
        )
        
        # Time the update phase
        start_time = time.time()
        for inp in test_inputs:
            handle.update(inp)
        update_time = time.time() - start_time
        
        # Time the quantization phase
        start_time = time.time()
        qweight, scale, zero = handle.quantize(bits=4)
        quant_time = time.time() - start_time
        
        awq_total = update_time + quant_time
        print(f"    Update time: {update_time:.2f}s")
        print(f"    Quantization time: {quant_time:.2f}s")
        print(f"    Total time: {awq_total:.2f}s")
        
        results[num_threads] = {
            'gptq_time': gptq_total,
            'awq_time': awq_total
        }
    
    # Print speedup summary
    print("\n" + "="*50)
    print("SPEEDUP SUMMARY")
    print("="*50)
    baseline_threads = thread_counts[0]
    baseline_gptq = results[baseline_threads]['gptq_time']
    baseline_awq = results[baseline_threads]['awq_time']
    
    print(f"\nBaseline ({baseline_threads} thread):")
    print(f"  GPTQ: {baseline_gptq:.2f}s")
    print(f"  AWQ: {baseline_awq:.2f}s")
    
    for num_threads in thread_counts[1:]:
        gptq_speedup = baseline_gptq / results[num_threads]['gptq_time']
        awq_speedup = baseline_awq / results[num_threads]['awq_time']
        print(f"\n{num_threads} threads:")
        print(f"  GPTQ speedup: {gptq_speedup:.2f}x")
        print(f"  AWQ speedup: {awq_speedup:.2f}x")

if __name__ == "__main__":
    test_parallel_performance()