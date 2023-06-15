# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm
import timm


def benchmark_random(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput
    

def benchmark_dataset(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    throw_out: float = 0.25,
    runs: int = -1,
    use_fp16: bool = False,
    verbose: bool = False,
    dataset: torch.utils.data.DataLoader = None
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    
    idx = 0
    total = 0
    total_correct = 0
    total_time = 0
    runs = len(dataset) if runs == -1 else runs

    for (input, label) in tqdm(dataset, disable=not verbose, desc="Benchmarking"):
        input = input.to(device)
        if use_fp16:
            input = input.half()

        with torch.autocast(device.type, enabled=use_fp16):
            with torch.no_grad():
                if idx == runs * throw_out:
                    total = 0
                    total_correct = 0
                    total_time = 0

                start = time.time()
                if is_cuda:
                    torch.cuda.synchronize()
                output = model(input)
                if is_cuda:
                    torch.cuda.synchronize()
                end = time.time()

                total_time += end - start
                total += input.shape[0]
                total_correct += (output.argmax(dim=1) == label.to(device)).sum().item()

                idx +=1
                if runs != -1 and idx == runs:
                    break

    elapsed = total_time
    throughput = total / elapsed
    accuracy = total_correct / total

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
        print(f"Accuracy: {accuracy*100:.2f}%")

    return throughput

if __name__ == "__main__":
    # Use any ViT model here (see timm.models.vision_transformer)
    model_name = "vit_base_patch16_224"

    # Load a pretrained model
    model = timm.create_model(model_name, pretrained=True)
    
    # Set this to be whatever device you want to benchmark on
    # If you don't have a GPU, you can use "cpu" but you probably want to set the # runs to be lower
    device = "cuda:0"
    runs = 50
    batch_size = 256  # Lower this if you don't have that much memory
    input_size = model.default_cfg["input_size"]

    # Baseline benchmark
    baseline_throughput = benchmark_random(
        model,
        device=device,
        verbose=True,
        runs=runs,
        batch_size=batch_size,
        input_size=input_size
    )