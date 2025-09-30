# Experiment Toolset README

This repository contains essential scripts and CUDA code for analyzing Large Language Model (LLM) inference performance and validating low-level GPU kernel optimizations.

---

## I. Command Line Inference (llama.cpp)

Use the `llama.cpp` CLI tool to get standardized load time, TTFT, TPOT metrics quickly.

[llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)

Prerequisite: Model must be converted to GGUF format.

```bash
./llama-cli -m <MODEL_PATH/model.gguf> \
            -p "Your Prompt" \
            -ngl <LAYERS_ON_GPU>
```

---

## II. LLM Inference Profiling (Nsight Tools)

The `kernel/test_qwen.py` and `kernel/test_gemma.py` scripts are used to perform LLM inference on PyTorch models. They serve as the benchmark foundation for NVIDIA Nsight tool profiling on Jetson and general-purpose GPUs.

### 1. System-Level Profiling (Nsight Systems - `nsys`)

Execute the LLM inference tasks to generate the necessary **operator-level CUDA activity trace** for subsequent profiling.

```bash
# Nsys Command Example (Qwen)
# Traces CUDA activity, runtime API calls, and Python activity
nsys profile -o qwen_profile_nsys --force-overwrite true python3 test_qwen.py

# Nsys Command Example (Gemma)
nsys profile -o gemma_profile_nsys --force-overwrite true python3 test_gemma.py
```

### 2. Kernel-Level Profiling (Nsight Compute - `ncu`)

Obtain detailed kernel statistics (Warp Occupancy, Memory Throughput) for low-level analysis.

```bash
# Ncu Command Example (Qwen)
# Profiling a single run for detailed metrics
ncu --set full -o profile_ncu python3 test_qwen.py

# Ncu Command Example (Gemma)
ncu --set full -o profile_ncu python3 test_gemma.py
```

## III. Kernel Fusion Proof-of-Concept
The `kernel_fusion/kernel_fusion.cu` code demonstrates the performance benefit of combining small operations (Add + Scale) into a single fused kernel.

### 1. Compilation and Execution

```bash
# Compile the CUDA code using nvcc
nvcc kernel_fusion.cu -o kernel_fusion

# Run the executable; output will show unfused vs. fused time (ms)
./kernel_fusion

# Profile using nsys profile to gather timeline information
nsys profile --stats=true -o fusion_report ./kernel_fusion

```