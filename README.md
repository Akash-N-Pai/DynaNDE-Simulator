# DynaNDE-MoE Simulator

This simulator extends the [NeuPIMs cycle-accurate simulator](https://github.com/casys-kaist/NeuPIMs) with comprehensive support for **Mixture of Experts (MoE)** architectures. The original NeuPIMs simulator integrates an open-source [NPU simulator](https://github.com/PSAL-POSTECH/ONNXim) and an in-house PIM (Processing-In-Memory) simulator based on DRAMsim3.

## Overview

This research project focuses on **improving load balancing between NPU and NDP/PIM (Near-Data Processing/Processing-In-Memory)** for MoE-based large language models. The simulator has been extended to support:

- **MoE model architectures**: Flame-moe, DeepSeek, Switch-base, and other MoE variants
- **Expert routing and execution**: Token-to-expert assignment with configurable routing strategies
- **Hybrid NPU-PIM execution**: Flexible assignment of expert computations to NPU or PIM
- **Trace-based simulation**: Support for routing traces generated from real model executions
- **Load balancing optimizations**: Realistic modeling of expert load imbalance and skew

### Original NeuPIMs Publication

- **Paper**: [NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing](https://dl.acm.org/doi/10.1145/3620666.3651380)
- **Authors**: Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, and Sanghyeon Lee (KAIST); Hyungkyu Ham and Gwangsun Kim (POSTECH); Divya Mahajan (Georgia Tech); Jongse Park (KAIST)

> **Note**: Citation for this MoE extension will be added upon publication.

---

## Getting Started

### Prerequisites

#### Python Package
- torch >= 1.10.1
- conan == 1.57.0
- onnxruntime >= 1.10.0

#### System Requirements
- cmake >= 3.22.1 (You need to build manually)
- gcc == 8.3

---

## Installation

### Method 1: Docker Image

```bash
$ git clone https://github.com/Akash-N-Pai/DynaNDE-Simulator.git
$ cd NeuPIMs
$ docker build . -t neupims
$ docker run -it -v .:/workspace/neupims-sim neupims
(docker) cd neupims-sim
(docker) git submodule update --recursive --init
(docker) ./build.sh
```

To run the simulator:
```bash
$ docker run -it -v .:/workspace/neupims-sim neupims
(docker) cd neupims-sim
(docker) ./brun_fast.sh
```
## ⚠️ IMPORTANT NOTE - Quick Start with Pre-Run Files

### Sample Pre-Run Simulation Data Available

**To save time, we have included pre-run simulation files for immediate analysis!**

The first phase of simulation (NPU and PIM execution data) has already been generated for both simulators FLAME-MoE:

#### **Prefiller Simulator** (`prefiller-simulator/`)
- **Dataset**: `Encoder-BS32-SL64/`
- **Includes**: 
  - Pre-computed NPU execution data (`npu/`)
  - Pre-computed PIM execution data (`pim/`)
  - Layer routing statistics files (`layer2.txt` through `layer9.txt`)
- **Usage**: Open `prefiller-simulator/MOE_simulator.ipynb` and run directly with `base_dir = 'Encoder-BS32-SL64'`

#### **Decoder Simulator** (`decoder-simulator/`)
- **Dataset**: `Decoder-BS16/`
- **Includes**:
  - Pre-computed NPU execution data for each token position (`1st/`, `2nd/`, ..., `10th/`)
  - Pre-computed PIM execution data for each token position
  - Layer routing statistics files (`layer2.txt` through `layer9.txt`) for each token position
- **Usage**: Open `decoder-simulator/MOE_simulator.ipynb` and run directly with `base_dir = 'Decoder-BS16'`

**You can immediately analyze NPU-only, PIM-only, MoNDE hybrid, and DynaNDE modes without running the time-consuming NeuPIMs-MoE simulator first.**

📖 **See detailed instructions in each folder's README.md:**
- `prefiller-simulator/README.md`
- `decoder-simulator/README.md`

### Automated Batch Run Script

A sample automated run script for decoder operation (`run_all_batches.sh`) is included. It can be used to simulate all decoder CSV routing trace files—approximately 240 traces—by running them sequentially. You may need to modify this script to match the structure and number of your trace files. A similar script can also be created for the encoder if needed.
or you can manulay enter each of these csv files in the  moe_routing_trace_path of the model_config files.
### Note on Dataset Size

Please note that not all trace files and pre-run outputs are included, as the full dataset is too large. The provided samples are sufficient for understanding the workflow and getting started quickly.

---

## Running MoE Simulations

### Quick Start

The simulator uses configuration files to specify model, memory, system, and client settings:

```bash
$ ./brun_fast.sh
```

This script runs a fast simulation with optimized settings. The script uses the following default configurations:
- **Memory config**: `./configs/memory_configs/neupims.json`
- **Model config**: `./configs/model_configs/Flame-moe-npu.json`
- **System config**: `./configs/system_configs/sub-batch-off.json`


### Custom Configuration

You can modify `brun_fast.sh` to use different configurations:

```bash
# Edit brun_fast.sh
config=./configs/systolic_ws_dev.json
mem_config=./configs/memory_configs/neupims.json
model_config=./configs/model_configs/Flame-moe-npu.json  # or Flame-moe-pim.json
sys_config=./configs/system_configs/sub-batch-off.json

```


### Key MoE Configuration Parameters

| Parameter | Description |
|:---------:|:------------|
| `moe_enabled` | Enable MoE mode (boolean) |
| `num_experts` | Total number of experts in the MoE layer |
| `experts_per_token` | Top-k experts selected per token (routing) |
| `expert_capacity_factor` | Capacity multiplier for expert buffers |
| `expert_load_imbalance` | Enable realistic expert load skew modeling |
| `expert_load_skew` | Skew factor (0.0-1.0) for load distribution |
| `moe_ffn_scaling` | Expert FFN width scaling: `"balanced"`, `"compute"`, or `"capacity"` |
| `moe_offchip_experts` | Experts stored in HBM (not on-chip) |
| `expert_load_latency` | Cycles to load one expert from HBM to SRAM |
| `expert_cache_size` | Number of experts that fit in on-chip cache |
| `moe_enable_parallelism` | Enable parallel expert execution |
| `moe_enable_double_buffering` | Overlap parameter load and compute |
| `moe_routing_trace_path` | Path to routing trace file (optional, empty for simulated routing) |
| `ffn_execution_mode` | FFN execution mode: `"npu"` or `"pim"` |

### Supported Models

The simulator has been tested with:
- **Flame-moe**: DeepSeek-based MoE architecture
- **DeepSeek**: DeepSeek MoE variants
- **Switch-base**: Switch Transformer architecture

Model configurations can be created by adapting the example JSON files in `configs/model_configs/`.

---

## Routing Trace Generation

The simulator supports two modes for expert routing:

1. **Trace-based routing**: Use pre-generated routing traces from actual model executions
   - Set `moe_routing_trace_path` to the trace file path
   - Traces can be generated using tools in `DynaNDE-Trcegenerator/`
   - Sample routing trace folder has the sample trace files for refference each csv file in this needs to be fed to the simulator through the model cofig file 

2. **Simulated routing**: Automatically generate token-to-expert assignments
   - Leave `moe_routing_trace_path` empty
   - Uses configurable load imbalance and skew factors

For detailed information on trace generation, see the README files in: `DynaNDE-Trcegenerator/`

---

## Output and Logging

Simulation results are saved in timestamped directories under `experiment_logs/`:

```
experiment_logs/
└── 2025-11-29_00:23:31/
    ├── config.log
    ├── terminal_output.log
    ├── SA_stage_E.tsv          # Stage E statistics
    └── ... (other log files)
```

The `SA_stage_E.tsv` file contains detailed statistics for MoE expert operations, including:
- Expert execution cycles
- Parameter load times
- Token distribution across experts

---

## Architecture Overview

### MoE Components

The simulator implements the following MoE-specific components:

1. **MoE Router**: Routes input tokens to selected experts using a gating network
2. **MoE Expert**: Individual expert FFN computation (FC1 → GELU → FC2)
3. **MoE Combine**: Weighted combination of expert outputs
4. **Expert Parameter Loading**: Models HBM-to-SRAM transfer for off-chip experts
5. **Expert Cache**: LRU cache for frequently-used experts
6. **Execution Planner**: Optimizes parallel expert execution with caching and double buffering

### NPU-PIM Load Balancing

The simulator models heterogeneous execution where:
- **NPU execution**: Experts run on the Neural Processing Unit (bigger systolic arrays connected through low bandwith interconnect)
- **PIM execution**: Experts run on Processing-In-Memory (smaller systolic arrays connected through High bandwith Memory)
- **Hybrid mode**: Different experts can be assigned to NPU or PIM based on configuration (MoNDE and DynaNDE)

This enables research into optimal load balancing strategies between NPU and PIM for MoE workloads.

---

## Baselines

The simulator supports the following execution modes:

1. **MoE-NPU**: MoE experts executed on NPU
2. **MoE-PIM**: MoE experts executed on PIM
3. **MoNDE**: Mixed NPU/PIM execution for MoE experts based on badwith diffrence
4. **DynaNDE**: Mixed NPU/PIM execution for MoE experts based on optimal execution time 
---

## Citation

If you use this MoE-extended simulator for your research, please cite:

1. **Original NeuPIMs paper**:
```
@inproceedings{10.1145/3620666.3651380,
author = {Heo, Guseul and Lee, Sangyeop and Cho, Jaehong and Choi, Hyunmin and Lee, Sanghyeon and Ham, Hyungkyu and Kim, Gwangsun and Mahajan, Divya and Park, Jongse},
title = {NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing},
year = {2024},
doi = {10.1145/3620666.3651380},
booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
series = {ASPLOS '24}
}
```

2. **This MoE extension**: Citation will be added upon publication.

---

## Additional Resources

- **Trace Generation Tools**: See `DynaNDE-Trcegenerator/README.md` 
- **Decoder Simulator**: See `decoder-simulator/README.md` 
- **Prefiller Simulator**: See `prefiller-simulator/README.md` 

---
