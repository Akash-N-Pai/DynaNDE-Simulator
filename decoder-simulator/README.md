# Decoder Simulator

Simulation tools for analyzing MoE expert execution in the **decoder/autoregressive phase** (sequential token generation). This simulator evaluates different execution strategies: NPU-only, PIM-only, MoNDE hybrid, and DynaNDE incremental optimization with cache awareness.

---

## Overview

The decoder simulator processes decoder-phase MoE execution where tokens are generated sequentially (autoregressive). It analyzes cycle-accurate execution data from the NeuPIMs-MoE simulator to compare different load balancing strategies between NPU and PIM.

**Key Feature**: The simulator processes **all layers across all token positions in first 10 forward pass**, aggregating execution cycles across all layers and token positions to provide **total compute cycles** for each execution mode (NPU-only, PIM-only, MoNDE, DynaNDE). This gives a complete picture of end-to-end decoder performance rather than per-token or per-layer analysis.

---

## Components

### 1. MOE Simulator (`MOE_simulator.ipynb`)

Main simulation notebook that analyzes four execution modes:

#### **NPU-Only Mode**
- **Description**: All experts execute on NPU (Neural Processing Unit) across all layers and token positions
- **Calculation**: 
  - For each token position and each layer: Sum of parameter load cycles for all experts + total compute cycles of the last expert (expert 63)
  - Sum across all token positions and all layers in the forward pass
- **Output**: **Total compute cycles** for the entire decoder forward pass (all layers, all token positions) in NPU-only mode

#### **PIM-Only Mode** (NDP - Near-Data Processing)
- **Description**: All experts execute on PIM (Processing-In-Memory) across all layers and token positions
- **Calculation**:
  - For each token position and each layer: Sum of total compute cycles for all experts (fc1 + gelu + fc2) + activation movement cycles (movement_1 + movement_2)
  - Sum across all token positions and all layers in the forward pass
- **Output**: **Total compute cycles** for the entire decoder forward pass (all layers, all token positions) in PIM-only mode

#### **MoNDE Hybrid Mode**
- **Description**: Dynamic hybrid execution based on **bandwidth difference** between NPU and PIM across all layers and token positions
- **Strategy**: 
  - Dynamically calculates number of experts on NPU based on bandwidth ratio: `num_on_npu = ceil(0.05882 × total_active_experts)`
  - Top N most active experts → NPU (N is calculated per layer/token position)
  - Remaining experts → PIM
- **Calculation**:
  - For each token position and each layer:
    - Identify active experts from routing statistics
    - Calculate dynamic NPU allocation based on bandwidth ratio
    - **NPU time**: Parameter load for top N experts + compute of Nth expert
    - **PIM time**: Total compute for remaining experts + activation movements
    - **Token/layer total**: `MAX(NPU_time, PIM_time)` (NPU and PIM execute in parallel)
  - Sum across all token positions and all layers in the forward pass
- **Output**: **Total compute cycles** for the entire decoder forward pass (all layers, all token positions) in MoNDE hybrid mode
- **Input**: Layer routing statistics files (`layer{num}.txt`) to identify active experts and their ordering per token position

#### **DynaNDE Incremental Optimization**
- **Description**: Dynamic optimization with **cache awareness** that finds the **optimal expert split** between NPU and PIM across all layers and token positions
- **Strategy**: Tests all configurations from 0 to total_active_experts on NPU per token position/layer
  - 0 experts on NPU = All on PIM
  - 1 to total_active-1 experts on NPU = Hybrid (top N on NPU, rest on PIM)
  - All active experts on NPU = All on NPU
- **Cache-Aware Optimization**:
  - Uses **LFU (Least Frequently Used) cache** with configurable cache size (e.g., 10-16 experts per layer, ~15-25% of 64 experts)
  - Cache persists across token positions and layers
  - **Benefit calculation**: For each expert, calculates `benefit = PIM_time - (NPU_time + param_load_cost)`
    - If expert is cached: `param_load_cost = 0` (zero cost)
    - If expert is not cached: `param_load_cost = param_load_cycles`
  - Experts sorted by benefit (highest benefit first) - cached experts naturally rank higher
  - For each configuration:
    - NPU time = param_load of non-cached experts + compute of cached experts (parallel) + compute of last expert (sequential)
    - PIM time = compute of remaining experts + activation movements
    - Token/layer total = `MAX(NPU_time, PIM_time)`
  - Selects configuration with **minimum total cycles** per token position/layer
  - Sums optimal configurations across all token positions and layers
- **Output**: Optimal number of experts on NPU per token position/layer and **total compute cycles** for the entire decoder forward pass

---

### 2. Analyze Experts (`analyze_experts.ipynb`)

Utility notebook for processing and analyzing simulation results:

- **Data extraction**: Parses `SA_stage_E.tsv` files from NPU and PIM simulations
- **Statistics generation**: Extracts expert operation cycles (param_load, fc1, gelu, fc2)
- **Token position processing**: Processes data organized by token positions (1st, 2nd, 3rd, etc.)
- **Layer file processing**: Reads routing statistics to identify active experts per token position

---

## Input Requirements

### Directory Structure

```
Decoder-BS{size}/
├── 1st/                    # First token position
│   ├── npu/
│   │   ├── 2/              # Experiment 2 (layer 2)
│   │   │   └── SA_stage_E.txt
│   │   ├── 3/              # Experiment 3 (layer 3)
│   │   └── ...
│   ├── pim/
│   │   ├── 2/
│   │   │   └── SA_stage_E.txt
│   │   └── ...
│   └── layer2.txt          # Routing statistics for this token position
├── 2nd/                    # Second token position
│   └── ...
└── ...
```

### Input Files

1. **`SA_stage_E.txt`** (NPU folder):
   - Expert operations table with columns:
     - Expert Number, param_load, fc1, gelu, fc2, total compute
   - One file per token position and experiment (layer)

2. **`SA_stage_E.txt`** (PIM folder):
   - Expert operations table (no param_load):
     - Expert Number, fc1, gelu, fc2, total compute
   - Activation movement cycles (movement_1, movement_2)

3. **`layer{num}.txt`** (Routing statistics):
   - Expert token distribution from trace generator
   - One file per token position
   - Used to identify active experts and their ordering for MoNDE and DynaNDE

---

## Execution Modes Comparison

| Mode | Experts on NPU | Experts on PIM | Decision Criteria | Cache Awareness |
|:----:|:--------------:|:--------------:|:-----------------|:---------------:|
| **NPU-Only** | All (64) | 0 | All experts on NPU | No |
| **PIM-Only** | 0 | All (64) | All experts on PIM | No |
| **MoNDE** | Dynamic (bandwidth-based) | Remaining | Bandwidth ratio (dynamic per layer/token) | No |
| **DynaNDE** | Optimal (varies) | Remaining | Execution time minimization + cache benefit | **Yes (LFU)** |

---

## Key Concepts

### Sequential Token Processing
Decoder processes tokens **sequentially** (1st, 2nd, 3rd, etc.). Each token position may have different expert routing patterns, requiring per-token optimization.

### Parallel Execution
NPU and PIM execute **simultaneously** in hybrid modes. Total execution time is:
```
Total = MAX(NPU_time, PIM_time)
```
This models realistic parallel execution where both units work concurrently.

### Cache-Aware Optimization (DynaNDE)
- **LFU Cache**: Tracks expert usage frequency across token positions and layers
- **Cache Benefit**: Cached experts have zero parameter load cost, making them more attractive for NPU execution
- **Benefit-Based Sorting**: Experts sorted by `PIM_time - NPU_time - param_load_cost`, prioritizing high-benefit experts
- **Persistent Cache**: Cache state persists across token positions, modeling realistic expert reuse

### Parameter Load Overhead
- **NPU**: Requires loading expert parameters from HBM to SRAM
- **PIM**: No parameter load (experts stored in memory)
- **Cache Effect**: Cached experts avoid parameter load, significantly reducing NPU execution time

### Activation Movement
- **PIM-only**: Requires moving activations to/from PIM
- Two movements: one for fc1 input, one for fc2 output
- Included in PIM total execution time

### Optimization Goal
DynaNDE finds the split that **minimizes total execution time** by balancing:
- NPU parameter load overhead vs. compute efficiency
- PIM compute efficiency vs. activation movement overhead
- Cache benefits (zero-cost parameter loads for cached experts)
- Load distribution across both units

---

## Usage

1. **Run NeuPIMs-MoE simulator** to generate `SA_stage_E.txt` files for NPU and PIM modes for each token position
2. **Generate routing statistics** using `DynaNDE-Trcegenerator/decoder.ipynb` to create `layer{num}.txt` files per token position
3. **Run `MOE_simulator.ipynb`**:
   - Configure `base_dir` to point to your experiment folder
   - Configure `numbered_folders` (e.g., `['1st', '2nd', ..., '10th']`)
   - Execute cells to analyze NPU-only, PIM-only, MoNDE, and DynaNDE modes
4. **Use `analyze_experts.ipynb`** for data preprocessing and detailed analysis

---

## Output

The simulator generates:
- **Summary tables** for each execution mode showing per-token-position and per-layer breakdown
- **Total compute cycles** for the entire decoder forward pass (sum across all token positions and all layers) in each mode
- **Optimal configurations** (DynaNDE) showing best expert split per token position/layer
- **Cache statistics** (DynaNDE) showing cache hit rates and benefit calculations
- **Performance comparison** across all modes for end-to-end decoder execution

---

## Notes

- **Batch size** is encoded in folder names (e.g., `Decoder-BS32`, `Decoder-BS64`)
- **Token positions**: Processes sequential decoder tokens (1st, 2nd, 3rd, etc.), each with potentially different expert routing
- **Forward pass simulation**: Processes all layers across all token positions in one forward pass, providing total compute cycles for end-to-end decoder execution
- **Layer cycling**: Experiments 2-12 typically cycle through layers 2-9 (8 layers)
- **Cache-aware optimization**: DynaNDE uses LFU cache to model realistic expert reuse, significantly improving performance
- **Real routing data**: Uses actual expert routing from model execution (not synthetic)
- **Cycle-accurate**: Based on detailed cycle-level simulation from NeuPIMs-MoE

---

## Differences from Prefiller Simulator

| Aspect | Prefiller | Decoder |
|:-------|:----------|:--------|
| **Token Processing** | Parallel (all tokens at once) | Sequential (1st, 2nd, 3rd, etc.) |
| **MoNDE Strategy** | Fixed top 4 experts | Dynamic (bandwidth-based calculation) |
| **DynaNDE Cache** | No cache awareness | LFU cache with persistent state |
| **Input Organization** | By layer | By token position → layer |
| **Optimization Scope** | Per-layer, then sum | Per token position/layer, then sum |

