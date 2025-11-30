# Prefiller Simulator

Simulation tools for analyzing MoE expert execution in the **encoder/prefill phase** (parallel token processing). This simulator evaluates different execution strategies: NPU-only, PIM-only, MoNDE hybrid, and DynaNDE incremental optimization.

---

## Overview

The prefiller simulator processes encoder-phase MoE execution where all tokens in a sequence are processed in parallel. It analyzes cycle-accurate execution data from the NeuPIMs-MoE simulator to compare different load balancing strategies between NPU and PIM.

**Key Feature**: The simulator processes **all layers in one forward pass**, aggregating execution cycles across all layers to provide **total compute cycles** for each execution mode (NPU-only, PIM-only, MoNDE, DynaNDE). This gives a complete picture of end-to-end performance rather than per-layer analysis.

---

## Components

### 1. MOE Simulator (`MOE_simulator.ipynb`)

Main simulation notebook that analyzes four execution modes:

#### **NPU-Only Mode**
- **Description**: All experts execute on NPU (Neural Processing Unit) across all layers
- **Calculation**: 
  - For each layer: Sum of parameter load cycles for all experts + total compute cycles of the last expert (expert 63)
  - Sum across all layers in the forward pass
- **Output**: **Total compute cycles** for the entire forward pass (all layers) in NPU-only mode

#### **PIM-Only Mode** (NDP - Near-Data Processing)
- **Description**: All experts execute on PIM (Processing-In-Memory) across all layers
- **Calculation**:
  - For each layer: Sum of total compute cycles for all experts (fc1 + gelu + fc2) + activation movement cycles (movement_1 + movement_2)
  - Sum across all layers in the forward pass
- **Output**: **Total compute cycles** for the entire forward pass (all layers) in PIM-only mode

#### **MoNDE Hybrid Mode**
- **Description**: Static hybrid execution based on **bandwidth difference** between NPU and PIM across all layers
- **Strategy**: 
  - Top 4 most active experts → NPU (based on routing statistics)
  - Remaining 60 experts → PIM
- **Calculation**:
  - For each layer:
    - **NPU time**: Parameter load for top 4 experts + compute of 4th expert
    - **PIM time**: Total compute for remaining 60 experts + activation movements
    - **Layer total**: `MAX(NPU_time, PIM_time)` (NPU and PIM execute in parallel)
  - Sum across all layers in the forward pass
- **Output**: **Total compute cycles** for the entire forward pass (all layers) in MoNDE hybrid mode
- **Input**: Layer routing statistics files (`layer{num}.txt`) to identify top 4 experts per layer

#### **DynaNDE Incremental Optimization**
- **Description**: Dynamic optimization that finds the **optimal expert split** between NPU and PIM across all layers
- **Strategy**: Tests all configurations from 0 to 64 experts on NPU per layer
  - 0 experts on NPU = All on PIM
  - 1-63 experts on NPU = Hybrid (top N on NPU, rest on PIM)
  - 64 experts on NPU = All on NPU
- **Optimization**:
  - For each layer and each configuration (0-64 experts on NPU):
    - NPU time = param_load of top N experts + compute of Nth expert
    - PIM time = compute of remaining experts + activation movements
    - Layer total = `MAX(NPU_time, PIM_time)`
  - Sum layer totals across all layers in the forward pass
  - Selects configuration with **minimum total cycles** for the entire forward pass
- **Output**: Optimal number of experts on NPU per layer and **total compute cycles** for the entire forward pass

---

### 2. Analyze Experts (`analyze_experts.ipynb`)

Utility notebook for processing and analyzing simulation results:

- **File management**: Renames timestamped folders to numbered folders
- **Data extraction**: Parses `SA_stage_E.txt` files from NPU and PIM simulations
- **Statistics generation**: Extracts expert operation cycles (param_load, fc1, gelu, fc2)
- **Activation movement analysis**: Processes PIM activation movement cycles
- **Layer file processing**: Reads routing statistics to identify top experts

---

## Input Requirements

### Directory Structure

```
Encoder-BS{size}-SL{length}/
├── npu/
│   ├── 2/
│   │   └── SA_stage_E.txt
│   ├── 3/
│   └── ...
└── pim/
    ├── 2/
    │   └── SA_stage_E.txt
    ├── 3/
    └── ...

Encoder-BS{size}-SL{length}/
├── layer2.txt    # Routing statistics (top experts)
├── layer3.txt
└── ...
```

### Input Files

1. **`SA_stage_E.txt`** (NPU folder):
   - Expert operations table with columns:
     - Expert Number, param_load, fc1, gelu, fc2, total compute
   - One file per experiment folder

2. **`SA_stage_E.txt`** (PIM folder):
   - Expert operations table (no param_load):
     - Expert Number, fc1, gelu, fc2, total compute
   - Activation movement cycles (movement_1, movement_2)

3. **`layer{num}.txt`** (Routing statistics):
   - Expert token distribution from trace generator
   - Used to identify top 4 experts for MoNDE
   - Used to order experts by activity for DynaNDE

---

## Execution Modes Comparison

| Mode | Experts on NPU | Experts on PIM | Decision Criteria |
|:----:|:--------------:|:--------------:|:-----------------|
| **NPU-Only** | 64 | 0 | All experts on NPU |
| **PIM-Only** | 0 | 64 | All experts on PIM |
| **MoNDE** | 4 (top) | 60 (rest) | Bandwidth difference (static) |
| **DynaNDE** | Optimal (varies) | Remaining | Execution time minimization (dynamic) |

---

## Key Concepts

### Parallel Execution
NPU and PIM execute **simultaneously** in hybrid modes. Total execution time is:
```
Total = MAX(NPU_time, PIM_time)
```
This models realistic parallel execution where both units work concurrently.

### Parameter Load Overhead
- **NPU**: Requires loading expert parameters from HBM to SRAM
- **PIM**: No parameter load (experts stored in memory)
- Parameter load cycles are significant and affect NPU execution time

### Activation Movement
- **PIM-only**: Requires moving activations to/from PIM
- Two movements: one for fc1 input, one for fc2 output
- Included in PIM total execution time

### Optimization Goal
DynaNDE finds the split that **minimizes total execution time** by balancing:
- NPU parameter load overhead vs. compute efficiency
- PIM compute efficiency vs. activation movement overhead
- Load distribution across both units

---

## Usage

1. **Run NeuPIMs-MoE simulator** to generate `SA_stage_E.txt` files for NPU and PIM modes
2. **Generate routing statistics** using `DynaNDE-Trcegenerator/Prefiller.ipynb` to create `layer{num}.txt` files
3. **Run `MOE_simulator.ipynb`**:
   - Configure `base_dir` to point to your experiment folder
   - Execute cells to analyze NPU-only, PIM-only, MoNDE, and DynaNDE modes
4. **Use `analyze_experts.ipynb`** for data preprocessing and detailed analysis

---

## Output

The simulator generates:
- **Summary tables** for each execution mode showing per-layer breakdown
- **Total compute cycles** for the entire forward pass (sum across all layers) in each mode
- **Optimal configurations** (DynaNDE) showing best expert split per layer
- **Performance comparison** across all modes for end-to-end execution

---

## Notes

- **Batch size and sequence length** are encoded in folder names (e.g., `BS32-SL512`)
- **Forward pass simulation**: Processes all layers in one forward pass, providing total compute cycles for end-to-end execution
- **Layer-specific optimization**: DynaNDE finds optimal split per layer (different layers may have different optimal configurations), then sums across all layers
- **Real routing data**: Uses actual expert routing from model execution (not synthetic)
- **Cycle-accurate**: Based on detailed cycle-level simulation from NeuPIMs-MoE

