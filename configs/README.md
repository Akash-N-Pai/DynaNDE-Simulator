# Configuration Files

Configuration files for the **DynaNDE simulator** (MoE extension of NeuPIMs).

---

## Model Configuration (Main)

**Location:** `model_configs/`

The model configuration is the **primary configuration file** for MoE models. This is where you specify MoE-specific parameters.

### Key MoE Parameters

| Parameter | Type | Description |
|:---------:|:---:|:------------|
| `moe_enabled` | boolean | Enable MoE mode |
| `num_experts` | int | Total number of experts |
| `experts_per_token` | int | Top-k experts per token (routing) |
| `moe_ffn_scaling` | string | Expert FFN scaling: `"balanced"`, `"compute"`, or `"capacity"` |
| `moe_offchip_experts` | boolean | Experts stored in HBM (off-chip) |
| `expert_load_latency` | int | Cycles to load one expert from HBM to SRAM |
| `expert_cache_size` | int | Number of experts that fit in on-chip cache |
| `moe_enable_parallelism` | boolean | Enable parallel expert execution |
| `moe_enable_double_buffering` | boolean | Overlap parameter load and compute |
| `moe_routing_trace_path` | string | Path to routing trace CSV file (optional) |
| `ffn_execution_mode` | string | FFN execution: `"npu"` or `"pim"` |

### Standard Model Parameters

| Parameter | Type | Description |
|:---------:|:---:|:------------|
| `model_name` | string | Model name (for logging) |
| `model_params_b` | int | Model parameters in billions |
| `model_vocab_size` | int | Vocabulary size |
| `model_n_layer` | int | Number of layers |
| `model_n_head` | int | Number of attention heads |
| `model_n_embd` | int | Embedding dimension |
| `model_n_ffn` | int | FFN dimension (defaults to 4 × n_embd if not set) |
| `n_tp` | int | Tensor parallelism degree |
| `n_pp` | int | Pipeline parallelism degree |

### Example Files

- `Flame-moe-npu.json`: Flame-moe model with NPU execution
- `Flame-moe-pim.json`: Flame-moe model with PIM execution

---

## Core Configuration

**File:** `systolic_ws_dev.json`

Hardware configuration for NPU cores (systolic arrays), SRAM, and interconnect. Same as original NeuPIMs simulator.

---

## Memory Configuration

**Location:** `memory_configs/`

DRAM/PIM memory system configuration. **Same as original NeuPIMs simulator.**

Key parameters:
- `dram_type`: Memory type (`dram`, `newton`, `neupims`)
- `dram_channels`: Number of DRAM channels
- `dram_req_size`: DRAM access granularity
- `pim_comp_coverage`: Number of multipliers per bank

---

## System Configuration

**Location:** `system_configs/`

System-level execution parameters. **Same as original NeuPIMs simulator.**

Key parameters:
- `run_mode`: `"npu"` or `"npu+pim"`
- `sub_batch_mode`: Sub-batch interleaving (boolean)
- `kernel_fusion`: Kernel fusion enabled (boolean)
- `max_batch_size`: Maximum batch size
- `max_active_reqs`: Maximum active requests
- `max_seq_len`: Maximum sequence length

---

## Notes

- **Model config** is the main file to modify for MoE experiments
- Memory and system configs are inherited from NeuPIMs (no MoE-specific changes)
- Routing trace files (CSV) can be generated using `DynaNDE-Trcegenerator/`
