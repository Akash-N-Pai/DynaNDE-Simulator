# Code Structure

## 4. Code Structure

The artifact extends an existing simulator framework (NeuPIM).

We do not modify the core functionality of the upstream simulator except where noted.

Below we document only the components relevant for reproducing the results in this paper.

### Upstream simulator (summary only)

The upstream NeuPIM simulator provides a cycle-accurate simulation framework for hybrid NPU-PIM (Processing-In-Memory) architectures. The simulator models a multi-core system where each core contains:

1. **Systolic Array (SA)**: Matrix multiplication accelerator with configurable width×height dimensions
2. **Vector Processing Unit (VU)**: Element-wise operations (LayerNorm, Softmax, GELU, Add)
3. **Memory hierarchy**: On-chip SRAM (scratchpad), off-chip HBM, and PIM-capable DRAM
4. **Instruction pipeline**: Three-stage pipeline with load (LD), execute (EX), and store (ST) queues

**Key architectural components:**

- **Core execution model** (`Core.cc/h`, `NeuPIMSCore.cc/h`): 
  - Base classes implementing the instruction pipeline
  - Separate queues for SA and PIM sub-batches (`_ld_inst_queue_for_sa`, `_ld_inst_queue_for_pim`, etc.)
  - Cycle-by-cycle simulation of instruction execution, memory access, and compute pipelines
  - SRAM management with double-buffering support

- **Memory system** (`Dram.cc/h`, `Sram.cc/h`, `Interconnect.cc/h`):
  - DRAM controller with channel, rank, bank, row, column addressing
  - SRAM scratchpad with allocation tracking and hit/miss detection
  - Interconnect modeling PCIe bandwidth and latency

- **Operation framework** (`operations/Operation.h`):
  - Base class for all neural network operations
  - Operations generate `Tile` structures containing `Instruction` sequences
  - Tiles are issued to cores and executed via the instruction pipeline

- **Tensor system** (`Tensor.cc/h`, `NPUTensor.cc/h`, `PIMTensor.cc/h`):
  - Multi-dimensional tensor representation with memory address mapping
  - Supports different buffer types (ACT, WGT, BIAS) and memory locations (HBM, SRAM, PIM)
  - Row-based address calculation for efficient memory access patterns

- **Simulation control** (`Simulator.cc/h`, `SimulationConfig.h`):
  - Main simulation loop coordinating cores, DRAM, and interconnect
  - Configuration system for model parameters, hardware specs, and execution modes
  - Statistics collection and logging infrastructure

**Execution model:**
1. Operations generate tiles containing instruction sequences (MOVIN, GEMM, LAYERNORM, MOVOUT, etc.)
2. Tiles are scheduled to cores based on resource availability
3. Instructions flow through LD → EX → ST queues
4. Compute instructions (GEMM, LAYERNORM) execute on SA/VU pipelines
5. Memory instructions (MOVIN, MOVOUT) access DRAM via interconnect
6. Tiles complete when all instructions finish; statistics are collected

The full upstream repository is included for completeness, but only the components listed below were modified or newly added for this work.

### New modules added in this artifact

#### MoE Execution Management

**`src/MoEExecution.cc/h`** - Expert Execution Planner and Optimizer

This module implements the core MoE execution planning logic with multiple optimization strategies. It serves as the analytical engine that calculates execution latencies and plans expert execution order.

**Key functionality:**
- **Expert task planning** (`plan_execution()`):
  - Takes token-to-expert assignment counts and calculates per-expert execution parameters
  - Creates `ExpertTask` structures containing: expert ID, token count, parameter load cycles, compute cycles, and total latency
  - Integrates with `MoEExpertCache` to determine if expert parameters are already cached (avoiding redundant loads)
  - Skips experts with zero token assignments (optimization #1: inactive expert elimination)

- **Latency calculation modes**:
  - **Serial execution**: Sum of all expert latencies (baseline, no parallelism)
  - **Parallel execution** (`calculate_parallel_latency()`): Wall-time = max(expert latencies) - assumes sufficient compute resources to run all experts simultaneously
  - **Double-buffered execution** (`calculate_double_buffered_latency()`): Overlaps parameter load of expert N+1 with computation of expert N, reducing total latency by hiding transfer overhead

- **Design decisions**:
  - Parameter load cycles are calculated externally (in `StageProgram`) based on expert weight size and PCIe bandwidth
  - Compute cycles are proportional to token count: `num_tokens × compute_cycles_per_token`
  - Cache hits are tracked to model expert parameter reuse across batches

**Integration points:**
- Called from `StageProgram::moe_ffn_block()` to plan execution before creating operations
- Uses `MoEExpertCache` for cache hit/miss tracking
- Results are used for logging and validation, but actual execution follows the operation dependency graph

---

**`src/MoERoutingTraceReader.cc/h`** - Expert Routing Trace File Parser

This module enables using real-world routing traces from actual MoE model inference, allowing the simulator to model realistic token-to-expert assignments rather than synthetic distributions.

**Key functionality:**
- **Trace file format** (CSV):
  ```
  layer_id,token_id,expert_0_prob,expert_1_prob,...,expert_N_prob
  0,0,0.45,0.32,0.10,0.08,0.05,...
  0,1,0.12,0.55,0.15,0.10,0.08,...
  ```
  - Each row represents one token's routing probabilities across all experts
  - Probabilities are normalized (sum to 1.0) and represent the gating network output

- **Assignment computation** (`compute_assignments()`):
  - For each token, selects top-k experts based on routing probabilities
  - Maintains two data structures:
    - `_expert_token_counts[layer_id]`: Vector of token counts per expert
    - `_expert_token_assignments[layer_id]`: Vector of token ID lists per expert
  - Handles total token count mismatches between trace file and simulation

- **Fallback mechanism**:
  - If trace file is missing or invalid, falls back to `MoETokenDispatcher` for simulated distribution
  - Provides logging to indicate which source is being used

**Design rationale:**
- Enables validation against real model behavior
- Supports different total token counts by truncating/padding trace data
- Caches computed assignments per layer to avoid recomputation

---

**`src/MoETokenDispatcher.cc/h`** - Synthetic Token-to-Expert Assignment Generator

This module generates realistic token-to-expert assignment patterns for simulation when trace files are unavailable. It models the load imbalance characteristic of real MoE systems.

**Key functionality:**
- **Uniform distribution** (`generate_uniform_distribution()`):
  - Round-robin assignment of tokens to experts
  - Each expert receives approximately `total_tokens × experts_per_token / num_experts` tokens
  - Used for baseline comparisons and balanced load scenarios

- **Skewed distribution** (`generate_skewed_distribution()`):
  - Models Pareto principle: top 5% of experts handle configurable percentage (default 80%) of tokens
  - Uses Zipf-like probability distribution: `P(expert_i) = 1/(i+1)` for top experts, scaled down for others
  - Configurable skew factor (0.0-1.0) controls imbalance severity
  - More realistic for production MoE systems where routing learns to favor certain experts

- **Statistics**:
  - Calculates load imbalance ratio: `max_tokens_per_expert / avg_tokens_per_expert`
  - Provides distribution histograms for validation

**Design rationale:**
- Enables controlled experiments with different load imbalance scenarios
- Reproducible via fixed random seed (42)
- Supports both uniform (ideal) and skewed (realistic) scenarios

---

**`src/MoEStats.cc/h`** - MoE Performance Statistics Collector

This module provides centralized statistics tracking for MoE-specific performance metrics, separate from the general simulator statistics.

**Key functionality:**
- **Statistics structure**:
  - `MoELayerStat`: Per-layer statistics including router cycles, expert utilization, combine cycles
  - `ExpertUtilization`: Per-expert metrics (tokens processed, compute cycles, load percentage)
  - Tracks load balance variance across experts

- **Recording functions**:
  - `record_router_completion()`: Logs router MatMul + Softmax cycles
  - `record_expert_completion()`: Logs per-expert compute cycles and token counts
  - `record_combine_completion()`: Logs output combination/gather cycles

- **Output**:
  - `print_stats()`: Console output with formatted statistics
  - `log_stats()`: TSV file output for post-processing and analysis

**Integration points:**
- Called from operation completion handlers (currently not fully integrated in baseline)
- Provides data for MoE performance analysis and bottleneck identification

#### MoE Operation Modules

**`src/operations/MoERouter.cc/h`** - MoE Gating Network Operation

This operation implements the MoE routing/gating network that determines which experts process which tokens.

**Architecture:**
- **Input**: `[total_tokens]` - normalized input tokens
- **Weights**: `[num_experts]` - router weight matrix
- **Output 1**: `[total_tokens, experts_per_token]` - routing weights (normalized probabilities for selected experts)
- **Output 2**: `[total_tokens, experts_per_token]` - expert indices (which experts were selected)

**Computation flow:**
1. **Router**: `input × router_weight → [total_tokens, num_experts]` logits
2. **Softmax**: Normalize logits to probabilities
3. **Top-k selection**: For each token, select `experts_per_token` experts with highest probabilities
4. **Weight normalization**: Renormalize selected expert probabilities to sum to 1.0


**Design rationale:**
- Separates routing logic from expert execution for modularity
- Enables future optimizations (e.g., caching routing decisions, approximate routing)

---

**`src/operations/MoEExpert.cc/h`** - Single Expert FFN Operation

This operation represents a single expert's feed-forward network computation.

**Architecture:**
- **Input**: `[num_tokens_assigned, d_model]` - tokens routed to this expert (token slicing)
- **Weights**: 
  - FC1 weight: `[d_model, d_ff_expert]`, FC1 bias: `[d_ff_expert]`
  - FC2 weight: `[d_ff_expert, d_model]`, FC2 bias: `[d_model]`
- **Output**: `[num_tokens_assigned, d_model]` - expert output

**Computation pipeline:**
1. **FC1**: `input × FC1_weight + FC1_bias → [num_tokens, d_ff_expert]`
2. **GELU**: Element-wise GELU activation
3. **FC2**: `gelu_out × FC2_weight + FC2_bias → [num_tokens, d_model]`


**Design rationale:**
- Encapsulates expert-specific logic (weight loading, statistics, caching)
- Enables per-expert optimization strategies
- Supports expert-specific configurations (e.g., different FFN dimensions per expert)

---

**`src/operations/MoECombine.cc/h`** - Expert Output Combination Operation

This operation combines outputs from multiple experts using routing weights to produce the final MoE output.

**Architecture:**
- **Inputs**:
  - Routing weights: `[total_tokens, experts_per_token]` - normalized weights
  - Expert indices: `[total_tokens, experts_per_token]` - which experts processed each token
  - Expert outputs: `num_experts` tensors, each `[num_tokens_assigned_to_expert, d_model]`
- **Output**: `[total_tokens, d_model]` - weighted combination

**Computation:**
For each token `t`:
```
output[t] = Σ(weight[t][i] × expert_output[expert_idx[t][i]][token_position_in_expert])
```
where the sum is over `i ∈ [0, experts_per_token)`.

**Implementation notes:**
- Currently creates a skip tile (combination is simplified in baseline)
- In a full implementation, this would:
  1. Scatter expert outputs back to original token positions
  2. Apply weighted combination for tokens processed by multiple experts
  3. Handle padding/alignment for variable expert token counts

**Design rationale:**
- Separates combination logic for future optimizations (e.g., sparse combination, quantization)
- Enables analysis of combination overhead separately from expert computation

---

**`src/operations/ActivationMovement.cc/h`** - Activation Transfer Overhead Modeling

This operation models the data movement overhead when executing MoE in PIM mode, where activations must be transferred between HBM and PIM memory.

**Architecture:**
- **Input**: `[total_tokens, d_model]` - activation tensor
- **Output**: `[total_tokens, d_model]` - same tensor (passthrough with timing overhead)

**Two movement phases:**
1. **Movement #1** (before expert processing):
   - Transfers all token activations from NPU to PIM memory
   - Size: `total_tokens × d_model × precision` bytes
   - Occurs after routing, before expert execution

2. **Movement #2** (after expert processing):
   - Transfers expert outputs back from PIM memory to NPU
   - Size: `total_tokens × d_model × precision` bytes
   - Occurs after all experts complete, before final combination

**Latency calculation** (`calculate_movement_cycles()`):
- **Transfer cycles**: `activation_size_bytes / bytes_per_cycle`
  - `bytes_per_cycle = (PCIe_bandwidth_GB/s × 1e9) / (core_freq_MHz × 1e6)`
- **Overhead components**:
  - Base latency: 1000 cycles (PCIe protocol setup)
  - Size-dependent overhead: Higher for small transfers (cache line operations)
  - HBM overhead: Additional latency for large transfers spanning multiple controllers
  - Write-back overhead: Extra cycles for movement #2 (write buffer flushing)
  - Variance: ±8% random variance to model real-world jitter

**Implementation:**
- Uses `Opcode::DUMMY` instructions with `size = _movement_cycles` to model transfer latency
- Creates dependency chain: movement #1 → experts → movement #2
- Passthrough design: output tensor has same dimensions as input, but execution includes transfer delay

**Design rationale:**
- Accurately models PIM mode overhead (critical for performance comparison)
- Separates transfer overhead from compute for analysis
- Realistic modeling of PCIe Gen4 x16 bandwidth (~22 GB/s sustained)

---

**`src/operations/ExpertParamLoad.cc/h`** - Expert Parameter Loading Operation

This operation models the overhead of loading expert parameters from HBM to NPU SRAM when experts are stored off-chip.

**Architecture:**
- **Input**: Dependency trigger (normalized input or previous expert's completion signal)
- **Weights**: Expert FC1 and FC2 weight tensors (optional, for NPU mode)
- **Output 1**: Data passthrough for FC1 computation
- **Output 2**: Completion signal for chaining to next expert

**Two execution modes:**

1. **NPU mode** (`use_parameter_load = true`):
   - **Purpose**: Load expert weights from HBM to NPU SRAM
   - **Weight size**: `2 × d_model × d_ff_expert × precision` bytes (FC1 + FC2)
   - **Transfer calculation**:
     - Transfer cycles: `weight_bytes / bytes_per_cycle`
     - Base latency: `Config::expert_load_latency` (default 1000 cycles)
     - Size-dependent overhead: Higher for small transfers
     - HBM overhead: Additional cycles for large transfers
     - Variance: ±5% random variance per expert (deterministic via expert_id seed)
   - **Dependency chain**: Expert N's param_load waits for Expert N-1's FC2 output (sequential execution)

2. **PIM mode** (`use_parameter_load = false`):
   - **Purpose**: Barrier operation to enforce sequential expert execution
   - **Weight size**: 0 (no actual parameter load)
   - **Dependency chain**: Expert N's barrier waits for Expert N-1's completion signal
   - **Design**: Reuses `ExpertParamLoad` infrastructure but with empty weights to create dependency without transfer overhead

**Latency calculation** (`calculate_load_cycles()`):
- **Realistic PCIe modeling**: Accounts for protocol overhead (~15-20%), TLP inefficiency, memory controller latency
- **Sustained bandwidth**: ~18-22 GB/s (not theoretical 32 GB/s peak)
- **Special case**: Returns 0 cycles if `_param_size_bytes == 0` (barrier-only mode)

**Implementation:**
- Uses `Opcode::DUMMY` instructions with `size = _load_cycles` to model transfer latency
- Two outputs enable double-buffering: data output for FC1, completion signal for next expert
- Sequential dependency: Each expert's param_load depends on previous expert's completion

**Design rationale:**
- Models critical overhead in off-chip expert scenarios
- Enables sequential execution modeling (experts execute one at a time)
- Supports both NPU (actual load) and PIM (barrier) modes with same interface
- Realistic PCIe bandwidth modeling for accurate performance prediction

### Modified modules (changes introduced for this paper)

**`src/NeuPIMSystolicWS.cc`** - Core Execution Statistics Enhancement

**Changes made:**
1. **Enhanced `update_stats()` method** (lines 299-379):
   - Added vector unit compute cycle tracking for MoE operations
   - Tracks cycles when `_vector_pipelines` contain instructions (expert FC1, FC2, GELU)
   - Increments `_stat.back().num_calculations` with `vector_core_width` per cycle for active vector pipelines
   - Maintains separate counters for different operation types: `_stat_layernorm_cycle`, `_stat_softmax_cycle`, `_stat_add_cycle`, `_stat_gelu_cycle`

2. **Vector unit cycle tracking**:
   - Iterates through all `_vector_pipelines` (multiple vector units) to track parallel execution
   - Updates parent tile statistics (`parent_tile->stat.compute_cycles++`) for accurate per-operation timing
   - Distinguishes between idle cycles (no compute), memory stall cycles (waiting for data), and compute cycles

**Purpose:**
- Enables accurate performance analysis of MoE FFN operations that execute on vector units
- Tracks expert-specific compute utilization separately from systolic array operations
- Provides data for identifying MoE performance bottlenecks (compute vs. memory vs. parameter loading)

**Integration:**
- No changes to core execution logic; only statistics collection
- Compatible with existing dense FFN operations (no regression)

---

**`src/StageProgram.cc`** - MoE FFN Block Implementation

**Changes made:**
1. **New method: `moe_ffn_block()`** (lines 317-732):
   - Complete implementation of MoE FFN stage execution pipeline
   - Replaces `ffn1_block()` + `ffn2_block()` when `Config::global_config.moe_enabled == true`

2. **Routing integration** (lines 333-395):
   - **Trace file support**: Uses `MoERoutingTraceReader` if trace file provided, otherwise `MoETokenDispatcher`
   - **Optimization**: Skips router MatMul and Softmax if trace file contains pre-computed routing

3. **Execution mode selection** (lines 467-475):
   - **NPU mode** (`ffn_execution_mode == "npu"`): Experts in HBM, loaded to SRAM sequentially
   - **PIM mode** (`ffn_execution_mode == "pim"`): Experts in PIM memory, activations moved to/from PIM
   - Creates appropriate `ActivationMovement` and `ExpertParamLoad` operations based on mode

4. **Expert execution loop** (lines 504-655):
   - Iterates through all experts, skipping inactive ones (0 tokens)
   - For each active expert:
     - Creates `ExpertParamLoad` operation (NPU mode) or barrier (PIM mode)
     - Creates FC1 MatMul with `set_row_count_override(num_tokens)` for token slicing
     - Creates GELU operation
     - Creates FC2 MatMul with `set_row_count_override(num_tokens)`
     - Enforces sequential execution via dependency chains

5. **Sequential dependency enforcement**:
   - **NPU mode**: Expert N's `ExpertParamLoad` depends on Expert N-1's FC2 output
   - **PIM mode**: Expert N's barrier depends on Expert N-1's completion signal
   - Ensures experts execute one at a time (no parallelism in baseline)

6. **Activation movement** (lines 477-491, 657-667):
   - **Movement #1**: Before expert processing (PIM mode only)
   - **Movement #2**: After all experts complete (PIM mode only)
   - Models transfer overhead for PIM execution mode

7. **Execution planning** (lines 397-445):
   - Uses `MoEExecution::plan_execution()` to calculate parameter load cycles
   - Computes transfer cycles based on expert weight size, PCIe bandwidth, and overhead
   - Logs execution plan for validation and debugging

8. **Output gathering** (lines 669-692):
   - Creates final gathered output tensor `[total_tokens, d_model]`
   - In full implementation, would scatter expert outputs back to original token positions
   - Currently simplified (all expert outputs accumulated)

**Key design decisions:**
- **Token slicing**: Each expert processes only `num_tokens` assigned tokens, not full batch
- **Sequential execution**: Experts execute one at a time (baseline; parallel execution is analytical only)
- **Dependency chains**: Uses operation graph dependencies to enforce execution order
- **Mode abstraction**: Same code path handles both NPU and PIM modes via conditional logic

**Integration points:**
- Called from `init_SA_program()` when `moe_enabled == true` and stage requires FFN (stages C, D, E, F)
- Integrates with `Model` to fetch expert weight tensors
- Uses existing `MatMul`, `Gelu`, `Add` operations for expert computation
- Creates new `ExpertParamLoad` and `ActivationMovement` operations

**Performance optimizations:**
- Skips inactive experts (0 tokens) entirely
- Token slicing reduces memory and compute requirements per expert
- Trace file support avoids expensive router computation
- Execution planning provides analytical latency estimates


---

**`src/operations/MatMul.cc`** - Token Slicing Support

**Changes made:**
1. **New member variables**:
   - `_use_row_override`: Boolean flag indicating row count override is active
   - `_row_count_override`: Override value for M dimension (number of rows to process)

2. **New method: `set_row_count_override()`**:
   - Sets `_use_row_override = true` and stores override value
   - Called from `StageProgram::moe_ffn_block()` for expert FC1 and FC2 MatMuls

3. **Modified `get_outputs()`** (lines 69-75):
   - Checks `_use_row_override` flag
   - If set, uses `_row_count_override` for output dimension calculation instead of input dimension
   - Example: Input `[2048, d_model]`, override `256` → Output `[256, d_ff]` (only processes 256 tokens)

4. **Modified `calculate_loops()`** (lines 474-482):
   - Uses `effective_m = _row_count_override` if `_use_row_override == true`
   - Logs override usage for debugging
   - Affects tile size calculation: smaller tiles for sliced experts

5. **Enhanced `initialize_instructions()`** (lines 312-331, 389-397):
   - **Tile dimension clamping**: Clamps `tile_m` and `tile_n` to respect row override
   - Handles both transposed and non-transposed cases
   - Prevents processing beyond assigned token count
   - Example: If override is 256 and current tile processes rows 250-260, clamps to 256

6. **Loop ordering optimization** (lines 93-137):
   - **K-optimized order**: When K dimension dominates (K > 2×max(M,N)), uses M→K→N loop order
   - **Standard order**: Otherwise uses M→N→K loop order
   - Improves data reuse for large K dimensions (common in expert FC layers)

**Token slicing mechanism:**
- **Problem**: Each expert should only process tokens assigned to it, not all tokens
- **Solution**: Override M dimension (row count) to `num_tokens_assigned_to_expert`
- **Effect**: 
  - Reduces memory requirements: Expert processes `num_tokens × d_model` instead of `total_tokens × d_model`
  - Reduces compute: Only assigned tokens participate in GEMM
  - Maintains correctness: Output dimensions match assigned token count

**Example:**
- Total tokens: 2048 tokens
- Expert 0 assigned: 256 tokens
- Expert 1 assigned: 512 tokens
- Without slicing: Both experts process 2048 tokens (wasteful)
- With slicing: Expert 0 processes 256, Expert 1 processes 512 (efficient)

**Design rationale:**
- **Memory efficiency**: Key optimization for MoE memory scaling
- **Compute efficiency**: Only processes necessary tokens
- **Backward compatible**: No impact on dense FFN (override not set)
- **Transpose handling**: Correctly handles both transposed and non-transposed MatMul cases

**Integration:**
- Called from `StageProgram::moe_ffn_block()` for each expert's FC1 and FC2 MatMuls
- Override value = `expert_token_counts[expert_id]` (number of tokens assigned to expert)
- Works seamlessly with existing tile generation and instruction creation logic

---

## Execution Flow and Data Flow

### MoE FFN Stage Execution Flow

The following describes the complete execution flow when MoE is enabled:

1. **Stage Initialization** (`StageProgram::init_SA_program()`):
   - Determines if MoE FFN is needed (stages C, D, E, F)
   - Calls `moe_ffn_block()` instead of `ffn1_block()` + `ffn2_block()`

2. **MoE Block Setup** (`StageProgram::moe_ffn_block()`):
   - **Input normalization**: LayerNorm operation on input tokens
   - **Routing decision**: 
     - If trace file exists: Load routing from `MoERoutingTraceReader`
     - Otherwise: Generate routing via `MoETokenDispatcher` (uniform or skewed)
   - **Execution planning**: `MoEExecution::plan_execution()` calculates latencies

3. **Router Computation** (optional, if no trace file):
   - Router MatMul: `[total_tokens, d_model] × [d_model, num_experts] → [total_tokens, num_experts]`
   - Router Softmax: Normalize to probabilities
   - Top-k selection: Select `experts_per_token` experts per token

4. **Activation Movement #1** (PIM mode only):
   - `ActivationMovement` operation transfers all tokens from HBM to PIM memory
   - Models PCIe transfer overhead

5. **Expert Execution Loop** (sequential, one expert at a time):
   For each active expert (with `num_tokens > 0`):
   - **Parameter Loading** (NPU mode):
     - `ExpertParamLoad` operation loads FC1 + FC2 weights from HBM to SRAM
     - Depends on previous expert's FC2 output (sequential dependency)
   - **FC1 Computation**:
     - MatMul with `row_count_override = num_tokens`
     - Only processes assigned tokens (token slicing)
   - **GELU Activation**:
     - Element-wise GELU on FC1 output
   - **FC2 Computation**:
     - MatMul with `row_count_override = num_tokens`
     - Output: `[num_tokens, d_model]`

6. **Activation Movement #2** (PIM mode only):
   - `ActivationMovement` operation transfers expert outputs back to HBM
   - Models write-back overhead

7. **Output Gathering**:
   - In full implementation: Scatters expert outputs back to original token positions
   - Applies weighted combination for tokens processed by multiple experts

8. **Residual Connection**:
   - Adds residual buffer to MoE output
   - Standard transformer residual connection

### Data Flow Diagram

```
Router 
  ↓
Top-k Selection → expert_token_counts, expert_token_assignments
  ↓
[PIM Mode Only] ActivationMovement #1: NPU → PIM
  ↓
For each expert (sequential):
  ├─ [NPU Mode] ExpertParamLoad: HBM → SRAM (weights)
  ├─ FC1 MatMul [num_tokens, d_model] × [d_model, d_ff] → [num_tokens, d_ff]
  ├─ GELU [num_tokens, d_ff]
  └─ FC2 MatMul [num_tokens, d_ff] × [d_ff, d_model] → [num_tokens, d_model]
  ↓
[PIM Mode Only] ActivationMovement #2: PIM → NPU
  ↓
Gather/Combine 
  ↓
Output 
```

### Instruction Pipeline Integration

MoE operations integrate with the existing instruction pipeline as follows:

1. **Operation → Tile Generation**:
   - Each operation (MatMul, GELU, ExpertParamLoad, etc.) generates `Tile` structures
   - Tiles contain sequences of `Instruction` objects (MOVIN, GEMM, LAYERNORM, MOVOUT, DUMMY)

2. **Tile Scheduling**:
   - Tiles are scheduled to cores by the scheduler
   - Dependency graph ensures correct execution order (e.g., Expert N waits for Expert N-1)

3. **Instruction Execution**:
   - **LD Queue**: MOVIN instructions load data from DRAM to SRAM
   - **EX Queue**: GEMM, LAYERNORM, GELU instructions execute on SA/VU
   - **ST Queue**: MOVOUT instructions store results from SRAM to DRAM
   - **DUMMY Instructions**: Model transfer overhead (parameter load, activation movement)

4. **Cycle-by-Cycle Simulation**:
   - Each cycle, cores process instructions from queues
   - Memory requests are sent to DRAM via interconnect
   - Compute pipelines (SA, VU) execute instructions when operands are ready
   - Statistics are collected per cycle

### Key Optimizations Implemented

1. **Token Slicing**: Each expert processes only assigned tokens via `row_count_override`
2. **Inactive Expert Skipping**: Experts with 0 tokens are skipped entirely
3. **Trace File Support**: Pre-computed routing avoids expensive router computation
4. **Sequential Execution**: Dependency chains enforce one-expert-at-a-time execution
5. **Realistic Overhead Modeling**: PCIe bandwidth, protocol overhead, HBM latency
6. **Execution Mode Abstraction**: Same code path handles NPU and PIM modes

### Configuration Parameters

MoE behavior is controlled via `SimulationConfig`:

- `moe_enabled`: Enable/disable MoE FFN
- `num_experts`: Number of experts in MoE layer
- `experts_per_token`: Top-k experts per token
- `moe_routing_trace_path`: Path to routing trace file (optional)
- `ffn_execution_mode`: "npu" or "pim"
- `expert_load_latency`: Base latency for parameter loading
- `expert_cache_size`: Number of experts that fit in cache
- `moe_enable_parallelism`: Enable parallel execution (analytical only)
- `moe_enable_double_buffering`: Enable double buffering (analytical only)

