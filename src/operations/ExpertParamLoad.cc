#include "ExpertParamLoad.h"

ExpertParamLoad::ExpertParamLoad(std::string name, uint32_t expert_id,
                                 std::vector<Ptr<NPUTensor>> expert_weights,
                                 Ptr<BTensor> data_tensor)
    : Operation(name), _expert_id(expert_id), _data_tensor(data_tensor), _expert_weights(expert_weights) {
    
    // Calculate total parameter size for this expert
    // FC1 + FC2 weights (bias is small, ignored)
    _param_size_bytes = 0;
    for (auto weight : expert_weights) {
        _param_size_bytes += weight->_inners[0]->_size;
    }
    
    _inputs.resize(expert_weights.size() + 1);
    for (size_t i = 0; i < expert_weights.size(); ++i) {
        _inputs[i + 1] = expert_weights[i];
    }
    
    calculate_load_cycles();
}

void ExpertParamLoad::calculate_load_cycles() {
    // Parameter Movement Overhead with REALISTIC modeling
    // Based on actual weight size passed to this operation
    
    uint64_t param_bytes = _param_size_bytes;
    
    // REALISTIC PCIe Gen4 x16 parameters:
    // - Theoretical peak: ~32 GB/s
    // - Sustained bandwidth in practice: 18-22 GB/s due to overheads
    // - Accounts for: protocol overhead (~15-20%), TLP inefficiency, memory controller, cache coherency
    uint32_t icnt_bandwidth_gbps = 22;  // Realistic sustained PCIe Gen4 x16
    uint32_t core_freq_mhz = _config.core_freq;  // 1000 MHz
    
    // Bytes per cycle at core frequency
    double bytes_per_cycle = (double)icnt_bandwidth_gbps * 1e9 / (core_freq_mhz * 1e6);
    
    // Cycles for transfer at core frequency
    uint64_t transfer_cycles = (uint64_t)(param_bytes / bytes_per_cycle);
    
    // REALISTIC OVERHEAD MODELING:
    // 1. Base PCIe transaction overhead
    uint64_t base_latency = _config.expert_load_latency;  // ~1000 cycles default
    
    // 2. Size-dependent overhead (larger transfers need more setup)
    uint64_t size_overhead = 0;
    if (param_bytes < 64 * 1024) {
        // Very small transfers: high relative overhead (cache line operations)
        size_overhead = 500;
    } else if (param_bytes < 1024 * 1024) {
        // Small transfers: moderate overhead
        size_overhead = 300;
    } else if (param_bytes < 8 * 1024 * 1024) {
        // Medium transfers: minimal overhead
        size_overhead = 100;
    }
    // Large transfers: no additional size overhead (already accounted in base_latency)
    
    // 3. Memory controller overhead (HBM access latency)
    // Larger transfers may hit multiple HBM channels, add some latency
    uint64_t hbm_overhead = 0;
    if (param_bytes > 4 * 1024 * 1024) {
        // Large transfers spread across multiple HBM controllers
        hbm_overhead = (param_bytes / (4 * 1024 * 1024)) * 200;  // ~200 cycles per controller
    }
    
    // 4. Real-world jitter/variance (±5-10%)
    // Simulate packet retransmission, cache misses, memory bank conflicts
    uint64_t variance = transfer_cycles * 0.07;  // ±7% average variance
    variance = variance / 2;  // Half the variance (average impact)
    
    _load_cycles = transfer_cycles + base_latency + size_overhead + hbm_overhead + variance;
    
    spdlog::info("Expert {} param load: {} bytes, {} cycles (transfer: {}, overhead: base={}, size={}, hbm={}, variance={})", 
                 _expert_id, param_bytes, _load_cycles, transfer_cycles, 
                 base_latency, size_overhead, hbm_overhead, variance);
}

std::vector<Ptr<BTensor>> ExpertParamLoad::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];  // This is the dependency trigger (normalized_input or prev signal)
    
    // TWO OUTPUTS for true double buffering:
    // Output 0: Data passthrough for FC1 (ALWAYS the data_tensor, not the input)
    // Output 1: Completion signal for chaining to next expert's param_load
    _outputs.resize(2);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_data_output",
        _data_tensor->get_dims(),  // Use stored data_tensor dimensions
        NPUTensorBufType::ACT, false);
    _outputs[1] = std::make_shared<NPUTensor>(
        _name + "_completion_signal",
        std::vector<uint32_t>{1},  // Tiny tensor - just a signal
        NPUTensorBufType::ACT, false);
    
    initialize_tiles();
    
    return _outputs;
}

void ExpertParamLoad::initialize_tiles() {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };
    
    // Model parameter load latency using DUMMY instruction
    // This adds the transfer overhead to the simulation timeline
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::DUMMY,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = _load_cycles,  // Models the parameter transfer latency
        .src_addrs = std::vector<addr_type>{},
    });
    
    _tiles.push_back(tile);
}

Tile ExpertParamLoad::initialize_instructions() {
    return Tile{};  // Created in initialize_tiles
}

