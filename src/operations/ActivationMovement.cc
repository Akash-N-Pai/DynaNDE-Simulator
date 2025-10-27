#include "ActivationMovement.h"

ActivationMovement::ActivationMovement(std::string name, uint32_t num_tokens, uint32_t E)
    : Operation(name), _num_tokens(num_tokens), _E(E) {
    
    // Calculate total activation size to move
    // All tokens × embedding_dim × precision
    _activation_size_bytes = (uint64_t)_num_tokens * _E * _config.precision;
    
    _inputs.resize(1);
    
    calculate_movement_cycles();
}

void ActivationMovement::calculate_movement_cycles() {
    // Activation Movement Overhead with REALISTIC modeling
    // Transfer latency calculation using PCIe bandwidth
    
    // REALISTIC PCIe Gen4 x16 parameters (same as parameter load):
    // - Sustained bandwidth: ~22 GB/s (not theoretical 32 GB/s)
    // - Accounts for protocol overhead, memory controller, etc.
    uint32_t icnt_bandwidth_gbps = 22;  // Realistic PCIe Gen4 x16 (~22 GB/s sustained)
    uint32_t core_freq_mhz = _config.core_freq;  // 1000 MHz
    
    // Bytes per cycle at core frequency
    double bytes_per_cycle = (double)icnt_bandwidth_gbps * 1e9 / (core_freq_mhz * 1e6);
    
    // Cycles for transfer at core frequency
    uint64_t transfer_cycles = (uint64_t)(_activation_size_bytes / bytes_per_cycle);
    
    // REALISTIC OVERHEAD MODELING:
    // 1. Base PCIe transaction overhead
    uint64_t base_latency = 1000;  // Protocol setup overhead
    
    // 2. Size-dependent overhead (similar to parameter load)
    uint64_t size_overhead = 0;
    if (_activation_size_bytes < 64 * 1024) {
        // Very small transfers: high relative overhead
        size_overhead = 500;
    } else if (_activation_size_bytes < 1024 * 1024) {
        // Small transfers: moderate overhead
        size_overhead = 300;
    } else if (_activation_size_bytes < 8 * 1024 * 1024) {
        // Medium transfers: minimal overhead
        size_overhead = 100;
    }
    
    // 3. Memory controller overhead (for large transfers)
    uint64_t hbm_overhead = 0;
    if (_activation_size_bytes > 4 * 1024 * 1024) {
        // Large transfers across multiple HBM controllers
        hbm_overhead = (_activation_size_bytes / (4 * 1024 * 1024)) * 200;
    }
    
    // 4. Activation-specific overhead: more random than parameters
    // Activation movement is less predictable than parameter load
    // (cache misses, address translation, etc.)
    uint64_t variance = transfer_cycles * 0.10;  // ±10% average variance (higher than param load)
    variance = variance / 2;  // Average impact
    
    // 5. Direction-specific overhead
    // Writing back to memory (movement #2) can have additional overhead
    // due to write buffer flushing, cache coherency
    uint64_t write_overhead = 0;
    if (_name.find("activation_movement_2") != std::string::npos) {
        // This is the second movement (SRAM → HBM)
        write_overhead = 200;  // Write-back overhead
    }
    
    _movement_cycles = transfer_cycles + base_latency + size_overhead + hbm_overhead + variance + write_overhead;
    
    spdlog::info("Activation movement: {} tokens × {} E × 2 bytes = {} bytes, {} cycles (transfer: {}, overhead: base={}, size={}, hbm={}, variance={}, write={})",
                 _num_tokens, _E, _activation_size_bytes, _movement_cycles, transfer_cycles, 
                 base_latency, size_overhead, hbm_overhead, variance, write_overhead);
}

std::vector<Ptr<BTensor>> ActivationMovement::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);
    
    assert(inputs.size() == 1);
    _inputs[0] = inputs[0];
    
    // Output: passthrough - same shape as input
    auto input_dim = inputs[0]->get_dims();
    _outputs.resize(1);
    _outputs[0] = std::make_shared<NPUTensor>(
        _name + "_output",
        input_dim,
        NPUTensorBufType::ACT, false);
    
    initialize_tiles();
    
    return _outputs;
}

void ActivationMovement::initialize_tiles() {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = 0,
        .accum = false,
    };
    
    // Model activation movement latency using DUMMY instruction
    // This adds the transfer overhead to the simulation timeline
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::DUMMY,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = _movement_cycles,  // Models the activation transfer latency
        .src_addrs = std::vector<addr_type>{},
    });
    
    _tiles.push_back(tile);
}

Tile ActivationMovement::initialize_instructions() {
    return Tile{};  // Created in initialize_tiles
}

