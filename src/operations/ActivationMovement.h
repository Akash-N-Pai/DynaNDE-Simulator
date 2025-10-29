#pragma once
#include "../tensor/NPUTensor.h"
#include "Operation.h"
#include <random>
#include <functional>

/**
 * ActivationMovement: Models activation transfer in MoE stage
 * 
 * First movement (before expert processing):
 * - After router, move all token activations from HBM to SRAM
 * - Size: batch_size * E * 2 bytes (all tokens * embedding_dim * FP16)
 * 
 * Second movement (after all experts complete):
 * - Move expert outputs back to HBM after gather phase
 * - Size: batch_size * E * 2 bytes
 */
class ActivationMovement : public Operation {
   public:
    ActivationMovement(std::string name, uint32_t num_tokens, uint32_t E);
    
    std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

   private:
    uint32_t _num_tokens;
    uint32_t _E;
    uint64_t _activation_size_bytes;  // Total activation size to move
    uint32_t _movement_cycles;       // Cycles needed for transfer
    
    void calculate_movement_cycles();
    void initialize_tiles();
    Tile initialize_instructions();
};

