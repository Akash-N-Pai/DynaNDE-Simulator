========================================
Simulation Automation Scripts
========================================

TWO SCRIPTS AVAILABLE:

1. run_all_layers.sh (USE THIS NOW - 80 runs)
   - Runs ONLY Batch_0-32 (files exist)
   - 10 iterations × 8 layers = 80 runs
   - Usage: ./run_all_layers.sh

2. run_all_batches.sh (FUTURE - 240 runs)
   - Runs ALL batches when files available
   - 3 batches × 10 iterations × 8 layers = 240 runs
   - Usage: ./run_all_batches.sh

========================================
Naming Conventions Per Batch:
========================================

Batch_0-16:
  shard1-0_to_1-0, 16x512
  Example: ...layer2_shard1-0_to_1-0_firstTokens_16x512.csv

Batch_0-32:
  shard0-0_to_0-0, 32x512
  Example: ...layer2_shard0-0_to_0-0_firstTokens_32x512.csv

Batch_0-64:
  shard0-0_to_0-1, 64x512
  Example: ...layer2_shard0-0_to_0-1_firstTokens_64x512.csv

========================================
Current Status:
========================================
✅ Batch_0-32: 80/80 files found
❌ Batch_0-16: 0/80 files (not ready)
❌ Batch_0-64: 0/80 files (not ready)

========================================
Quick Start:
========================================
# Run the 80 simulations for Batch_0-32:
./run_all_layers.sh

# When you have all batches, run all 240:
./run_all_batches.sh
========================================
