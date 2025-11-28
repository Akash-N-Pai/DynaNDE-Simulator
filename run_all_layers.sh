#!/bin/bash

# Run simulations for all batches, iterations, and layers
echo "============================================"
echo "Running ALL simulations:"
echo "  3 batches × 10 iterations × 8 layers = 240 runs"
echo "============================================"
echo ""

total=0
completed=0

# Batch_0-16: shard1-0_to_1-0, 16x512
echo ""
echo "################################################"
echo "# BATCH: Batch_0-16"
echo "################################################"
echo ""

for iteration in 1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th
do
    echo ""
    echo "############################################"
    echo "# Batch_0-16 / Iteration: $iteration"
    echo "############################################"
    echo ""
    
    for layer in 2 3 4 5 6 7 8 9
    do
        total=$((total + 1))
        
        echo "====================================="
        echo "Run $total/240: Batch_0-16 / $iteration / Layer $layer"
        echo "====================================="
        
        # Batch_0-16 naming: shard1-0_to_1-0, 16x512
        trace_path="Batch_0-16/${iteration}/Layer_${layer}/flame-moe-290m_runid31066_epoch5473_layer${layer}_shard1-0_to_1-0_firstTokens_16x512.csv"
        
        sed -i "s|\"moe_routing_trace_path\": \".*\"|\"moe_routing_trace_path\": \"${trace_path}\"|" configs/model_configs/gpt3-7B-moe.json
        
        echo "Updated config: $trace_path"
        
        ./brun_fast.sh
        
        if [ $? -eq 0 ]; then
            completed=$((completed + 1))
            echo "✅ Completed: Run $total/240"
        else
            echo "❌ Failed: Run $total/240"
        fi
        
        echo ""
    done
done



echo ""
echo "============================================"
echo "ALL SIMULATIONS COMPLETED!"
echo "============================================"
echo "Total runs: $total"
echo "Completed: $completed"
echo "Failed: $((total - completed))"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($completed/$total)*100}")%"
echo "============================================"
