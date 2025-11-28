#!/bin/bash

# Run simulations for ALL batches when files are available
# 3 batches × 10 iterations × 8 layers = 240 runs

echo "============================================"
echo "Running ALL batches:"
echo "  3 batches × 10 iterations × 8 layers = 240 runs"
echo "============================================"
echo ""

total=0
completed=0
skipped=0

for batch in Batch_0-16 Batch_0-32 Batch_0-64
do
    # Set the correct naming pattern for each batch
    if [ "$batch" = "Batch_0-16" ]; then
        naming="shard1-0_to_1-0_firstTokens_16x512"
    elif [ "$batch" = "Batch_0-32" ]; then
        naming="shard0-0_to_0-0_firstTokens_32x512"
    else  # Batch_0-64
        naming="shard0-0_to_0-1_firstTokens_64x512"
    fi
    
    echo ""
    echo "################################################"
    echo "# BATCH: $batch (pattern: $naming)"
    echo "################################################"
    echo ""
    
    for iteration in 1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th
    do
        echo ""
        echo "############################################"
        echo "# $batch / Iteration: $iteration"
        echo "############################################"
        echo ""
        
        for layer in 2 3 4 5 6 7 8 9
        do
            total=$((total + 1))
            
            # Build trace file path with correct naming pattern
            trace_file="${batch}/${iteration}/Layer_${layer}/flame-moe-290m_runid31066_epoch5473_layer${layer}_${naming}.csv"
            
            if [ ! -f "$trace_file" ]; then
                echo "⏭️  Run $total/240: Skipping (file not found: $trace_file)"
                skipped=$((skipped + 1))
                continue
            fi
            
            echo "====================================="
            echo "Run $total/240: $batch / $iteration / Layer $layer"
            echo "====================================="
            
            # Update the trace path in config
            sed -i "s|\"moe_routing_trace_path\": \".*\"|\"moe_routing_trace_path\": \"${trace_file}\"|" configs/model_configs/gpt3-7B-moe.json
            
            echo "Updated config: $trace_file"
            
            # Run the simulation
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
done

echo ""
echo "============================================"
echo "ALL SIMULATIONS COMPLETED!"
echo "============================================"
echo "Total runs attempted: $total"
echo "Completed: $completed"
echo "Failed: $((total - completed - skipped))"
echo "Skipped (missing files): $skipped"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($completed/($total-$skipped))*100}")%"
echo "============================================"

