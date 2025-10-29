#!/usr/bin/env python3
"""
Create hybrid NPU/PIM results based on H value from stats file.
"""

import re

def read_stats_file(stats_filename):
    """Read H value and expert order from stats file."""
    h_value = None
    expert_order = []
    
    with open(stats_filename, 'r') as f:
        for line in f:
            if 'H (Gating Output):' in line:
                h_value = int(line.split(':')[1].strip())
            # Look for lines like "Expert 28    73"
            match = re.match(r'Expert (\d+)\s+\d+', line)
            if match:
                expert_num = int(match.group(1))
                expert_order.append(expert_num)
    
    return h_value, expert_order

def read_results(filename):
    """Read NPU or PIM results and extract expert data."""
    expert_data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Find the expert data section
        for line in lines:
            if line.strip() and not line.startswith('Note') and not line.startswith('Expert') and not line.startswith('-') and not line.startswith('TOTAL') and not line.startswith('activation'):
                parts = line.split()
                if parts and parts[0].isdigit():
                    expert_num = int(parts[0])
                    # NPU file has: Expert, param_load, fc1, gelu, fc2, Total, 32units, Total+load
                    # PIM file has: Expert, fc1, gelu, fc2, Total
                    if len(parts) >= 5:
                        expert_data[expert_num] = parts
    
    return expert_data

def calculate_execution_cycles(npu_data, pim_data, expert_order, h_value, activation_movement_1, activation_movement_2):
    """Calculate execution cycles for a given H value."""
    # Assign experts: top H go to NPU, rest go to PIM (parallel)
    npu_experts = expert_order[:h_value]
    pim_experts = expert_order[h_value:]
    
    # Calculate total cycles
    npu_total = 0
    pim_total = 0
    
    # NPU experts use Total+load (column 8 from NPU file, index 7)
    for expert in npu_experts:
        if expert in npu_data:
            npu_total += int(npu_data[expert][7])
    
    # PIM experts use Total (column 5 from PIM file, index 4)
    for expert in pim_experts:
        if expert in pim_data:
            pim_total += int(pim_data[expert][4])
    
    # PIM total includes activation movements
    pim_total_with_activation = activation_movement_1 + pim_total + activation_movement_2
    
    # Parallel execution: max(NPU_time, PIM_time_with_activation)
    parallel_time = max(npu_total, pim_total_with_activation)
    total_execution_cycles = parallel_time
    
    return npu_experts, pim_experts, npu_total, pim_total, pim_total_with_activation, parallel_time, total_execution_cycles

def main():
    stats_filename = 'flame-moe-290m_runid31066_epoch1080_layer2_shard0-0_512tokens_top2_stats.txt'
    npu_results_filename = 'npu_results.txt'
    pim_results_filename = 'pim_results.txt'
    output_filename = 'hybrid_results.txt'
    
    # Read stats to get expert order
    _, expert_order = read_stats_file(stats_filename)
    
    # Read NPU and PIM results
    npu_data = read_results(npu_results_filename)
    pim_data = read_results(pim_results_filename)
    
    # Read activation movements from PIM file
    activation_movement_1 = 0
    activation_movement_2 = 0
    
    with open(pim_results_filename, 'r') as f:
        for line in f:
            if 'activation_movement_1:' in line:
                activation_movement_1 = int(line.split()[-1])
            elif 'activation_movement_2:' in line:
                activation_movement_2 = int(line.split()[-1])
    
    # Try all H values from 0 to 64
    all_results = []
    
    for h_value in range(0, len(expert_order) + 1):
        npu_experts, pim_experts, npu_total, pim_total, pim_total_with_activation, parallel_time, total_execution_cycles = calculate_execution_cycles(
            npu_data, pim_data, expert_order, h_value, activation_movement_1, activation_movement_2
        )
        
        all_results.append({
            'h': h_value,
            'npu_experts': npu_experts,
            'pim_experts': pim_experts,
            'npu_total': npu_total,
            'pim_total': pim_total,
            'pim_total_with_activation': pim_total_with_activation,
            'parallel_time': parallel_time,
            'total_execution_cycles': total_execution_cycles
        })
    
    # Find optimal (minimum cycles)
    optimal = min(all_results, key=lambda x: x['total_execution_cycles'])
    
    # Write results to file
    with open(output_filename, 'w') as f:
        f.write(f"Hybrid NPU/PIM Execution Analysis - Optimal Configuration Search\n")
        f.write(f"=================================================================\n\n")
        
        # Write all configurations summary
        f.write(f"All Configurations Summary:\n")
        f.write(f"{'H':<6} {'NPU Experts':<15} {'NPU Total':<15} {'PIM Total':<15} {'Parallel Time':<15} {'Total Cycles':<15}\n")
        f.write("-" * 82 + "\n")
        
        for res in all_results:
            f.write(f"{res['h']:<6} {len(res['npu_experts']):<15} {res['npu_total']:<15} {res['pim_total_with_activation']:<15} {res['parallel_time']:<15} {res['total_execution_cycles']:<15}\n")
        
        f.write(f"\n")
        f.write(f"{'='*100}\n")
        f.write(f"DETAILED RESULTS FOR EACH H VALUE:\n")
        f.write(f"{'='*100}\n\n")
        
        # Write detailed results for each H value
        for res in all_results:
            f.write(f"=================================================================\n")
            f.write(f"H (Gating Output): {res['h']}\n")
            f.write(f"=================================================================\n\n")
            
            f.write(f"NPU Experts ({len(res['npu_experts'])}): {res['npu_experts']}\n")
            f.write(f"PIM Experts ({len(res['pim_experts'])}): {res['pim_experts']}\n")
            f.write(f"\n")
            
            f.write(f"NPU Execution Details:\n")
            f.write(f"  Experts: {res['npu_experts']}\n")
            for expert in res['npu_experts']:
                if expert in npu_data:
                    f.write(f"    Expert {expert}: Total+load = {npu_data[expert][7]} cycles\n")
            f.write(f"  NPU Total Cycles: {res['npu_total']}\n")
            f.write(f"\n")
            
            f.write(f"PIM Execution Details:\n")
            f.write(f"  Experts: {res['pim_experts']}\n")
            f.write(f"  PIM Sum of Experts: {res['pim_total']}\n")
            f.write(f"\n")
            
            f.write(f"Activation Movements:\n")
            f.write(f"  activation_movement_1: {activation_movement_1}\n")
            f.write(f"  activation_movement_2: {activation_movement_2}\n")
            f.write(f"  PIM Total Cycles (with activations): {res['pim_total_with_activation']}\n")
            f.write(f"\n")
            
            f.write(f"Parallel Execution:\n")
            f.write(f"  max(NPU_time, PIM_time_with_activation) = max({res['npu_total']}, {res['pim_total_with_activation']}) = {res['parallel_time']}\n")
            f.write(f"\n")
            
            f.write(f"TOTAL EXECUTION CYCLES: {res['total_execution_cycles']}\n")
            f.write(f"\n\n")
        
        f.write(f"{'='*100}\n")
        f.write(f"OPTIMAL CONFIGURATION HIGHLIGHT:\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Optimal H value: {optimal['h']}\n")
        f.write(f"Minimum total execution cycles: {optimal['total_execution_cycles']}\n")
        f.write(f"See details for H={optimal['h']} above in the detailed results section.\n")
    
    print(f"Results written to {output_filename}")
    print(f"Optimal H: {optimal['h']}")
    print(f"TOTAL EXECUTION CYCLES: {optimal['total_execution_cycles']}")

if __name__ == '__main__':
    main()

