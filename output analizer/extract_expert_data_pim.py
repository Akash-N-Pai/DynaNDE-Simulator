#!/usr/bin/env python3
"""
Extract fc1, fc2, gelu, and activation movements for experts 0-63 from the PIM TSV file.
"""

import csv
from collections import defaultdict
import re

def parse_data(filename):
    """Parse the TSV file and extract expert data."""
    expert_data = defaultdict(lambda: {'fc1': 0, 'fc2': 0, 'gelu': 0, 'activation_movement_1': 0, 'activation_movement_2': 0})
    activation_movements = {'activation_movement_1': 0, 'activation_movement_2': 0}
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            op_name = row['OpName']
            total_cycle = int(row['TotalCycle'])
            
            # Parse the operation name to extract expert number and operation type
            if 'moe_expert' in op_name:
                try:
                    # Extract expert number
                    expert_num = int(op_name.split('moe_expert.')[1].split('.')[0])
                    
                    # Extract operation type
                    if op_name.endswith('.fc1'):
                        expert_data[expert_num]['fc1'] = total_cycle
                    elif op_name.endswith('.fc2'):
                        expert_data[expert_num]['fc2'] = total_cycle
                    elif op_name.endswith('.gelu'):
                        expert_data[expert_num]['gelu'] = total_cycle
                except (ValueError, IndexError):
                    continue
            
            # Parse activation movements (these are not expert-specific)
            elif 'activation_movement_1' in op_name:
                activation_movements['activation_movement_1'] = total_cycle
            elif 'activation_movement_2' in op_name:
                activation_movements['activation_movement_2'] = total_cycle
    
    return expert_data, activation_movements

def parse_expert_order(stats_filename):
    """Parse the stats file to get expert order by token count."""
    expert_order = []
    try:
        with open(stats_filename, 'r') as f:
            for line in f:
                # Look for lines like "Expert 28    73"
                match = re.match(r'Expert (\d+)\s+\d+', line)
                if match:
                    expert_num = int(match.group(1))
                    expert_order.append(expert_num)
    except FileNotFoundError:
        print(f"Warning: Stats file {stats_filename} not found. Using default order.")
        expert_order = list(range(64))
    return expert_order

def write_formatted_data(expert_data, activation_movements, expert_order, output_filename):
    """Write the data to a formatted file."""
    with open(output_filename, 'w') as f:
        f.write("Note: Data is for 8 units of 16x16 systolic array\n")
        f.write("      Max MAC per cycle: 2048 (16x16x8 = 2048)\n")
        f.write("\n")
        f.write(f"{'Expert':<10} {'fc1':<15} {'gelu':<15} {'fc2':<15} {'Total':<15}\n")
        f.write("-" * 70 + "\n")
        
        total_all_experts = 0
        
        # Write data for each expert
        for expert_num in expert_order:
            if expert_num in expert_data:
                data = expert_data[expert_num]
                # Divide fc1 and fc2 by 2 for PIM data
                fc1_display = data['fc1'] // 2
                fc2_display = data['fc2'] // 2
                total = fc1_display + data['gelu'] + fc2_display
                total_all_experts += total
                f.write(f"{expert_num:<10} {fc1_display:<15} {data['gelu']:<15} {fc2_display:<15} {total:<15}\n")
            else:
                # Expert not found, write zeros
                f.write(f"{expert_num:<10} {'0':<15} {'0':<15} {'0':<15} {'0':<15}\n")
        
        # Write activation movements and total execution cycles at the end
        f.write("-" * 70 + "\n")
        f.write(f"{'':<10} {'--':<15} {'--':<15} {'--':<15} {'--':<15}\n")
        f.write(f"{'activation_movement_1:':<25} {activation_movements['activation_movement_1']:<15}\n")
        f.write(f"{'Sum of all experts:':<25} {total_all_experts:<15}\n")
        f.write(f"{'activation_movement_2:':<25} {activation_movements['activation_movement_2']:<15}\n")
        f.write(f"{'TOTAL EXECUTION CYCLES:':<25} {activation_movements['activation_movement_1'] + total_all_experts + activation_movements['activation_movement_2']:<15}\n")

if __name__ == '__main__':
    filename = 'SA_stage_E_pim2.tsv'
    stats_filename = 'flame-moe-290m_runid31066_epoch1080_layer2_shard0-0_512tokens_top2_stats.txt'
    output_filename = 'pim_results.txt'
    expert_data, activation_movements = parse_data(filename)
    expert_order = parse_expert_order(stats_filename)
    write_formatted_data(expert_data, activation_movements, expert_order, output_filename)
    print(f"Results written to {output_filename}")

