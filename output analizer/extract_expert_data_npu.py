#!/usr/bin/env python3
"""
Extract fc1, fc2, gelu, and param_load operations for experts 0-63 from the TSV file.
"""

import csv
from collections import defaultdict
import re

def parse_data(filename):
    """Parse the TSV file and extract expert data."""
    expert_data = defaultdict(lambda: {'fc1': 0, 'fc2': 0, 'gelu': 0, 'param_load': 0})
    
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
                    elif op_name.endswith('.param_load'):
                        expert_data[expert_num]['param_load'] = total_cycle
                except (ValueError, IndexError):
                    continue
    
    return expert_data

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

def write_formatted_data(expert_data, expert_order, output_filename):
    """Write the data to a formatted file."""
    with open(output_filename, 'w') as f:
        f.write("Note: Data is for 8 units of 32x32 systolic array\n")
        f.write("      Max MAC per cycle: 8192 (32x32x8 = 8192)\n")
        f.write("\n")
        f.write(f"{'Expert':<10} {'param_load':<15} {'fc1':<15} {'gelu':<15} {'fc2':<15} {'Total':<15} {'32units':<15} {'Total+load':<15}\n")
        f.write("-" * 120 + "\n")
        
        total_all_experts = 0
        
        for expert_num in expert_order:
            if expert_num in expert_data:
                data = expert_data[expert_num]
                # Total excludes param_load, only fc1 + gelu + fc2
                total = data['fc1'] + data['gelu'] + data['fc2']
                # For 32 units (4x scaling from 8 units), divide by 4
                total_32units = total // 4
                # Total for 32 units + param_load
                total_32units_with_load = total_32units + data['param_load']
                total_all_experts += total_32units_with_load
                f.write(f"{expert_num:<10} {data['param_load']:<15} {data['fc1']:<15} {data['gelu']:<15} {data['fc2']:<15} {total:<15} {total_32units:<15} {total_32units_with_load:<15}\n")
            else:
                # Expert not found, print zeros
                f.write(f"{expert_num:<10} {'0':<15} {'0':<15} {'0':<15} {'0':<15} {'0':<15} {'0':<15} {'0':<15}\n")
        
        # Write total execution cycles at the end
        f.write("-" * 120 + "\n")
        f.write(f"{'TOTAL EXECUTION CYCLES:':<25} {total_all_experts:<15}\n")

if __name__ == '__main__':
    filename = 'SA_stage_E_npu2.tsv'
    stats_filename = 'flame-moe-290m_runid31066_epoch1080_layer2_shard0-0_512tokens_top2_stats.txt'
    output_filename = 'npu_results.txt'
    expert_data = parse_data(filename)
    expert_order = parse_expert_order(stats_filename)
    write_formatted_data(expert_data, expert_order, output_filename)
    print(f"Results written to {output_filename}")

