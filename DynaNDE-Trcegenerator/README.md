# DynaNDE Trace Generator

Tools for generating expert routing traces from MoE model executions for use with the NeuPIMs-MoE simulator.

---

## Decoder Trace Generator (`decoder.ipynb`)

Generates routing traces for **decoder/autoregressive** inference (sequential token generation).

### Input

**Required directory structure:**
```
samples/{model}/{runid}/1-{shard_idx}.pt          # Token samples
actives/{model}/{runid}/{epoch}/{layer}/1-{shard_idx}.pt  # Routing data (scores, indices)
```

**Configuration:**
- `model`, `runid`, `epoch`: Model identification
- `sequence_length`: Sequence length (e.g., 512)
- `num_tokens_to_extract`: Number of token positions to extract (e.g., 10 for positions 0-9)
- `layers_to_process`: Layer range (e.g., [2, 3, 4, 5, 6, 7, 8, 9])
- `start_shard_idx`, `num_shards`: Shard range to process

### Output

**Directory structure:**
```
Batch_{batch_id}/
├── 1st/Layer_{layer}/     # First token position
├── 2nd/Layer_{layer}/     # Second token position
└── ...
```

**Files per layer:**
- `{model}_runid{runid}_epoch{epoch}_layer{layer}_..._firstTokens_{num_seqs}x{seq_len}.csv` - Routing scores (simulator input)
- `{filename}_top{k}_stats.txt` and `{filename}_top{k}_stats.csv` - Statistics for top-1, top-2, top-6 experts

**CSV format:**
```csv
layer_id,token_id,expert_0,expert_1,...,expert_63
0,0,0.000000,0.123456,...,0.000000
```

---

## Prefiller Trace Generator (`Prefiller.ipynb`)

Generates routing traces for **encoder/prefill** phase (all tokens processed in parallel).

### Input

**Required directory structure:**
```
samples/{model}/{runid}/0-{shard_idx}.pt          # Token samples (encoder uses 0- prefix)
actives/{model}/{runid}/{epoch}/{layer}/0-{shard_idx}.pt  # Routing data
```

**Configuration:**
- Same as decoder, but typically shorter `sequence_length` (e.g., 128)
- Processes **all tokens** from all sequences (no token position filtering)

### Output

**Directory structure:**
```
Encoder_Batch_{batch_id}/
├── Layer_2/
├── Layer_3/
└── ...
```

**Files per layer:**
- `{model}_runid{runid}_epoch{epoch}_layer{layer}_..._encoder_{num_seqs}seqs_{num_tokens}tokens.csv` - Routing scores for all tokens
- `{filename}_top{k}_stats.txt` and `{filename}_top{k}_stats.csv` - Statistics

**CSV format:** Same as decoder

---

## Usage

1. **Configure parameters** in the notebook (model, runid, epoch, shard range, etc.)
2. **Run all cells** in Jupyter Notebook
3. **Output CSV files** can be used in simulator by setting `moe_routing_trace_path` in model config

---

## Requirements

- Python 3.x, PyTorch, pandas, numpy, Jupyter Notebook

---

## Notes

- **Decoder**: Extracts specific token positions (1st, 2nd, 3rd, etc.) - organized by position then layer
- **Prefiller**: Extracts all tokens from all sequences - organized by layer only
- **Shard naming**: Decoder uses `1-{idx}.pt`, Prefiller uses `0-{idx}.pt`
- **CSV files** contain routing scores (0.0-1.0) for all 64 experts per token
