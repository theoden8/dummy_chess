# NNUE Training

## Quick Start

```bash
cd training

# 1. Get data
mkdir -p data && cd data
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d lichess_db_puzzle.csv.zst
cd ..

# 2. Preprocess
uv run python preprocess_puzzles.py -n 1000000  # 1M positions

# 3. Train
uv run python train.py --epochs 30 --batch-size 8192

# 4. Deploy
cp network.nnue ../
```

## Performance

| Hardware | Batch Size | Speed | 1M pos / 30 epochs |
|----------|-----------|-------|-------------------|
| RTX 3090 | 16384 | ~50k/s | ~10 min |
| RTX 4090 | 32768 | ~100k/s | ~5 min |
| CPU 8-core | 2048 | ~2k/s | ~2.5 hours |

## Key Optimizations

- **`nn.EmbeddingBag`** - Sparse lookups in single CUDA kernel
- **`IterableDataset`** - Streams from disk, constant memory
- **Mixed precision** - FP16 on GPU via `torch.amp`
- **OneCycleLR** - Fast convergence with super-convergence
- **Large batches** - 8192+ for better GPU utilization

## Data Options

### Lichess Puzzles (default)
~4M tactical positions. Download:
```bash
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
```

### With Stockfish (better quality)
```bash
uv run python preprocess_puzzles.py -s /path/to/stockfish -n 1000000
```

### Lichess Evals (best quality)
Pre-computed Stockfish evals from https://database.lichess.org/#evals

## Architecture

```
HalfKP[41024] → 256 ─┐
                     ├→ 512 → 32 → 32 → 1
HalfKP[41024] → 256 ─┘
```

- Shared feature transformer weights
- Quantized int8/int16 for C++ inference
- ~21MB weights, ~1M evals/sec

## Files

- `preprocess_puzzles.py` - Convert raw data to training CSV
- `train.py` - PyTorch training with sparse features
- `network.nnue` - Output weights for C++ engine
- `network.pt` - PyTorch checkpoint
