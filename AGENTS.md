# Agent Guidelines for dummy_chess

This document provides guidelines for AI coding agents working on the dummy_chess codebase.

## Project Overview

dummy_chess is a UCI-compliant chess engine written in C++20 with Python and Rust bindings. It features NNUE (efficiently updatable neural network) evaluation with HalfKP and HalfKAv2 architectures.

## Build Commands

### C++ Engine
```bash
# Configure and build (Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build with Profile-Guided Optimization
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPGO_GENERATE=ON
cmake --build build && ./build/dummy_chess_bench
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPGO_USE=ON
cmake --build build
```

### Python Package
```bash
# Install/reinstall the main package
uv sync --reinstall-package dummy-chess

# Build the _preprocess C++ extension (torch tensor output for training)
cd training && uv run python setup.py build_ext --inplace

# Build with CUDA support (_preprocess + _extract_cuda extensions)
cd training && CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH uv run python setup.py build_ext --inplace

# Install/reinstall the training package (includes Cython extensions)
cd training && uv sync --reinstall-package dummy-chess-training
```

### Rust
```bash
cd rust && cargo build --release
```

## Test Commands

### Python Tests
```bash
# Run all tests in training/
cd training && uv run pytest

# Run a single test file
cd training && uv run pytest tests/test_features.py

# Run a single test function
cd training && uv run pytest tests/test_features.py::TestHalfKPFeatures::test_features_match_reference

# Run tests matching a pattern
cd training && uv run pytest -k "halfkp"

# Run with verbose output
cd training && uv run pytest -v

# Skip benchmark tests
cd training && uv run pytest -m "not benchmark"
```

### C++ Tests (UCI Protocol)
```bash
# Run all UCI expect tests
./test/run_tests.sh ./build/dummy_chess_uci

# Run with verbose output
./test/run_tests.sh -v ./build/dummy_chess_uci

# Stop on first failure
./test/run_tests.sh --exitfirst ./build/dummy_chess_uci
```

### Rust Tests
```bash
cd rust && cargo test --release --lib
```

### Perft Validation
```bash
# Compare perft results against Stockfish
uv run python scripts/perft_test.py
```

## Code Style Guidelines

### Python

**Imports:**
- NO `from X import Y` style imports - use fully qualified names
- NO import aliases EXCEPT these allowed shorthand: `np` (numpy), `pd` (pandas), `pl` (polars), `mpl` (matplotlib), `plt` (matplotlib.pyplot), `sp` (scipy), `sns` (seaborn), `ipd` (IPython.display), `tqdm.auto as tqdm`
- Correct: `import numpy` then use `numpy.array()`, or `import numpy as np` then use `np.array()`
- **FORBIDDEN**: Local/inline imports inside functions, methods, or classes. ALL imports MUST be at module top level. No exceptions.
- Organize: standard library, third-party, local (separated by blank lines)

**Naming:**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

**Types:**
- Use modern Python 3.11+ type hints: `list[str]`, `tuple[int, int]`, `dict[str, Any]`
- Use `typing` module for complex types

**Error Handling:**
- Use exceptions with descriptive messages
- NO bare `try/except` - always specify exception types

**Example:**
```python
import dataclasses
import pathlib

import numpy as np
import torch

BATCH_SIZE = 8192

@dataclasses.dataclass
class Config:
    learning_rate: float = 1e-3
    epochs: int = 30

def process_batch(data: np.ndarray) -> torch.Tensor:
    """Process a batch of training data."""
    return torch.from_numpy(data)
```

### C++

**Style:**
- C++20 standard
- Header-only design (most code in `.hpp` files)
- Use angle brackets for local includes: `#include <Board.hpp>`

**Naming:**
- Classes: `PascalCase`
- Member variables: `snake_case_` (trailing underscore)
- Functions/methods: `snake_case`
- Constants/enums: `SCREAMING_SNAKE_CASE`
- Macros: `SCREAMING_SNAKE_CASE`

**Error Handling:**
- NO try/catch - use error checking with return values or assertions
- Use `assert()` liberally for invariants

**The `self` Macro:**
- `#define self (*this)` is used locally in files that need it
- Each file that uses `self` must have its own `#define`/`#undef` pair
- This allows external code (like Cython) to include headers without conflicts

### Rust

**Formatting:**
- 2-space indentation (configured in `rustfmt.toml`)
- Standard Rust naming conventions

**FFI Safety:**
- Explicit `unsafe` blocks with safety documentation
- RAII wrappers for C++ objects (implement `Drop` trait)

## Memory Constraints

**FORBIDDEN:** Never bulk-load an entire parquet file into memory. Data files can be 8GB+ (350M+ rows). Always stream row-group-by-row-group, keeping at most one row group (~100K rows) in memory at a time. This applies to both host and GPU memory allocation during initialization.

**Training:** Run with memory limit to prevent OOM:
```bash
ulimit -v 8388608  # 8GB limit
uv run python train.py ...
```

**Important:** Do NOT use `ulimit` when running CUDA training - GPU memory allocation needs unrestricted virtual memory.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `*.hpp` | C++ header-only implementation |
| `python/` | Python bindings (pybind11) |
| `rust/` | Rust bindings and FFI |
| `training/` | NNUE training code (Python + Cython) |
| `training/csrc/` | C++ and CUDA extensions for training |
| `test/` | UCI protocol tests (Expect scripts) |
| `scripts/` | Utility scripts |
| `m42/` | Bitboard library (submodule) |
| `fathom/` | Syzygy tablebase probing (submodule) |

## Common Tasks

### Adding a New Python Test
1. Create test file in `training/tests/test_*.py`
2. Use pytest fixtures and parametrization
3. Run: `cd training && uv run pytest tests/test_newfile.py -v`

### Modifying C++ Headers
1. Edit the `.hpp` file
2. Rebuild: `uv sync --reinstall-package dummy-chess`
3. Test: `uv run python -c "import dummy_chess; ..."`

### Updating Cython Extensions
1. Edit `.pyx` files in `training/`
2. Rebuild: `cd training && uv sync --reinstall-package dummy-chess-training`
3. Test: `cd training && uv run pytest tests/test_features.py`

## Gotchas

### Parquet FEN column is `binary_view`, NOT `large_binary`
The `fen` column in preprocessed parquet files uses Arrow's `binary_view` type, which has a fragmented multi-buffer layout (dozens of small buffers). The C++ feature extraction functions (`_preprocess`, `dummy_chess`) expect contiguous `large_binary` layout with `bufs[1]` = int64 offsets and `bufs[2]` = data. **You must cast before accessing raw buffers:**
```python
fen_col = table.column("fen")
if fen_col.type != pyarrow.large_binary():
    fen_col = fen_col.cast(pyarrow.large_binary())
fen_col = fen_col.combine_chunks() if fen_col.num_chunks > 1 else fen_col.chunk(0)
bufs = fen_col.buffers()
data_ptr, off_ptr = bufs[2].address, bufs[1].address
```
Accessing `bufs[1]`/`bufs[2]` on a `binary_view` column will **segfault** (signal 139) because the buffer layout is completely different.

## Training

### CPU extraction (default)
```bash
cd training && uv run python train.py --batch-size 8192 --epochs 30 data/preprocessed_puzzles_shuffled.parquet
```

### GPU extraction (CUDA kernels, requires _extract_cuda extension)
```bash
cd training && CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --gpu-extract --batch-size 65536 --epochs 30 \
    data/preprocessed_puzzles_shuffled.parquet
```

GPU extraction loads all compressed FEN bytes and scores onto the GPU at startup, then extracts features on-device using CUDA kernels (`_extract_cuda`). This eliminates the CPU extraction bottleneck entirely.

### Performance (RTX 4090, HalfKP, GPU extraction)

| Batch Size | Batches/s | ms/batch | Samples/s | GPU-Util |
|-----------|-----------|----------|-----------|----------|
| 8192 | 219 | 4.6 ms | 1.8M | ~28% |
| 16384 | 215 | 4.6 ms | 3.5M | ~55% |
| 65536 | 125 | 8.0 ms | 8.2M | **97%** |

### CLI flags
- `--gpu-extract`: Use CUDA kernels for feature extraction (requires `_extract_cuda`)
- `--grad-acc-steps N`: Gradient accumulation. Effective batch = batch_size * N
- `--arch {halfkp,halfkav2}`: NNUE architecture
- `--flip-augment`: Augment data with flipped positions (2x data)
- `--prefetch N`: Batches to prefetch for CPU extraction (default 4)

### Training extensions

| Extension | Source | Purpose |
|-----------|--------|---------|
| `_preprocess` | `csrc/preprocess.cpp` | C++ multi-threaded feature extraction (CPU) |
| `_extract_cuda` | `csrc/extract_cuda.cu` | CUDA feature extraction kernels (GPU) |

The `_extract_cuda` extension is only built when `CUDA_HOME` is set. The `_preprocess` extension uses a spawn-join thread pool with `std::atomic` work-stealing (CHUNK_SIZE=1024) and falls back to single-threaded for small batches.

### Compressed FEN format
Byte 0 = flags (bit 0 = side to move: 0=white, 1=black; bit 2 = crazyhouse). Remaining bytes are packed nibbles (high nibble first). Nibbles 0-5 = white KQRBNP, 6-11 = black kqrbnp, 0xC = empty run (next nibble = count-1), 0xF = padding. Board is rank-8-first. Square mapping: `sq = (7 - board_idx/8)*8 + board_idx%8`.

## Architecture Notes

### NNUE Network
- HalfKP: `41024 -> 256x2 -> 32 -> 32 -> 1`
- HalfKAv2: `7692 -> 256x2 -> 32 -> 32 -> 1`
- Feature transformer uses sparse EmbeddingBag
- Quantization scales: FT=127, weights=64

### Key Constants (must match between C++ and Python)
- `HALFKP_SIZE = 41025`
- `HALFKAV2_SIZE = 7693`
- `FT_OUT = 256`
- `L1_OUT = L2_OUT = 32`
