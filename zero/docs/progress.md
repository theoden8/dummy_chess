# dc0 Progress

## Current Status

**Phase 5 complete**: CLI binary `dc0` builds and runs. Full training loop verified end-to-end on GPU.
Phases 1-5 done. Next: UCI integration (Phase 6), then optimizations (Phase 7).

## Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Move encoding | Done |
| 1b | Neural network | Done |
| 1c | Board encoding | Done |
| 2 | MCTS | Done |
| 3 | Self-play pipeline | Done |
| 4 | Training loop | Done |
| 5 | CLI binary | Done |
| 6 | UCI integration | Not started |
| 7 | Optimizations | Not started |

## Phase 1: Neural Network + Move Encoding (Done)

### 1a: Move Encoding
- `MoveEncoding.hpp`: C++ constexpr lookup tables, encode/decode between engine `move_t` and AlphaZero policy index (0-4671)
- `move_encoding.py`: Python reference implementation
- 1924 valid moves out of 4672 slots. All roundtrip correctly.
- `test_move_encoding.cpp`: all passing

### 1b: Neural Network
- `Network.hpp`: SE-ResNet in libtorch. SEBlock, ResBlock, DC0NetworkImpl with policy (4672) + WDL (3) heads.
- `model.py`: Python reference matching C++ architecture
- Default config: 6 blocks, 128 filters = 2,146,988 parameters
- GPU inference verified on RTX 4090. Save/load weights working.
- `test_network.cpp`: all passing (CPU + GPU)

### 1c: Board Encoding
- `BoardEncoding.hpp`: 22-plane encoding (pieces, repetition, color, fullmove, castling, en passant, halfmove clock)
- Legal move mask generation via `encode_legal_moves()`
- `test_board_encoding.cpp`: all passing

### Build System
- `CMakeLists.txt`: libtorch integration with CUDA 12.4, engine headers, all workarounds
- `pyproject.toml`: Python env with torch==2.5.1+cu124, numpy, jupyter, matplotlib

## Phase 2: MCTS (Done)

- `MCTS.hpp`: PUCT search with configurable `c_puct` (default 2.5)
- `MCTSNode` with `vector<MCTSEdge>`, lazy child expansion via `unique_ptr<MCTSNode>`
- `MCTSTree` manages root board (via `unique_ptr<Board>` — Board has deleted copy assignment due to const members)
- Dirichlet noise at root for training exploration
- Temperature-based move selection (temp=0 for greedy, temp=1 for proportional)
- Tree reuse via `advance(move, new_board)` — subtree preservation
- `expand_node()` returns value to avoid double NN evaluation per leaf
- `test_mcts.cpp`: 7 tests, all passing

## Phase 3: Self-Play Pipeline (Done)

- `SelfPlay.hpp`:
  - `TrainingExample`: 1408 floats (board) + 4672 floats (policy) + 1 float (result) = 24324 bytes
  - `TrainingData`: binary serialization with magic header `0xDC000001`, version 1
  - `NNEvaluator`: bridges DC0Network to `EvalFunction` for MCTS. Handles board encoding, legal move masking, probability-to-logit conversion.
  - `play_self_play_game()`: single game with MCTS, collects examples, assigns results post-game
  - `run_self_play()`: N sequential games with progress reporting, writes binary output
- `test_selfplay.cpp`: 7 tests (serialization, random eval game, NN evaluator, NN game, save/load), all passing

## Phase 4: Training Loop (Done)

- `Training.hpp`:
  - `TrainingDataset`: libtorch Dataset wrapping binary format. Returns `(input[22,8,8], target[4673])` where target = policy ++ [result].
  - `Trainer`: SGD with momentum 0.9, weight decay 1e-4. Policy cross-entropy + WDL cross-entropy loss. LR schedule with configurable step drops.
  - `evaluate_models()`: head-to-head games between two models, alternating colors
  - `run_generation()`: full generation loop: self-play -> train -> evaluate -> conditionally promote
  - `GenerationConfig`: all hyperparameters in one struct
- `test_training.cpp`: 6 tests (dataset shape, training step, loss decreases, checkpoint roundtrip, LR schedule, trained-model-as-evaluator), all passing
- Verified: loss drops from ~8.8 to ~4.5 over 5 training passes on random self-play data

## Phase 5: CLI Binary (Done)

- `dc0_main.cpp`: compiled and linked as the `dc0` target
- Three modes: `train`, `selfplay`, `eval`
- Full argument parsing with `--model`, `--games`, `--sims`, `--blocks`, `--filters`, etc.
- Smoke-tested: 3-generation training run with 2-block/32-filter network on RTX 4090
  - Self-play generates games and writes binary training data
  - Training reduces policy loss (8.0 -> 2.98 within one generation)
  - Evaluation pits new vs old model; promotes on win_rate >= 55%
  - Generation 2 model promoted with 75% win rate

### Usage
```bash
# Full training loop (small test run)
CUDA_VISIBLE_DEVICES=0 ./build/dc0 train \
  --blocks 2 --filters 32 \
  --games 5 --sims 50 --generations 3 \
  --eval-games 4 --eval-sims 20 \
  --batch-size 16 --epochs 3 \
  --output dc0_output

# Production-scale training
CUDA_VISIBLE_DEVICES=0 ./build/dc0 train \
  --blocks 6 --filters 128 \
  --games 100 --sims 800 --generations 100 \
  --batch-size 256 --lr 0.02 \
  --output dc0_output

# Self-play data generation only
CUDA_VISIBLE_DEVICES=0 ./build/dc0 selfplay \
  --model dc0_output/best_model.bin \
  --games 100 --sims 800

# Evaluate model quality
CUDA_VISIBLE_DEVICES=0 ./build/dc0 eval \
  --model dc0_output/best_model.bin \
  --games 20 --sims 200
```

## Test Suites

All 6 test binaries build and pass:

| Binary | Requires | Tests |
|--------|----------|-------|
| `dc0_test_move_encoding` | — | encode/decode roundtrip, valid count |
| `dc0_test_board_encoding` | Board, m42 | plane values, legal mask, promotions |
| `dc0_test_network` | libtorch, CUDA | shapes, param count, GPU inference, save/load |
| `dc0_test_mcts` | Board, m42 | search, deterministic select, policy output, terminal, noise, tree reuse, game play |
| `dc0_test_selfplay` | libtorch, Board | serialization, random game, NN evaluator, NN game, data roundtrip |
| `dc0_test_training` | libtorch, Board | dataset, training step, loss decrease, checkpoint, LR, train+eval cycle |

### Build commands
```bash
cd zero
CUDACXX=/usr/local/cuda-12.4/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j$(nproc)
```

### Run all tests
```bash
./build/dc0_test_move_encoding
./build/dc0_test_board_encoding
CUDA_VISIBLE_DEVICES=0 ./build/dc0_test_network
./build/dc0_test_mcts
CUDA_VISIBLE_DEVICES=0 ./build/dc0_test_selfplay
CUDA_VISIBLE_DEVICES=0 ./build/dc0_test_training
```

## Discoveries & Gotchas

1. **Board copy assignment deleted**: `Board` has `const bool chess960` and `const bool crazyhouse` members. Must use `unique_ptr<Board>` or copy construction, never assignment.

2. **Move encoding format**: Engine `move_t` = `uint16_t` = `(from_sq << 8) | to_byte`. `to_byte` bits 5-0 = destination square, bits 7-6 = promotion type (0=knight, 64=bishop, 128=rook, 192=queen).

3. **Board square layout**: A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63. `file = sq % 8`, `rank = sq / 8`.

4. **libtorch workarounds**:
   - `CUDACXX=/usr/local/cuda-12.4/bin/nvcc` required for cmake
   - `CMAKE_CUDA_STANDARD 17` (cmake 3.25 doesn't know CUDA20)
   - Dummy `CUDA::nvToolsExt` target (missing in newer CUDA)
   - Installed `cuda-nvrtc-dev-12-4` and `libnvjitlink-dev-12-4` for linking

5. **RTX 4090 on device 0**, GTX 1080 Ti on device 1 (incompatible with torch 2.5.1+cu124). Always use `CUDA_VISIBLE_DEVICES=0`.

6. **Double-eval bug (fixed)**: Original MCTS `expand_node` discarded the NN value, then the caller called `eval_fn` again. Fixed by having `expand_node` return the value.

7. **NNEvaluator logit conversion**: Network `predict()` returns probabilities (post-softmax), but MCTS `expand()` does its own softmax. `NNEvaluator` converts back to log-space: `log(p)` for p > 1e-8, else -30.

## File Inventory

```
zero/
  docs/
    DESIGN.md                # Architecture and phase plan
    progress.md              # This file

  # C++ core
  MoveEncoding.hpp           # move_t <-> policy index (constexpr tables)
  BoardEncoding.hpp          # Board -> (22,8,8) tensor + legal move mask
  Network.hpp                # SE-ResNet (libtorch nn::Module)
  MCTS.hpp                   # PUCT tree search (single + batched)
  SelfPlay.hpp               # Self-play games + binary data format + NNEvaluator
  Training.hpp               # Training loop, loss, eval, generation loop
  DC0Engine.hpp              # UCI-facing MCTS engine wrapper
  Logging.hpp                # Log levels, timestamps, configurable verbosity
  dc0_main.cpp               # CLI binary: train / selfplay / eval
  CMakeLists.txt             # Build system: dc0, dc0_uci, dc0_tests targets

  # Tests (in tests/)
  tests/main.cpp             # gtest global environment
  tests/test_move_encoding.cpp
  tests/test_board_encoding.cpp
  tests/test_network.cpp
  tests/test_mcts.cpp
  tests/test_selfplay.cpp
  tests/test_training.cpp

  # Python reference
  model.py                   # Reference SE-ResNet (PyTorch)
  move_encoding.py           # Reference move encoding
  pyproject.toml             # Python env config
```

## Next Steps

1. **Multi-game parallelism**: Play N games concurrently, batching NN evals across all trees for higher GPU utilization
2. **Optimizations (Phase 7)**: FP16 inference, transposition table
3. **Production training run**: Full-scale with 6 blocks, 128 filters, 800 sims, 100+ generations
