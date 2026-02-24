# dc0 Progress

## Current Status

**Phase 6 complete**: UCI integration done. `dc0_uci` binary plays via UCI protocol using MCTS+NN.
Phases 1-6 done. Next: multi-game parallelism for higher GPU utilization, then optimizations (Phase 7).

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
| 6 | UCI integration | Done |
| 7 | Optimizations | In progress |

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
- `predict_logits()` method added to skip softmax when returning to MCTS (optimization)
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
- Batched MCTS with virtual loss for GPU batch inference
- `test_mcts.cpp`: 11 tests (7 unbatched + 4 batched), all passing

## Phase 3: Self-Play Pipeline (Done)

- `SelfPlay.hpp`:
  - `TrainingExample`: 1408 floats (board) + 4672 floats (policy) + 1 float (result) = 24324 bytes
  - `TrainingData`: binary serialization with magic header `0xDC000001`, version 1
  - `NNEvaluator`: bridges DC0Network to MCTS eval functions. Uses `predict_logits()` to pass logits directly (avoids softmax->log->softmax round-trip).
  - `play_self_play_game_batched()`: batched MCTS game with per-move quality metrics
  - `run_self_play()`: N sequential games with progress reporting + metric aggregation
- Per-move metrics collected: policy entropy, root Q, NN prior of MCTS-selected move
- `test_selfplay.cpp`: 8 tests, all passing

## Phase 4: Training Loop (Done)

- `Training.hpp`:
  - `TrainingDataset`: libtorch Dataset wrapping binary format. Returns `(input[22,8,8], target[4673])` where target = policy ++ [result].
  - `Trainer`: SGD with momentum 0.9, weight decay 1e-4. Policy cross-entropy + WDL cross-entropy loss. LR schedule with configurable step drops.
  - `evaluate_models()`: head-to-head games between two models, alternating colors
  - `run_generation()`: full generation loop: self-play -> train -> evaluate -> conditionally promote
  - `GenerationConfig`: all hyperparameters in one struct
- `test_training.cpp`: 6 tests, all passing
- Verified: loss drops from ~8.8 to ~4.5 over 5 training passes on random self-play data

## Phase 5: CLI Binary (Done)

- `dc0_main.cpp`: compiled and linked as the `dc0` target
- Three modes: `train`, `selfplay`, `eval`
- Full argument parsing with `--model`, `--games`, `--sims`, `--blocks`, `--filters`, etc.
- Smoke-tested: 3-generation training run with 2-block/32-filter network on RTX 4090

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

## Phase 6: UCI Integration (Done)

- `DC0Engine.hpp`: self-contained MCTS+NN engine wrapper for UCI
  - `init()`, `new_game()`, `set_position()`, `go()` with SearchParams, info callback, stop check
  - Configurable simulations, batch size, c_puct
- `UCI.hpp` modified with `#ifdef DC0_ENABLED` guards:
  - Options: `SearchMode` (alphabeta/dc0), `DC0ModelPath`, `DC0Device`, `DC0Simulations`, `DC0BatchSize`, `DC0Blocks`, `DC0Filters`
  - `perform_go_dc0()`: maps UCI go params to dc0 search, emits info lines, handles time control
  - `ensure_dc0_engine()`: lazy initialization on first use
  - Torch headers included before engine headers to avoid fathom macro clashes (`square`, `diag`, `rank`)
- `dc0_uci` target in CMakeLists: compiles `uci.cpp` with `-DDC0_ENABLED` + libtorch
- Tested: ~3,800 nps with batch_size=64 on RTX 4090, proper UCI info lines and bestmove output

### UCI usage
```bash
# Start UCI engine with dc0 neural network search
CUDA_VISIBLE_DEVICES=0 ./build/dc0_uci

# In the UCI session:
# setoption name SearchMode value dc0
# setoption name DC0ModelPath value /path/to/model.bin
# setoption name DC0Simulations value 800
# position startpos
# go wtime 60000 btime 60000
```

## Self-Play Quality Metrics

Per-game and per-generation metrics are logged during self-play:

| Metric | Description | Untrained | Learning signal |
|--------|-------------|-----------|-----------------|
| **entropy** | Policy entropy: -sum(p*log(p)) of MCTS visit distribution | ~3.0 (uniform over ~20 moves) | Should decrease as model gets confident |
| **\|Q\|** | Average absolute root Q-value | ~0.01 (model uncertain) | Increases as model evaluates positions |
| **prior** | NN prior probability of MCTS-selected move | ~0.05 (1/20 random) | Should increase (NN agrees with search) |
| **avg_len** | Average game length | ~150 (random, many draws) | Stabilizes; decisive games get shorter |
| **policy_loss** | Cross-entropy: MCTS target vs NN output | ~7-8 | Should decrease over generations |
| **value_loss** | WDL cross-entropy: game result vs NN prediction | ~0.7-1.1 | Should decrease over generations |

Example output:
```
Game 1/3: 1/2 in 88 moves (1.6s) | entropy=3.28 |Q|=0.015 prior=0.049 | 2747 nodes/s, 1.6 s/game
Quality: entropy=3.01 |Q|=0.014 prior=0.093 avg_len=149 W/D/L=0/3/0
Training done: policy_loss=7.3180 value_loss=0.7316 total=8.0495 (448 examples)
```

## Optimizations Done

1. **Batched MCTS**: virtual loss for path diversification, ~7,100 nps at bs=128 (21x over unbatched)
2. **Direct logit passing**: `NNEvaluator` uses `predict_logits()` instead of `predict()`, eliminating the softmax->log->softmax round-trip. Logits go directly from network to MCTS `expand()`.
3. **Bulk mask copy**: legal move mask built via `torch::from_blob` on bool array instead of per-element accessor loop.

## Test Suites

Single unified test binary `dc0_tests` with 46 tests across 6 suites:

| Suite | Tests | Coverage |
|-------|-------|----------|
| MoveEncoding | 8 | encode/decode roundtrip, valid count, symmetry |
| BoardEncoding | 7 | plane values, legal mask, promotions, positions |
| Network | 6 | shapes, param count, GPU inference, save/load |
| MCTS | 11 | single + batched search, policy, terminal, noise, tree reuse |
| SelfPlay | 8 | serialization, random game, NN evaluator, batched game, save/load |
| Training | 6 | dataset, training step, loss decrease, checkpoint, LR, train+eval |

### Build commands
```bash
cd zero
CUDACXX=/usr/local/cuda-12.4/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j$(nproc)
```

### Run all tests
```bash
CUDA_VISIBLE_DEVICES=0 ./build/dc0_tests
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
   - `_GLIBCXX_USE_CXX11_ABI=0` globally (libtorch forces old ABI; gtest must match)

5. **RTX 4090 on device 0**, GTX 1080 Ti on device 1 (incompatible with torch 2.5.1+cu124). Always use `CUDA_VISIBLE_DEVICES=0`.

6. **Double-eval bug (fixed)**: Original MCTS `expand_node` discarded the NN value, then the caller called `eval_fn` again. Fixed by having `expand_node` return the value.

7. **NNEvaluator logit conversion (fixed)**: Previously: `predict()` softmax -> probabilities, then `evaluate_batch()` -> `log(p)`, then `expand()` -> softmax again. Now: `predict_logits()` returns masked logits directly for MCTS.

8. **Fathom macro clashes**: `tbchess.c` defines `#define square(r,f)`, `#define diag(s)`, `#define rank(s)` which collide with libtorch's `at::Tensor::square()` etc. Fixed by including `<torch/torch.h>` before engine headers in `UCI.hpp`.

9. **Engine header ODR issues**: Headers like `Zobrist.hpp`, `FEN.hpp`, `Piece.hpp` have non-inline function definitions. Fixed with `LINKER:--allow-multiple-definition`.

## File Inventory

```
zero/
  docs/
    DESIGN.md                # Architecture and phase plan
    progress.md              # This file

  # C++ core
  MoveEncoding.hpp           # move_t <-> policy index (constexpr tables)
  BoardEncoding.hpp          # Board -> (22,8,8) tensor + legal move mask
  Network.hpp                # SE-ResNet (libtorch nn::Module), predict + predict_logits
  MCTS.hpp                   # PUCT tree search (single + batched with virtual loss)
  SelfPlay.hpp               # Self-play games + binary data + NNEvaluator + quality metrics
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

1. **Multi-game parallelism**: Play N games concurrently, batching NN evals across all trees for higher GPU utilization (currently ~10-16% GPU util, single-game bottleneck)
2. **FP16 inference**: Half-precision forward pass for ~2x throughput
3. **Production training run**: Full-scale with 6 blocks, 128 filters, 800 sims, 100+ generations
