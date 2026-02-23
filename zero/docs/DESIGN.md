# dc0: AlphaZero-style Engine for dummy_chess

## Overview

dc0 is an AlphaZero/LC0-style chess engine built on top of the dummy_chess
board representation. It replaces the alpha-beta search with Monte Carlo Tree
Search (MCTS) guided by a neural network that provides both a **policy** (move
probabilities) and a **value** (position evaluation).

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Training Loop   |     |    Self-Play      |     |   UCI Engine      |
|   (C++ libtorch)  |     |    (C++)          |     |   (C++)           |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   Neural Network  |     |      MCTS         |     |      MCTS         |
|   (C++ libtorch)  |     |      (C++)        |     |      (C++)        |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   SGD optimizer   |     |   libtorch        |     |   libtorch        |
|   (C++ libtorch)  |     |   inference       |     |   inference       |
+-------------------+     +-------------------+     +-------------------+
                                   |                         |
                                   v                         v
                          +-------------------+     +-------------------+
                          |   Board.hpp       |     |   Board.hpp       |
                          |   (bitboard)      |     |   (bitboard)      |
                          +-------------------+     +-------------------+
```

## Components

### 1. Neural Network Architecture

**Input encoding** (22 planes of 8x8 = 22x64 = 1408 floats):
- 12 piece planes: 6 piece types x 2 colors (binary: 1 if piece present)
- 2 repetition planes: has position occurred 1x, 2x (binary)
- 1 color plane: all 1s if white to move, all 0s if black
- 1 move count plane: fullmove number / 100 (scalar broadcast)
- 4 castling planes: KQkq rights (binary, all 1s or all 0s per plane)
- 1 en passant plane: 1 on the en passant target square, 0 elsewhere
- 1 halfmove clock plane: halfmove clock / 100 (scalar broadcast)

This is simpler than AlphaZero's 119-plane encoding (which includes 8 steps of
history). We start without history planes and can add them later.

**Network body**: SE-ResNet (Squeeze-Excitation Residual Network)
- Initial conv: 3x3 conv, `N_FILTERS` filters, batch norm, ReLU
- `N_BLOCKS` SE-residual blocks, each containing:
  - 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN
  - SE: global avg pool -> FC(N_FILTERS, N_FILTERS/4) -> ReLU -> FC(N_FILTERS/4, 2*N_FILTERS) -> sigmoid -> scale
  - Residual connection + ReLU
- Default config: `N_BLOCKS=6`, `N_FILTERS=128` (small), scalable to 15/256 (large)

**Policy head** (move probabilities):
- 1x1 conv -> BN -> ReLU -> 1x1 conv to 73 planes -> flatten
- Output: 73 x 64 = 4672 logits (AlphaZero move encoding)
  - 56 "queen moves" (7 distances x 8 directions)
  - 8 "knight moves"
  - 9 "underpromotions" (3 piece types x 3 directions)
- Masked by legal moves, then softmax

**Value head** (position evaluation):
- 1x1 conv -> BN -> ReLU -> flatten -> FC(256) -> ReLU -> FC(3)
- Output: 3 logits for WDL (win/draw/loss), softmax
- Final value = P(win) - P(loss), in [-1, 1]

WDL output is preferred over scalar tanh because:
- Better gradient properties
- Can distinguish "likely draw" from "unclear position"
- Matches modern LC0 practice

**Sizes** (6 blocks, 128 filters):
- Parameters: ~2.5M
- Forward pass: ~0.5ms on RTX 4090 (single), ~50us batched

### 2. Move Encoding

AlphaZero encodes moves as (from_square, move_type) -> 64 x 73 = 4672:

```
Move types (73 total):
  Queen moves: 7 distances x 8 directions = 56
    Directions: N, NE, E, SE, S, SW, W, NW
    Distances: 1-7
  Knight moves: 8
    NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW
  Underpromotions: 9
    3 directions (left-capture, forward, right-capture) x 3 pieces (knight, bishop, rook)
    (Queen promotion is already encoded as queen move 1 square forward/diagonal)
```

Move encoding maps from engine `move_t` (from_sq, to_sq, promo) to policy index.
Move decoding maps from policy index back to `move_t`.

These are precomputed lookup tables.

### 3. MCTS (Monte Carlo Tree Search)

Each node in the search tree stores:
```cpp
struct MCTSNode {
    // Tree structure
    MCTSNode* parent;
    std::vector<MCTSEdge> children;
    
    // State (only stored at root and for transpositions)
    // Child states are computed on-the-fly via make_move/retract_move
    
    // Statistics
    uint32_t visit_count;       // N(s)
    float value_sum;            // W(s) - total value from MCTS backups
    float virtual_loss;         // for parallel search
};

struct MCTSEdge {
    move_t move;                // the move from parent to child
    float prior;                // P(s,a) from policy network  
    MCTSNode* child;            // nullptr until expanded (lazy)
    uint32_t visit_count;       // N(s,a)
    float value_sum;            // Q(s,a) * N(s,a)
};
```

**Selection** (PUCT):
```
UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

where:
  Q(s, a) = W(s, a) / N(s, a)    (mean value)
  P(s, a) = prior from policy net
  N(s)    = parent visit count
  N(s, a) = edge visit count
  c_puct  = exploration constant (default: 2.5, tunable)
```

**Expansion**: When a leaf is reached, run the neural network to get (policy, value).
Create child edges with priors from the policy. Backpropagate the value.

**Backpropagation**: Walk up the tree, incrementing visit counts and adding value
(negated at each level since players alternate).

**Temperature**: At move selection:
- Training: tau=1.0 for first 30 moves (proportional to visit count), tau->0 after
- Playing: tau->0 (always pick most visited)

**Dirichlet noise** (training only): At root, mix policy with Dir(alpha=0.3):
```
P'(s, a) = 0.75 * P(s, a) + 0.25 * Dir(0.3)
```

### 4. Self-Play Pipeline (C++)

```
SelfPlayManager:
  - Manages N concurrent games
  - Collects positions needing evaluation into batches
  - Runs batched inference via libtorch
  - Returns results to MCTS trees
  
  Loop:
    1. For each active game, run MCTS selection until leaf
    2. Collect all leaf positions into a batch
    3. Run neural network on batch (GPU)
    4. Distribute results back to trees
    5. Continue MCTS (backprop, more selection) until sim budget
    6. Select moves, record training data
    7. Check for game termination
```

**Training data format** (per position):
```
struct TrainingExample {
    float planes[22][8][8];     // input features
    float policy[4672];         // MCTS visit count distribution
    float result;               // game result from this player's perspective {-1, 0, 1}
};
```

Stored as flat binary files or numpy-compatible format for PyTorch DataLoader.

### 5. Training Loop (C++ with libtorch)

The entire pipeline runs in C++ via libtorch. No Python in the production loop.

```
dc0_train binary:
  for generation in 0..N:
    1. Self-play: generate games using current model (batched MCTS inference)
       -> writes training examples to binary files
    2. Train: load data, run SGD with libtorch autograd
       -> saves model checkpoint (.pt via torch::save)
    3. Evaluate: pit new model vs previous best in head-to-head games
       -> if win_rate > 55%, promote to new best
```

**Loss function**:
```
L = L_value + L_policy + c * L_reg

L_value  = cross_entropy(z, v)     # WDL cross-entropy
L_policy = -pi^T * log(p)          # policy cross-entropy
L_reg    = c * ||theta||^2         # L2 weight decay in SGD
```

**Training hyperparameters** (AlphaZero-like):
- Optimizer: SGD with momentum 0.9, weight decay 1e-4
- Learning rate: 0.2, dropped to 0.02 at 100K steps, 0.002 at 300K steps
- Batch size: 1024 (or 4096 for larger runs)
- Training window: last 500K games (or all data for small scale)

Python code (model.py, move_encoding.py) is retained as reference implementation
for correctness testing and notebook-based analysis of saved checkpoints.

### 6. UCI Integration

dc0 operates as an alternative search mode within the existing UCI framework:

```
setoption name SearchMode value dc0      # switch to MCTS
setoption name SearchMode value alphabeta # switch back to traditional
setoption name DC0Model value /path/to/model.ts  # load TorchScript model
setoption name DC0Simulations value 800  # MCTS simulations per move
setoption name DC0Cpuct value 2.5        # exploration constant
setoption name DC0Temperature value 0    # 0 = deterministic play
```

The `go` command works as normal. `go nodes N` maps to N MCTS simulations.
`go movetime T` runs simulations until time expires.

### 7. File Structure

```
zero/
  docs/
    DESIGN.md            # this file
    progress.md          # detailed progress tracking
  
  # C++ core (libtorch, all production code)
  MoveEncoding.hpp       # move_t <-> policy index conversion tables
  BoardEncoding.hpp      # Board -> input tensor encoding
  Network.hpp            # SE-ResNet model definition (libtorch modules)
  MCTS.hpp               # MCTS tree, node, edge, PUCT search
  SelfPlay.hpp           # self-play game manager, batched inference
  Training.hpp           # training loop, loss, optimizer, data loading
  DC0Engine.hpp          # ties everything together for UCI
  Logging.hpp            # log levels, timestamps, configurable verbosity
  dc0_main.cpp           # main binary: train / selfplay / eval modes
  CMakeLists.txt         # builds dc0, dc0_uci, dc0_tests binaries
  
  # Python (reference impl + notebook analysis)
  model.py               # PyTorch reference model (mirrors Network.hpp)
  move_encoding.py       # Python reference move encoding (mirrors MoveEncoding.hpp)
  pyproject.toml         # Python env with torch + jupyter for notebooks
```

## Phases

### Phase 1: Neural Network + Move Encoding
- Move encoding tables in C++ (MoveEncoding.hpp) + Python reference
- SE-ResNet model in C++ libtorch (Network.hpp) + Python reference
- Board -> tensor encoding in C++ (BoardEncoding.hpp)
- CMakeLists.txt linking libtorch + dummy_chess headers
- Tests: forward pass shapes, move encode/decode roundtrips, Python parity

### Phase 2: MCTS (C++)
- MCTSNode/MCTSEdge data structures
- PUCT selection
- Expansion + backpropagation
- Temperature-based move selection
- Dirichlet noise at root
- Test with random policy/value (should play legal chess, prefer captures)

### Phase 3: Self-Play Pipeline (C++)
- SelfPlayManager: N concurrent games
- Batched leaf evaluation via libtorch
- Training data serialization (binary format)
- CLI: num_games, model_path, output_dir, etc.

### Phase 4: Training Loop (C++ with libtorch)
- DataLoader for self-play binary data
- SGD optimizer with momentum, weight decay, LR schedule
- Loss: policy cross-entropy + WDL cross-entropy
- Model checkpointing (torch::save / torch::load)
- Evaluation: pit new vs old model in head-to-head games
- Full loop: self-play -> train -> evaluate -> repeat

### Phase 5: UCI Integration
- DC0Engine class composing MCTS + Network + Board
- SearchMode option in UCI.hpp
- go command dispatching to MCTS
- Model loading via setoption

### Phase 6: Optimizations
- FP16 inference
- Tree reuse between moves
- Batched inference with multiple search threads
- Transposition table for MCTS (DAG, not tree)
- Endgame tablebase integration in MCTS
- Pondering (background search during opponent's turn)

## Key Design Decisions

1. **Everything in C++**: self-play, training, and play all via libtorch. No Python in production loop.
2. **WDL value head** over scalar tanh: better training signal, matches LC0
3. **SE-ResNet** over plain ResNet: proven improvement in LC0, marginal cost
4. **AlphaZero move encoding** (73x64): universal, no need for move lists
5. **libtorch** for all NN ops: training + inference in same language, same binary
6. **No history planes initially**: simplifies input, add later if needed
7. **Python as reference only**: model.py + move_encoding.py for correctness testing and notebooks
