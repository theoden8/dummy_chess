# Super-Efficient HalfKP NNUE Design

## Architecture Overview

### Supported Architectures

| Architecture | Features | Parameters | Description |
|-------------|----------|------------|-------------|
| **HalfKP** | 41,025 | ~10.5M | Original, full king square granularity |
| **HalfKAv2** | 7,693 | ~2.0M | King buckets + horizontal mirroring |

### Network Topology
```
Perspective 1 (STM):   Features[N] ─┐
                                    ├─> Concat[512] -> L1[512->256] -> L2[256->32] -> L3[32->32] -> Out[32->1]
Perspective 2 (NSTM):  Features[N] ─┘

Where N = 41,025 (HalfKP) or 7,693 (HalfKAv2)
```

### Key Design Decisions

1. **Larger Feature Transformer**: 256 neurons (vs original 256, same as Stockfish)
2. **Wider Hidden Layers**: 256 -> 32 -> 32 -> 1 (compact but expressive)
3. **All Integer Arithmetic**: int16 accumulators, int8 weights for hidden layers
4. **SIMD Throughout**: AVX-512 > AVX2 > SSE4.2 > Scalar fallback

## Memory Layout

### Feature Transformer Weights
```cpp
// Transposed for efficient sparse accumulation
// Layout: [HALFKP_SIZE][FT_OUT] but stored as [FT_OUT][HALFKP_SIZE] transposed
alignas(64) int16_t ft_weights[FT_OUT][HALFKP_SIZE];  // 256 * 41024 * 2 = ~21MB
alignas(64) int16_t ft_biases[FT_OUT];                 // 256 * 2 = 512B
```

### Hidden Layer Weights
```cpp
// Dense layers - column-major for efficient SIMD
alignas(64) int8_t l1_weights[L1_IN][L1_OUT];   // 512 * 256 = 128KB
alignas(64) int32_t l1_biases[L1_OUT];          // 256 * 4 = 1KB

alignas(64) int8_t l2_weights[L2_IN][L2_OUT];   // 256 * 32 = 8KB  
alignas(64) int32_t l2_biases[L2_OUT];          // 32 * 4 = 128B

alignas(64) int8_t l3_weights[L3_IN][L3_OUT];   // 32 * 32 = 1KB
alignas(64) int32_t l3_biases[L3_OUT];          // 32 * 4 = 128B

alignas(64) int8_t out_weights[OUT_IN];         // 32 * 1 = 32B
alignas(64) int32_t out_bias;                   // 4B
```

## Accumulator Design

### Stack-Based Incremental Updates
```cpp
struct Accumulator {
    alignas(64) int16_t values[2][FT_OUT];  // [perspective][neurons]
    bool computed[2];                        // validity flags
};

// Stack for make/unmake - max depth ~128 plies
thread_local Accumulator accumulator_stack[MAX_PLY];
thread_local int accumulator_sp = 0;
```

### Update Operations
```cpp
// On make_move: push new accumulator, compute delta
void push_accumulator(const Move& m) {
    accumulator_stack[++accumulator_sp] = accumulator_stack[accumulator_sp - 1];
    apply_delta(m);
}

// On unmake_move: just pop
void pop_accumulator() {
    --accumulator_sp;
}

// Delta update: O(1) feature changes vs O(N) recomputation
void apply_delta(const Move& m) {
    // King moves: full refresh for that perspective
    // Other moves: add/subtract 1-4 feature vectors
}
```

## SIMD Implementations

### AVX-512 (512-bit = 32 x int16 or 64 x int8)
```cpp
// Accumulator update: add feature vector
void add_feature_avx512(int16_t* acc, const int16_t* weights, size_t size) {
    for (size_t i = 0; i < size; i += 32) {
        __m512i a = _mm512_load_si512(acc + i);
        __m512i w = _mm512_load_si512(weights + i);
        _mm512_store_si512(acc + i, _mm512_add_epi16(a, w));
    }
}

// Dense layer with int8 weights
int32_t dense_layer_avx512(const int8_t* input, const int8_t* weights, 
                           int32_t bias, size_t in_size) {
    __m512i sum = _mm512_setzero_si512();
    for (size_t i = 0; i < in_size; i += 64) {
        __m512i x = _mm512_load_si512(input + i);
        __m512i w = _mm512_load_si512(weights + i);
        sum = _mm512_dpbusd_epi32(sum, x, w);  // VNNI instruction
    }
    return bias + _mm512_reduce_add_epi32(sum);
}
```

### AVX2 (256-bit = 16 x int16 or 32 x int8)
```cpp
void add_feature_avx2(int16_t* acc, const int16_t* weights, size_t size) {
    for (size_t i = 0; i < size; i += 16) {
        __m256i a = _mm256_load_si256((__m256i*)(acc + i));
        __m256i w = _mm256_load_si256((__m256i*)(weights + i));
        _mm256_store_si256((__m256i*)(acc + i), _mm256_add_epi16(a, w));
    }
}

// Dot product with int8 using madd
int32_t dot_product_avx2(const int8_t* a, const int8_t* b, size_t size) {
    __m256i sum = _mm256_setzero_si256();
    for (size_t i = 0; i < size; i += 32) {
        __m256i va = _mm256_load_si256((__m256i*)(a + i));
        __m256i vb = _mm256_load_si256((__m256i*)(b + i));
        // maddubs: multiply i8*u8 -> i16, then hadd pairs
        __m256i prod = _mm256_maddubs_epi16(va, vb);
        sum = _mm256_add_epi16(sum, prod);
    }
    // Horizontal sum
    return hadd_epi16_to_i32(sum);
}
```

### SSE4.2 (128-bit = 8 x int16 or 16 x int8)
```cpp
void add_feature_sse42(int16_t* acc, const int16_t* weights, size_t size) {
    for (size_t i = 0; i < size; i += 8) {
        __m128i a = _mm_load_si128((__m128i*)(acc + i));
        __m128i w = _mm_load_si128((__m128i*)(weights + i));
        _mm_store_si128((__m128i*)(acc + i), _mm_add_epi16(a, w));
    }
}
```

## HalfKP Feature Index Calculation

```cpp
constexpr size_t KING_SQUARES = 64;
constexpr size_t PIECE_SQUARES = 64;
constexpr size_t PIECE_TYPES = 10;  // 5 pieces * 2 colors (excluding kings)
constexpr size_t HALFKP_SIZE = KING_SQUARES * (PIECE_TYPES * PIECE_SQUARES + 1);
// = 64 * (10 * 64 + 1) = 64 * 641 = 41024

// Feature index for piece at 'sq' from perspective of king at 'ksq'
inline size_t halfkp_index(bool white_pov, int ksq, int sq, int piece_type, bool piece_white) {
    int oriented_ksq = white_pov ? ksq : (63 - ksq);
    int oriented_sq = white_pov ? sq : (63 - sq);
    int piece_index = piece_type * 2 + (piece_white == white_pov ? 0 : 1);
    return oriented_ksq * 641 + piece_index * 64 + oriented_sq + 1;
}
```

## HalfKAv2 Architecture

HalfKAv2 reduces the feature space from 41K to 7.7K features using two key techniques:

1. **King Buckets**: Groups strategically similar king positions into 12 buckets instead of 64 squares
2. **Horizontal Mirroring**: Exploits kingside/queenside symmetry (files e-h mirror to a-d)

### Benefits
- **Faster training**: ~5x smaller embedding table means faster gradient updates
- **Better generalization**: Grouping similar king positions reduces overfitting
- **Lower memory**: Smaller model size for deployment

### King Bucket Layout

The 12 buckets prioritize back-rank granularity where kings typically reside:

```
Rank 1 (back rank): 4 buckets (0-3) - most granular for castled positions
  a1 → bucket 0, b1 → bucket 1, c1 → bucket 2, d1 → bucket 3
  
Rank 2: 2 buckets (4-5)
  a2,b2 → bucket 4, c2,d2 → bucket 5
  
Rank 3: 2 buckets (6-7)
  a3,b3 → bucket 6, c3,d3 → bucket 7
  
Rank 4: 1 bucket (8)
Rank 5: 1 bucket (9)
Rank 6: 1 bucket (10)
Ranks 7-8: 1 bucket (11) - rare king positions
```

Files e-h are mirrored to a-d before bucket lookup.

### Feature Index Calculation

```cpp
constexpr size_t KING_BUCKETS = 12;
constexpr size_t HALFKAV2_SIZE = KING_BUCKETS * (PIECE_TYPES * PIECE_SQUARES + 1);
// = 12 * (10 * 64 + 1) = 12 * 641 = 7692 + 1 = 7693

// King bucket table: maps (rank * 4 + file) to bucket for files a-d
static constexpr int KING_BUCKET[32] = {
    0, 1, 2, 3,     // Rank 1: buckets 0-3
    4, 4, 5, 5,     // Rank 2: buckets 4-5
    6, 6, 7, 7,     // Rank 3: buckets 6-7
    8, 8, 8, 8,     // Rank 4: bucket 8
    9, 9, 9, 9,     // Rank 5: bucket 9
    10, 10, 10, 10, // Rank 6: bucket 10
    11, 11, 11, 11, // Ranks 7-8: bucket 11
};

// Get bucket and mirror flag for king square
inline std::pair<int, bool> get_bucket_and_mirror(int king_sq) {
    int file = king_sq % 8;
    int rank = king_sq / 8;
    bool mirror = (file >= 4);
    if (mirror) file = 7 - file;  // Mirror e-h to d-a
    return {KING_BUCKET[rank * 4 + file], mirror};
}

// Feature index with mirroring
inline size_t halfkav2_index(int bucket, bool mirror, int piece_sq, int piece_index) {
    if (mirror) piece_sq ^= 7;  // Mirror file
    return bucket * 641 + piece_index * 64 + piece_sq + 1;
}
```

### Training Usage

```bash
# Train with HalfKAv2 (smaller, faster)
uv run python train.py data/evals.parquet --shuffle --arch halfkav2 --epochs 30

# Train with HalfKP (original, more capacity)
uv run python train.py data/evals.parquet --shuffle --arch halfkp --epochs 30
```

## Clipped ReLU Implementation

```cpp
// ClippedReLU: max(0, min(127, x))
// For int16 accumulators -> int8 output

// AVX2 version
void clipped_relu_avx2(const int16_t* input, int8_t* output, size_t size) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max_val = _mm256_set1_epi8(127);
    
    for (size_t i = 0; i < size; i += 32) {
        __m256i lo = _mm256_load_si256((__m256i*)(input + i));
        __m256i hi = _mm256_load_si256((__m256i*)(input + i + 16));
        
        // Pack int16 to int8 with saturation
        __m256i packed = _mm256_packs_epi16(lo, hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);  // fix lane crossing
        
        // Clamp to [0, 127]
        packed = _mm256_max_epi8(packed, zero);
        packed = _mm256_min_epi8(packed, max_val);
        
        _mm256_store_si256((__m256i*)(output + i), packed);
    }
}
```

## File Format (Compatible with Stockfish .nnue)

```
Header:
  uint32_t version
  uint32_t hash
  uint32_t arch_string_len
  char[arch_string_len] arch_string

Feature Transformer:
  uint32_t header
  int16_t biases[FT_OUT]
  int16_t weights[HALFKP_SIZE * FT_OUT]  // row-major

Network Layers:
  uint32_t header
  For each layer:
    int32_t biases[out_size]
    int8_t weights[in_size * out_size]   // column-major for hidden
```

## Performance Expectations

| Operation | Cycles (AVX2) | Cycles (AVX-512) |
|-----------|---------------|------------------|
| Feature add/sub | ~8 | ~4 |
| Full FT refresh | ~2000 | ~1000 |
| L1 forward | ~200 | ~100 |
| L2 forward | ~30 | ~15 |
| L3 forward | ~30 | ~15 |
| Output | ~10 | ~5 |
| **Total incremental** | **~300** | **~150** |
| **Total from scratch** | **~2300** | **~1150** |

At 3GHz: ~100ns incremental, ~400ns full refresh (AVX2)

## Training Infrastructure

### Overview

Training uses PyTorch with a streaming data pipeline. The model trains in float32, then weights are quantized for C++ inference.

### Data Pipeline

```
Lichess Data (.csv.zst)
        |
        v
preprocess.py (streaming)
        |
        v
Parquet files (compressed FEN + score)
        |
        v
ShuffledParquetDataset (row-group interleaving)
        |
        v
C++ get_halfkp_features_batch()
        |
        v
PyTorch DataLoader
```

### Key Components

**`training/preprocess.py`**
- `process_evals()`: Lichess evaluation database
- `process_puzzles()`: Lichess puzzles (2 positions per puzzle: start + after move 1)
- `process_endgames()`: Random endgame positions with tablebase scores
- All use streaming and batch FEN compression
- `--engine` / `-e`: UCI engine for evaluation (required for puzzles)
- `--tablebase`: Path to Syzygy tablebases for endgame positions

**`training/train.py`**
- `SplitConfig`: Controls train/val/test splits with seed for reproducibility
- `ShuffledParquetDataset`: Memory-efficient shuffled streaming (see below)
- `LazyFrameDataset`: Sequential streaming (legacy, for compatibility)
- `HalfKPNet`: PyTorch model matching the target architecture
- Uses `SparseAdam` for HalfKP embeddings, `AdamW` for dense layers

**`python/Bindings.cpp`**
- `compress_fen()` / `decompress_fen()`: FEN compression (single)
- `compress_fens_batch()` / `decompress_fens_batch()`: Batch compression
- `get_halfkp_features_batch(fens, flip)`: ~250x faster than Python feature extraction

### ShuffledParquetDataset

Memory-efficient shuffled dataset using multi-row-group interleaving.

**How it works:**
1. Shuffles row group order each epoch
2. Maintains N "active" row groups simultaneously
3. Samples randomly from all active group buffers
4. When a group is exhausted, replaces with next from shuffled list

**Memory usage formula:**
```
RAM = num_active_groups × buffer_per_group × ~50 bytes/sample
```

| Configuration | Memory |
|---------------|--------|
| Default (8 × 4096) | ~1.6 MB |
| Large (16 × 8192) | ~6.5 MB |
| Aggressive (32 × 16384) | ~26 MB |

**Constructor:**
```python
ShuffledParquetDataset(
    paths,                    # List of paths OR list of (path, weight) tuples
    split="train",            # "train", "val", "test", or None for all
    split_config=SplitConfig(),
    num_active_groups=8,      # More = better shuffle, more RAM
    buffer_per_group=4096,    # Samples buffered per group
    flip_augment=False,       # Include flipped positions (2x data)
    seed=None,                # Random seed (changes each epoch if None)
)
```

**Weighted multi-source training:**
```python
# Mix datasets with different weights
dataset = ShuffledParquetDataset(
    paths=[
        ("data/evals.parquet", 1.0),      # Full weight
        ("data/puzzles.parquet", 2.0),    # 2x sampling rate
        ("data/endgames.parquet", 0.5),   # 0.5x sampling rate
    ],
    split="train",
    num_active_groups=16,
)
```

**From a notebook:**
```python
from train import ShuffledParquetDataset, SplitConfig
import torch

split_config = SplitConfig(val_ratio=0.05, test_ratio=0.05)

train_dataset = ShuffledParquetDataset(
    paths=[
        ("data/evals.parquet", 1.0),
        ("data/puzzles.parquet", 1.5),
    ],
    split="train",
    split_config=split_config,
    num_active_groups=8,
    buffer_per_group=4096,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8192,
    num_workers=0,  # Must be 0 for IterableDataset
    collate_fn=collate_sparse,
)

# Training loop
for epoch in range(30):
    train_dataset.set_epoch(epoch)  # Reshuffle each epoch
    for features, targets in train_loader:
        # train...
        pass
```

### Flip Augmentation

Flip augmentation (`--flip-augment`) doubles training data by generating a color-swapped version of each position. This helps the network learn symmetric evaluation.

**What happens when `flip=True`:**

1. **STM (side to move) is inverted**: `stm_flipped = 1 - stm`
   - White to move (0) becomes Black to move (1)
   - Black to move (1) becomes White to move (0)

2. **HalfKP features are swapped between perspectives**:
   - Original: white_features → white accumulator, black_features → black accumulator
   - Flipped: black_features → white accumulator, white_features → black accumulator
   
3. **Score is negated**: `score_flipped = -score`
   - A position that's +100 for white becomes -100 (i.e., +100 for black)

**Example:**
```
Original position: White to move, white is +150cp ahead
  STM = 0 (white)
  Score = +150
  White features: [king on e1 sees rook on a1, ...]
  Black features: [king on e8 sees rook on a8, ...]

Flipped position: Black to move, black is +150cp ahead  
  STM = 1 (black)
  Score = -150
  White features: [king on e8 sees rook on a8, ...] (was black's view)
  Black features: [king on e1 sees rook on a1, ...] (was white's view)
```

**Why this works:**
- HalfKP features already encode positions from each king's perspective with board flipping (black's view is vertically mirrored)
- Swapping the feature sets is equivalent to swapping which color "we" are
- The network learns that `eval(white_view, black_view, white_to_move)` = `-eval(black_view, white_view, black_to_move)`

**Benefits:**
- 2x training data without additional positions
- Forces symmetric evaluation (same position evaluated from opposite sides should give opposite scores)
- Reduces bias toward one color

**Implementation** (in `Bindings.cpp`):
```cpp
if (flip) {
  // Swap white and black features
  white_indices.push_back(black_feat);
  black_indices.push_back(white_feat);
  // STM is also flipped
  stm[pos_idx] = 1 - stm_val;
}
// Caller negates the score: -float(scores[i])
```

### Training Features

- **Flip augmentation**: `--flip-augment` yields both original and color-swapped positions (2x data, zero-mean scores)
- **Shuffled streaming**: `--shuffle` enables ShuffledParquetDataset (recommended)
- **Multi-source mixing**: Pass multiple files with weights for balanced training
- **Memory efficient**: True streaming, configurable RAM budget
- **Mixed precision**: Optional AMP for faster training on supported GPUs

### CLI Usage

```bash
cd training

# Preprocess data
uv run python preprocess.py evals -n 100000 -o data/evals.parquet
uv run python preprocess.py puzzles -e /path/to/stockfish -n 50000 -o data/puzzles.parquet

# Train with shuffling (recommended)
uv run python train.py data/evals.parquet --shuffle --epochs 30 --batch-size 8192

# Train with multiple sources
uv run python train.py data/evals.parquet data/puzzles.parquet --shuffle --epochs 30

# Tune memory/shuffle tradeoff
uv run python train.py data/evals.parquet --shuffle \
    --num-active-groups 16 \
    --buffer-per-group 8192 \
    --epochs 30 --batch-size 8192

# With flip augmentation
uv run python train.py data/evals.parquet --shuffle --flip-augment --epochs 30

# Rebuild C++ module after Bindings.cpp changes
uv run --reinstall-package dummy-chess python -c "import dummy_chess"
```

### CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `data` | (required) | One or more parquet files |
| `--output` | `network.nnue` | Output file path |
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 8192 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--val-ratio` | 0.05 | Validation split ratio |
| `--test-ratio` | 0.05 | Test split ratio |
| `--flip-augment` | False | Double data with flipped positions |
| `--shuffle` | False | Use ShuffledParquetDataset |
| `--num-active-groups` | 8 | Row groups sampled simultaneously |
| `--buffer-per-group` | 4096 | Samples buffered per group |
| `--arch` | `halfkp` | Architecture: `halfkp` (41K features) or `halfkav2` (7.7K features) |

## TODO

### Inference (C++)
- [ ] Implement SIMD-optimized forward pass
- [ ] Implement incremental accumulator updates
- [ ] Add weight quantization export from PyTorch
- [ ] Integrate with search

### Training
- [x] Streaming data pipeline
- [x] C++ batched feature extraction
- [x] Flip augmentation for zero-mean scores
- [x] Memory-efficient batch iteration
- [ ] Learning rate scheduling
- [ ] Validation/early stopping
- [ ] Train on full Lichess eval dataset
