# Super-Efficient HalfKP NNUE Design

## Architecture Overview

### Network Topology
```
Perspective 1 (STM):   HalfKP[41024] ─┐
                                      ├─> Concat[512] -> L1[512->256] -> L2[256->32] -> L3[32->32] -> Out[32->1]
Perspective 2 (NSTM):  HalfKP[41024] ─┘
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
