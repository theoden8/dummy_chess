# NNUE Analysis - Existing Implementation on `nnue` Branch

## Summary

The `nnue` branch contains a working but basic HalfKP NNUE implementation. This document analyzes the existing code and identifies optimization opportunities for a super-efficient reimplementation.

## Current Architecture (from nnue branch)

### Network Structure
```
Input: HalfKP features (41024 sparse features per perspective)
       |
       v
Feature Transformer: 41024 -> 256 (shared weights, ClippedReLU)
       |
       v
Concatenate perspectives: 256 * 2 = 512
       |
       v
Hidden 1: 512 -> 32 (NNUEReLU: x/64 clamped to [0,127])
       |
       v
Hidden 2: 32 -> 32 (NNUEReLU)
       |
       v
Output: 32 -> 1 (linear)
```

### HalfKP Feature Encoding
- **Total features**: 41024 per perspective (64 king squares * 64 piece squares * 10 piece types + 1)
- **Feature index formula**: `orient(pos) + piece_index(p, color) + (10 * 64 + 1) * king_square`
- **Orientation**: White perspective keeps positions as-is, Black flips (63 - pos)

### Data Types Used
| Layer | Weight Type | Bias Type | Output Type |
|-------|-------------|-----------|-------------|
| Feature Transformer | int16_t | int16_t | int8_t (after ClippedReLU) |
| Hidden 1 | int8_t | int32_t | int8_t |
| Hidden 2 | int8_t | int32_t | int8_t |
| Output | int8_t | int32_t | int32_t |

### Activation Functions
1. **ClippedReLU**: `max(0, min(127, x))` - used after feature transformer
2. **NNUEReLU**: `max(0, min(127, x/64))` - used in hidden layers

## Identified Issues & Optimization Opportunities

### 1. No SIMD Vectorization
The current implementation uses scalar loops:
```cpp
for(size_t j = 0; j < OUT; ++j) {
  OUT_T dotprod = b[j];
  for(size_t i = 0; i < IN; ++i) {
    dotprod += OUT_T(A[i * OUT + j]) * OUT_T(x[i]);
  }
  y[j] = dotprod;
}
```

**Optimization**: Use AVX2/AVX-512 for 16-32x speedup on dense layers.

### 2. Non-Incremental Feature Transformer
Currently rebuilds all features from scratch:
```cpp
for(size_t i : input_indices[c]) {
  for(size_t j = 0; j < M; ++j) {
    _y[j] += int16_t(affine.A[i * M + j]);
  }
}
```

**Optimization**: Use accumulator stack with incremental add/sub on make/unmake move.

### 3. Memory Layout Not Cache-Friendly
Weight matrices stored row-major but accessed column-wise in some cases.

**Optimization**: Use transposed layouts optimized for SIMD access patterns.

### 4. No Lazy Evaluation
Always computes full forward pass even when not needed (e.g., in QSearch cutoffs).

**Optimization**: Lazy evaluation - only compute when score is actually needed.

### 5. Training Infrastructure
- Uses TensorFlow with Python bindings
- Preprocessing pipeline: Lichess puzzles -> FEN + score -> sparse features
- Loss: MSE between predicted and actual score

## File Organization (nnue branch)

```
NNUE.hpp          - Core NNUE implementation (Layer, Affine, ClippedReLU, halfkp)
nnue.cpp          - Simple test driver
training/
  train.py        - TensorFlow training script
  preprocess_puzzles.py - Dataset preparation
python/
  Bindings.cpp    - Python bindings for NNUEDummy class
```

## Stockfish NNUE Reference

For comparison, Stockfish's NNUE uses:
- Feature transformer: HalfKA (king-aware) with 1024 hidden units
- Perspective-specific accumulators with lazy updates
- Heavy SIMD optimization (AVX-512, AVX2, SSE4.2, NEON)
- Quantized int8/int16 arithmetic throughout
- Cache-aligned data structures

## Recommendations for Super-Efficient Implementation

1. **SIMD-First Design**: Structure all data for vectorized operations
2. **Incremental Accumulators**: Stack-based accumulator with delta updates
3. **Cache-Aligned Allocations**: 64-byte alignment for all weight matrices
4. **Compile-Time Dispatch**: Use `#ifdef` for AVX-512/AVX2/SSE4.2/scalar fallback
5. **Lazy Evaluation**: Defer forward pass until score is actually needed
6. **Memory Pooling**: Pre-allocate accumulator stack to avoid heap allocations
7. **Prefetching**: Use `__builtin_prefetch` for weight matrix access patterns

## Performance Targets

| Operation | Current (scalar) | Target (AVX2) | Target (AVX-512) |
|-----------|------------------|---------------|------------------|
| Feature Transformer Update | ~500ns | ~50ns | ~25ns |
| Full Forward Pass | ~2000ns | ~200ns | ~100ns |
| Incremental Update | N/A | ~30ns | ~15ns |
