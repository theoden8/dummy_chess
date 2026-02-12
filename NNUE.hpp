#pragma once

// Super-Efficient HalfKP NNUE Implementation
// Optimized for AVX2/AVX-512/SSE4.2 with incremental accumulator updates

#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>

#include "Optimizations.hpp"
#include "Piece.hpp"

// SIMD detection and includes
#if defined(__AVX512F__) && defined(__AVX512BW__)
  #define NNUE_USE_AVX512 1
  #include <immintrin.h>
#elif defined(__AVX2__)
  #define NNUE_USE_AVX2 1
  #include <immintrin.h>
#elif defined(__SSE4_2__) || defined(__SSE4_1__)
  #define NNUE_USE_SSE42 1
  #include <smmintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #define NNUE_USE_NEON 1
  #include <arm_neon.h>
#else
  #define NNUE_USE_SCALAR 1
#endif

namespace nnue {

// ============================================================================
// Architecture Constants
// ============================================================================

// HalfKP feature dimensions
constexpr size_t PIECE_TYPES = 10;  // 5 piece types * 2 colors (no kings)
constexpr size_t HALFKP_FEATURES = board::SIZE * (PIECE_TYPES * board::SIZE + 1);
// = 64 * 641 = 41024 (number of actual HalfKP feature indices)
constexpr size_t HALFKP_SIZE = HALFKP_FEATURES + 1;
// = 41025 (EmbeddingBag size in training, includes zero/padding index)

// Network architecture
constexpr size_t FT_OUT = 256;          // Feature transformer output
constexpr size_t L1_IN = FT_OUT * 2;    // 512 (concatenated perspectives)
constexpr size_t L1_OUT = 32;
constexpr size_t L2_IN = L1_OUT;
constexpr size_t L2_OUT = 32;
constexpr size_t OUT_IN = L2_OUT;

// Quantization constants
constexpr int FT_SCALE = 127;           // Feature transformer activation scale
constexpr int NET_SCALE = 64;           // Hidden layer activation scale
constexpr int SIGMOID_SCALE = 400;      // Must match training SIGMOID_SCALE

// Maximum search depth for accumulator stack
constexpr size_t MAX_PLY = 256;

// Cache line alignment
constexpr size_t CACHE_LINE = 64;

// ============================================================================
// SIMD Helpers
// ============================================================================

#if defined(NNUE_USE_AVX512)
constexpr size_t SIMD_WIDTH = 64;
using simd_reg_t = __m512i;
#define SIMD_ALIGNMENT 64
#elif defined(NNUE_USE_AVX2)
constexpr size_t SIMD_WIDTH = 32;
using simd_reg_t = __m256i;
#define SIMD_ALIGNMENT 32
#elif defined(NNUE_USE_SSE42)
constexpr size_t SIMD_WIDTH = 16;
using simd_reg_t = __m128i;
#define SIMD_ALIGNMENT 16
#elif defined(NNUE_USE_NEON)
constexpr size_t SIMD_WIDTH = 16;
using simd_reg_t = int16x8_t;
#define SIMD_ALIGNMENT 16
#else
constexpr size_t SIMD_WIDTH = 8;
using simd_reg_t = uint64_t;
#define SIMD_ALIGNMENT 8
#endif

// Aligned allocation helper
template<typename T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    
    T* allocate(size_t n) {
        void* ptr = nullptr;
        int result = posix_memalign(&ptr, Alignment, n * sizeof(T));
        assert(result == 0 && "posix_memalign failed");
        (void)result;
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_t) {
        free(ptr);
    }
};

template<typename T, size_t N, size_t Alignment = CACHE_LINE>
using AlignedArray = std::array<T, N>;

// ============================================================================
// Compressed FEN decoder with continuation-passing per piece
// ============================================================================

// Nibble encoding (from FEN.hpp): K=0,Q=1,R=2,B=3,N=4,P=5,k=6,q=7,r=8,b=9,n=10,p=11, 0xC=empty run
// Nibble-to-PIECE lookup: nib%6 → {KING,QUEEN,ROOK,BISHOP,KNIGHT,PAWN}
static constexpr PIECE NIB_TO_PIECE[6] = {KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN};

// Decode compressed FEN and call func(sq, piece, color) for each non-king piece.
// Writes king positions and active player through output references.
// Single pass: collects kings and non-king pieces from the nibble stream,
// then invokes the callback with king positions already set.
template<typename F>
inline void foreach_compressed_fen(
    const uint8_t* data, size_t length,
    COLOR& active_out, pos_t& wk_out, pos_t& bk_out,
    F&& func)
{
    wk_out = board::nopos;
    bk_out = board::nopos;
    active_out = WHITE;
    if (length < 8) return;

    active_out = (data[0] & 1) ? BLACK : WHITE;
    bool is_crazyhouse = data[0] & 4;
    size_t meta_size = 6 + (is_crazyhouse ? 13 : 0);
    size_t board_end = length - meta_size;

    // Unpack nibbles
    uint8_t nibs[128];
    size_t n_nibs = 0;
    for (size_t i = 1; i < board_end; ++i) {
        nibs[n_nibs++] = (data[i] >> 4) & 0xF;
        nibs[n_nibs++] = data[i] & 0xF;
    }

    // First pass: find kings and collect non-king pieces
    // Buffer sized for worst case (crazyhouse can exceed standard chess limits).
    struct PieceEntry { pos_t sq; PIECE piece; COLOR color; };
    PieceEntry pieces[board::SIZE - NO_COLORS];
    int n_pieces = 0;

    int board_idx = 0;
    for (size_t ni = 0; ni < n_nibs && board_idx < 64; ) {
        uint8_t nib = nibs[ni];
        if (nib == 0xC) {
            uint8_t cnt = (ni + 1 < n_nibs) ? nibs[ni + 1] + 1 : 1;
            board_idx += cnt;
            ni += 2;
        } else if (nib < 12) {
            pos_t sq = static_cast<pos_t>((7 - board_idx / 8) * 8 + board_idx % 8);
            COLOR color = (nib < 6) ? WHITE : BLACK;
            PIECE piece = NIB_TO_PIECE[nib % 6];
            if (piece == KING) {
                if (color == WHITE) wk_out = sq; else bk_out = sq;
            } else {
                pieces[n_pieces++] = {sq, piece, color};
            }
            ++board_idx;
            ++ni;
        } else {
            break;
        }
    }

    // Second pass: invoke callback for each non-king piece
    for (int i = 0; i < n_pieces; ++i) {
        func(pieces[i].sq, pieces[i].piece, pieces[i].color);
    }
}

// ============================================================================
// HalfKP Feature Index Calculation
// ============================================================================

struct HalfKP {
    // Piece type indices (0-9) for HalfKP encoding
    // White pieces: 0,2,4,6,8 | Black pieces: 1,3,5,7,9
    static constexpr int piece_to_index[NO_PIECES][NO_COLORS] = {
        {0, 1},   // PAWN
        {2, 3},   // KNIGHT
        {4, 5},   // BISHOP
        {6, 7},   // ROOK
        {8, 9},   // QUEEN
        {-1, -1}  // KING (not encoded)
    };

    // Orient square for perspective
    INLINE static constexpr pos_t orient(bool white_pov, pos_t sq) {
        return white_pov ? sq : (63 - sq);
    }

    // Calculate HalfKP feature index
    // king_sq: king square from perspective's view (oriented)
    // piece_sq: piece square (not oriented yet)
    // piece_type: 0-9 index
    INLINE static constexpr size_t index(bool white_pov, pos_t king_sq, pos_t piece_sq, int piece_type) {
        const pos_t oriented_sq = orient(white_pov, piece_sq);
        // Index = king_sq * 641 + piece_type * 64 + piece_sq + 1
        return static_cast<size_t>(king_sq) * (PIECE_TYPES * board::SIZE + 1) 
             + static_cast<size_t>(piece_type) * board::SIZE 
             + static_cast<size_t>(oriented_sq) + 1;
    }

    // Get piece type index for HalfKP encoding
    INLINE static constexpr int get_piece_type(PIECE p, COLOR piece_color, bool white_pov) {
        if (p == KING || p == EMPTY) return -1;
        // From white's perspective: white pieces are even, black pieces are odd
        // From black's perspective: it's reversed
        bool same_side = (piece_color == WHITE) == white_pov;
        return piece_to_index[p][same_side ? 0 : 1];
    }

    // Result of feature extraction for a single position
    struct Features {
        static constexpr int MAX_FEATURES = board::SIZE - NO_COLORS; // all squares minus the two kings
        int32_t white[MAX_FEATURES];  // White perspective feature indices
        int32_t black[MAX_FEATURES];  // Black perspective feature indices
        int count = 0;                // Number of features (same for both)
        int64_t stm = 0;             // Side to move: 0=white, 1=black
    };

    // Extract HalfKP features from a Board position.
    // If flip=true, swap white/black perspectives and flip STM.
    template<typename BoardT>
    static Features get_features(const BoardT& board, bool flip = false) {
        Features f;
        f.stm = (board.activePlayer() == WHITE) ? 0 : 1;
        if (flip) f.stm = 1 - f.stm;

        const pos_t wk_raw = board.pos_king[WHITE];
        const pos_t bk_raw = board.pos_king[BLACK];
        const pos_t wk = orient(true, wk_raw);   // White king from white's POV
        const pos_t bk = orient(false, bk_raw);   // Black king from black's POV

        const piece_bitboard_t occupied = board.bits[WHITE] | board.bits[BLACK];
        const piece_bitboard_t king_bits = board.get_king_bits();

        bitmask::foreach(occupied & ~king_bits, [&](pos_t sq) {
            const auto p = board[sq];
            const int wpt = get_piece_type(p.value, p.color, true);
            const int bpt = get_piece_type(p.value, p.color, false);
            if (wpt >= 0 && bpt >= 0 && f.count < Features::MAX_FEATURES) {
                int32_t wi = static_cast<int32_t>(index(true, wk, sq, wpt));
                int32_t bi = static_cast<int32_t>(index(false, bk, sq, bpt));
                if (flip) {
                    f.white[f.count] = bi;
                    f.black[f.count] = wi;
                } else {
                    f.white[f.count] = wi;
                    f.black[f.count] = bi;
                }
                ++f.count;
            }
        });
        return f;
    }

    // Extract features directly from compressed FEN bytes (no Board/FeatureBoard)
    static Features get_features_compressed(const uint8_t* data, size_t length, bool flip = false) {
        Features f;
        COLOR active; pos_t wk, bk;
        foreach_compressed_fen(data, length, active, wk, bk,
            [&](pos_t sq, PIECE piece, COLOR color) {
                const int wpt = get_piece_type(piece, color, true);
                const int bpt = get_piece_type(piece, color, false);
                if (wpt >= 0 && bpt >= 0 && f.count < Features::MAX_FEATURES) {
                    int32_t wi = static_cast<int32_t>(index(true, orient(true, wk), sq, wpt));
                    int32_t bi = static_cast<int32_t>(index(false, orient(false, bk), sq, bpt));
                    if (flip) {
                        f.white[f.count] = bi;
                        f.black[f.count] = wi;
                    } else {
                        f.white[f.count] = wi;
                        f.black[f.count] = bi;
                    }
                    ++f.count;
                }
            });
        f.stm = (active == WHITE) ? 0 : 1;
        if (flip) f.stm = 1 - f.stm;
        return f;
    }
};

// ============================================================================
// HalfKAv2 Feature Index Calculation (king buckets + horizontal mirroring)
// ============================================================================

struct HalfKAv2 {
    // King bucket table: maps king square (files a-d only) to bucket (0-11).
    // Files e-h are mirrored to a-d before lookup.
    static constexpr int KING_BUCKET[32] = {
        0, 1, 2, 3,       // Rank 1 (a1-d1)
        4, 4, 5, 5,       // Rank 2 (a2-d2)
        6, 6, 7, 7,       // Rank 3 (a3-d3)
        8, 8, 8, 8,       // Rank 4 (a4-d4)
        9, 9, 9, 9,       // Rank 5 (a5-d5)
        10, 10, 10, 10,   // Rank 6 (a6-d6)
        11, 11, 11, 11,   // Ranks 7-8
    };

    static constexpr int NUM_BUCKETS = 12;

    struct BucketInfo {
        int bucket;
        bool mirror; // true if king was on kingside (files e-h)
    };

    INLINE static constexpr BucketInfo get_bucket(int king_sq) {
        int file = king_sq % 8;
        int rank = king_sq / 8;
        bool mirror = (file >= 4);
        if (mirror) file = 7 - file;
        int idx = rank * 4 + file;
        if (idx >= 32) idx = 31;
        return {KING_BUCKET[idx], mirror};
    }

    // Calculate HalfKAv2 feature index
    INLINE static constexpr size_t index(int bucket, bool mirror, int piece_type, pos_t piece_sq) {
        int sq = mirror ? (static_cast<int>(piece_sq) ^ 7) : static_cast<int>(piece_sq);
        return static_cast<size_t>(bucket) * (nnue::PIECE_TYPES * board::SIZE + 1)
             + static_cast<size_t>(piece_type) * board::SIZE
             + static_cast<size_t>(sq) + 1;
    }

    using Features = HalfKP::Features;

    template<typename BoardT>
    static Features get_features(const BoardT& board, bool flip = false) {
        Features f;
        f.stm = (board.activePlayer() == WHITE) ? 0 : 1;
        if (flip) f.stm = 1 - f.stm;

        const pos_t wk_raw = board.pos_king[WHITE];
        const pos_t bk_raw = board.pos_king[BLACK];

        // White perspective: orient white king, get bucket
        auto [w_bucket, w_mirror] = get_bucket(static_cast<int>(wk_raw));
        // Black perspective: orient black king (flip vertically first)
        auto [b_bucket, b_mirror] = get_bucket(63 - static_cast<int>(bk_raw));

        const piece_bitboard_t occupied = board.bits[WHITE] | board.bits[BLACK];
        const piece_bitboard_t king_bits = board.get_king_bits();

        bitmask::foreach(occupied & ~king_bits, [&](pos_t sq) {
            const auto p = board[sq];
            const int wpt = HalfKP::get_piece_type(p.value, p.color, true);
            const int bpt = HalfKP::get_piece_type(p.value, p.color, false);
            if (wpt >= 0 && bpt >= 0 && f.count < Features::MAX_FEATURES) {
                int32_t wi = static_cast<int32_t>(index(w_bucket, w_mirror, wpt, sq));
                int32_t bi = static_cast<int32_t>(index(b_bucket, b_mirror, bpt, 63 - sq));
                if (flip) {
                    f.white[f.count] = bi;
                    f.black[f.count] = wi;
                } else {
                    f.white[f.count] = wi;
                    f.black[f.count] = bi;
                }
                ++f.count;
            }
        });
        return f;
    }

    // Extract features directly from compressed FEN bytes (no Board/FeatureBoard)
    static Features get_features_compressed(const uint8_t* data, size_t length, bool flip = false) {
        Features f;
        COLOR active; pos_t wk, bk;
        foreach_compressed_fen(data, length, active, wk, bk,
            [&](pos_t sq, PIECE piece, COLOR color) {
                auto [w_bucket, w_mirror] = get_bucket(static_cast<int>(wk));
                auto [b_bucket, b_mirror] = get_bucket(63 - static_cast<int>(bk));
                const int wpt = HalfKP::get_piece_type(piece, color, true);
                const int bpt = HalfKP::get_piece_type(piece, color, false);
                if (wpt >= 0 && bpt >= 0 && f.count < Features::MAX_FEATURES) {
                    int32_t wi = static_cast<int32_t>(index(w_bucket, w_mirror, wpt, sq));
                    int32_t bi = static_cast<int32_t>(index(b_bucket, b_mirror, bpt, 63 - sq));
                    if (flip) {
                        f.white[f.count] = bi;
                        f.black[f.count] = wi;
                    } else {
                        f.white[f.count] = wi;
                        f.black[f.count] = bi;
                    }
                    ++f.count;
                }
            });
        f.stm = (active == WHITE) ? 0 : 1;
        if (flip) f.stm = 1 - f.stm;
        return f;
    }
};

// ============================================================================
// Accumulator - Stores incremental feature transformer state
// ============================================================================

struct alignas(CACHE_LINE) Accumulator {
    alignas(CACHE_LINE) int16_t values[NO_COLORS][FT_OUT];
    bool computed[NO_COLORS];
    
    Accumulator() {
        computed[0] = computed[1] = false;
    }
    
    void clear() {
        computed[0] = computed[1] = false;
    }
};

// ============================================================================
// Delta - Represents changes to accumulator from a move
// ============================================================================

struct AccumulatorDelta {
    static constexpr size_t MAX_CHANGES = 4;  // Max features changed per move per perspective
    
    size_t removed[MAX_CHANGES];
    size_t added[MAX_CHANGES];
    uint8_t n_removed;
    uint8_t n_added;
    
    AccumulatorDelta() : n_removed(0), n_added(0) {}
    
    void add_removed(size_t idx) {
        assert(n_removed < MAX_CHANGES);
        removed[n_removed++] = idx;
    }
    
    void add_added(size_t idx) {
        assert(n_added < MAX_CHANGES);
        added[n_added++] = idx;
    }
    
    void clear() {
        n_removed = 0;
        n_added = 0;
    }
};

// ============================================================================
// Network Weights
// ============================================================================

struct alignas(CACHE_LINE) NetworkWeights {
    // Feature transformer (largest component ~21MB)
    alignas(CACHE_LINE) int16_t ft_biases[FT_OUT];
    alignas(CACHE_LINE) int16_t ft_weights[HALFKP_SIZE * FT_OUT];
    
    // Hidden layers
    alignas(CACHE_LINE) int32_t l1_biases[L1_OUT];
    alignas(CACHE_LINE) int8_t l1_weights[L1_IN * L1_OUT];
    
    alignas(CACHE_LINE) int32_t l2_biases[L2_OUT];
    alignas(CACHE_LINE) int8_t l2_weights[L2_IN * L2_OUT];
    
    // Output layer
    alignas(CACHE_LINE) int32_t out_bias;
    alignas(CACHE_LINE) int8_t out_weights[OUT_IN];
};

// ============================================================================
// SIMD Operations
// ============================================================================

namespace simd {

// Add feature vector to accumulator
INLINE void add_feature(int16_t* acc, const int16_t* weights) {
#if defined(NNUE_USE_AVX512)
    for (size_t i = 0; i < FT_OUT; i += 32) {
        __m512i a = _mm512_load_si512(reinterpret_cast<const __m512i*>(acc + i));
        __m512i w = _mm512_load_si512(reinterpret_cast<const __m512i*>(weights + i));
        _mm512_store_si512(reinterpret_cast<__m512i*>(acc + i), _mm512_add_epi16(a, w));
    }
#elif defined(NNUE_USE_AVX2)
    for (size_t i = 0; i < FT_OUT; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, w));
    }
#elif defined(NNUE_USE_SSE42)
    for (size_t i = 0; i < FT_OUT; i += 8) {
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(acc + i));
        __m128i w = _mm_load_si128(reinterpret_cast<const __m128i*>(weights + i));
        _mm_store_si128(reinterpret_cast<__m128i*>(acc + i), _mm_add_epi16(a, w));
    }
#elif defined(NNUE_USE_NEON)
    for (size_t i = 0; i < FT_OUT; i += 8) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t w = vld1q_s16(weights + i);
        vst1q_s16(acc + i, vaddq_s16(a, w));
    }
#else
    for (size_t i = 0; i < FT_OUT; ++i) {
        acc[i] += weights[i];
    }
#endif
}

// Subtract feature vector from accumulator
INLINE void sub_feature(int16_t* acc, const int16_t* weights) {
#if defined(NNUE_USE_AVX512)
    for (size_t i = 0; i < FT_OUT; i += 32) {
        __m512i a = _mm512_load_si512(reinterpret_cast<const __m512i*>(acc + i));
        __m512i w = _mm512_load_si512(reinterpret_cast<const __m512i*>(weights + i));
        _mm512_store_si512(reinterpret_cast<__m512i*>(acc + i), _mm512_sub_epi16(a, w));
    }
#elif defined(NNUE_USE_AVX2)
    for (size_t i = 0; i < FT_OUT; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, w));
    }
#elif defined(NNUE_USE_SSE42)
    for (size_t i = 0; i < FT_OUT; i += 8) {
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(acc + i));
        __m128i w = _mm_load_si128(reinterpret_cast<const __m128i*>(weights + i));
        _mm_store_si128(reinterpret_cast<__m128i*>(acc + i), _mm_sub_epi16(a, w));
    }
#elif defined(NNUE_USE_NEON)
    for (size_t i = 0; i < FT_OUT; i += 8) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t w = vld1q_s16(weights + i);
        vst1q_s16(acc + i, vsubq_s16(a, w));
    }
#else
    for (size_t i = 0; i < FT_OUT; ++i) {
        acc[i] -= weights[i];
    }
#endif
}

// Copy accumulator
INLINE void copy_accumulator(int16_t* dst, const int16_t* src) {
#if defined(NNUE_USE_AVX512)
    for (size_t i = 0; i < FT_OUT; i += 32) {
        __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i*>(src + i));
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + i), v);
    }
#elif defined(NNUE_USE_AVX2)
    for (size_t i = 0; i < FT_OUT; i += 16) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
#elif defined(NNUE_USE_SSE42)
    for (size_t i = 0; i < FT_OUT; i += 8) {
        __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_store_si128(reinterpret_cast<__m128i*>(dst + i), v);
    }
#else
    std::memcpy(dst, src, FT_OUT * sizeof(int16_t));
#endif
}

// ClippedReLU: int16 -> int8, clamped to [0, 127]
INLINE void clipped_relu(const int16_t* input, int8_t* output, size_t size) {
#if defined(NNUE_USE_AVX512)
    const __m512i zero = _mm512_setzero_si512();
    const __m512i max_val = _mm512_set1_epi8(127);
    for (size_t i = 0; i < size; i += 64) {
        __m512i in0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(input + i));
        __m512i in1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(input + i + 32));
        // Pack with saturation and permute to fix lane ordering
        __m512i packed = _mm512_packs_epi16(in0, in1);
        // Clamp to [0, 127]
        packed = _mm512_max_epi8(packed, zero);
        packed = _mm512_min_epi8(packed, max_val);
        // Fix interleaving from packs
        packed = _mm512_permutexvar_epi64(_mm512_setr_epi64(0,2,4,6,1,3,5,7), packed);
        _mm512_store_si512(reinterpret_cast<__m512i*>(output + i), packed);
    }
#elif defined(NNUE_USE_AVX2)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max_val = _mm256_set1_epi8(127);
    for (size_t i = 0; i < size; i += 32) {
        __m256i in0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));
        __m256i in1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i + 16));
        // Pack int16 to int8 with saturation
        __m256i packed = _mm256_packs_epi16(in0, in1);
        // Fix lane crossing from packs (0,1,4,5,2,3,6,7 -> 0,1,2,3,4,5,6,7)
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        // Clamp to [0, 127]
        packed = _mm256_max_epi8(packed, zero);
        packed = _mm256_min_epi8(packed, max_val);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), packed);
    }
#elif defined(NNUE_USE_SSE42)
    const __m128i zero = _mm_setzero_si128();
    const __m128i max_val = _mm_set1_epi8(127);
    for (size_t i = 0; i < size; i += 16) {
        __m128i in0 = _mm_load_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i in1 = _mm_load_si128(reinterpret_cast<const __m128i*>(input + i + 8));
        __m128i packed = _mm_packs_epi16(in0, in1);
        packed = _mm_max_epi8(packed, zero);
        packed = _mm_min_epi8(packed, max_val);
        _mm_store_si128(reinterpret_cast<__m128i*>(output + i), packed);
    }
#elif defined(NNUE_USE_NEON)
    for (size_t i = 0; i < size; i += 16) {
        int16x8_t in0 = vld1q_s16(input + i);
        int16x8_t in1 = vld1q_s16(input + i + 8);
        // Narrow with saturation
        int8x8_t lo = vqmovn_s16(in0);
        int8x8_t hi = vqmovn_s16(in1);
        int8x16_t packed = vcombine_s8(lo, hi);
        // Clamp to [0, 127]
        packed = vmaxq_s8(packed, vdupq_n_s8(0));
        packed = vminq_s8(packed, vdupq_n_s8(127));
        vst1q_s8(reinterpret_cast<int8_t*>(output + i), packed);
    }
#else
    for (size_t i = 0; i < size; ++i) {
        int16_t v = input[i];
        output[i] = static_cast<int8_t>(std::clamp<int16_t>(v, 0, 127));
    }
#endif
}

// Dense layer: int8 input, int8 weights, int32 output (before activation)
INLINE int32_t dot_product_i8(const int8_t* a, const int8_t* b, size_t size) {
#if defined(NNUE_USE_AVX512) && defined(__AVX512VNNI__)
    __m512i sum = _mm512_setzero_si512();
    for (size_t i = 0; i < size; i += 64) {
        __m512i va = _mm512_load_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_load_si512(reinterpret_cast<const __m512i*>(b + i));
        // VNNI: multiply and accumulate
        sum = _mm512_dpbusd_epi32(sum, va, vb);
    }
    return _mm512_reduce_add_epi32(sum);
#elif defined(NNUE_USE_AVX2)
    __m256i sum = _mm256_setzero_si256();
    for (size_t i = 0; i < size; i += 32) {
        __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i));
        // maddubs: treats first arg as unsigned, second as signed
        // We need signed*signed, so we use a workaround
        __m256i prod = _mm256_maddubs_epi16(_mm256_abs_epi8(va), vb);
        __m256i sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), va);
        __m256i neg_prod = _mm256_maddubs_epi16(_mm256_abs_epi8(va), _mm256_sign_epi8(vb, va));
        // Actually simpler approach: use madd with sign correction
        prod = _mm256_madd_epi16(neg_prod, _mm256_set1_epi16(1));
        sum = _mm256_add_epi32(sum, prod);
    }
    // Horizontal sum
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
#elif defined(NNUE_USE_SSE42)
    __m128i sum = _mm_setzero_si128();
    for (size_t i = 0; i < size; i += 16) {
        __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i));
        // Sign-aware multiply
        __m128i prod = _mm_maddubs_epi16(_mm_abs_epi8(va), _mm_sign_epi8(vb, va));
        prod = _mm_madd_epi16(prod, _mm_set1_epi16(1));
        sum = _mm_add_epi32(sum, prod);
    }
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
#else
    int32_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return sum;
#endif
}

} // namespace simd

// ============================================================================
// NNUE Evaluator
// ============================================================================

class Evaluator {
public:
    // Network weights (shared across threads via pointer)
    const NetworkWeights* weights_;
    
    // Per-thread accumulator stack
    Accumulator accumulator_stack_[MAX_PLY];
    int accumulator_sp_;
    
    // Temporary buffers for forward pass
    alignas(CACHE_LINE) int8_t ft_output_[L1_IN];
    alignas(CACHE_LINE) int8_t l1_output_[L1_OUT];
    alignas(CACHE_LINE) int8_t l2_output_[L2_OUT];
    
    Evaluator() : weights_(nullptr), accumulator_sp_(0) {}
    
    void set_weights(const NetworkWeights* w) {
        weights_ = w;
    }
    
    // Reset accumulator stack (call at root of search)
    void reset() {
        accumulator_sp_ = 0;
        accumulator_stack_[0].clear();
    }
    
    // Get current accumulator
    Accumulator& current_accumulator() {
        return accumulator_stack_[accumulator_sp_];
    }
    
    // Push accumulator (on make_move)
    void push_accumulator() {
        assert(accumulator_sp_ < MAX_PLY - 1);
        ++accumulator_sp_;
        // Copy from parent
        auto& parent = accumulator_stack_[accumulator_sp_ - 1];
        auto& child = accumulator_stack_[accumulator_sp_];
        for (int c = 0; c < NO_COLORS; ++c) {
            if (parent.computed[c]) {
                simd::copy_accumulator(child.values[c], parent.values[c]);
                child.computed[c] = true;
            } else {
                child.computed[c] = false;
            }
        }
    }
    
    // Pop accumulator (on unmake_move)
    void pop_accumulator() {
        assert(accumulator_sp_ > 0);
        --accumulator_sp_;
    }
    
    // Initialize accumulator from board position
    template<typename BoardT>
    void refresh_accumulator(const BoardT& board, COLOR perspective) {
        assert(weights_ != nullptr);
        
        auto& acc = current_accumulator();
        const int c = static_cast<int>(perspective);
        
        // Start with biases
        std::memcpy(acc.values[c], weights_->ft_biases, FT_OUT * sizeof(int16_t));
        
        // Get king position for this perspective
        const bool white_pov = (perspective == WHITE);
        const pos_t king_sq = HalfKP::orient(white_pov, board.pos_king[perspective]);
        
        // Iterate over all pieces (excluding kings)
        const piece_bitboard_t occupied = board.bits[WHITE] | board.bits[BLACK];
        const piece_bitboard_t king_bits = board.get_king_bits();
        
        bitmask::foreach(occupied & ~king_bits, [&](pos_t sq) {
            const Piece p = board[sq];
            const int piece_type = HalfKP::get_piece_type(p.value, p.color, white_pov);
            if (piece_type >= 0) {
                const size_t idx = HalfKP::index(white_pov, king_sq, sq, piece_type);
                simd::add_feature(acc.values[c], &weights_->ft_weights[idx * FT_OUT]);
            }
        });
        
        acc.computed[c] = true;
    }
    
    // Apply incremental update to accumulator
    void apply_delta(COLOR perspective, const AccumulatorDelta& delta) {
        assert(weights_ != nullptr);
        
        auto& acc = current_accumulator();
        const int c = static_cast<int>(perspective);
        
        // Remove old features
        for (uint8_t i = 0; i < delta.n_removed; ++i) {
            simd::sub_feature(acc.values[c], &weights_->ft_weights[delta.removed[i] * FT_OUT]);
        }
        
        // Add new features
        for (uint8_t i = 0; i < delta.n_added; ++i) {
            simd::add_feature(acc.values[c], &weights_->ft_weights[delta.added[i] * FT_OUT]);
        }
    }
    
    // Compute delta for a move (call before making the move)
    template<typename BoardT>
    void compute_delta(const BoardT& board, move_t m, 
                       AccumulatorDelta& delta_stm, AccumulatorDelta& delta_nstm) {
        delta_stm.clear();
        delta_nstm.clear();
        
        const pos_t from = bitmask::first(m);
        const pos_t to = bitmask::second(m) & board::MOVEMASK;
        
        // Handle special cases
        if (from & board::CRAZYHOUSE_DROP) {
            // Crazyhouse drop - only add, no remove
            // ... handle drop
            return;
        }
        
        const COLOR stm = board.activePlayer();
        const COLOR nstm = enemy_of(stm);
        const Piece moving = board[from];
        const Piece captured = board[to];
        
        // King moves require full refresh
        if (moving.value == KING) {
            // Mark perspective as needing refresh
            current_accumulator().computed[static_cast<int>(stm)] = false;
            // For opponent's perspective, moving piece changes (their relative king didn't move)
            // Still need to update if there's a capture
            if (captured.value != EMPTY && captured.value != KING) {
                // Remove captured piece from non-side-to-move perspective
                const bool white_pov_nstm = (nstm == WHITE);
                const pos_t nstm_king_sq = HalfKP::orient(white_pov_nstm, board.pos_king[nstm]);
                const int cap_type = HalfKP::get_piece_type(captured.value, captured.color, white_pov_nstm);
                if (cap_type >= 0) {
                    delta_nstm.add_removed(HalfKP::index(white_pov_nstm, nstm_king_sq, to, cap_type));
                }
            }
            return;
        }
        
        // Regular move - compute deltas for both perspectives
        for (int c = 0; c < NO_COLORS; ++c) {
            const COLOR persp = static_cast<COLOR>(c);
            AccumulatorDelta& delta = (persp == stm) ? delta_stm : delta_nstm;
            
            const bool white_pov = (persp == WHITE);
            const pos_t king_sq = HalfKP::orient(white_pov, board.pos_king[persp]);
            
            // Remove moving piece from old square
            const int moving_type = HalfKP::get_piece_type(moving.value, moving.color, white_pov);
            if (moving_type >= 0) {
                delta.add_removed(HalfKP::index(white_pov, king_sq, from, moving_type));
            }
            
            // Handle capture
            if (captured.value != EMPTY && captured.value != KING) {
                const int cap_type = HalfKP::get_piece_type(captured.value, captured.color, white_pov);
                if (cap_type >= 0) {
                    delta.add_removed(HalfKP::index(white_pov, king_sq, to, cap_type));
                }
            }
            
            // Add moving piece to new square (handle promotion)
            PIECE final_piece = moving.value;
            if (moving.value == PAWN && (board::_y(to) == 0 || board::_y(to) == 7)) {
                final_piece = board::get_promotion_as(bitmask::second(m));
            }
            const int final_type = HalfKP::get_piece_type(final_piece, moving.color, white_pov);
            if (final_type >= 0) {
                delta.add_added(HalfKP::index(white_pov, king_sq, to, final_type));
            }
        }
        
        // Handle en passant
        if (moving.value == PAWN && board::_x(from) != board::_x(to) && captured.value == EMPTY) {
            // En passant capture
            const pos_t ep_sq = to + ((stm == WHITE) ? -8 : 8);
            for (int c = 0; c < NO_COLORS; ++c) {
                const COLOR persp = static_cast<COLOR>(c);
                AccumulatorDelta& delta = (persp == stm) ? delta_stm : delta_nstm;
                
                const bool white_pov = (persp == WHITE);
                const pos_t king_sq = HalfKP::orient(white_pov, board.pos_king[persp]);
                const int pawn_type = HalfKP::get_piece_type(PAWN, nstm, white_pov);
                if (pawn_type >= 0) {
                    delta.add_removed(HalfKP::index(white_pov, king_sq, ep_sq, pawn_type));
                }
            }
        }
        
        // Handle castling rook movement
        // ... (similar pattern for rook in castling)
    }
    
    // Forward pass through network
    int32_t forward(COLOR stm) {
        assert(weights_ != nullptr);
        
        auto& acc = current_accumulator();
        assert(acc.computed[0] && acc.computed[1]);
        
        // ClippedReLU on concatenated accumulators
        // Order: STM perspective first, then NSTM
        const int stm_idx = static_cast<int>(stm);
        const int nstm_idx = 1 - stm_idx;
        
        simd::clipped_relu(acc.values[stm_idx], ft_output_, FT_OUT);
        simd::clipped_relu(acc.values[nstm_idx], ft_output_ + FT_OUT, FT_OUT);
        
        // L1: 512 -> 32
        for (size_t j = 0; j < L1_OUT; ++j) {
            int32_t sum = weights_->l1_biases[j];
            sum += simd::dot_product_i8(ft_output_, &weights_->l1_weights[j * L1_IN], L1_IN);
            // NNUEReLU: max(0, min(127, x/64))
            sum = std::clamp(sum / NET_SCALE, 0, 127);
            l1_output_[j] = static_cast<int8_t>(sum);
        }
        
        // L2: 32 -> 32
        for (size_t j = 0; j < L2_OUT; ++j) {
            int32_t sum = weights_->l2_biases[j];
            sum += simd::dot_product_i8(l1_output_, &weights_->l2_weights[j * L2_IN], L2_IN);
            sum = std::clamp(sum / NET_SCALE, 0, 127);
            l2_output_[j] = static_cast<int8_t>(sum);
        }
        
        // Output: 32 -> 1
        int32_t output = weights_->out_bias;
        output += simd::dot_product_i8(l2_output_, weights_->out_weights, OUT_IN);
        
        return output;
    }
    
    // Convert raw network output to centipawns
    static int32_t to_centipawns(int32_t raw) {
        // Remove quantization scaling and apply SIGMOID_SCALE from training.
        // During training: output = linear(x) * SIGMOID_SCALE, but the export
        // only quantizes linear(x), so we must apply SIGMOID_SCALE here.
        return raw * SIGMOID_SCALE / (FT_SCALE * NET_SCALE);
    }
    
    // Full evaluation (refresh + forward)
    template<typename BoardT>
    int32_t evaluate(const BoardT& board) {
        // Ensure both perspectives are computed
        if (!current_accumulator().computed[WHITE]) {
            refresh_accumulator(board, WHITE);
        }
        if (!current_accumulator().computed[BLACK]) {
            refresh_accumulator(board, BLACK);
        }
        
        int32_t raw = forward(board.activePlayer());
        return to_centipawns(raw);
    }
};

// ============================================================================
// Network File I/O
// ============================================================================

class NetworkLoader {
public:
    static bool load(const char* filename, NetworkWeights& weights) {
        FILE* fp = fopen(filename, "rb");
        if (!fp) return false;
        
        // Read header
        uint32_t version, hash, arch_len;
        if (fread(&version, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
        if (fread(&hash, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
        if (fread(&arch_len, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
        
        // Skip architecture string
        fseek(fp, arch_len, SEEK_CUR);
        
        // Read feature transformer header
        uint32_t ft_header;
        if (fread(&ft_header, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
        
        // Read feature transformer
        if (fread(weights.ft_biases, sizeof(int16_t), FT_OUT, fp) != FT_OUT) { fclose(fp); return false; }
        if (fread(weights.ft_weights, sizeof(int16_t), HALFKP_SIZE * FT_OUT, fp) != HALFKP_SIZE * FT_OUT) { 
            fclose(fp); return false; 
        }
        
        // Read network header
        uint32_t net_header;
        if (fread(&net_header, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return false; }
        
        // Read L1
        if (fread(weights.l1_biases, sizeof(int32_t), L1_OUT, fp) != L1_OUT) { fclose(fp); return false; }
        // Note: file may store in different order, adjust as needed
        if (fread(weights.l1_weights, sizeof(int8_t), L1_IN * L1_OUT, fp) != L1_IN * L1_OUT) { 
            fclose(fp); return false; 
        }
        
        // Read L2
        if (fread(weights.l2_biases, sizeof(int32_t), L2_OUT, fp) != L2_OUT) { fclose(fp); return false; }
        if (fread(weights.l2_weights, sizeof(int8_t), L2_IN * L2_OUT, fp) != L2_IN * L2_OUT) { 
            fclose(fp); return false; 
        }
        
        // Read output
        if (fread(&weights.out_bias, sizeof(int32_t), 1, fp) != 1) { fclose(fp); return false; }
        if (fread(weights.out_weights, sizeof(int8_t), OUT_IN, fp) != OUT_IN) { fclose(fp); return false; }
        
        fclose(fp);
        return true;
    }
    
    static bool save(const char* filename, const NetworkWeights& weights) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) return false;
        
        // Write header
        uint32_t version = 0x7AF32F20;  // NNUE version
        uint32_t hash = 0;
        const char* arch = "Features=HalfKP(Friend)[41024->256x2]->[32->32]->1";
        uint32_t arch_len = strlen(arch);
        
        fwrite(&version, sizeof(uint32_t), 1, fp);
        fwrite(&hash, sizeof(uint32_t), 1, fp);
        fwrite(&arch_len, sizeof(uint32_t), 1, fp);
        fwrite(arch, 1, arch_len, fp);
        
        // Feature transformer
        uint32_t ft_header = 0x5D69D5B9;
        fwrite(&ft_header, sizeof(uint32_t), 1, fp);
        fwrite(weights.ft_biases, sizeof(int16_t), FT_OUT, fp);
        fwrite(weights.ft_weights, sizeof(int16_t), HALFKP_SIZE * FT_OUT, fp);
        
        // Network
        uint32_t net_header = 0x0;
        fwrite(&net_header, sizeof(uint32_t), 1, fp);
        fwrite(weights.l1_biases, sizeof(int32_t), L1_OUT, fp);
        fwrite(weights.l1_weights, sizeof(int8_t), L1_IN * L1_OUT, fp);
        fwrite(weights.l2_biases, sizeof(int32_t), L2_OUT, fp);
        fwrite(weights.l2_weights, sizeof(int8_t), L2_IN * L2_OUT, fp);
        fwrite(&weights.out_bias, sizeof(int32_t), 1, fp);
        fwrite(weights.out_weights, sizeof(int8_t), OUT_IN, fp);
        
        fclose(fp);
        return true;
    }
};

} // namespace nnue
