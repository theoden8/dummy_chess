/**
 * CUDA kernel for HalfKP/HalfKAv2 feature extraction.
 *
 * Decompresses compressed FEN bytes and computes NNUE feature indices
 * entirely on the GPU, eliminating the CPU extraction bottleneck.
 *
 * Compressed FEN format (from NNUE.hpp):
 *   Byte 0: flags (bit 0 = side to move, bit 2 = crazyhouse)
 *   Bytes 1..: packed nibbles (high nibble first), encoding pieces
 *     Nibbles 0-5  = white K,Q,R,B,N,P
 *     Nibbles 6-11 = black k,q,r,b,n,p
 *     0xC = empty-square run (next nibble = count-1)
 *     0xF = padding (end of nibbles)
 *   Trailing bytes: castling, en passant, etc. (ignored for features)
 *
 * Board is traversed rank-8 to rank-1, file a-h (standard FEN order).
 * Internal square = (7 - board_idx/8)*8 + board_idx%8
 */

#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

// ============================================================================
// Constants (must match NNUE.hpp)
// ============================================================================

// Piece types: K=0, Q=1, R=2, B=3, N=4, P=5
// piece_to_index[piece_type][0=friendly, 1=enemy]
__constant__ int d_piece_to_index[6][2] = {
    {-1, -1},  // King (excluded)
    {8, 9},    // Queen
    {6, 7},    // Rook
    {4, 5},    // Bishop
    {2, 3},    // Knight
    {0, 1},    // Pawn
};

// HalfKAv2 king bucket table (files a-d only; e-h mirrored)
__constant__ int d_king_bucket[32] = {
    0, 1, 2, 3,       // Rank 1
    4, 4, 5, 5,       // Rank 2
    6, 6, 7, 7,       // Rank 3
    8, 8, 9, 9,       // Rank 4
    8, 8, 9, 9,       // Rank 5
    10, 10, 11, 11,   // Rank 6
    10, 10, 11, 11,   // Rank 7
    10, 10, 11, 11,   // Rank 8
};

// Maximum non-king pieces
static constexpr int MAX_FEATURES = 62;

// ============================================================================
// Device helpers
// ============================================================================

// Orient square for a given perspective
__device__ __forceinline__ int orient_sq(bool is_white, int sq) {
    return is_white ? sq : (63 - sq);
}

// HalfKP feature index: king_sq * 641 + piece_type * 64 + sq + 1
__device__ __forceinline__ int halfkp_index(int king_sq, int piece_type, int sq) {
    return king_sq * 641 + piece_type * 64 + sq + 1;
}

// Get piece type index from perspective
// piece: 0=K, 1=Q, 2=R, 3=B, 4=N, 5=P
// color: 0=white, 1=black
// white_pov: true for white perspective
__device__ __forceinline__ int get_piece_type(int piece, int color, bool white_pov) {
    if (piece == 0) return -1;  // King
    bool same_side = (color == 0) == white_pov;
    return d_piece_to_index[piece][same_side ? 0 : 1];
}

// HalfKAv2: get bucket and mirror flag for a king square
__device__ __forceinline__ void get_bucket(int king_sq, int& bucket, bool& mirror) {
    int file = king_sq & 7;
    mirror = (file >= 4);
    int mapped_sq = mirror ? (king_sq ^ 7) : king_sq;  // Mirror file if kingside
    // mapped_sq is now in files a-d; convert to bucket table index
    int rank = mapped_sq >> 3;
    int mapped_file = mapped_sq & 3;
    bucket = d_king_bucket[rank * 4 + mapped_file];
}

// HalfKAv2 feature index
__device__ __forceinline__ int halfkav2_index(int bucket, bool mirror, int sq, int piece_type) {
    int msq = mirror ? (sq ^ 7) : sq;  // Mirror horizontally if needed
    return bucket * 641 + piece_type * 64 + msq + 1;
}

// ============================================================================
// Kernel: count features per position (pass 1)
// ============================================================================

__global__ void count_features_kernel(
    const uint8_t* __restrict__ data,
    const int64_t* __restrict__ offsets,
    int64_t start,
    int64_t count,
    int32_t* __restrict__ counts)    // [count] output: features per position
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int64_t idx = start + tid;
    const uint8_t* fen = data + offsets[idx];
    // int64_t len = offsets[idx + 1] - offsets[idx];  // not needed for counting

    // Skip byte 0 (flags)
    int byte_pos = 1;
    int nibble_hi = 1;  // Start with high nibble
    int board_idx = 0;
    int n_features = 0;

    while (board_idx < 64) {
        uint8_t byte_val = fen[byte_pos];
        int nibble = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);

        if (!nibble_hi) byte_pos++;
        nibble_hi = !nibble_hi;

        if (nibble <= 11) {
            // Piece: 0-5 = white KQRBNP, 6-11 = black kqrbnp
            int piece = nibble % 6;  // 0=K, 1=Q, 2=R, 3=B, 4=N, 5=P
            if (piece != 0) {  // Non-king
                n_features++;
            }
            board_idx++;
        } else if (nibble == 0xC) {
            // Empty run: next nibble is count-1
            byte_val = fen[byte_pos];
            int run_nib = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);
            if (!nibble_hi) byte_pos++;
            nibble_hi = !nibble_hi;
            board_idx += run_nib + 1;
        } else if (nibble == 0xF) {
            break;  // Padding
        } else {
            board_idx++;  // Unknown nibble, skip
        }
    }

    counts[tid] = n_features;
}

// ============================================================================
// Kernel: extract HalfKP features (pass 2)
// ============================================================================

__global__ void extract_halfkp_kernel(
    const uint8_t* __restrict__ data,
    const int64_t* __restrict__ offsets,
    int64_t start,
    int64_t count,
    bool flip,
    const int64_t* __restrict__ prefix_sums,  // [count] exclusive prefix sum of feature counts
    int64_t* __restrict__ w_idx,     // output: white feature indices
    int64_t* __restrict__ w_off,     // output: per-position offsets into w_idx
    int64_t* __restrict__ b_idx,     // output: black feature indices
    int64_t* __restrict__ b_off,     // output: per-position offsets into b_idx
    int64_t* __restrict__ stm_out)   // output: side to move
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int64_t idx = start + tid;
    const uint8_t* fen = data + offsets[idx];
    uint8_t flags = fen[0];

    // Decode board
    int byte_pos = 1;
    int nibble_hi = 1;
    int board_idx = 0;

    // Temporary storage for pieces (stack-local per thread)
    int piece_sqs[MAX_FEATURES];
    int piece_types[MAX_FEATURES];  // 0=K,1=Q,2=R,3=B,4=N,5=P
    int piece_colors[MAX_FEATURES]; // 0=white, 1=black
    int n_pieces = 0;
    int wk = -1, bk = -1;

    while (board_idx < 64) {
        uint8_t byte_val = fen[byte_pos];
        int nibble = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);

        if (!nibble_hi) byte_pos++;
        nibble_hi = !nibble_hi;

        if (nibble <= 11) {
            int piece = nibble % 6;   // 0=K,1=Q,2=R,3=B,4=N,5=P
            int color = nibble / 6;   // 0=white, 1=black

            // Convert board_idx to internal square
            int sq = (7 - board_idx / 8) * 8 + (board_idx % 8);

            if (piece == 0) {  // King
                if (color == 0) wk = sq;
                else bk = sq;
            } else if (n_pieces < MAX_FEATURES) {
                piece_sqs[n_pieces] = sq;
                piece_types[n_pieces] = piece;
                piece_colors[n_pieces] = color;
                n_pieces++;
            }
            board_idx++;
        } else if (nibble == 0xC) {
            byte_val = fen[byte_pos];
            int run_nib = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);
            if (!nibble_hi) byte_pos++;
            nibble_hi = !nibble_hi;
            board_idx += run_nib + 1;
        } else if (nibble == 0xF) {
            break;
        } else {
            board_idx++;
        }
    }

    // Compute feature indices
    int64_t out_pos = prefix_sums[tid];
    w_off[tid] = out_pos;
    b_off[tid] = out_pos;

    if (wk >= 0 && bk >= 0) {
        int wk_oriented = orient_sq(true, wk);
        int bk_oriented = orient_sq(false, bk);

        for (int i = 0; i < n_pieces; i++) {
            int sq = piece_sqs[i];
            int piece = piece_types[i];
            int color = piece_colors[i];

            int wpt = get_piece_type(piece, color, true);
            int bpt = get_piece_type(piece, color, false);

            if (wpt >= 0 && bpt >= 0) {
                int wi = halfkp_index(wk_oriented, wpt, orient_sq(true, sq));
                int bi = halfkp_index(bk_oriented, bpt, orient_sq(false, sq));

                if (flip) {
                    w_idx[out_pos] = bi;
                    b_idx[out_pos] = wi;
                } else {
                    w_idx[out_pos] = wi;
                    b_idx[out_pos] = bi;
                }
                out_pos++;
            }
        }
    }

    // Side to move
    int64_t stm = (flags & 1) ? 1 : 0;
    if (flip) stm = 1 - stm;
    stm_out[tid] = stm;
}

// ============================================================================
// Kernel: extract HalfKAv2 features (pass 2)
// ============================================================================

__global__ void extract_halfkav2_kernel(
    const uint8_t* __restrict__ data,
    const int64_t* __restrict__ offsets,
    int64_t start,
    int64_t count,
    bool flip,
    const int64_t* __restrict__ prefix_sums,
    int64_t* __restrict__ w_idx,
    int64_t* __restrict__ w_off,
    int64_t* __restrict__ b_idx,
    int64_t* __restrict__ b_off,
    int64_t* __restrict__ stm_out)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int64_t idx = start + tid;
    const uint8_t* fen = data + offsets[idx];
    uint8_t flags = fen[0];

    // Decode board (same as HalfKP)
    int byte_pos = 1;
    int nibble_hi = 1;
    int board_idx = 0;

    int piece_sqs[MAX_FEATURES];
    int piece_types[MAX_FEATURES];
    int piece_colors[MAX_FEATURES];
    int n_pieces = 0;
    int wk = -1, bk = -1;

    while (board_idx < 64) {
        uint8_t byte_val = fen[byte_pos];
        int nibble = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);
        if (!nibble_hi) byte_pos++;
        nibble_hi = !nibble_hi;

        if (nibble <= 11) {
            int piece = nibble % 6;
            int color = nibble / 6;
            int sq = (7 - board_idx / 8) * 8 + (board_idx % 8);
            if (piece == 0) {
                if (color == 0) wk = sq;
                else bk = sq;
            } else if (n_pieces < MAX_FEATURES) {
                piece_sqs[n_pieces] = sq;
                piece_types[n_pieces] = piece;
                piece_colors[n_pieces] = color;
                n_pieces++;
            }
            board_idx++;
        } else if (nibble == 0xC) {
            byte_val = fen[byte_pos];
            int run_nib = nibble_hi ? (byte_val >> 4) : (byte_val & 0xF);
            if (!nibble_hi) byte_pos++;
            nibble_hi = !nibble_hi;
            board_idx += run_nib + 1;
        } else if (nibble == 0xF) {
            break;
        } else {
            board_idx++;
        }
    }

    int64_t out_pos = prefix_sums[tid];
    w_off[tid] = out_pos;
    b_off[tid] = out_pos;

    if (wk >= 0 && bk >= 0) {
        // White perspective
        int w_bucket, b_bucket;
        bool w_mirror, b_mirror;
        get_bucket(orient_sq(true, wk), w_bucket, w_mirror);
        get_bucket(orient_sq(false, bk), b_bucket, b_mirror);

        for (int i = 0; i < n_pieces; i++) {
            int sq = piece_sqs[i];
            int piece = piece_types[i];
            int color = piece_colors[i];

            int wpt = get_piece_type(piece, color, true);
            int bpt = get_piece_type(piece, color, false);

            if (wpt >= 0 && bpt >= 0) {
                int wi = halfkav2_index(w_bucket, w_mirror, orient_sq(true, sq), wpt);
                int bi = halfkav2_index(b_bucket, b_mirror, orient_sq(false, sq), bpt);

                if (flip) {
                    w_idx[out_pos] = bi;
                    b_idx[out_pos] = wi;
                } else {
                    w_idx[out_pos] = wi;
                    b_idx[out_pos] = bi;
                }
                out_pos++;
            }
        }
    }

    int64_t stm = (flags & 1) ? 1 : 0;
    if (flip) stm = 1 - stm;
    stm_out[tid] = stm;
}

// ============================================================================
// Host-side entry points
// ============================================================================

// Extract features on GPU.
// data_gpu: uint8 tensor of all compressed FEN bytes (on GPU)
// offsets_gpu: int64 tensor of byte offsets (on GPU)
// Returns (w_idx, w_off, b_idx, b_off, stm) all on GPU.
template <typename KernelFn>
static py::tuple extract_gpu_impl(
    torch::Tensor data_gpu,
    torch::Tensor offsets_gpu,
    int64_t start,
    int64_t count,
    bool flip,
    KernelFn kernel_fn)
{
    TORCH_CHECK(data_gpu.is_cuda(), "data must be on GPU");
    TORCH_CHECK(offsets_gpu.is_cuda(), "offsets must be on GPU");
    TORCH_CHECK(data_gpu.dtype() == torch::kUInt8, "data must be uint8");
    TORCH_CHECK(offsets_gpu.dtype() == torch::kInt64, "offsets must be int64");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(data_gpu.device());
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(data_gpu.device());

    const uint8_t* data_ptr = data_gpu.data_ptr<uint8_t>();
    const int64_t* off_ptr = offsets_gpu.data_ptr<int64_t>();

    // Pass 1: count features per position
    auto counts = torch::empty({count}, opts_i32);
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        count_features_kernel<<<blocks, threads, 0, stream>>>(
            data_ptr, off_ptr, start, count, counts.data_ptr<int32_t>());
    }

    // Prefix sum (exclusive) to get output offsets
    auto prefix = counts.to(torch::kInt64);
    auto total_tensor = prefix.sum();
    prefix = prefix.cumsum(0) - prefix;  // exclusive prefix sum

    int64_t total = total_tensor.item<int64_t>();

    // Allocate output tensors
    auto w_idx = torch::empty({total}, opts_i64);
    auto b_idx = torch::empty({total}, opts_i64);
    auto w_off = torch::empty({count}, opts_i64);
    auto b_off = torch::empty({count}, opts_i64);
    auto stm   = torch::empty({count}, opts_i64);

    // Pass 2: extract features
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        kernel_fn<<<blocks, threads, 0, stream>>>(
            data_ptr, off_ptr, start, count, flip,
            prefix.data_ptr<int64_t>(),
            w_idx.data_ptr<int64_t>(),
            w_off.data_ptr<int64_t>(),
            b_idx.data_ptr<int64_t>(),
            b_off.data_ptr<int64_t>(),
            stm.data_ptr<int64_t>());
    }

    return py::make_tuple(w_idx, w_off, b_idx, b_off, stm);
}

// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(_extract_cuda, m) {
    m.doc() = "CUDA-accelerated NNUE feature extraction";

    m.def("extract_halfkp_gpu",
        [](torch::Tensor data_gpu, torch::Tensor offsets_gpu,
           int64_t start, int64_t count, bool flip) {
            return extract_gpu_impl(data_gpu, offsets_gpu, start, count, flip,
                                    extract_halfkp_kernel);
        },
        py::arg("data_gpu"), py::arg("offsets_gpu"),
        py::arg("start"), py::arg("count"), py::arg("flip") = false,
        "Extract HalfKP features on GPU from compressed FEN bytes.");

    m.def("extract_halfkav2_gpu",
        [](torch::Tensor data_gpu, torch::Tensor offsets_gpu,
           int64_t start, int64_t count, bool flip) {
            return extract_gpu_impl(data_gpu, offsets_gpu, start, count, flip,
                                    extract_halfkav2_kernel);
        },
        py::arg("data_gpu"), py::arg("offsets_gpu"),
        py::arg("start"), py::arg("count"), py::arg("flip") = false,
        "Extract HalfKAv2 features on GPU from compressed FEN bytes.");
}
