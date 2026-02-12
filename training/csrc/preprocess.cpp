/**
 * Torch-native feature extraction for NNUE training.
 *
 * Produces torch::Tensor directly from compressed FEN bytes,
 * eliminating numpy intermediary and torch.from_numpy() overhead.
 *
 * Uses a persistent std::thread pool (independent of OpenMP/MKL) to
 * parallelise per-position feature extraction across CPU cores.
 * All extraction + tensor building happens under a single GIL release.
 *
 * Build: via torch.utils.cpp_extension.CppExtension in setup.py
 */

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include <NNUE.hpp>

namespace py = pybind11;

// ============================================================================
// Feature extraction
// ============================================================================

// Chunk size for per-thread extraction.
// 1024 positions × 508 bytes ≈ 0.5 MB, fits comfortably in L2 cache.
static constexpr size_t CHUNK_SIZE = 1024;

// Number of worker threads for parallel extraction.
static int get_n_threads() {
  static int n = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
  return n;
}

// Build torch tensor result tuple from a flat array of Features.
// Returns (w_idx, w_off, b_idx, b_off, stm) as int64 CPU tensors.
static py::tuple build_result_torch(
    const std::vector<nnue::HalfKP::Features>& feats)
{
  const int64_t n = static_cast<int64_t>(feats.size());
  int64_t total = 0;
  for (const auto& f : feats) total += f.count;

  auto opts = torch::TensorOptions().dtype(torch::kInt64);

  auto w_arr     = torch::empty({total}, opts);
  auto b_arr     = torch::empty({total}, opts);
  auto w_off_arr = torch::empty({n}, opts);
  auto b_off_arr = torch::empty({n}, opts);
  auto stm_arr   = torch::empty({n}, opts);

  auto* w       = w_arr.data_ptr<int64_t>();
  auto* b       = b_arr.data_ptr<int64_t>();
  auto* w_off   = w_off_arr.data_ptr<int64_t>();
  auto* b_off   = b_off_arr.data_ptr<int64_t>();
  auto* stm_out = stm_arr.data_ptr<int64_t>();

  int64_t pos = 0;
  for (int64_t i = 0; i < n; ++i) {
    w_off[i] = pos;
    b_off[i] = pos;
    stm_out[i] = feats[i].stm;
    for (int j = 0; j < feats[i].count; ++j) {
      w[pos + j] = feats[i].white[j];
      b[pos + j] = feats[i].black[j];
    }
    pos += feats[i].count;
  }
  return py::make_tuple(w_arr, w_off_arr, b_arr, b_off_arr, stm_arr);
}

// Parallel feature extraction + direct tensor building.
// Everything under a single GIL release: spawns short-lived threads for
// cache-friendly chunks, then builds tensors from the results.
template <typename FeatureFn>
static py::tuple extract_parallel_torch(
    uintptr_t data_ptr, uintptr_t offsets_ptr,
    size_t start, size_t count, bool flip,
    FeatureFn feature_fn)
{
  // For very small batches, skip threading overhead
  if (count <= CHUNK_SIZE) {
    std::vector<nnue::HalfKP::Features> feats(count);
    {
      py::gil_scoped_release nogil;
      const auto* data = reinterpret_cast<const uint8_t*>(data_ptr);
      const auto* offsets = reinterpret_cast<const int64_t*>(offsets_ptr);
      for (size_t i = 0; i < count; ++i) {
        size_t idx = start + i;
        feats[i] = feature_fn(
          data + offsets[idx],
          static_cast<size_t>(offsets[idx + 1] - offsets[idx]),
          flip
        );
      }
    }
    return build_result_torch(feats);
  }

  // --- Parallel path ---
  const size_t n_chunks = (count + CHUNK_SIZE - 1) / CHUNK_SIZE;

  // Each chunk gets its own contiguous Features vector (cache-friendly).
  std::vector<std::vector<nnue::HalfKP::Features>> chunk_feats(n_chunks);
  for (size_t c = 0; c < n_chunks; ++c) {
    size_t chunk_count = std::min(CHUNK_SIZE, count - c * CHUNK_SIZE);
    chunk_feats[c].resize(chunk_count);
  }

  // Per-chunk feature totals for prefix sum
  std::vector<int64_t> chunk_totals(n_chunks, 0);

  {
    py::gil_scoped_release nogil;

    const auto* data = reinterpret_cast<const uint8_t*>(data_ptr);
    const auto* offsets = reinterpret_cast<const int64_t*>(offsets_ptr);

    // Phase 1: parallel extraction using atomic work-stealing
    std::atomic<size_t> next_chunk{0};
    int n_threads = std::min(get_n_threads(), static_cast<int>(n_chunks));

    auto worker = [&]() {
      while (true) {
        size_t c = next_chunk.fetch_add(1, std::memory_order_relaxed);
        if (c >= n_chunks) break;
        size_t chunk_start = c * CHUNK_SIZE;
        size_t chunk_count = chunk_feats[c].size();
        auto& feats = chunk_feats[c];
        int64_t total = 0;
        for (size_t i = 0; i < chunk_count; ++i) {
          size_t idx = start + chunk_start + i;
          feats[i] = feature_fn(
            data + offsets[idx],
            static_cast<size_t>(offsets[idx + 1] - offsets[idx]),
            flip
          );
          total += feats[i].count;
        }
        chunk_totals[c] = total;
      }
    };

    // Spawn n_threads-1 workers, use current thread as well
    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 0; t < n_threads - 1; ++t)
      threads.emplace_back(worker);
    worker();  // Current thread participates
    for (auto& t : threads) t.join();
  }

  // Phase 2: compute prefix sums (needs counts from phase 1)
  int64_t grand_total = 0;
  std::vector<int64_t> chunk_offsets(n_chunks);
  std::vector<int64_t> chunk_pos_offsets(n_chunks);
  {
    int64_t pos_offset = 0;
    for (size_t c = 0; c < n_chunks; ++c) {
      chunk_offsets[c] = grand_total;
      chunk_pos_offsets[c] = pos_offset;
      grand_total += chunk_totals[c];
      pos_offset += static_cast<int64_t>(chunk_feats[c].size());
    }
  }

  // Phase 3: allocate tensors (needs GIL) then fill in parallel
  const int64_t n = static_cast<int64_t>(count);
  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  auto w_arr     = torch::empty({grand_total}, opts);
  auto b_arr     = torch::empty({grand_total}, opts);
  auto w_off_arr = torch::empty({n}, opts);
  auto b_off_arr = torch::empty({n}, opts);
  auto stm_arr   = torch::empty({n}, opts);

  auto* w       = w_arr.data_ptr<int64_t>();
  auto* b       = b_arr.data_ptr<int64_t>();
  auto* w_off   = w_off_arr.data_ptr<int64_t>();
  auto* b_off   = b_off_arr.data_ptr<int64_t>();
  auto* stm_out = stm_arr.data_ptr<int64_t>();

  {
    py::gil_scoped_release nogil;

    std::atomic<size_t> next_chunk2{0};
    int n_threads = std::min(get_n_threads(), static_cast<int>(n_chunks));

    auto filler = [&]() {
      while (true) {
        size_t c = next_chunk2.fetch_add(1, std::memory_order_relaxed);
        if (c >= n_chunks) break;
        const auto& feats = chunk_feats[c];
        int64_t feat_pos = chunk_offsets[c];
        int64_t pos_base = chunk_pos_offsets[c];
        for (size_t i = 0; i < feats.size(); ++i) {
          int64_t global_i = pos_base + static_cast<int64_t>(i);
          w_off[global_i] = feat_pos;
          b_off[global_i] = feat_pos;
          stm_out[global_i] = feats[i].stm;
          for (int j = 0; j < feats[i].count; ++j) {
            w[feat_pos + j] = feats[i].white[j];
            b[feat_pos + j] = feats[i].black[j];
          }
          feat_pos += feats[i].count;
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 0; t < n_threads - 1; ++t)
      threads.emplace_back(filler);
    filler();
    for (auto& t : threads) t.join();
  }

  return py::make_tuple(w_arr, w_off_arr, b_arr, b_off_arr, stm_arr);
}

// Feature extraction functions (file-scope, no lifetime issues)
static nnue::HalfKP::Features halfkp_fn(
    const uint8_t* data, size_t len, bool flip) {
  return nnue::HalfKP::get_features_compressed(data, len, flip);
}

static nnue::HalfKP::Features halfkav2_fn(
    const uint8_t* data, size_t len, bool flip) {
  return nnue::HalfKAv2::get_features_compressed(data, len, flip);
}

PYBIND11_MODULE(_preprocess, m) {
  m.doc() = "Torch-native feature extraction for NNUE training";

  m.def("get_halfkp_features_torch", [](
      uintptr_t data_ptr, uintptr_t offsets_ptr,
      size_t start, size_t count, bool flip) {
    return extract_parallel_torch(data_ptr, offsets_ptr, start, count, flip, halfkp_fn);
  }, py::arg("data_ptr"), py::arg("offsets_ptr"),
  py::arg("start"), py::arg("count"), py::arg("flip") = false,
  "Extract HalfKP features via raw pointers. Returns torch tensors (w_idx, w_off, b_idx, b_off, stm).");

  m.def("get_halfkav2_features_torch", [](
      uintptr_t data_ptr, uintptr_t offsets_ptr,
      size_t start, size_t count, bool flip) {
    return extract_parallel_torch(data_ptr, offsets_ptr, start, count, flip, halfkav2_fn);
  }, py::arg("data_ptr"), py::arg("offsets_ptr"),
  py::arg("start"), py::arg("count"), py::arg("flip") = false,
  "Extract HalfKAv2 features via raw pointers. Returns torch tensors (w_idx, w_off, b_idx, b_off, stm).");

  m.attr("HALFKP_SIZE") = 64 * 641 + 1;   // 41025
  m.attr("HALFKAV2_SIZE") = 12 * 641 + 1;  // 7693
}
