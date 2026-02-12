//! NNUE Training using Burn
//!
//! Architecture: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1
//!
//! Performance optimizations:
//! - Parallel batch loading with rayon
//! - Prefetching batches while GPU trains
//! - WDL sigmoid loss for better convergence
//! - Batched feature extraction
//!
//! Usage:
//!   # Single source
//!   cargo run --release -- data/evals.parquet --epochs 30 --batch-size 8192
//!
//!   # Multiple sources with weights (70% evals, 30% puzzles)
//!   cargo run --release -- data/evals.parquet:0.7 data/puzzles.parquet:0.3 --epochs 30
//!
//!   # Using config file
//!   cargo run --release -- --config train.toml

use arrow::array::AsArray;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Int;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;

use serde::Deserialize;
use std::path::PathBuf;
use std::sync::mpsc;

// ============================================================================
// Memory Debugging (only enabled in debug builds)
// ============================================================================

#[cfg(debug_assertions)]
fn get_memory_usage_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

macro_rules! debug_mem {
    ($msg:expr) => {
        #[cfg(debug_assertions)]
        eprintln!("[DEBUG MEM] {}: {:.1} MB", $msg, get_memory_usage_mb());
    };
}

// ============================================================================
// Architecture Constants (must match NNUE.hpp and train.py)
// ============================================================================

const HALFKP_SIZE: usize = 41024;
const FT_OUT: usize = 256;
const L1_OUT: usize = 32;
const L2_OUT: usize = 32;

/// Sigmoid scaling factor for WDL loss (Stockfish uses 410)
const SIGMOID_SCALE: f32 = 400.0;

// ============================================================================
// Config File Support
// ============================================================================

/// TOML config file format for training
///
/// Example train.toml:
/// ```toml
/// output = "network.nnue"
/// epochs = 30
/// batch_size = 8192
/// lr = 0.001
/// val_ratio = 0.05
/// test_ratio = 0.05
/// flip_augment = false
/// backend = "gpu"
/// prefetch = 4
///
/// [[sources]]
/// path = "data/evals.parquet"
/// weight = 0.7
///
/// [[sources]]
/// path = "data/puzzles.parquet"
/// weight = 0.2
///
/// [[sources]]
/// path = "data/endgames.parquet"
/// weight = 0.1
/// ```
#[derive(Debug, Deserialize)]
struct TrainConfig {
    sources: Vec<SourceConfig>,
    #[serde(default = "default_output")]
    output: PathBuf,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_lr")]
    lr: f64,
    #[serde(default = "default_val_ratio")]
    val_ratio: f64,
    #[serde(default = "default_test_ratio")]
    test_ratio: f64,
    #[serde(default)]
    flip_augment: bool,
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default = "default_prefetch")]
    prefetch: usize,
    #[serde(default)]
    quiet: bool,
}

#[derive(Debug, Deserialize)]
struct SourceConfig {
    path: PathBuf,
    #[serde(default = "default_weight")]
    weight: f64,
    /// Optional: specify row count to avoid scanning the file
    #[serde(default)]
    rows: Option<usize>,
}

fn default_output() -> PathBuf {
    PathBuf::from("network.nnue")
}
fn default_epochs() -> usize {
    30
}
fn default_batch_size() -> usize {
    8192
}
fn default_lr() -> f64 {
    1e-3
}
fn default_val_ratio() -> f64 {
    0.05
}
fn default_test_ratio() -> f64 {
    0.05
}
fn default_backend() -> String {
    "gpu".to_string()
}
fn default_prefetch() -> usize {
    4
}
fn default_weight() -> f64 {
    1.0
}

impl TrainConfig {
    fn load(path: &PathBuf) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse config file: {}", e))
    }

    fn to_sources(&self) -> Vec<DataSource> {
        self.sources
            .iter()
            .map(|s| DataSource {
                path: s.path.clone(),
                weight: s.weight,
                rows: s.rows,
            })
            .collect()
    }

    fn backend_type(&self) -> BackendType {
        match self.backend.to_lowercase().as_str() {
            "cpu" => BackendType::Cpu,
            _ => BackendType::Gpu,
        }
    }
}

// ============================================================================
// Split Configuration
// ============================================================================

/// Configuration for deterministic train/val/test splitting.
/// Data should be pre-shuffled during preprocessing.
#[derive(Clone, Debug)]
struct SplitConfig {
    val_ratio: f64,
    test_ratio: f64,
}

impl SplitConfig {
    fn new(val_ratio: f64, test_ratio: f64) -> Self {
        Self {
            val_ratio,
            test_ratio,
        }
    }

    /// Get start and end indices for a split
    fn get_split_indices(&self, n: usize, split: &str) -> (usize, usize) {
        let n_test = (n as f64 * self.test_ratio) as usize;
        let n_val = (n as f64 * self.val_ratio) as usize;
        let n_train = n - n_val - n_test;

        match split {
            "train" => (0, n_train),
            "val" => (n_train, n_train + n_val),
            "test" => (n_train + n_val, n),
            _ => (0, n),
        }
    }
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            val_ratio: 0.05,
            test_ratio: 0.05,
        }
    }
}

// ============================================================================
// Data Source
// ============================================================================

/// A data source with path, weight, and optional pre-specified row count
#[derive(Clone, Debug)]
struct DataSource {
    path: PathBuf,
    weight: f64,
    /// If specified, skip scanning the file for row count
    rows: Option<usize>,
}

impl DataSource {
    fn parse(s: &str) -> Self {
        if let Some((path, weight)) = s.rsplit_once(':') {
            if let Ok(w) = weight.parse::<f64>() {
                return Self {
                    path: PathBuf::from(path),
                    weight: w,
                    rows: None,
                };
            }
        }
        Self {
            path: PathBuf::from(s),
            weight: 1.0,
            rows: None,
        }
    }
}

// ============================================================================
// HalfKP Feature Extraction
// ============================================================================

/// Piece char to (piece_type 0-4, is_white)
fn piece_info(c: char) -> Option<(usize, bool)> {
    match c {
        'P' => Some((0, true)),
        'N' => Some((1, true)),
        'B' => Some((2, true)),
        'R' => Some((3, true)),
        'Q' => Some((4, true)),
        'p' => Some((0, false)),
        'n' => Some((1, false)),
        'b' => Some((2, false)),
        'r' => Some((3, false)),
        'q' => Some((4, false)),
        _ => None,
    }
}

/// Extract HalfKP features from FEN string
/// Returns (white_features, black_features, side_to_move)
fn get_halfkp_features(fen: &str, flip: bool) -> (Vec<i64>, Vec<i64>, i64) {
    let parts: Vec<&str> = fen.split(' ').collect();
    let board_str = parts[0];
    let mut stm = if parts.len() > 1 && parts[1] == "b" {
        1
    } else {
        0
    };

    let mut wk: i64 = -1;
    let mut bk: i64 = -1;
    let mut pieces: Vec<(i64, usize, bool)> = Vec::new();

    let mut sq: i64 = 56;
    for c in board_str.chars() {
        match c {
            '/' => sq -= 16,
            '1'..='8' => sq += c.to_digit(10).unwrap() as i64,
            'K' => {
                wk = sq;
                sq += 1;
            }
            'k' => {
                bk = sq;
                sq += 1;
            }
            _ => {
                if let Some((pt, is_white)) = piece_info(c) {
                    pieces.push((sq, pt, is_white));
                }
                sq += 1;
            }
        }
    }

    let mut white_feats = Vec::with_capacity(pieces.len());
    let mut black_feats = Vec::with_capacity(pieces.len());

    for (piece_sq, pt, is_white) in pieces {
        let w_idx = if is_white { 0 } else { 1 };
        let b_idx = if is_white { 1 } else { 0 };

        let white_feat = wk * 641 + (pt as i64 * 2 + w_idx) * 64 + piece_sq + 1;
        let black_feat = (63 - bk) * 641 + (pt as i64 * 2 + b_idx) * 64 + (63 - piece_sq) + 1;

        white_feats.push(white_feat);
        black_feats.push(black_feat);
    }

    // If flipping, swap white/black perspectives and flip STM
    if flip {
        stm = 1 - stm;
        (black_feats, white_feats, stm)
    } else {
        (white_feats, black_feats, stm)
    }
}

/// Extract HalfKP features from compressed FEN bytes
fn get_halfkp_features_compressed(data: &[u8], flip: bool) -> (Vec<i64>, Vec<i64>, i64) {
    // Decompress using the format from compress.hpp
    // Format: flags(1) + board(32) + castling(1) + ep(1) + halfmove(1) + fullmove(2)
    if data.is_empty() {
        return (vec![], vec![], 0);
    }

    let flags = data[0];
    let mut stm = if flags & 1 != 0 { 1i64 } else { 0i64 }; // bit 0 = black to move

    // Parse board (32 bytes = 64 nibbles)
    let mut wk: i64 = -1;
    let mut bk: i64 = -1;
    let mut pieces: Vec<(i64, usize, bool)> = Vec::new();

    for sq in 0..64i64 {
        let byte_idx = 1 + (sq as usize / 2);
        if byte_idx >= data.len() {
            break;
        }
        let nibble = if sq % 2 == 0 {
            data[byte_idx] & 0x0F
        } else {
            (data[byte_idx] >> 4) & 0x0F
        };

        // Nibble encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
        match nibble {
            0 => {} // empty
            1..=5 => pieces.push((sq, (nibble - 1) as usize, true)),
            6 => wk = sq,
            7..=11 => pieces.push((sq, (nibble - 7) as usize, false)),
            12 => bk = sq,
            _ => {}
        }
    }

    let mut white_feats = Vec::with_capacity(pieces.len());
    let mut black_feats = Vec::with_capacity(pieces.len());

    for (piece_sq, pt, is_white) in pieces {
        let w_idx = if is_white { 0 } else { 1 };
        let b_idx = if is_white { 1 } else { 0 };

        let white_feat = wk * 641 + (pt as i64 * 2 + w_idx) * 64 + piece_sq + 1;
        let black_feat = (63 - bk) * 641 + (pt as i64 * 2 + b_idx) * 64 + (63 - piece_sq) + 1;

        white_feats.push(white_feat);
        black_feats.push(black_feat);
    }

    if flip {
        stm = 1 - stm;
        (black_feats, white_feats, stm)
    } else {
        (white_feats, black_feats, stm)
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Module, Debug)]
pub struct NNUE<B: Backend> {
    ft: Embedding<B>,
    ft_bias: Param<Tensor<B, 1>>,
    l1: Linear<B>,
    l2: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> NNUE<B> {
    pub fn new(device: &B::Device) -> Self {
        let ft = EmbeddingConfig::new(HALFKP_SIZE, FT_OUT).init(device);
        let ft_bias = Param::from_tensor(Tensor::zeros([FT_OUT], device));
        let l1 = LinearConfig::new(FT_OUT * 2, L1_OUT).init(device);
        let l2 = LinearConfig::new(L1_OUT, L2_OUT).init(device);
        let out = LinearConfig::new(L2_OUT, 1).init(device);

        Self {
            ft,
            ft_bias,
            l1,
            l2,
            out,
        }
    }

    /// Forward pass
    /// w_idx, b_idx: [batch, max_features] padded indices
    /// w_mask, b_mask: [batch, max_features] masks (1.0 for valid, 0.0 for padding)
    /// stm: [batch] side to move (0=white, 1=black)
    pub fn forward(
        &self,
        w_idx: Tensor<B, 2, Int>,
        w_mask: Tensor<B, 2>,
        b_idx: Tensor<B, 2, Int>,
        b_mask: Tensor<B, 2>,
        stm: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        let w_emb = self.ft.forward(w_idx);
        let b_emb = self.ft.forward(b_idx);

        let w_mask = w_mask.unsqueeze_dim(2);
        let b_mask = b_mask.unsqueeze_dim(2);

        // sum_dim returns [batch, 1, FT_OUT], squeeze to [batch, FT_OUT]
        let w_ft: Tensor<B, 2> =
            (w_emb * w_mask).sum_dim(1).squeeze(1) + self.ft_bias.val().unsqueeze_dim(0);
        let b_ft: Tensor<B, 2> =
            (b_emb * b_mask).sum_dim(1).squeeze(1) + self.ft_bias.val().unsqueeze_dim(0);

        let w_ft = w_ft.clamp(0.0, 1.0);
        let b_ft = b_ft.clamp(0.0, 1.0);

        // Concatenate based on side to move
        let stm_f: Tensor<B, 2> = stm.float().unsqueeze_dim(1);
        let one_minus_stm = stm_f.clone().neg().add_scalar(1.0);

        // Broadcast stm [batch, 1] with ft [batch, FT_OUT]
        let first = w_ft.clone() * one_minus_stm.clone() + b_ft.clone() * stm_f.clone();
        let second = b_ft * one_minus_stm + w_ft * stm_f;
        let ft = Tensor::cat(vec![first, second], 1);

        let x = self.l1.forward(ft).clamp(0.0, 1.0);
        let x = self.l2.forward(x).clamp(0.0, 1.0);
        self.out.forward(x)
    }
}

// ============================================================================
// Data Loading
// ============================================================================

#[derive(Clone)]
struct Sample {
    w_feats: Vec<i64>,
    b_feats: Vec<i64>,
    stm: i64,
    score: f32,
}

/// Source info with computed split boundaries
struct SourceInfo {
    source: DataSource,
    start_idx: usize,
    end_idx: usize,
}

/// Streaming batch iterator - reads one batch at a time using slice+collect
struct BatchIterator {
    source_infos: Vec<SourceInfo>,
    batch_size: usize,
    flip_augment: bool,
    current_source: usize,
    current_offset: usize,
}

impl BatchIterator {
    fn new(
        sources: Vec<DataSource>,
        split: &str,
        split_config: &SplitConfig,
        batch_size: usize,
        flip_augment: bool,
    ) -> Self {
        // Get row counts and compute boundaries
        let source_infos: Vec<SourceInfo> = sources
            .into_iter()
            .map(|src| {
                // Use pre-specified row count or read from parquet metadata (cheap)
                let n = src
                    .rows
                    .unwrap_or_else(|| Self::get_row_count_from_metadata(&src.path));
                let (start, end) = split_config.get_split_indices(n, split);
                SourceInfo {
                    source: src,
                    start_idx: start,
                    end_idx: end,
                }
            })
            .collect();

        // Apply weight-based limiting
        let source_infos = Self::apply_weights(source_infos);

        Self {
            source_infos,
            batch_size,
            flip_augment,
            current_source: 0,
            current_offset: 0,
        }
    }

    /// Read row count from parquet file metadata (very cheap, no data loading)
    fn get_row_count_from_metadata(path: &PathBuf) -> usize {
        debug_mem!("get_row_count_from_metadata: opening file");
        let file = std::fs::File::open(path).expect("Failed to open parquet file");
        let reader = SerializedFileReader::new(file).expect("Failed to read parquet");
        let metadata = reader.metadata();
        let row_groups = metadata.row_groups();
        let row_count: i64 = row_groups.iter().map(|rg| rg.num_rows()).sum();
        debug_mem!("get_row_count_from_metadata: done");
        row_count as usize
    }

    fn apply_weights(mut infos: Vec<SourceInfo>) -> Vec<SourceInfo> {
        if infos.is_empty() {
            return infos;
        }

        let total_weight: f64 = infos.iter().map(|s| s.source.weight).sum();
        let norm_weights: Vec<f64> = infos
            .iter()
            .map(|s| s.source.weight / total_weight)
            .collect();

        let split_lens: Vec<usize> = infos.iter().map(|s| s.end_idx - s.start_idx).collect();

        // Find max total that respects strict ratio
        let max_total = split_lens
            .iter()
            .zip(&norm_weights)
            .map(|(&avail, &w)| {
                if w > 0.0 {
                    avail as f64 / w
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::INFINITY, f64::min);

        // Apply limits
        for (i, info) in infos.iter_mut().enumerate() {
            let limit = (max_total * norm_weights[i]) as usize;
            info.end_idx = info.end_idx.min(info.start_idx + limit);
        }

        infos
    }

    fn total_samples(&self) -> usize {
        let base: usize = self
            .source_infos
            .iter()
            .map(|s| s.end_idx - s.start_idx)
            .sum();
        if self.flip_augment {
            base * 2
        } else {
            base
        }
    }

    fn total_batches(&self) -> usize {
        (self.total_samples() + self.batch_size - 1) / self.batch_size
    }

    /// Load next batch of samples using parquet crate directly (true streaming)
    fn next_batch(&mut self) -> Option<Vec<Sample>> {
        while self.current_source < self.source_infos.len() {
            let info = &self.source_infos[self.current_source];
            let abs_offset = info.start_idx + self.current_offset;

            if abs_offset >= info.end_idx {
                self.current_source += 1;
                self.current_offset = 0;
                continue;
            }

            let remaining = info.end_idx - abs_offset;
            let batch_len = self.batch_size.min(remaining);

            debug_mem!(format!(
                "Before read_batch (batch {}, offset {}, len {})",
                self.current_offset / self.batch_size,
                abs_offset,
                batch_len
            ));

            let samples = self.read_batch_direct(&info.source.path, abs_offset, batch_len);

            debug_mem!(format!("After read_batch ({} samples)", samples.len()));

            self.current_offset += batch_len;
            return Some(samples);
        }
        None
    }

    /// Read batch using Arrow columnar reader - fast column projection
    /// Uses rayon for parallel feature extraction
    fn read_batch_direct(&self, path: &PathBuf, offset: usize, len: usize) -> Vec<Sample> {
        let file = std::fs::File::open(path).expect("Failed to open parquet file");

        // Build Arrow reader with column projection (only fen and score)
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to create reader builder");

        let metadata = builder.metadata().clone();

        // Find which row groups contain our data
        let mut row_groups_to_read = Vec::new();
        let mut cumulative_rows = 0usize;

        for (rg_idx, rg_meta) in metadata.row_groups().iter().enumerate() {
            let rg_rows = rg_meta.num_rows() as usize;
            let rg_start = cumulative_rows;
            let rg_end = cumulative_rows + rg_rows;

            if rg_end > offset && rg_start < offset + len {
                row_groups_to_read.push(rg_idx);
            }

            cumulative_rows += rg_rows;
            if cumulative_rows >= offset + len {
                break;
            }
        }

        if row_groups_to_read.is_empty() {
            eprintln!(
                "WARNING: No row groups to read for offset={}, len={}",
                offset, len
            );
            return Vec::new();
        }

        // Reopen file for the reader (builder consumes file)
        let file = std::fs::File::open(path).expect("Failed to reopen parquet file");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to create reader builder");

        // Get column indices from the schema
        let schema = builder.schema().clone();
        let fen_idx = schema.index_of("fen").expect("No fen column");
        let score_idx = schema.index_of("score").expect("No score column");

        // Build reader selecting only needed row groups
        let reader = builder
            .with_row_groups(row_groups_to_read)
            .with_batch_size(len.max(1024))
            .build()
            .expect("Failed to build reader");

        // Calculate how many rows to skip within selected row groups
        let mut skip_in_first_rg = 0usize;
        let mut cumulative = 0usize;
        for rg_meta in metadata.row_groups().iter() {
            let rg_rows = rg_meta.num_rows() as usize;
            if cumulative + rg_rows > offset {
                skip_in_first_rg = offset - cumulative;
                break;
            }
            cumulative += rg_rows;
        }
        let rows_to_skip = skip_in_first_rg;

        // First pass: collect raw data (FEN bytes/strings and scores)
        let mut raw_data: Vec<(RawFen, f32)> = Vec::with_capacity(len);
        let mut rows_skipped = 0usize;

        for batch_result in reader {
            let batch = batch_result.expect("Failed to read batch");
            let batch_len = batch.num_rows();

            let fen_col = batch.column(fen_idx);
            let score_col = batch.column(score_idx);

            use arrow::datatypes::DataType;

            for i in 0..batch_len {
                if rows_skipped < rows_to_skip {
                    rows_skipped += 1;
                    continue;
                }

                if raw_data.len() >= len {
                    break;
                }

                let score = match score_col.data_type() {
                    DataType::Int64 => score_col
                        .as_primitive::<arrow::datatypes::Int64Type>()
                        .value(i) as f32,
                    DataType::Int32 => score_col
                        .as_primitive::<arrow::datatypes::Int32Type>()
                        .value(i) as f32,
                    _ => 0.0,
                };

                let raw_fen = match fen_col.data_type() {
                    DataType::BinaryView => {
                        RawFen::Bytes(fen_col.as_binary_view().value(i).to_vec())
                    }
                    DataType::Binary => {
                        RawFen::Bytes(fen_col.as_binary::<i32>().value(i).to_vec())
                    }
                    DataType::Utf8View => {
                        RawFen::String(fen_col.as_string_view().value(i).to_string())
                    }
                    DataType::Utf8 => {
                        RawFen::String(fen_col.as_string::<i32>().value(i).to_string())
                    }
                    _ => continue,
                };

                raw_data.push((raw_fen, score));
            }

            if raw_data.len() >= len {
                break;
            }
        }

        // Second pass: parallel feature extraction using rayon
        let flip_augment = self.flip_augment;
        let samples: Vec<Sample> = raw_data
            .par_iter()
            .flat_map(|(raw_fen, score)| {
                let (w, b, stm) = match raw_fen {
                    RawFen::Bytes(bytes) => get_halfkp_features_compressed(bytes, false),
                    RawFen::String(fen) => get_halfkp_features(fen, false),
                };

                let mut result = vec![Sample {
                    w_feats: w,
                    b_feats: b,
                    stm,
                    score: *score,
                }];

                if flip_augment {
                    let (w_f, b_f, stm_f) = match raw_fen {
                        RawFen::Bytes(bytes) => get_halfkp_features_compressed(bytes, true),
                        RawFen::String(fen) => get_halfkp_features(fen, true),
                    };
                    result.push(Sample {
                        w_feats: w_f,
                        b_feats: b_f,
                        stm: stm_f,
                        score: -*score,
                    });
                }

                result
            })
            .collect();

        samples
    }
}

/// Raw FEN data before feature extraction
enum RawFen {
    Bytes(Vec<u8>),
    String(String),
}

fn collate_batch<B: Backend>(
    samples: &[Sample],
    device: &B::Device,
) -> (
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 1, Int>,
    Tensor<B, 2>,
) {
    let batch_size = samples.len();
    let max_w = samples.iter().map(|s| s.w_feats.len()).max().unwrap_or(1);
    let max_b = samples.iter().map(|s| s.b_feats.len()).max().unwrap_or(1);

    let mut w_idx_data = vec![0i64; batch_size * max_w];
    let mut w_mask_data = vec![0.0f32; batch_size * max_w];
    let mut b_idx_data = vec![0i64; batch_size * max_b];
    let mut b_mask_data = vec![0.0f32; batch_size * max_b];
    let mut stm_data = vec![0i64; batch_size];
    let mut target_data = vec![0.0f32; batch_size];

    for (i, s) in samples.iter().enumerate() {
        for (j, &idx) in s.w_feats.iter().enumerate() {
            w_idx_data[i * max_w + j] = idx;
            w_mask_data[i * max_w + j] = 1.0;
        }
        for (j, &idx) in s.b_feats.iter().enumerate() {
            b_idx_data[i * max_b + j] = idx;
            b_mask_data[i * max_b + j] = 1.0;
        }
        stm_data[i] = s.stm;
        target_data[i] = s.score;
    }

    let w_idx: Tensor<B, 1, Int> = Tensor::from_data(w_idx_data.as_slice(), device);
    let w_idx = w_idx.reshape([batch_size, max_w]);

    let w_mask: Tensor<B, 1> = Tensor::from_data(w_mask_data.as_slice(), device);
    let w_mask = w_mask.reshape([batch_size, max_w]);

    let b_idx: Tensor<B, 1, Int> = Tensor::from_data(b_idx_data.as_slice(), device);
    let b_idx = b_idx.reshape([batch_size, max_b]);

    let b_mask: Tensor<B, 1> = Tensor::from_data(b_mask_data.as_slice(), device);
    let b_mask = b_mask.reshape([batch_size, max_b]);

    let stm: Tensor<B, 1, Int> = Tensor::from_data(stm_data.as_slice(), device);

    let target: Tensor<B, 1> = Tensor::from_data(target_data.as_slice(), device);
    let target = target.reshape([batch_size, 1]);

    (w_idx, w_mask, b_idx, b_mask, stm, target)
}

// ============================================================================
// Loss Functions
// ============================================================================

/// WDL-style loss using sigmoid scaling.
/// Converts both prediction and target to win probabilities using sigmoid,
/// then computes MSE. This naturally handles the wide score range (-15000 to +15000).
fn wdl_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    use burn::tensor::activation::sigmoid;
    let pred_prob = sigmoid(pred / SIGMOID_SCALE);
    let target_prob = sigmoid(target / SIGMOID_SCALE);
    (pred_prob - target_prob).powf_scalar(2.0).mean()
}

// ============================================================================
// Training
// ============================================================================

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendType {
    Cpu,
    Gpu,
}

type GpuBackend = Autodiff<Wgpu>;
type CpuBackend = Autodiff<NdArray>;

/// Prefetching batch iterator that loads batches in a background thread
struct PrefetchingIterator {
    receiver: mpsc::Receiver<Vec<Sample>>,
    _handle: std::thread::JoinHandle<()>,
}

impl PrefetchingIterator {
    fn new(mut batch_iter: BatchIterator, prefetch_count: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(prefetch_count);

        let handle = std::thread::spawn(move || {
            while let Some(batch) = batch_iter.next_batch() {
                if sender.send(batch).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        Self {
            receiver,
            _handle: handle,
        }
    }
}

impl Iterator for PrefetchingIterator {
    type Item = Vec<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

fn train_with_backend<B: burn::tensor::backend::AutodiffBackend>(
    sources: Vec<DataSource>,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    split_config: SplitConfig,
    flip_augment: bool,
    prefetch: usize,
    quiet: bool,
    device: B::Device,
) where
    B::InnerBackend: burn::tensor::backend::Backend<Device = B::Device>,
{
    println!("Device: {:?}", device);
    println!("Prefetch: {} batches", prefetch);
    debug_mem!("Start of train_with_backend");

    // Get total counts first (lightweight metadata read)
    let train_iter_info = BatchIterator::new(
        sources.clone(),
        "train",
        &split_config,
        batch_size,
        flip_augment,
    );
    let val_iter_info = BatchIterator::new(
        sources.clone(),
        "val",
        &split_config,
        batch_size,
        false,
    );

    let n_train_samples = train_iter_info.total_samples();
    let n_val_samples = val_iter_info.total_samples();
    let n_train_batches = train_iter_info.total_batches();
    let n_val_batches = val_iter_info.total_batches();

    println!(
        "Train: {} samples ({} batches), Val: {} samples ({} batches)",
        n_train_samples, n_train_batches, n_val_samples, n_val_batches
    );
    println!("Using WDL loss with scale={}", SIGMOID_SCALE);

    debug_mem!("Before creating model");
    let model: NNUE<B> = NNUE::new(&device);
    debug_mem!("After creating model");

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4)))
        .init();
    debug_mem!("After creating optimizer");

    let total = (n_train_batches + n_val_batches) * epochs;

    // Create progress bar (hidden if quiet)
    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    };

    let mut best_val = f32::INFINITY;
    let mut model = model;

    for epoch in 0..epochs {
        let epoch_start = std::time::Instant::now();

        // Create fresh iterators with prefetching for each epoch
        let train_iter = BatchIterator::new(
            sources.clone(),
            "train",
            &split_config,
            batch_size,
            flip_augment,
        );
        let train_prefetch = PrefetchingIterator::new(train_iter, prefetch);

        // Training - stream batches with prefetching
        let mut train_loss = 0.0f32;
        let mut train_batch_count = 0;
        debug_mem!(format!("Epoch {} start training", epoch + 1));

        for batch in train_prefetch {
            let (w_idx, w_mask, b_idx, b_mask, stm, target) = collate_batch::<B>(&batch, &device);

            // Drop batch early to free memory
            drop(batch);

            let pred = model.forward(w_idx, w_mask, b_idx, b_mask, stm);
            let loss = wdl_loss(pred, target);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            train_loss += loss.into_scalar().elem::<f32>();
            train_batch_count += 1;
            pb.inc(1);
        }
        train_loss /= train_batch_count.max(1) as f32;

        // Validation with prefetching
        let val_iter = BatchIterator::new(
            sources.clone(),
            "val",
            &split_config,
            batch_size,
            false,
        );
        let val_prefetch = PrefetchingIterator::new(val_iter, prefetch);

        let mut val_loss = 0.0f32;
        let mut val_batch_count = 0;
        let model_valid = model.valid();

        for batch in val_prefetch {
            let (w_idx, w_mask, b_idx, b_mask, stm, target) =
                collate_batch::<B::InnerBackend>(&batch, &device);
            let pred = model_valid.forward(w_idx, w_mask, b_idx, b_mask, stm);
            let loss = wdl_loss(pred, target);
            val_loss += loss.into_scalar().elem::<f32>();
            val_batch_count += 1;
            pb.inc(1);
        }
        val_loss /= val_batch_count.max(1) as f32;

        let epoch_elapsed = epoch_start.elapsed();

        if quiet {
            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, time={:.1}s",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                epoch_elapsed.as_secs_f32()
            );
        } else {
            pb.set_message(format!(
                "epoch {}/{} train={:.4} val={:.4}",
                epoch + 1,
                epochs,
                train_loss,
                val_loss
            ));
        }

        if val_loss < best_val {
            best_val = val_loss;
            if !quiet {
                println!("\nNew best val={:.4}, saving to {:?}", val_loss, output);
            }
            // TODO: save model in NNUE format
        }
    }

    pb.finish_with_message("Done");
    println!("Training complete. Best val_loss={:.6}", best_val);
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(about = "Train NNUE network using Burn")]
#[command(long_about = "Train NNUE network using Burn.\n\n\
    Supports multiple data sources with weights:\n\
    \n\
    Single source:\n\
      train data/evals.parquet\n\
    \n\
    Multiple sources with weights (70% evals, 30% puzzles):\n\
      train data/evals.parquet:0.7 data/puzzles.parquet:0.3\n\
    \n\
    Using config file:\n\
      train --config train.toml\n\
    \n\
    Without weights (equal mixing):\n\
      train data/evals.parquet data/puzzles.parquet")]
struct Args {
    /// Input parquet files, optionally with weights (path:weight)
    #[arg(num_args = 0..)]
    data: Vec<String>,

    /// Config file (TOML format) - overrides CLI args
    #[arg(long, short)]
    config: Option<PathBuf>,

    /// Output file
    #[arg(long, default_value = "network.nnue")]
    output: PathBuf,

    /// Number of epochs
    #[arg(long, default_value_t = 30)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Validation ratio
    #[arg(long, default_value_t = 0.05)]
    val_ratio: f64,

    /// Test ratio
    #[arg(long, default_value_t = 0.05)]
    test_ratio: f64,

    /// Augment data by including flipped positions (2x data)
    #[arg(long)]
    flip_augment: bool,

    /// Backend to use for training
    #[arg(long, value_enum, default_value_t = BackendType::Gpu)]
    backend: BackendType,

    /// Number of batches to prefetch in background thread
    #[arg(long, default_value_t = 4)]
    prefetch: usize,

    /// Suppress progress bar, only print epoch summaries
    #[arg(long)]
    quiet: bool,
}

fn main() {
    let args = Args::parse();

    // Load from config file or CLI args
    let (sources, output, epochs, batch_size, lr, val_ratio, test_ratio, flip_augment, backend, prefetch, quiet) =
        if let Some(config_path) = &args.config {
            let config = TrainConfig::load(config_path).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            let backend = config.backend_type();
            (
                config.to_sources(),
                config.output,
                config.epochs,
                config.batch_size,
                config.lr,
                config.val_ratio,
                config.test_ratio,
                config.flip_augment,
                backend,
                config.prefetch,
                config.quiet,
            )
        } else {
            if args.data.is_empty() {
                eprintln!("Error: No data sources specified. Use positional args or --config");
                std::process::exit(1);
            }
            (
                args.data.iter().map(|s| DataSource::parse(s)).collect(),
                args.output,
                args.epochs,
                args.batch_size,
                args.lr,
                args.val_ratio,
                args.test_ratio,
                args.flip_augment,
                args.backend,
                args.prefetch,
                args.quiet,
            )
        };

    println!("NNUE Training (Burn)");
    println!("  Sources:");
    for src in &sources {
        println!("    {:?} (weight={:.2})", src.path, src.weight);
    }
    println!("  Output: {:?}", output);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  LR: {}", lr);
    println!("  Val ratio: {}", val_ratio);
    println!("  Test ratio: {}", test_ratio);
    println!("  Flip augment: {}", flip_augment);
    println!("  Backend: {:?}", backend);
    println!("  Prefetch: {}", prefetch);

    let split_config = SplitConfig::new(val_ratio, test_ratio);

    match backend {
        BackendType::Gpu => {
            train_with_backend::<GpuBackend>(
                sources,
                output,
                epochs,
                batch_size,
                lr,
                split_config,
                flip_augment,
                prefetch,
                quiet,
                WgpuDevice::default(),
            );
        }
        BackendType::Cpu => {
            train_with_backend::<CpuBackend>(
                sources,
                output,
                epochs,
                batch_size,
                lr,
                split_config,
                flip_augment,
                prefetch,
                quiet,
                NdArrayDevice::Cpu,
            );
        }
    }
}
