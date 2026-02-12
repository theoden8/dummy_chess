#include <chrono>
#include <regex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Engine.hpp>
#include <PGN.hpp>

namespace py = pybind11;
using namespace std::chrono;

struct MoveBindings {
  const int64_t m_code;
  const std::string m_algebraic;
  const std::string m_pgn;

  MoveBindings() = delete;

  MoveBindings(Board &board, move_t m):
    m_code(m), m_algebraic(board._move_str(m)), m_pgn(pgn::_move_str(board, m))
  {}

  bool operator==(const MoveBindings &other) const {
    return m_code == other.m_code;
  }

  operator move_t() const {
    return move_t(m_code);
  }

  operator std::string() const {
    return m_pgn;
  }

  std::string repr() const {
    return m_pgn;
  }
};


struct MoveScopeBindings {
  Engine &engine;
  const MoveBindings &m;

  MoveScopeBindings(Engine &engine, const MoveBindings &m):
    engine(engine), m(m)
  {}

  void enter() {
    engine.make_move(m);
  }

  void exit(py::object type, py::object value, py::object traceback) {
    engine.retract_move();
  }
};


class BoardBindings {
public:
  enum Status { ONGOING, DRAW, CHECKMATE };

  BoardBindings():
    engine()
  {}

  explicit BoardBindings(const std::string &fenstring):
    engine(fen::load_from_string(fenstring))
  {}

  void make_move(const MoveBindings &py_m) {
    if (!contains_move(list_legal_moves(), py_m)) {
      throw std::runtime_error("invalid move <" + std::string(py_m.m_pgn) + ">");
    }
    engine.make_move(move_t(py_m));
  }

  void retract_move() {
    engine.retract_move();
  }

  void set_fen(const std::string &fenstring) {
    engine.set_fen(fen::load_from_string(fenstring));
  }

  MoveScopeBindings mscope(const MoveBindings &py_m) {
    if(!contains_move(list_legal_moves(), py_m)) {
      throw std::runtime_error("invalid move <" + std::string(py_m.m_pgn) + ">");
    }
    return MoveScopeBindings(engine, py_m);
  }

  std::vector<MoveBindings> list_legal_moves() {
    std::vector<MoveBindings> legalmoves;
    engine.iter_moves([&](pos_t i, pos_t j) mutable noexcept -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      legalmoves.emplace_back(engine.as_board(), m);
    });
    return legalmoves;
  }

  decltype(auto) sample() {
    return MoveBindings(engine.as_board(), engine.get_random_move());
  }

  Status status() const {
    if(engine.is_checkmate()) {
      return Status::CHECKMATE;
    } else if(engine.is_draw()) {
      return Status::DRAW;
    }
    return Status::ONGOING;
  }

  std::string fen() const {
    return fen::export_as_string(engine.export_as_fen());
  }

  std::vector<std::vector<std::string>> get_mailbox_repr() const {
    std::vector<std::vector<std::string>> repr;
    repr.reserve(board::LEN);
    for(pos_t i = 0; i < board::LEN; ++i) {
      std::vector<std::string> row;
      row.reserve(board::LEN);
      for(pos_t j = 0; j < board::LEN; ++j) {
        const pos_t ind = board::_pos(A+j, 1+i);
        row.emplace_back(1, engine[ind].str());
      }
      repr.emplace_back(std::move(row));
    }
    return repr;
  }

  size_t size() const {
    return engine.state_hist.size();
  }

  // NOTE operator[]
  std::string get_piece(pos_t ind) const {
    return std::string(1, engine[ind].str());
  }

  py::array_t<float> get_simple_feature_set() const {
    const size_t n_features = size_t(NO_COLORS) * size_t(NO_PIECES) * board::SIZE;
    auto result = py::array_t<float>(n_features);
    auto buf = result.mutable_data();
    std::fill(buf, buf + n_features, 0.0f);
    
    size_t start = 0;
    for(COLOR c : {WHITE, BLACK}) {
      for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        bitmask::foreach(engine.as_board().get_mask(Piece(p, c)), [&](pos_t i) mutable noexcept -> void {
          buf[start + i] = 1.0f;
        });
        start += board::SIZE;
      }
    }
    return result;
  }

  MoveBindings get_move_from_move_t(int64_t action) {
    return MoveBindings(engine.as_board(), move_t(action));
  }

  int64_t perft(int depth) {
    return engine.perft(Engine::depth_t(depth));
  }

  std::tuple<MoveBindings, double> get_fixed_depth_move(int depth) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const Engine::depth_t d = Engine::depth_t(depth);
    move_t bestmove = engine.start_thinking(d, idstate);
    return std::make_tuple(
        MoveBindings(engine.as_board(), bestmove),
        float(idstate.eval) / Engine::MATERIAL_PAWN
    );
  }

  py::dict _make_callback_info(double time_spent, const Engine::iddfs_state &idstate, bool verbose=true) {
    py::dict info;
    info["depth"] = idstate.curdepth;
    info["pline"] = py::cast(pgn::_line_str(engine.as_board(), idstate.pline));
    info["score"] = double(float(idstate.eval) / Engine::MATERIAL_PAWN);
    info["time_elapsed"] = time_spent;
    info["stat_nodes"] = engine.nodes_searched;
    info["stat_nps"] = double(engine.nodes_searched) / (1e-9+time_spent);
    info["stat_tt_hit_rate"] = double(engine.zb_hit) / double(1e-9+engine.zb_hit + engine.zb_miss);
    info["stat_tt_hash_full"] = double(engine.zb_occupied) / engine.zobrist_size;
    info["important"] = bool(verbose);
    return info;
  }

  std::tuple<MoveBindings, double> start_thinking(py::function visitor_func) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const auto start = system_clock::now();
    const move_t bestmove = engine.start_thinking(1000, idstate, [&](bool verbose) mutable -> bool {
      const double time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
      return visitor_func(_make_callback_info(time_spent, idstate, verbose)).cast<bool>();
    }, {});
    return std::make_tuple(
        MoveBindings(engine.as_board(), bestmove),
        float(idstate.eval) / Engine::MATERIAL_PAWN
    );
  }

  std::tuple<MoveBindings, double> iterate_depths(int maxdepth, py::function visitor_func) {
    init_abstorage_scope();
    Engine::depth_t dmax = Engine::depth_t(maxdepth);
    Engine::iddfs_state idstate;
    const auto start = system_clock::now();
    const move_t bestmove = engine.start_thinking(1000, idstate, [&](bool verbose) mutable -> bool {
      const double time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
      if(verbose) {
        visitor_func(_make_callback_info(time_spent, idstate));
        if(idstate.curdepth == dmax) {
          return false;
        }
      }
      return idstate.curdepth < dmax;
    }, {});
    return std::make_tuple(
        MoveBindings(engine.as_board(), bestmove),
        float(idstate.eval) / Engine::MATERIAL_PAWN
    );
  }

    double evaluate() {
        return float(engine.evaluate()) / Engine::MATERIAL_PAWN;
    }

private:
  Engine engine;
  std::shared_ptr<typename Engine::ab_storage_t> engine_ab_storage;

  void init_abstorage_scope() {
    if (!engine_ab_storage) {
      engine_ab_storage.reset(new typename Engine::ab_storage_t(engine.get_zobrist_alphabeta_scope()));
    }
  }

  static bool contains_move(const std::vector<MoveBindings>& moves, const MoveBindings& move) {
    return std::find_if(moves.begin(), moves.end(),
        [&move](const MoveBindings& m) { return m == move; }) != moves.end();
  }
};

// Register preprocessing functions on a module (used for both main module and submodule)
void register_preprocess_functions(py::module_ &m) {
  // FEN compression functions
  m.def("compress_fen", [](const std::string &fenstring) {
    fen::FEN f = fen::load_from_string(fenstring);
    std::vector<uint8_t> compressed = fen::compress::compress_fen(f);
    return py::bytes(reinterpret_cast<const char*>(compressed.data()), compressed.size());
  }, "Compress a FEN string to bytes");

  m.def("decompress_fen", [](const py::bytes &data) {
    char *buffer;
    Py_ssize_t length;
    PyBytes_AsStringAndSize(data.ptr(), &buffer, &length);
    std::vector<uint8_t> v(buffer, buffer + length);
    fen::FEN f = fen::compress::decompress_fen(v);
    return fen::export_as_string(f);
  }, "Decompress bytes to a FEN string");

  m.def("compress_fens_batch", [](const std::vector<std::string> &fens) {
    py::list result;
    for (const auto &fenstring : fens) {
      fen::FEN f = fen::load_from_string(fenstring);
      std::vector<uint8_t> compressed = fen::compress::compress_fen(f);
      result.append(py::bytes(reinterpret_cast<const char*>(compressed.data()), compressed.size()));
    }
    return result;
  }, "Compress a batch of FEN strings to a list of bytes");

  m.def("decompress_fens_batch", [](const std::vector<py::bytes> &data_list) {
    py::list result;
    for (const auto &data : data_list) {
      char *buffer;
      Py_ssize_t length;
      PyBytes_AsStringAndSize(data.ptr(), &buffer, &length);
      std::vector<uint8_t> v(buffer, buffer + length);
      fen::FEN f = fen::compress::decompress_fen(v);
      result.append(fen::export_as_string(f));
    }
    return result;
  }, "Decompress a batch of bytes to a list of FEN strings");

  m.def("get_variant", [](const py::bytes &data) {
    char *buffer;
    Py_ssize_t length;
    PyBytes_AsStringAndSize(data.ptr(), &buffer, &length);
    std::vector<uint8_t> v(buffer, buffer + length);
    fen::FEN f = fen::compress::decompress_fen(v);
    if (f.chess960) return std::string("chess960");
    if (f.crazyhouse) return std::string("crazyhouse");
    return std::string("standard");
  }, "Get variant name from compressed FEN: 'standard', 'chess960', or 'crazyhouse'");

  m.def("get_variant_batch", [](const std::vector<py::bytes> &data_list) {
    py::list result;
    for (const auto &data : data_list) {
      char *buffer;
      Py_ssize_t length;
      PyBytes_AsStringAndSize(data.ptr(), &buffer, &length);
      std::vector<uint8_t> v(buffer, buffer + length);
      fen::FEN f = fen::compress::decompress_fen(v);
      if (f.chess960) result.append("chess960");
      else if (f.crazyhouse) result.append("crazyhouse");
      else result.append("standard");
    }
    return result;
  }, "Get variant names from batch of compressed FENs. Returns list of 'standard', 'chess960', or 'crazyhouse'.");

  // HalfKP feature extraction from compressed FEN bytes
  // Returns (white_indices, white_offsets, black_indices, black_offsets, stm)
  // where indices are flat arrays and offsets mark boundaries for each position
  // If flip=true, swaps white/black perspectives and flips STM (caller should negate scores)
  m.def("get_halfkp_features_batch", [](const std::vector<py::bytes> &data_list, bool flip) {
    const size_t n_positions = data_list.size();
    
    // Pre-allocate with estimated sizes (max 30 pieces per position)
    std::vector<int32_t> white_indices;
    std::vector<int32_t> black_indices;
    std::vector<int64_t> white_offsets(n_positions);
    std::vector<int64_t> black_offsets(n_positions);
    std::vector<int64_t> stm(n_positions);
    
    white_indices.reserve(n_positions * 16);
    black_indices.reserve(n_positions * 16);
    
    for (size_t pos_idx = 0; pos_idx < n_positions; ++pos_idx) {
      // Record offsets
      white_offsets[pos_idx] = static_cast<int64_t>(white_indices.size());
      black_offsets[pos_idx] = static_cast<int64_t>(black_indices.size());
      
      // Decompress FEN
      char *buffer;
      Py_ssize_t length;
      PyBytes_AsStringAndSize(data_list[pos_idx].ptr(), &buffer, &length);
      std::vector<uint8_t> v(buffer, buffer + length);
      fen::FEN f = fen::compress::decompress_fen(v);
      
      // Set side to move (0 = white, 1 = black), flip if requested
      int64_t stm_val = (f.active_player == WHITE) ? 0 : 1;
      stm[pos_idx] = flip ? (1 - stm_val) : stm_val;
      
      // Parse board string to find pieces and king positions
      int wk = -1, bk = -1;
      std::vector<std::tuple<int, int, bool>> pieces; // (square, piece_type, is_white)
      
      int sq = 56; // Start at a8 (rank 8, file a)
      for (char c : f.board) {
        if (sq < 0) break;
        
        int piece_type = -1;
        bool is_white = false;
        
        switch (c) {
          case 'P': piece_type = 0; is_white = true; break;
          case 'N': piece_type = 1; is_white = true; break;
          case 'B': piece_type = 2; is_white = true; break;
          case 'R': piece_type = 3; is_white = true; break;
          case 'Q': piece_type = 4; is_white = true; break;
          case 'K': wk = sq; break;
          case 'p': piece_type = 0; is_white = false; break;
          case 'n': piece_type = 1; is_white = false; break;
          case 'b': piece_type = 2; is_white = false; break;
          case 'r': piece_type = 3; is_white = false; break;
          case 'q': piece_type = 4; is_white = false; break;
          case 'k': bk = sq; break;
          case ' ': break; // empty square
          default: break;
        }
        
        if (piece_type >= 0) {
          pieces.emplace_back(sq, piece_type, is_white);
        }
        
        // Move to next square (row by row, left to right)
        sq++;
        if (sq % 8 == 0) {
          sq -= 16; // Move to start of previous rank
        }
      }
      
      // Compute HalfKP features for each piece
      for (const auto& [piece_sq, pt, is_white] : pieces) {
        // White perspective: friend=0, enemy=1
        int w_idx = is_white ? 0 : 1;
        // Black perspective: friend=0, enemy=1 (inverted)
        int b_idx = is_white ? 1 : 0;
        
        // Feature index formula: king_sq * 641 + piece_index * 64 + piece_sq + 1
        // piece_index = piece_type * 2 + color_offset
        int32_t white_feat = wk * 641 + (pt * 2 + w_idx) * 64 + piece_sq + 1;
        int32_t black_feat = (63 - bk) * 641 + (pt * 2 + b_idx) * 64 + (63 - piece_sq) + 1;
        
        if (flip) {
          // Swap white and black features
          white_indices.push_back(black_feat);
          black_indices.push_back(white_feat);
        } else {
          white_indices.push_back(white_feat);
          black_indices.push_back(black_feat);
        }
      }
    }
    
    // Convert to numpy arrays
    auto white_idx_arr = py::array_t<int32_t>(white_indices.size());
    auto black_idx_arr = py::array_t<int32_t>(black_indices.size());
    auto white_off_arr = py::array_t<int64_t>(n_positions);
    auto black_off_arr = py::array_t<int64_t>(n_positions);
    auto stm_arr = py::array_t<int64_t>(n_positions);
    
    std::copy(white_indices.begin(), white_indices.end(), white_idx_arr.mutable_data());
    std::copy(black_indices.begin(), black_indices.end(), black_idx_arr.mutable_data());
    std::copy(white_offsets.begin(), white_offsets.end(), white_off_arr.mutable_data());
    std::copy(black_offsets.begin(), black_offsets.end(), black_off_arr.mutable_data());
    std::copy(stm.begin(), stm.end(), stm_arr.mutable_data());
    
    return py::make_tuple(white_idx_arr, white_off_arr, black_idx_arr, black_off_arr, stm_arr);
  }, py::arg("data_list"), py::arg("flip") = false,
  "Extract HalfKP features from batch of compressed FENs. Returns (w_idx, w_off, b_idx, b_off, stm). If flip=true, swaps perspectives (caller should negate scores)");

  // HalfKAv2 feature extraction
  // Uses king buckets (12 buckets) + horizontal mirroring to reduce feature space
  // and improve generalization. Buckets group strategically similar king positions.
  //
  // King bucket layout (from white's perspective, mirrored for black):
  //   Files a-d are mirrored to e-h (horizontal symmetry)
  //   Buckets based on king safety zones:
  //     0-3: back rank (a1-d1 -> buckets by file groups)
  //     4-7: second rank  
  //     8-11: ranks 3-8 (less granular, king rarely there in middlegame)
  //
  // Feature index = bucket * 640 + piece_type * 2 * 64 + piece_color * 64 + piece_sq + 1
  // Total features = 12 * 640 + 1 = 7681

  m.def("get_halfkav2_features_batch", [](const std::vector<py::bytes> &data_list, bool flip) {
    // King bucket table: maps king square (0-63) to bucket (0-11)
    // For files e-h, we mirror to a-d first, so this table is for files a-d only
    // Layout prioritizes back ranks where kings usually are
    static constexpr int KING_BUCKET[32] = {
      // Rank 1 (a1-d1): buckets 0-3 (most granular - castled king positions)
      0, 1, 2, 3,
      // Rank 2 (a2-d2): buckets 4-5 (king moved up one)
      4, 4, 5, 5,
      // Rank 3 (a3-d3): buckets 6-7
      6, 6, 7, 7,
      // Rank 4 (a4-d4): bucket 8
      8, 8, 8, 8,
      // Rank 5 (a5-d5): bucket 9
      9, 9, 9, 9,
      // Rank 6 (a6-d6): bucket 10
      10, 10, 10, 10,
      // Ranks 7-8 (a7-d8): bucket 11 (rare positions)
      11, 11, 11, 11,
    };
    
    const size_t n_positions = data_list.size();
    
    std::vector<int32_t> white_indices;
    std::vector<int32_t> black_indices;
    std::vector<int64_t> white_offsets(n_positions);
    std::vector<int64_t> black_offsets(n_positions);
    std::vector<int64_t> stm(n_positions);
    
    white_indices.reserve(n_positions * 16);
    black_indices.reserve(n_positions * 16);
    
    auto get_bucket_and_mirror = [](int king_sq) -> std::pair<int, bool> {
      int file = king_sq % 8;
      int rank = king_sq / 8;
      bool mirror = (file >= 4);
      if (mirror) {
        file = 7 - file;  // Mirror e-h to d-a
      }
      // Index into bucket table: rank * 4 + file (for files a-d)
      int bucket_idx = rank * 4 + file;
      if (bucket_idx >= 32) bucket_idx = 31;  // Safety clamp
      return {KING_BUCKET[bucket_idx], mirror};
    };
    
    for (size_t pos_idx = 0; pos_idx < n_positions; ++pos_idx) {
      white_offsets[pos_idx] = static_cast<int64_t>(white_indices.size());
      black_offsets[pos_idx] = static_cast<int64_t>(black_indices.size());
      
      // Decompress FEN
      char *buffer;
      Py_ssize_t length;
      PyBytes_AsStringAndSize(data_list[pos_idx].ptr(), &buffer, &length);
      std::vector<uint8_t> v(buffer, buffer + length);
      fen::FEN f = fen::compress::decompress_fen(v);
      
      int64_t stm_val = (f.active_player == WHITE) ? 0 : 1;
      stm[pos_idx] = flip ? (1 - stm_val) : stm_val;
      
      // Parse board
      int wk = -1, bk = -1;
      std::vector<std::tuple<int, int, bool>> pieces;
      
      int sq = 56;
      for (char c : f.board) {
        if (sq < 0) break;
        
        int piece_type = -1;
        bool is_white = false;
        
        switch (c) {
          case 'P': piece_type = 0; is_white = true; break;
          case 'N': piece_type = 1; is_white = true; break;
          case 'B': piece_type = 2; is_white = true; break;
          case 'R': piece_type = 3; is_white = true; break;
          case 'Q': piece_type = 4; is_white = true; break;
          case 'K': wk = sq; break;
          case 'p': piece_type = 0; is_white = false; break;
          case 'n': piece_type = 1; is_white = false; break;
          case 'b': piece_type = 2; is_white = false; break;
          case 'r': piece_type = 3; is_white = false; break;
          case 'q': piece_type = 4; is_white = false; break;
          case 'k': bk = sq; break;
          default: break;
        }
        
        if (piece_type >= 0) {
          pieces.emplace_back(sq, piece_type, is_white);
        }
        
        sq++;
        if (sq % 8 == 0) {
          sq -= 16;
        }
      }
      
      // Get king buckets and mirror flags
      auto [w_bucket, w_mirror] = get_bucket_and_mirror(wk);
      auto [b_bucket, b_mirror] = get_bucket_and_mirror(63 - bk);  // Flip for black's perspective
      
      // Compute HalfKAv2 features for each piece
      for (const auto& [piece_sq, pt, is_white] : pieces) {
        int w_idx = is_white ? 0 : 1;
        int b_idx = is_white ? 1 : 0;
        
        // Apply horizontal mirroring if king is on kingside
        int w_piece_sq = w_mirror ? (piece_sq ^ 7) : piece_sq;  // XOR with 7 mirrors file
        int b_piece_sq_flipped = 63 - piece_sq;  // Vertical flip for black
        int b_piece_sq = b_mirror ? (b_piece_sq_flipped ^ 7) : b_piece_sq_flipped;
        
        // Feature index: bucket * 640 + piece_index * 64 + piece_sq + 1
        int32_t white_feat = w_bucket * 641 + (pt * 2 + w_idx) * 64 + w_piece_sq + 1;
        int32_t black_feat = b_bucket * 641 + (pt * 2 + b_idx) * 64 + b_piece_sq + 1;
        
        if (flip) {
          white_indices.push_back(black_feat);
          black_indices.push_back(white_feat);
        } else {
          white_indices.push_back(white_feat);
          black_indices.push_back(black_feat);
        }
      }
    }
    
    // Convert to numpy arrays
    auto white_idx_arr = py::array_t<int32_t>(white_indices.size());
    auto black_idx_arr = py::array_t<int32_t>(black_indices.size());
    auto white_off_arr = py::array_t<int64_t>(n_positions);
    auto black_off_arr = py::array_t<int64_t>(n_positions);
    auto stm_arr = py::array_t<int64_t>(n_positions);
    
    std::copy(white_indices.begin(), white_indices.end(), white_idx_arr.mutable_data());
    std::copy(black_indices.begin(), black_indices.end(), black_idx_arr.mutable_data());
    std::copy(white_offsets.begin(), white_offsets.end(), white_off_arr.mutable_data());
    std::copy(black_offsets.begin(), black_offsets.end(), black_off_arr.mutable_data());
    std::copy(stm.begin(), stm.end(), stm_arr.mutable_data());
    
    return py::make_tuple(white_idx_arr, white_off_arr, black_idx_arr, black_off_arr, stm_arr);
  }, py::arg("data_list"), py::arg("flip") = false,
  "Extract HalfKAv2 features from batch of compressed FENs. Uses 12 king buckets + horizontal mirroring. Returns (w_idx, w_off, b_idx, b_off, stm).");

  // Export constants for Python
  m.attr("HALFKP_SIZE") = 64 * 641 + 1;   // 41025
  m.attr("HALFKAV2_SIZE") = 12 * 641 + 1; // 7693

  // Parse [%eval X.XX] or [%eval #N] from comment string
  // Returns score in centipawns, or nullopt if not found
  auto parse_eval_from_comment = [](const std::string &comment) -> std::optional<int> {
    // Find [%eval ...]
    size_t pos = comment.find("[%eval ");
    if (pos == std::string::npos) return std::nullopt;
    
    size_t start = pos + 7;
    size_t end = comment.find(']', start);
    if (end == std::string::npos) return std::nullopt;
    
    std::string eval_str = comment.substr(start, end - start);
    
    // Trim whitespace
    while (!eval_str.empty() && isspace(eval_str.front())) eval_str.erase(0, 1);
    while (!eval_str.empty() && isspace(eval_str.back())) eval_str.pop_back();
    
    if (eval_str.empty()) return std::nullopt;
    
    // Handle mate scores like "#5" or "#-3"
    if (eval_str[0] == '#') {
      const char* mate_start = eval_str.c_str() + 1;
      char* mate_end = nullptr;
      long mate_in = std::strtol(mate_start, &mate_end, 10);
      if (mate_end == mate_start) return std::nullopt;  // No digits parsed
      return (mate_in > 0) ? 10000 : -10000;
    }
    
    // Handle centipawn scores (Lichess gives in pawns like "1.23")
    const char* score_start = eval_str.c_str();
    char* score_end = nullptr;
    double score = std::strtod(score_start, &score_end);
    if (score_end == score_start) return std::nullopt;  // No digits parsed
    return static_cast<int>(score * 100);
  };

  // Parse a single PGN game and extract positions with [%eval] annotations
  // Returns list of (compressed_fen, score_cp, ply_index) tuples
  // Returns empty list for invalid/corrupted PGNs (unless strict=true, then asserts)
  m.def("parse_pgn_with_evals", [parse_eval_from_comment](const std::string &pgn_text, bool strict=false) {
    py::list results;
    
    Board board;
    pgn::PGN pgn(board);
    
    if (!pgn.read(pgn_text, true, strict)) {  // true = store comments
      // Invalid PGN (corrupted game) - return empty list
      return results;
    }
    
    // pgn.read() has played all moves, board is at final position
    // Walk backwards to collect positions with evals
    // We need (ply_index, compressed_fen, score) for positions BEFORE each move
    
    size_t num_moves = pgn.ply.size();
    if (num_moves == 0) {
      return results;
    }
    
    // Collect results in reverse order by retracting moves
    std::vector<std::tuple<int, std::vector<uint8_t>, int>> reversed_results;
    
    for (size_t i = num_moves; i > 0; --i) {
      size_t idx = i - 1;
      
      // Retract the move to get position BEFORE this move
      board.retract_move();
      
      // Check if this move has an eval comment
      if (idx < pgn.comments.size() && !pgn.comments[idx].empty()) {
        auto score = parse_eval_from_comment(pgn.comments[idx]);
        if (score.has_value()) {
          fen::FEN f = board.export_as_fen();
          std::vector<uint8_t> compressed = fen::compress::compress_fen(f);
          reversed_results.emplace_back(static_cast<int>(idx), std::move(compressed), score.value());
        }
      }
    }
    
    // Reverse to get chronological order
    for (auto it = reversed_results.rbegin(); it != reversed_results.rend(); ++it) {
      results.append(py::make_tuple(
        py::bytes(reinterpret_cast<const char*>(std::get<1>(*it).data()), std::get<1>(*it).size()),
        std::get<2>(*it),
        std::get<0>(*it)
      ));
    }
    
    return results;
  }, py::arg("pgn_text"), py::arg("strict") = false,
  "Parse a PGN game and extract positions with [%eval] annotations. "
  "Returns list of (compressed_fen, score_cp, ply_index) tuples. "
  "If strict=True, asserts on invalid PGNs instead of returning empty list.");

}

PYBIND11_MODULE(_dummychess, m) {
  // Create preprocess submodule
  py::module_ preprocess = m.def_submodule("preprocess", "Preprocessing utilities for NNUE training");
  register_preprocess_functions(preprocess);
  
  // Also register on main module for backward compatibility
  register_preprocess_functions(m);

  py::enum_<BoardBindings::Status>(m, "Status")
    .value("ONGOING", BoardBindings::Status::ONGOING)
    .value("DRAW", BoardBindings::Status::DRAW)
    .value("CHECKMATE", BoardBindings::Status::CHECKMATE);

  py::class_<MoveBindings>(m, "Move")
    .def(py::init<Board &, move_t>())
    .def_readonly("m_code", &MoveBindings::m_code)
    .def_readonly("m_algebraic", &MoveBindings::m_algebraic)
    .def_readonly("m_pgn", &MoveBindings::m_pgn)
    .def("__repr__", &MoveBindings::repr)
    .def("__int__", [](const MoveBindings &m)
      { return m.m_code; })
    .def("__eq__", &MoveBindings::operator==);

  py::class_<MoveScopeBindings>(m, "MoveScope")
    .def(py::init<Engine &, const MoveBindings &>())
    .def("__enter__", &MoveScopeBindings::enter)
    .def("__exit__", &MoveScopeBindings::exit);

  py::class_<BoardBindings>(m, "ChessDummy")
    .def(py::init<>())
    .def(py::init<const std::string &>())
    .def_property_readonly("status", &BoardBindings::status)
    .def_property_readonly("legal_moves", &BoardBindings::list_legal_moves)
    .def_property_readonly("fen", &BoardBindings::fen)
    .def("__len__", &BoardBindings::size)
    .def("as_list", &BoardBindings::get_mailbox_repr)
    .def("__getitem__", &BoardBindings::get_piece)
    .def("sample", &BoardBindings::sample)
    .def("step", &BoardBindings::make_move)
    .def("undo", &BoardBindings::retract_move)
    .def("set_fen", &BoardBindings::set_fen)
    .def("step_scope", &BoardBindings::mscope)
    .def("perft", &BoardBindings::perft)
    .def("get_depth_move", &BoardBindings::get_fixed_depth_move)
    .def("start_thinking", &BoardBindings::start_thinking)
    .def("iterate_depths", &BoardBindings::iterate_depths)
    .def("evaluate", &BoardBindings::evaluate)
    .def("get_simple_features", &BoardBindings::get_simple_feature_set)
    .def("get_move_from_move_t", &BoardBindings::get_move_from_move_t);
}
