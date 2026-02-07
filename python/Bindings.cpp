#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Engine.hpp>

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

PYBIND11_MODULE(_dummychess, m) {
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
