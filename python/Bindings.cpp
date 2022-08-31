#include <chrono>

#include <boost/python.hpp>
#include <boost/python/call.hpp>
#include <boost/python/numpy.hpp>

#define CONST_COPYABLE
#define CONST_IS_EMPTY

#include <Engine.hpp>


using namespace std::chrono;

namespace python = boost::python;
namespace np = python::numpy;


struct MoveBindings {
  const python::long_ m_code;
  const python::str m_algebraic;
  const python::str m_pgn;

  MoveBindings() = delete;

  MoveBindings(Board &board, move_t m):
    m_code(m), m_algebraic(board._move_str(m)), m_pgn(pgn::_move_str(board, m))
  {}

  bool operator==(const MoveBindings &other) {
    return m_code == other.m_code;
  }

  operator move_t() const {
    return python::extract<move_t>(m_code);
  }

  operator std::string() const {
    return python::extract<std::string>(m_pgn);
  }

  operator python::long_() const {
    return m_code;
  }

  operator python::str() const {
    return m_pgn;
  }
};


struct MoveScopeBindings {
  std::shared_ptr<Engine> engine;
  const MoveBindings &m;

  MoveScopeBindings(std::shared_ptr<Engine> &engine, const MoveBindings &m):
    engine(engine), m(m)
  {}

  void open() {
    engine->make_move(m);
  }

  void close(const python::object &type, const python::object &value, const python::object &traceback) {
    engine->retract_move();
  }
};


struct BoardBindings {
  std::shared_ptr<Engine> engine;
  std::shared_ptr<typename Engine::ab_storage_t> engine_ab_storage;

  enum Status { ONGOING, DRAW, CHECKMATE };

  BoardBindings():
    engine(new Engine())
  {
    np::initialize();
  }

  explicit BoardBindings(const python::str &fenstring):
    BoardBindings()
  {
    set_fen(fenstring);
  }

  void set_fen(const python::str &fenstring) {
    const fen::FEN f = fen::load_from_string(python::extract<std::string>(fenstring));
    engine->set_fen(f);
  }

  void make_move(const MoveBindings &py_m) {
    if(!list_legal_moves().contains(py_m)) {
      throw std::runtime_error("invalid move <"s + std::string(py_m) + ">"s);
    }
    engine->make_move(move_t(py_m));
  }

  void retract_move() {
    engine->retract_move();
  }

  MoveScopeBindings mscope(const MoveBindings &py_m) {
    if(!list_legal_moves().contains(py_m)) {
      throw std::runtime_error("invalid move <"s + std::string(py_m) + ">"s);
    }
    return MoveScopeBindings(engine, py_m);
  }

  python::list list_legal_moves() const {
    python::list legalmoves;
    engine->iter_moves([&](pos_t i, pos_t j) mutable noexcept -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      legalmoves.append(MoveBindings(engine->as_board(), m));
    });
    return legalmoves;
  }

  decltype(auto) sample() const {
    return MoveBindings(engine->as_board(), engine->get_random_move());
  }

  Status status() const {
    if(engine->is_checkmate()) {
      return Status::CHECKMATE;
    } else if(engine->is_draw()) {
      return Status::DRAW;
    }
    return Status::ONGOING;
  }

  decltype(auto) fen() const {
    return fen::export_as_string(engine->export_as_fen());
  }

  python::list get_mailbox_repr() const {
    python::list repr;
    for(pos_t i = 0; i < board::LEN; ++i) {
      python::list row;
      for(pos_t j = 0; j < board::LEN; ++j) {
        const pos_t ind = board::_pos(A+j, 1+i);
        row.append((*engine)[ind].str());
      }
      repr.append(row);
    }
    return repr;
  }

  python::long_ size() const {
    return python::long_(engine->state_hist.size());
  }

  python::str operator[](const python::long_ &ind) const {
    return python::str((*engine)[python::extract<pos_t>(ind)].str());
  }

  np::ndarray get_simple_feature_set() const {
    size_t n_features = size_t(NO_COLORS) * size_t(NO_PIECES) * board::SIZE;
    np::ndarray phi = np::zeros(python::make_tuple(n_features), np::dtype::get_builtin<float>());
    size_t start = 0;
    for(COLOR c : {WHITE, BLACK}) {
      for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        bitmask::foreach(engine->as_board().get_mask(Piece(p, c)), [&](pos_t i) mutable noexcept -> void {
          phi[start + i] = 1;
        });
        start += board::SIZE;
      }
    }
    return phi;
  }

  MoveBindings get_move_from_move_t(const python::long_ &action) {
    return MoveBindings(*engine, python::extract<move_t>(action));
  }

  // perft stuff
  python::long_ perft(const python::long_ &depth) {
    Engine::depth_t d = python::extract<Engine::depth_t>(depth);
    return python::long_(engine->perft(d));
  }

  // engine stuff
  void init_abstorage_scope() {
    if(!engine_ab_storage) {
      engine_ab_storage.reset(new typename Engine::ab_storage_t(engine->get_zobrist_alphabeta_scope()));
    }
  }

  inline python::tuple _return_move_score(move_t m, Engine::score_t score) {
    const double py_score = Engine::score_float(score);
    return python::make_tuple(MoveBindings(engine->as_board(), m), py_score);
  }

  python::tuple get_fixed_depth_move(const python::long_ &depth) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const Engine::depth_t d = python::extract<Engine::depth_t>(depth);
    const move_t bestmove = engine->start_thinking(d, idstate);
    return _return_move_score(bestmove, idstate.eval);
  }

  python::dict _make_callback_info(double time_spent, const Engine::iddfs_state &idstate, bool verbose=true) {
      python::dict info;
      info[python::str("depth")] = python::long_(idstate.curdepth);
      info[python::str("pline")] = python::str(pgn::_line_str(engine->as_board(), idstate.pline)).split();
      info[python::str("score")] = double(float(idstate.eval) / Engine::MATERIAL_PAWN);
      info[python::str("time_elapsed")] = time_spent;
      info[python::str("stat_nodes")] = python::long_(engine->nodes_searched);
      info[python::str("stat_nps")] = double(engine->nodes_searched) / (1e-9+time_spent);
      info[python::str("stat_tt_hit_rate")] = double(engine->zb_hit) / double(1e-9+engine->zb_hit + engine->zb_miss);
      info[python::str("stat_tt_hash_full")] = double(engine->zb_occupied) / engine->zobrist_size;
      info[python::str("important")] = bool(verbose);
      return info;
  }

  python::tuple start_thinking(const python::object &visitor) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const auto start = system_clock::now();
    const move_t bestmove = engine->start_thinking(1000, idstate, [&](bool verbose) mutable -> bool {
      if(!verbose)return true;
      const double time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
      return visitor(_make_callback_info(time_spent, idstate, verbose));
    }, {});
    return _return_move_score(bestmove, idstate.eval);
  }

  python::tuple iterate_depths(python::long_ maxdepth, const python::object &visitor) {
    init_abstorage_scope();
    Engine::depth_t dmax = python::extract<Engine::depth_t>(maxdepth);
    Engine::iddfs_state idstate;
    const auto start = system_clock::now();
    const move_t bestmove = engine->start_thinking(1000, idstate, [&](bool verbose) mutable -> bool {
      const double time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
      if(verbose) {
        visitor(_make_callback_info(time_spent, idstate));
        if(idstate.curdepth == dmax) {
          return false;
        }
      }
      return idstate.curdepth < dmax;
    }, {});
    return _return_move_score(bestmove, idstate.eval);
  }

  python::long_ score() const {
    return python::long_(engine->evaluate());
  }

  double evaluate() const {
    return float(engine->evaluate()) / Engine::MATERIAL_PAWN;
  }
};


#include <NNUE.hpp>

struct NNUEBindings {
  std::shared_ptr<nn::halfkp> nnue;
  fen::FEN position = fen::starting_pos;

  NNUEBindings():
    nnue(new nn::halfkp())
  {
    np::initialize();
    nnue->init_halfkp_features(fen::board_view(position));
  }

  explicit NNUEBindings(const python::str &fenstring):
    nnue(new nn::halfkp())
  {
    np::initialize();
    set_fen(fenstring);
  }

  void set_fen(const python::str &fenstring) {
    position = fen::load_from_string(python::extract<std::string>(fenstring));
    nnue->init_halfkp_features(fen::board_view(position));
  }

  python::list halfkp_features() const {
    python::list result;
    using T = typename std::remove_reference_t<decltype(nnue->input_indices[0])>::value_type;
    for(size_t c = 0; c < NO_COLORS; ++c) {
      const auto &indices = nnue->input_indices[c];
      np::ndarray a = np::from_data(indices.data(), np::dtype::get_builtin<size_t>(),
                                    python::make_tuple(indices.size()),
                                    python::make_tuple(sizeof(size_t)),
                                    python::object());
      result.append(a);
    }
    return result;
  }

  template <typename T, size_t N>
  static np::ndarray _make_ndarray(const python::tuple &shape, const std::array<T, N> &arr) {
    python::list strides(shape);
    strides.pop(0);
    strides.append(sizeof(T));
    return np::from_data(arr.data(), np::dtype::get_builtin<T>(), python::list(shape), strides, python::object());
  }

  template <typename AffineT>
  static python::list _get_parameter_list_affine(const AffineT &affine) {
    python::list parameters;
    parameters.append(_make_ndarray(python::make_tuple(AffineT::in_shape, AffineT::out_shape), affine.A));
    parameters.append(_make_ndarray(python::make_tuple(AffineT::out_shape), affine.b));
    parameters.append(_make_ndarray(python::make_tuple(AffineT::out_shape), affine.y));
    return parameters;
  }


  python::list get_layers() const {
    python::list layers;
    layers.append(_get_parameter_list_affine(nnue->model.feature_transformer.affine));
    layers.append(_get_parameter_list_affine(nnue->model.hidden1.affine));
    layers.append(_get_parameter_list_affine(nnue->model.hidden2.affine));
    layers.append(_get_parameter_list_affine(nnue->model.output));
    return layers;
  }

  float evaluate() const {
    const float eval = nn::halfkp::value_to_centipawn(nnue->init_forward_pass());
    return eval;
  }

  void load(const std::string &filename) {
    nnue->model.load(filename);
  }

  void save(const std::string &filename) {
    nnue->model.save(filename);
  }
};


BOOST_PYTHON_MODULE(dummy_chess) {
  python::enum_<BoardBindings::Status>("Status")
    .value("ONGOING", BoardBindings::Status::ONGOING)
    .value("DRAW", BoardBindings::Status::DRAW)
    .value("CHECKMATE", BoardBindings::Status::CHECKMATE)
  ;
  python::class_<MoveBindings>("Move", python::init<Board &, move_t>())
    .def_readonly("m_code", &MoveBindings::m_code)
    .def_readonly("m_algebraic", &MoveBindings::m_algebraic)
    .def_readonly("m_pgn", &MoveBindings::m_pgn)
    .def("__repr__", &MoveBindings::operator python::str)
    .def("__int__", &MoveBindings::operator python::long_)
    .def("__eq__", &MoveBindings::operator==)
  ;
  python::class_<MoveScopeBindings>("MoveScope", python::init<std::shared_ptr<Engine> &, const MoveBindings &>())
    .def("__enter__", &MoveScopeBindings::open)
    .def("__exit__", &MoveScopeBindings::close)
  ;
  python::class_<BoardBindings>("ChessDummy")
    .def(python::init<const python::str &>())
    .def("set_fen", &BoardBindings::set_fen)
    .add_property("status", &BoardBindings::status)
    .add_property("legal_moves", &BoardBindings::list_legal_moves)
    .add_property("fen", &BoardBindings::fen)
    .def("__len__", &BoardBindings::size)
    .def("as_list", &BoardBindings::get_mailbox_repr)
    .def("__getitem__", &BoardBindings::operator[])
    .def("sample", &BoardBindings::sample)
    .def("step", &BoardBindings::make_move)
    .def("undo", &BoardBindings::retract_move)
    .def("step_scope", &BoardBindings::mscope)
    .def("perft", &BoardBindings::perft)
    .def("get_depth_move", &BoardBindings::get_fixed_depth_move)
    .def("start_thinking", &BoardBindings::start_thinking)
    .def("iterate_depths", &BoardBindings::iterate_depths)
    .def("score", &BoardBindings::score)
    .def("evaluate", &BoardBindings::evaluate)
    .def("get_simple_features", &BoardBindings::get_simple_feature_set)
    .def("get_move_from_move_t", &BoardBindings::get_move_from_move_t)
  ;
  python::class_<NNUEBindings>("NNUEDummy")
    .def(python::init<const python::str &>())
    .add_property("halfkp", &NNUEBindings::halfkp_features)
    .add_property("layers", &NNUEBindings::get_layers)
    .def("set_fen", &NNUEBindings::set_fen)
    .def("evaluate", &NNUEBindings::evaluate)
    .def("load", &NNUEBindings::load)
    .def("save", &NNUEBindings::save)
  ;
}
