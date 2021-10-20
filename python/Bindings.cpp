#include <chrono>

#include <boost/python.hpp>
#include <boost/python/call.hpp>

#include <Engine.hpp>

using namespace std::chrono;

struct BoardBindings {
  Engine engine;
  typename Engine::ab_storage_t *engine_ab_storage = nullptr;

  enum Status { ONGOING, DRAW, CHECKMATE };

  BoardBindings():
    engine()
  {}

  BoardBindings(const boost::python::str &fenstring):
    engine(fen::load_from_string(boost::python::extract<std::string>(fenstring)))
  {}

  void make_move(const std::string &s) {
    if(!list_legal_moves().contains(s)) {
      throw std::runtime_error("invalid move <"s + s + ">"s);
    }
    pgn::PGN pgn(engine.self);
    bool flag1, flag2;
    const move_t m = pgn.read_move_with_flags(s, flag1, flag2);
    engine.self.make_move(m);
  }

  void retract_move() {
    engine.self.retract_move();
  }

  boost::python::list list_legal_moves() const {
    boost::python::list legalmoves;
    engine.iter_moves([&](pos_t i, pos_t j) mutable noexcept -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      legalmoves.append(pgn::_move_str(engine.self, m));
    });
    return legalmoves;
  }

  decltype(auto) sample() const {
    return pgn::_move_str(engine.self, engine.get_random_move());
  }

  Status status() const {
    if(engine.is_checkmate()) {
      return Status::CHECKMATE;
    } else if(engine.is_draw()) {
      return Status::DRAW;
    }
    return Status::ONGOING;
  }

  decltype(auto) fen() const {
    return fen::export_as_string(engine.export_as_fen());
  }

  void init_abstorage_scope() {
    if(engine_ab_storage == nullptr) {
      engine_ab_storage = new typename Engine::ab_storage_t(engine.get_zobrist_alphabeta_scope());
    }
  }

  boost::python::tuple get_fixed_depth_move(const boost::python::long_ depth) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const Engine::depth_t d = boost::python::extract<Engine::depth_t>(depth);
    move_t bestmove = engine.start_thinking(d, idstate);
    std::string py_bestmove = pgn::_move_str(engine.self, bestmove);
    double py_score = float(idstate.eval) / Engine::MATERIAL_PAWN;
    return boost::python::make_tuple(py_bestmove, py_score);
  }

  boost::python::tuple start_thinking(boost::python::object &visitor) {
    init_abstorage_scope();
    Engine::iddfs_state idstate;
    const auto start = system_clock::now();
    const move_t bestmove = engine.start_thinking(1000, idstate, [&](bool verbose) mutable -> bool {
      const double time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
      boost::python::dict info;
      info[boost::python::str("depth")] = boost::python::long_(idstate.curdepth);
      info[boost::python::str("pline")] = boost::python::str(pgn::_line_str(engine.self, idstate.pline)).split();
      info[boost::python::str("score")] = double(float(idstate.eval) / Engine::MATERIAL_PAWN);
      info[boost::python::str("time_elapsed")] = time_spent;
      info[boost::python::str("stat_nodes")] = boost::python::long_(engine.nodes_searched);
      info[boost::python::str("stat_nps")] = double(engine.nodes_searched) / (1e-9+time_spent);
      info[boost::python::str("stat_tt_hit_rate")] = double(engine.zb_hit) / double(1e-9+engine.zb_hit + engine.zb_miss);
      info[boost::python::str("stat_tt_hash_full")] = double(engine.zb_occupied) / engine.zobrist_size;
      info[boost::python::str("important")] = bool(verbose);
      return visitor(info);
    }, {});
    std::string py_bestmove = pgn::_move_str(engine.self, bestmove);
    double py_score = float(idstate.eval) / Engine::MATERIAL_PAWN;
    return boost::python::make_tuple(py_bestmove, py_score);
  }

  boost::python::list get_mailbox_repr() const {
    boost::python::list repr;
    for(pos_t i = 0; i < board::LEN; ++i) {
      boost::python::list row;
      for(pos_t j = 0; j < board::LEN; ++j) {
        const pos_t ind = board::_pos(A+j, 1+i);
        row.append(engine[ind].str());
      }
      repr.append(row);
    }
    return repr;
  }

  ~BoardBindings() {
    if(engine_ab_storage != nullptr) {
      delete engine_ab_storage;
    }
  }
};

BOOST_PYTHON_MODULE(dummy_chess) {
  boost::python::enum_<BoardBindings::Status>("Status")
    .value("ONGOING", BoardBindings::Status::ONGOING)
    .value("DRAW", BoardBindings::Status::DRAW)
    .value("CHECKMATE", BoardBindings::Status::CHECKMATE);
  boost::python::class_<BoardBindings>("ChessDummy")
    .def(boost::python::init<boost::python::str>())
    .def("step", &BoardBindings::make_move)
    .def("undo", &BoardBindings::retract_move)
    .add_property("legal_moves", &BoardBindings::list_legal_moves)
    .def("sample", &BoardBindings::sample)
    .add_property("status", &BoardBindings::status)
    .add_property("fen", &BoardBindings::fen)
    .def("get_depth_move", &BoardBindings::get_fixed_depth_move)
    .def("start_thinking", &BoardBindings::start_thinking)
    .def("as_list", &BoardBindings::get_mailbox_repr)
  ;
}