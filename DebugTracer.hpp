#pragma once


#include <MoveLine.hpp>
#include <PGN.hpp>


template <typename EngineT>
struct DebugTracer {
  using board_info = typename EngineT::board_info;
  using tt_ab_entry = typename EngineT::tt_ab_entry;

  EngineT &engine;
  bool show_everything = false;
  bool exact_line = false;
  MoveLine debug_moveline = MoveLine(std::vector<move_t>{
  });
  board_info debug_board_info;
  int16_t debug_depth = 0;

  DebugTracer(EngineT &engine):
    engine(engine)
  {
    if(!debug_moveline.empty()) {
      debug_board_info = engine.get_info_from_line(debug_moveline);
    } else {
      debug_board_info.unset();
    }
  }

  void set_depth(int16_t depth) {
    debug_depth = depth;
  }

  std::string tab(int16_t depth) const {
    return std::string(debug_depth - depth, ' ');
  }

  std::string actinfo(double alpha, double beta, double bestscore, double score) const {
    if(score >= beta) {
      return "beta-cutoff"s;
    } else if(score > bestscore) {
      return "new-best"s;
    }
    return "ignored"s;
  }

  std::string actinfo_mem(double alpha, double beta, const tt_ab_entry &zb) const {
    if(zb.ndtype == EngineT::TT_CUTOFF) {
      return "tt_cutoff"s;
    } else if(zb.ndtype == EngineT::TT_ALLNODE) {
      return "tt_allnode"s;
    }
    return "tt_exact"s;
  }

  std::string actinfo_standpat(double alpha, double beta, double score, int nchecks) const {
    if(score >= beta) {
      return "q-beta-cutoff"s;
    } else if(nchecks < 0) {
      return "q-checks"s;
    } else if(score > alpha) {
      return "q-new-best"s;
    }
    return "q-ignored"s;
  }

  std::string _line_str_full(const MoveLine &mline) const {
    if(!engine.check_valid_sequence(mline)) {
      return "[invalid "s + engine._line_str_full(mline) + "]"s;
    }
    return pgn::_line_str_full(engine, mline);
  }

  void update_line(int16_t depth, double alpha, double beta, const MoveLine &pline) {
    #ifndef NDEBUG
    if(!exact_line && !debug_moveline.empty() && engine.state.info == debug_board_info && !pline.startswith(debug_moveline)) {
      _printf("%s depth=%d <changed moveline> %s (%.6f, %.6f) from %s\n",
              tab(depth).c_str(), depth, _line_str_full(pline).c_str(), alpha, beta,
              _line_str_full(debug_moveline.as_past()).c_str());
      debug_moveline = pline.get_past();
    }
    #endif
  }

  bool filter_moveline(const MoveLine &pline_alt) {
    if(show_everything || engine.state.info == debug_board_info) {
      return true;
    }
    return !debug_moveline.empty() && (
        (!exact_line && pline_alt.full().startswith(debug_moveline))
        || (exact_line && pline_alt.full() == debug_moveline)
    );
  }

  void update(int16_t depth, double alpha, double beta, double bestscore, double score, move_t m,
              const MoveLine &pline, const MoveLine &pline_alt, std::string extra_inf=""s)
  {
    #ifndef NDEBUG
    if(filter_moveline(pline_alt)) {
      const char ast = (pline_alt.full().startswith(debug_moveline)) ? ' ' : '*';
      _printf("%s%cdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s %s\n",
        tab(depth).c_str(), ast, depth, pgn::_move_str(engine, m).c_str(), score, beta, alpha,
        _line_str_full(pline_alt).c_str(), actinfo(alpha, beta, bestscore, score).c_str(),
        extra_inf.c_str());
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_mem(int16_t depth, double alpha, double beta, const tt_ab_entry &zb, const MoveLine &pline) {
    #ifndef NDEBUG
    MoveLine pline_alt = pline.get_past().as_past();
    pline_alt.replace_line(zb.subpline);
    if(filter_moveline(pline_alt)) {
      _printf("%s depth=%d, %s (%.6f, %.6f) %s -- %s (score=%.6f)\n",
          tab(depth).c_str(), depth, pgn::_move_str(engine, zb.m).c_str(), alpha, beta,
          _line_str_full(pline_alt).c_str(), actinfo_mem(alpha, beta, zb).c_str(),
          zb.score);
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_standpat(int16_t depth, double alpha, double beta, double score, const MoveLine &pline, int nchecks) {
    #ifndef NDEBUG
    if(filter_moveline(pline)) {
      const char ast = (pline.full().startswith(debug_moveline)) ? ' ' : '*';
      _printf("%s%cdepth=%d, score=%.6f (%.6f, %.6f) %s -- %s (standpat)\n",
        tab(depth).c_str(), ast, depth, score, beta, alpha,
        _line_str_full(pline).c_str(), actinfo_standpat(alpha, beta, score, nchecks).c_str());
      update_line(depth, alpha, beta, pline);
    }
    assert(engine.check_valid_sequence(pline));
    #endif
  }

  void check_score(int16_t depth, double score, const MoveLine &pline) {
    #ifndef NDEBUG
    const float adraw = std::abs(-1e-4);
    const float pvscore = engine.get_pvline_score(pline);
    if(!engine.check_pvline_score(pline, score) && std::abs(std::abs(score) - adraw) > 1e-6 && std::abs(std::abs(pvscore) - adraw) > 1e-6) {
      _printf("%s pvline %s\n", tab(depth).c_str(), _line_str_full(pline).c_str());
      _printf("%s score %.6f\n", tab(depth).c_str(), score);
      _printf("%s pvscore %.6f\n", tab(depth).c_str(), pvscore);
      abort();
    }
    assert(engine.check_pvline_score(pline, score) || std::abs(std::abs(score) - adraw) < 1e-6 || std::abs(std::abs(pvscore) - adraw) < 1e-6);
    #endif
  }

  void check_length(int16_t depth, const MoveLine &pline) {
    assert(pline.size() >= size_t(depth) || engine.check_line_terminates(pline));
  }
};
