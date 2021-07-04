#pragma once


#include <MoveLine.hpp>
#include <PGN.hpp>


template <typename EngineT>
struct DebugTracer {
  using board_info = typename EngineT::board_info;
  using tt_ab_entry = typename EngineT::tt_ab_entry;

  EngineT &engine;
  MoveLine debug_moveline = {
//    bitmask::_pos_pair(G1, H3),
//    bitmask::_pos_pair(D7, D5),
//    bitmask::_pos_pair(D2, D4),
//    bitmask::_pos_pair(D8, D6),
//    bitmask::_pos_pair(D1, D3),
//    bitmask::_pos_pair(C8, H3),
//    bitmask::_pos_pair(D3, H3),
//    bitmask::_pos_pair(D6, B4),
//    bitmask::_pos_pair(B1, D2),
//    bitmask::_pos_pair(B4, D4),
  };
  board_info debug_board_info;
  int16_t debug_depth = 0;

  DebugTracer(EngineT &engine):
    engine(engine)
  {
    debug_board_info = engine.get_info_from_line(debug_moveline);
  }

  std::string tab(int16_t depth) const {
    std::string tab = ""s; for(int16_t i=0; i<debug_depth-depth;++i)tab+=" "s;
    return tab;
  }

  std::string actinfo(double alpha, double beta, double bestscore, double score) const {
    if(score >= beta) {
      return "beta cut-off"s;
    } else if(score > bestscore) {
      return "new-best"s;
    }
    return "ignored"s;
  }

  std::string actinfo_mem(double alpha, double beta, const tt_ab_entry &zb) const {
    std::string actinfo = "mem"s;
    if(zb.lowerbound >= beta) {
      actinfo += "-beta-cutoff"s;
    } else if(zb.upperbound <= alpha) {
      actinfo += "-new-best"s;
    }
    return actinfo;
  }

  void update_line(int16_t depth, double alpha, double beta, const MoveLine &pline) {
    #ifndef NDEBUG
    if(!debug_moveline.empty() && engine.state.info == debug_board_info) {
      _printf("%sdepth=%d changed moveline %s (%.6f, %.6f)\n",
              tab(depth).c_str(), depth, pgn::_line_str_full(engine, pline).c_str(), alpha, beta);
      debug_moveline = pline.get_past();
    }
    #endif
  }

  void update(int16_t depth, double alpha, double beta, double bestscore, double score, move_t m,
              const MoveLine &pline, const MoveLine &pline_alt) {
    #ifndef NDEBUG
    if(!debug_moveline.empty() && pline_alt.full().startswith(debug_moveline)) {
      _printf("%sdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s\n",
        tab(depth).c_str(), depth, pgn::_move_str(engine, m).c_str(), score, beta, alpha,
        pgn::_line_str_full(engine, pline_alt).c_str(), actinfo(alpha, beta, bestscore, score).c_str());
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_mem(int16_t depth, double alpha, double beta, const tt_ab_entry &zb, const MoveLine &pline) {
    #ifndef NDEBUG
    MoveLine pline_alt = pline.branch_from_past();
    pline_alt.replace_line(zb.subpline);
    if(!debug_moveline.empty() && pline.full().startswith(debug_moveline)) {
      _printf("%sdepth=%d, %s (%.6f, %.6f) %s -- %s (%.6f, %.6f)\n",
          tab(depth).c_str(), depth, pgn::_move_str(engine, zb.m).c_str(), alpha, beta,
          pgn::_line_str_full(engine, pline_alt).c_str(), actinfo_mem(alpha, beta, zb).c_str(),
          zb.lowerbound, zb.upperbound);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void check_score(int16_t depth, double score, const MoveLine &pline) {
    #ifndef NDEBUG
    if(!engine.check_pvline_score(pline, score)) {
      str::pdebug(tab(depth), "pvline", pgn::_line_str_full(engine, pline));
      str::pdebug(tab(depth), "score", score);
      str::pdebug(tab(depth), "pvscore", engine.get_pvline_score(pline));
      abort();
    }
    assert(engine.check_pvline_score(pline, score));
    #endif
  }

  void check_length(int16_t depth, const MoveLine &pline) {
    assert(pline.size() >= size_t(depth) || engine.check_line_terminates(pline));
  }
};
