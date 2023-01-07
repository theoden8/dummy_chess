#pragma once


#include <MoveLine.hpp>
#include <PGN.hpp>


template <typename EngineT>
struct DebugTracer {
  using board_info = typename EngineT::board_info;
  using tt_ab_entry = typename EngineT::tt_ab_entry;

  using score_t = typename EngineT::score_t;
  using depth_t = typename EngineT::depth_t;

  EngineT &engine;
  FILE *fp = stdout;
  static constexpr bool show_pv = true;
  static constexpr bool show_zw = true && show_pv;
  static constexpr bool show_q = true && show_zw;
  static constexpr bool show_mem = true;
  static constexpr bool change_line_to_info = false;
  static constexpr bool exact_line = true;
  MoveLine debug_moveline = MoveLine(std::vector<move_t>{
  });
  static constexpr depth_t show_maxdepth = 10;
  board_info debug_board_info;
  depth_t debug_depth = 0;

  explicit DebugTracer(EngineT &engine):
    engine(engine)
  {
    if(!debug_moveline.empty()) {
      debug_board_info = engine.get_info_from_line(debug_moveline);
    } else {
      debug_board_info.unset();
    }
  }

  void set_depth(depth_t depth) {
    debug_depth = depth;
  }

  #ifndef NDEBUG
  std::string tab(depth_t depth) const {
    return std::string(debug_depth - depth, ' ');
  }

  std::string actinfo(score_t alpha, score_t beta, score_t bestscore, score_t score) const {
    if(score >= beta) {
      return "beta-cutoff"s;
    } else if(score > bestscore) {
      return "new-best"s;
    }
    return "ignored"s;
  }

  std::string actinfo_mem(score_t alpha, score_t beta, const tt_ab_entry &zb) const {
    if(zb.ndtype == EngineT::TT_CUTOFF) {
      return "tt_cutoff"s;
    } else if(zb.ndtype == EngineT::TT_ALLNODE) {
      return "tt_allnode"s;
    }
    return "tt_exact"s;
  }

  std::string actinfo_standpat(score_t alpha, score_t beta, score_t score, int nchecks) const {
    if(score >= beta) {
      return "q-beta-cutoff"s;
    } else if(nchecks < 0) {
      return "q-checks"s;
    } else if(score > alpha) {
      return "q-new-best"s;
    }
    return "q-ignored"s;
  }

  std::string _line_str_full(const MoveLine &mline) {
    if(!engine.check_valid_sequence(mline)) {
      str::pdebug("[invalid "s + engine._line_str_full(mline, false) + "]"s);
      str::pdebug("fen", fen::export_as_string(engine.export_as_fen()));
      abort();
    }
    char s[1024];
    snprintf(s, 1024, "%s %s", mline.pgn_full(engine).c_str(), mline.tb?"TB":"  ");
    return s;
  }
  #endif


  void update_line(depth_t depth, score_t alpha, score_t beta, const MoveLine &pline) {
    #ifndef NDEBUG
    if(!exact_line && change_line_to_info && !debug_moveline.empty() && engine.state.info == debug_board_info && !pline.startswith(debug_moveline)) {
      _fprintf(fp, "%s depth=%d <changed moveline> %s (%s, %s) from %s\n",
              tab(depth).c_str(), (int)depth, _line_str_full(pline).c_str(),
              EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
              _line_str_full(debug_moveline.as_past()).c_str());
      debug_moveline = pline.get_past();
    }
    #endif
  }

  bool filter_moveline(depth_t depth, const MoveLine &pline_alt) const {
    if(engine.state.info == debug_board_info) {
      return true;
    }
    return (
        (!exact_line && pline_alt.full().startswith(debug_moveline))
        || (exact_line && debug_moveline.startswith(pline_alt.full()))
    ) && (debug_depth < depth + show_maxdepth);
  }

  void log_variation(depth_t depth, const MoveLine &pline, const std::string &message) {
    #ifndef NDEBUG
    if(filter_moveline(depth, pline) && show_pv) {
      _fprintf(fp, "%s depth=%d LOG, %s\n", tab(depth).c_str(), int(depth), message.c_str());
    }
    #endif
  }

  void update_standpat(depth_t depth, score_t alpha, score_t beta, score_t score, const MoveLine &pline, int nchecks) {
    #ifndef NDEBUG
    if(filter_moveline(depth, pline) && show_q) {
      const char ast = (pline.get_past().startswith(debug_moveline)) ? ' ' : '*';
      _fprintf(fp, "%s%cdepth=%d STP, score=%s (%s, %s) %s -- %s (standpat)\n",
              tab(depth).c_str(), ast, (int)depth,
              EngineT::score_string(score).c_str(), EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
              _line_str_full(pline).c_str(), actinfo_standpat(alpha, beta, score, nchecks).c_str());
      update_line(depth, alpha, beta, pline);
    }
    assert(engine.check_valid_sequence(pline));
    #endif
  }

  void update_q(depth_t depth, score_t alpha, score_t beta, score_t bestscore, score_t score, move_t m,
              const MoveLine &pline, const MoveLine &pline_alt, const std::string &extra_inf=""s)
  {
    #ifndef NDEBUG
    if(filter_moveline(depth, pline_alt) && show_q) {
      const char ast = (pline_alt.full().startswith(debug_moveline)) ? ' ' : '*';
      _fprintf(fp, "%s%cdepth=%d QSC, %s, score=%s (%s, %s) %s -- %s %s Q\n",
              tab(depth).c_str(), ast, (int)depth, pgn::_move_str(engine, m).c_str(),
              EngineT::score_string(score).c_str(), EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
              _line_str_full(pline_alt).c_str(), actinfo(alpha, beta, bestscore, score).c_str(),
              extra_inf.c_str());
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_mem(depth_t depth, score_t alpha, score_t beta, const tt_ab_entry &zb, const MoveLine &pline) {
    #ifndef NDEBUG
    MoveLine pline_alt = pline.get_past().as_past();
    pline_alt.replace_line(zb.subpline);
    if(filter_moveline(depth, pline_alt) && show_mem) {
      _fprintf(fp, "%s depth=%d MEM, %s (%s, %s) %s -- %s (score=%s)\n",
              tab(depth).c_str(), int(depth), pgn::_move_str(engine, zb.m_hint.front()).c_str(),
              EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
              _line_str_full(pline_alt).c_str(), actinfo_mem(alpha, beta, zb).c_str(),
              EngineT::score_string(zb.score).c_str());
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_zw(depth_t depth, score_t alpha, score_t beta, score_t bestscore, score_t score, move_t m,
              const MoveLine &pline, const MoveLine &pline_alt, const std::string &extra_inf=""s)
  {
    #ifndef NDEBUG
    assert(alpha + 1 == beta);
    if(filter_moveline(depth, pline_alt) && show_zw) {
      if(score<bestscore)return;
      const char ast = (pline_alt.full().startswith(debug_moveline)) ? ' ' : '*';
      _fprintf(fp, "%s%cdepth=%d ZWS, %s, score=%s (%s, %s) %s -- %s %s ZW\n",
              tab(depth).c_str(), ast, int(depth), pgn::_move_str(engine, m).c_str(),
              EngineT::score_string(score).c_str(), EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
              _line_str_full(pline_alt).c_str(), actinfo(alpha, beta, bestscore, score).c_str(),
              extra_inf.c_str());
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void update_pv(depth_t depth, score_t alpha, score_t beta, score_t bestscore, score_t score, move_t m,
              const MoveLine &pline, const MoveLine &pline_alt, const std::string &extra_inf=""s)
  {
    #ifndef NDEBUG
    if(filter_moveline(depth, pline_alt) && show_pv) {
      if(score<bestscore)return;
      const char ast = (pline_alt.full().startswith(debug_moveline)) ? ' ' : '*';
      _fprintf(fp, "%s%cdepth=%d PVS, %s, score=%s (%s, %s) %s -- %s %s PV\n",
        tab(depth).c_str(), ast, int(depth), pgn::_move_str(engine, m).c_str(),
        EngineT::score_string(score).c_str(), EngineT::score_string(alpha).c_str(), EngineT::score_string(beta).c_str(),
        _line_str_full(pline_alt).c_str(), actinfo(alpha, beta, bestscore, score).c_str(),
        extra_inf.c_str());
      update_line(depth, alpha, beta, pline_alt);
    }
    assert(engine.check_valid_sequence(pline_alt));
    #endif
  }

  void check_score(depth_t depth, score_t score, const MoveLine &pline) const {
    #ifndef NDEBUG
    assert(EngineT::ENABLE_SYZYGY || !pline.tb);
    const score_t adraw = std::abs(0);
    const score_t pvscore = engine.get_pvline_score(pline);
    if(!engine.check_pvline_score(pline, score) && std::abs(score) != adraw && std::abs(pvscore) != adraw) {
      _fprintf(fp, "%s pvline %s\n", tab(depth).c_str(), pline.pgn_full(engine).c_str());
      _fprintf(fp, "%s score %s (%d)\n", tab(depth).c_str(), EngineT::score_string(score).c_str(), (int)score);
      _fprintf(fp, "%s pvscore %s (%d)\n", tab(depth).c_str(), EngineT::score_string(pvscore).c_str(), (int)pvscore);
      _fprintf(fp, "%s fen %s\n", tab(depth).c_str(), fen::export_as_string(engine.export_as_fen()).c_str());
      abort();
    }
    assert(engine.check_pvline_score(pline, score) || std::abs(score) == adraw || std::abs(pvscore) == adraw);
    #endif
  }

  void check_pathdep(bool pathdep, depth_t depth, score_t score, const MoveLine &pline) const {
    assert(!pathdep || (score == 0 || engine.is_repeated_thought(pline)));
  }

  void check_length(depth_t depth, const MoveLine &pline) const {
    #ifndef NDEBUG
    assert(pline.size() >= size_t(depth) || engine.check_line_terminates(pline));
    #endif
  }
};
