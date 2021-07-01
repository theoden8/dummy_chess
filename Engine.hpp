#pragma once


#include <cfloat>
#include <algorithm>
#include <list>
#include <map>
#include <optional>

#include <FEN.hpp>
#include <Board.hpp>


class Engine : public Board {
public:
  Engine(const fen::FEN fen=fen::starting_pos):
    Board(fen)
  {
    const std::vector<PIECE> piece_types = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    std::vector<std::pair<double, PIECE>> pieces;
    for(PIECE p : piece_types) {
      pieces.emplace_back(material_of(p), p);
    }
    std::sort(pieces.begin(), pieces.end());
    for(const auto &[_, p] : pieces) {
      MATERIAL_PIECE_ORDERING.emplace_back(p);
    }
  }

  static constexpr std::array<pos_t, 4> PROMOTION_PIECES = {
    board::PROMOTE_KNIGHT, board::PROMOTE_BISHOP,
    board::PROMOTE_ROOK, board::PROMOTE_QUEEN
  };

  template <typename F>
  INLINE void iter_moves_from(pos_t i, F &&func) const {
    bitmask::foreach(get_moves_from(i), [&](pos_t j) mutable -> void {
      if(is_promotion_move(i, j)) {
        for(pos_t promotion : PROMOTION_PIECES) {
          func(i, j | promotion);
        }
      } else {
        func(i, j);
      }
    });
  }

  template <typename F>
  INLINE void iter_quiesc_moves_from(pos_t i, F &&func) const {
    const COLOR c = self.color_at_pos(i);
    const piece_bitboard_t foes = bits[enemy_of(c)];
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      if(is_promotion_move(i, j)
          || is_enpassant_take_move(i, j)
          || is_capture_move(i, j)
          || is_naively_checking_move(i, j))
      {
        func(i, j);
      }
    });
  }

  piece_bitboard_t get_quiesc_moves_from(pos_t i) const {
    piece_bitboard_t pb = 0x00;
    iter_quiesc_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      pb |= piece::pos_mask(j & board::MOVEMASK);
    });
    return pb;
  }

  template <typename F>
  INLINE void iter_moves(F &&func) const {
    bitmask::foreach(bits[activePlayer()], [&](pos_t i) mutable -> void {
      iter_moves_from(i, func);
    });
  }

  template <typename F>
  INLINE void iter_quiesc_moves(F &&func) const {
    bitmask::foreach(bits[activePlayer()], [&](pos_t i) mutable -> void {
      iter_quiesc_moves_from(i, func);
    });
  }

  INLINE size_t count_moves(COLOR c) const {
    assert(c == WHITE || c == BLACK);
    int16_t no_moves = 0;
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      pos_t moves_from = bitmask::count_bits(get_moves_from(i));
      if(piece::is_set(bits_pawns, i) && (board::_y(i) == 2-1 || board::_y(i) == 7-1)
          && (
            (self.color_at_pos(i) == WHITE && 1+board::_y(i) == 7)
            || (self.color_at_pos(i) == BLACK && 1+board::_y(i) == 2))
        )
      {
        moves_from *= 4;
      }
      no_moves += moves_from;
    });
    return no_moves;
  }

  move_t get_random_move() const {
    std::vector<move_t> moves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nomove;
    return moves[rand() % moves.size()];
  }

  // for MC-style testing
  move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nomove;
    return moves[rand() % moves.size()];
  }

  const double MATERIAL_PAWN = 1,
               MATERIAL_KNIGHT = 3.25,
               MATERIAL_BISHOP = 3.25,
               MATERIAL_ROOK = 5,
               MATERIAL_QUEEN = 10,
               MATERIAL_KING = 1e7;
  std::vector<PIECE> MATERIAL_PIECE_ORDERING;

  INLINE double material_of(PIECE p) const {
    switch(p) {
      case EMPTY:return 0;
      case PAWN:return MATERIAL_PAWN;
      case KNIGHT:return MATERIAL_KNIGHT;
      case BISHOP:return MATERIAL_BISHOP;
      case ROOK:return MATERIAL_ROOK;
      case QUEEN:return MATERIAL_QUEEN;
      case KING:return MATERIAL_KING;
    }
    return 0;
  }

  INLINE double count_material(piece_bitboard_t mask) const {
    double m = .0;
    m += piece::size(mask & bits_pawns) * material_of(PAWN);
    m += piece::size(mask & bits_slid_diag & bits_slid_orth) * material_of(QUEEN);
    m += piece::size(mask & bits_slid_diag & ~bits_slid_orth) * material_of(BISHOP);
    m += piece::size(mask & ~bits_slid_diag & bits_slid_orth) * material_of(ROOK);
    m += piece::size(mask & get_knight_bits()) * material_of(KNIGHT);
    return m;
  }

  INLINE double h_attack_cells(COLOR c) const {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    double attacks = .0;
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      auto a = self.get_attacks_from(i);
      attacks += piece::size(a & occupied) + piece::size(a);
    });
    return attacks;
  }

  INLINE double heuristic_of(COLOR c) const {
    double h = .0;
    h += count_material(bits[c]);
    h += count_material(state_pins[enemy_of(c)]) * 1e-4;
//    h += count_moves(c) * 2e-4;
    h += h_attack_cells(c) * 1e-4;
    return h;
  }

  INLINE double evaluate() const {
    if(self.is_draw() || self.can_draw_repetition()){
      return 0;
    } else if(self.is_checkmate()) {
      return -MATERIAL_KING;
    }
    const COLOR c = activePlayer();
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  INLINE double move_heuristic(pos_t i, pos_t j) const {
    double val = material_of(self[j & board::MOVEMASK].value);
    if(is_enpassant_take_move(i, j)) {
      val += material_of(PAWN);
    } else if(is_promotion_move(i, j)) {
      val += material_of(board::get_promotion_as(j)) - material_of(PAWN);
    }
    if(!self.empty_at_pos(j & board::MOVEMASK)) {
      val -= material_of(self[i].value)*.05;
    }
    return val;
  }

  struct tt_ab_entry {
    board_info info;
    int16_t depth;
    double eval;
    double lowerbound, upperbound;
    move_t m;
    MoveLine subpline;
    size_t age;

    INLINE bool can_apply(const board_info &_info, int16_t _depth, size_t _age) const {
      return info == _info && depth >= _depth && age == _age;
    }

    INLINE bool is_inactive(size_t cur_age) const {
      return info.is_unset() || cur_age != age;
    }
  };

  INLINE bool score_is_mate(double score) const {
    return std::abs(score) > MATERIAL_KING * 1e-2;
  }

  INLINE double score_decay(double score) const {
    if(score_is_mate(score)) {
      score -= score / std::abs(score);
    }
    return score;
  }

  INLINE int score_mate_in(double score) const {
    assert(score_is_mate(score));
    const int mate_in = int(MATERIAL_KING - std::abs(score));
    return (score < 0) ? -mate_in : mate_in;
  }

  INLINE double get_pvline_score(const MoveLine &pline) {
    assert(check_valid_sequence(pline));
    auto rec = self.recursive_move_scope();
    for(const move_t &m : pline) {
      rec.scope(m);
    }
    double pvscore = evaluate();
    for(size_t i = 0; i < pline.size(); ++i) {
      pvscore = -score_decay(pvscore);
    }
    return pvscore;
  }

  INLINE bool check_pvline_score(const MoveLine &pline, double score) {
    return std::abs(score - get_pvline_score(pline)) < 1e-6;
  }

  piece_bitboard_t get_least_valuable_piece(piece_bitboard_t mask) const {
    // short-hands for bits
    const piece_bitboard_t diag = self.bits_slid_diag, orth = self.bits_slid_orth;
    const piece_bitboard_t kings = piece::pos_mask(pos_king[WHITE]) | piece::pos_mask(pos_king[BLACK]);
    // get some piece that is minimal
    piece_bitboard_t found = 0x00ULL;
    for(const auto p : MATERIAL_PIECE_ORDERING) {
      switch(p) {
        case PAWN: if(mask & self.bits_pawns) found = mask & self.bits_pawns; break;
        case KNIGHT: if(mask & self.get_knight_bits()) found = mask & self.get_knight_bits(); break;
        case BISHOP: if(mask & diag & ~orth) found = mask & diag & ~orth; break;
        case ROOK: if(mask & ~diag & orth) found = mask & ~diag & orth; break;
        case QUEEN: if(mask & diag & orth) found = mask & diag & orth; break;
        case KING: if(mask & kings) found = mask & kings; break;
        default: break;
      }
      if(found)break;
    }
    return found ? bitmask::highest_bit(found) : found;
  }

  double static_exchange_evaluation(pos_t i, pos_t j) const {
//    str::pdebug("move", _move_str(bitmask::_pos_pair(i, j)));
//    str::pdebug(fen::export_as_string(export_as_fen()));
    std::array<double, 64> gain = {0.};
    int depth = 0;
    const piece_bitboard_t may_xray = bits_slid_diag | bits_slid_orth | bits_pawns;
    piece_bitboard_t from_set = piece::pos_mask(i);
    piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    piece_bitboard_t attadef = get_attacks_to(j, BOTH);
    gain[depth] = material_of(self[j].value);
    do {
      const pos_t from = bitmask::log2_of_exp2(from_set);
      ++depth;
      gain[depth] = material_of(self[from].value) - gain[depth - 1];
//      str::pdebug("SEE: depth:", depth, "piece:", ""s + self[from].str(), "score:"s, gain[depth]);
      if(std::max(-gain[depth - 1], gain[depth]) < 0) {
//        str::pdebug("SEE BREAK: max(", -gain[depth-1], ",", gain[depth], ") < 0");
        break;
      }
      attadef &= ~from_set;
      occupied &= ~from_set;
      if(from_set & may_xray) {
        attadef |= get_attacks_to(j, BOTH, occupied);
      }
      const COLOR c = self.color_at_pos((depth & 1) ? j : i);
      from_set = get_least_valuable_piece(attadef & bits[c]);
    } while(from_set);
    int maxdepth = depth;
    while(--depth) {
      gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }
//    for(int i = 0; i < maxdepth; ++i) {
//      str::pdebug("gain[", i, "] = ", gain[i]);
//    }
    return gain[0];
  }

  INLINE decltype(auto) get_ab_ttable_zobrist(int16_t depth, zobrist::ttable<tt_ab_entry> &ab_ttable) {
    std::optional<double> maybe_score;
    const auto info = self.get_board_info();
    const zobrist::key_t k = self.zb_hash();
    if(ab_ttable[k].can_apply(info, depth, tt_age)) {
      ++zb_hit;
      ++nodes_searched;
      maybe_score.emplace(ab_ttable[k].eval);
    } else {
      ++zb_miss;
    }
    return std::make_tuple(maybe_score, k, info);
  }

  int16_t debug_depth = -1;
//  const board_info debug_board_info = board_info::import_dbg("255|4360946792109482756|34562222993|17835716876999589888|2594073419725146112|9583660041404153985|1");
  MoveLine debug_moveline = MoveLine(std::vector<move_t>{
  });

  INLINE decltype(auto) ab_get_quiesc_moves(int16_t depth, MoveLine &pline, int8_t delta, bool king_in_check) {
    std::vector<std::tuple<double, move_t, bool>> quiescmoves;
    quiescmoves.reserve(8);
    if(!king_in_check) {
      iter_quiesc_moves([&](pos_t i, pos_t j) mutable -> void {
        double val = move_heuristic(i, j);
        const move_t m = bitmask::_pos_pair(i, j);
        bool reduce_delta = false;
        if(is_promotion_move(i, j) || is_enpassant_take_move(i, j) || material_of(self[i].value) <= material_of(self[j].value)) {
          ;
        } else {
          // captures and checks
          const bool is_checking = delta > 0 && depth > -5 && is_naively_checking_move(i, j);
          if(is_capture_move(i, j)) {
            double see = static_exchange_evaluation(i, j);
            if(see < 0 && !king_in_check && !is_checking)return;
            val = see;
          } else if(is_checking) {
            reduce_delta = true;
          } else {
            return;
          }
        }
        if(pline.find_in_mainline(m)) {
          val += 10.;
        }
        quiescmoves.emplace_back(-val, m, reduce_delta);
      });
    } else {
      iter_moves([&](pos_t i, pos_t j) mutable -> void {
        double val = move_heuristic(i, j);
        const move_t m = bitmask::_pos_pair(i, j);
        if(pline.find_in_mainline(m)) {
          val += 10.;
        }
        quiescmoves.emplace_back(-val, m, false);
      });
    }
    std::sort(quiescmoves.begin(), quiescmoves.end());
    // no need for reverse sort: values are already negated
    return quiescmoves;
  }

  double alpha_beta_quiescence(double alpha, double beta, int16_t depth, MoveLine &pline,
                               zobrist::ttable<tt_ab_entry> &ab_ttable, int8_t delta)
  {
    double score = -MATERIAL_KING;
    double bestscore = -MATERIAL_KING;
    const bool king_in_check = state_checkline[activePlayer()] != ~0ULL;
    if(!king_in_check) {
      score = evaluate();
      if(score >= beta) {
        ++nodes_searched;
        return score;
      } else if(score > bestscore) {
        bestscore = score;
        if(score > alpha) {
          alpha = score;
        }
      }
    }

    // zobrist
    const auto [zbscore, k, info] = get_ab_ttable_zobrist(depth, ab_ttable);
    if(zbscore.has_value()) {
      const auto &zb = ab_ttable[k];
//      if(!debug_moveline.empty() && pline.get_past().startswith(debug_moveline)) {
//        std::string tab=""s; for(int i=debug_depth;i>depth;--i)tab+=" ";
//
//        std::string actinfo = "memoized"s;
//        MoveLine pline_alt = pline.branch_from_past();
//        pline_alt.replace_line(zb.subpline);
//        _printf("%sdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s (%.6f, %.6f)\n",
//            tab.c_str(), depth, board::_move_str(zb.m).c_str(), zb.eval, alpha, beta,
//            _line_str_full(pline_alt).c_str(), actinfo.c_str(),
//            zb.lowerbound, zb.upperbound);
//      }
      if(zb.lowerbound >= beta) {
        pline.replace_line(ab_ttable[k].subpline);
        return zb.lowerbound;
      }
      if(zb.upperbound <= alpha) {
        pline.replace_line(ab_ttable[k].subpline);
        return zb.upperbound;
      }
      alpha = std::max(alpha, zb.lowerbound);
      beta = std::min(beta, zb.upperbound);
    }
    const bool overwrite = (depth >= ab_ttable[k].depth || ab_ttable[k].is_inactive(tt_age));
    decltype(auto) quiescmoves = ab_get_quiesc_moves(depth, pline, delta, king_in_check);
    if(quiescmoves.empty() || delta <= 0) {
      ++nodes_searched;
      if(king_in_check) {
        score = evaluate();
      }
      return score;
    }
    assert(pline.empty());
    move_t m_best = board::nomove;
    for(const auto &[_, m, reduce_delta] : quiescmoves) {
      MoveLine pline_alt = pline.branch_from_past();
      {
        volatile auto mscope = move_scope(m);
        pline_alt.premove(m);
        assert(pline_alt.empty());
        score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_ttable, reduce_delta ? delta - 1 : delta));
        assert(check_valid_sequence(pline_alt));
//        if(!debug_moveline.empty() && pline_alt.get_past().startswith(debug_moveline)) {
//          std::string tab=""s; for(int i=0;i<debug_depth-depth;++i)tab+=" "s;
//          std::string actinfo = "ignored"s;
//          if(score >= beta) {
//            actinfo = "beta-curoff"s;
//          } else if(score > bestscore) {
//            actinfo = "new best"s;
//          }
//          _printf("%sdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s\n",
//            tab.c_str(), depth, board::_move_str(m).c_str(), score, beta, alpha,
//            _line_str_full(pline_alt).c_str(), actinfo.c_str());
//        }
        pline_alt.recall();
      }
      assert(check_pvline_score(pline_alt, score));
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite && (ab_ttable[k].is_inactive(tt_age) || ab_ttable[k].lowerbound != ab_ttable[k].upperbound)) {
          if(ab_ttable[k].info.is_unset())++zb_occupied;
          ab_ttable[k] = { .info=info, .depth=depth, .eval=score, .lowerbound=score, .upperbound=DBL_MAX,
                           .m=m, .subpline=pline.get_future(), .age=tt_age };
        }
        return score;
      } else if(score > bestscore) {
        pline.replace_line(pline_alt);
        bestscore = score;
        m_best = m;
        if(score > alpha) {
          alpha = score;
        }
      }
    }
    if(overwrite && m_best != board::nomove) {
      if(bestscore <= alpha) {
        if(ab_ttable[k].is_inactive(tt_age) || ab_ttable[k].lowerbound != ab_ttable[k].upperbound) {
          if(ab_ttable[k].info.is_unset())++zb_occupied;
          ab_ttable[k] = { .info=info, .depth=depth, .eval=bestscore, .lowerbound=-DBL_MAX, .upperbound=bestscore,
                            .m=m_best, .subpline=pline.get_future(), .age=tt_age };
        }
      } else {
        if(ab_ttable[k].is_inactive(tt_age))++zb_occupied;
        ab_ttable[k] = { .info=info, .depth=depth, .eval=bestscore, .lowerbound=bestscore, .upperbound=bestscore,
                         .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      }
    }
    return bestscore;
  }

  INLINE decltype(auto) ab_get_ordered_moves(MoveLine &pline) {
    std::vector<std::pair<double, move_t>> moves;
    moves.reserve(16);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      // principal variation move ordering
      if(pline.front_in_mainline() == m) {
        val += 100.;
      } else if(pline.find_in_mainline(m)) {
        val += 10.;
      }
      moves.emplace_back(-val, bitmask::_pos_pair(i,j));
    });
    std::sort(moves.begin(), moves.end());
    return moves;
  }

  double alpha_beta(double alpha, double beta, int16_t depth, MoveLine &pline,
                                  zobrist::ttable<tt_ab_entry> &ab_ttable)
  {
    if(depth == 0) {
      const int delta = 3;
      assert(pline.empty());
      const double score = alpha_beta_quiescence(alpha, beta, depth, pline, ab_ttable, delta);
      return score;
    }

    // zobrist
    const auto [zbscore, k, info] = get_ab_ttable_zobrist(depth, ab_ttable);
//    if(info == debug_board_info) {
//      str::pdebug(depth, "depth", "changed moveline", _line_str_full(pline), "("s, _beta, ", "s, _alpha, ")"s);
//      debug_moveline = pline.full();
//    }
    if(zbscore.has_value()) {
      const auto &zb = ab_ttable[k];
//      if(!debug_moveline.empty() && pline.get_past().startswith(debug_moveline)) {
//        std::string tab=""s; for(int i=debug_depth;i>depth;--i)tab+=" ";
//
//        std::string actinfo = "memoized"s;
//        MoveLine pline_alt = pline.branch_from_past();
//        pline_alt.replace_line(zb.subpline);
//        _printf("%sdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s (%.6f, %.6f)\n",
//            tab.c_str(), depth, board::_move_str(zb.m).c_str(), zb.eval, alpha, beta,
//            _line_str_full(pline_alt).c_str(), actinfo.c_str(),
//            zb.lowerbound, zb.upperbound);
//      }
      if(zb.lowerbound >= beta) {
        pline.replace_line(ab_ttable[k].subpline);
        ++nodes_searched;
        return zb.lowerbound;
      } else if(zb.upperbound <= alpha) {
        pline.replace_line(ab_ttable[k].subpline);
        ++nodes_searched;
        return zb.upperbound;
      }
      alpha = std::max(alpha, zb.lowerbound);
      beta = std::min(beta, zb.upperbound);
    }
    const bool overwrite = (depth >= ab_ttable[k].depth || ab_ttable[k].is_inactive(tt_age));
    decltype(auto) moves = ab_get_ordered_moves(pline);
    if(moves.empty()) {
      ++nodes_searched;
      return evaluate();
    }
    move_t m_best = board::nomove;
    double bestscore = -DBL_MAX;
    for(const auto [_, m] : moves) {
      MoveLine pline_alt = pline.branch_from_past();
      double score;
      {
        volatile auto mscope = move_scope(m);
        pline_alt.premove(m);
        assert(pline_alt.empty());
        score = -score_decay(alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_ttable));
//        if(!debug_moveline.empty() && pline_alt.get_past().startswith(debug_moveline)) {
//          std::string tab=""s; for(int i=0;i<debug_depth-depth;++i)tab+=" "s;
//          std::string actinfo = "ignored"s;
//          if(score >= beta) {
//            actinfo = "beta-curoff"s;
//          } else if(score > bestscore) {
//            actinfo = "new best"s;
//          }
//          _printf("%sdepth=%d, %s, score=%.6f (%.6f, %.6f) %s -- %s\n",
//            tab.c_str(), depth, board::_move_str(m).c_str(), score, beta, alpha,
//            _line_str_full(pline_alt).c_str(), actinfo.c_str());
//        }
        pline_alt.recall();
      }
      assert(check_pvline_score(pline_alt, score));
      if(score >= beta) {
        assert(pline_alt.size() >= size_t(depth) || check_line_terminates(pline_alt));
        pline.replace_line(pline_alt);
        if(overwrite && (ab_ttable[k].is_inactive(tt_age) || ab_ttable[k].lowerbound != ab_ttable[k].upperbound)) {
          if(ab_ttable[k].is_inactive(tt_age))++zb_occupied;
          ab_ttable[k] = { .info=info, .depth=depth, .eval=score, .lowerbound=score, .upperbound=DBL_MAX,
                           .m=m, .subpline=pline.get_future(), .age=tt_age };
        }
        return score;
      } else if(score > bestscore) {
        assert(pline_alt.size() >= size_t(depth) || check_line_terminates(pline_alt));
        m_best = m;
        pline.replace_line(pline_alt);
        bestscore = score;
        if(score > alpha) {
          alpha = score;
        }
      }
    };
    if(overwrite && m_best != board::nomove) {
      if(bestscore < alpha) {
        if(ab_ttable[k].is_inactive(tt_age) || ab_ttable[k].lowerbound != ab_ttable[k].upperbound) {
          if(ab_ttable[k].is_inactive(tt_age))++zb_occupied;
          ab_ttable[k] = { .info=info, .depth=depth, .eval=bestscore, .lowerbound=-DBL_MAX, .upperbound=bestscore,
                           .m=m_best, .subpline=pline.get_future(), .age=tt_age };
        }
      } else {
        if(ab_ttable[k].is_inactive(tt_age))++zb_occupied;
        ab_ttable[k] = { .info=info, .depth=depth, .eval=bestscore, .lowerbound=bestscore, .upperbound=bestscore,
                         .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      }
    }
    return bestscore;
  }

  template <typename F>
  decltype(auto) iterative_deepening_dfs(int16_t depth, const std::unordered_set<move_t> &searchmoves,
                                                    zobrist::ttable<tt_ab_entry> &ab_ttable, F &&callback_f)
  {
    if(depth == 0)return std::make_tuple(MATERIAL_KING, board::nomove);
    std::vector<std::tuple<double, int, move_t>> bestmoves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      if(searchmoves.empty() || searchmoves.find(m) != std::end(searchmoves)) {
        bestmoves.emplace_back(-val,0,m);
      }
    });
    std::sort(bestmoves.begin(), bestmoves.end());
    std::map<move_t, MoveLine> pline;
    bool should_stop_iddfs = false;
    for(int16_t d = 0; d < depth; ++d) {
      double eval_best = -MATERIAL_KING;
      for(size_t i = 0; i < bestmoves.size(); ++i) {
        auto &[eval, curdepth, m] = bestmoves[i];
        if(d == 0) {
          pline[m].premove(m);
        }
        str::pdebug("IDDFS:", d, "move:", _move_str(m), "pre-eval:", eval);
        {
          volatile auto mscope = move_scope(m);
          MoveLine mline = pline[m];
          eval = -alpha_beta(-MATERIAL_KING, MATERIAL_KING, d, mline, ab_ttable);
          ++curdepth;
          const bool fail_low = (pline[m].full().size() < size_t(d) && !check_line_terminates(pline[m]));
          if(!fail_low) {
            if(eval > eval_best) {
              eval_best = eval;
            }
            pline[m].replace_line(mline);
          }
          // front node must be m
          assert(pline[m].full().front() == m);
        }
        {
          const auto [_, d_best, m_best] = *std::max_element(std::begin(bestmoves), std::end(bestmoves));
          // check that search was successful
          assert(pline[m_best].full().size() >= size_t(d) || check_line_terminates(pline[m_best]));
          if(!callback_f(d_best, m_best, eval_best, pline[m_best].full(), m) || (score_is_mate(eval_best) && d > 1)) {
            should_stop_iddfs = true;
            break;
          }
        }
        //if(d+1==depth)str::pdebug("eval:", eval, "move:", _move_str(m), "pvline", _line_str(pline[m]));
      }
      std::sort(bestmoves.begin(), bestmoves.end());
      std::reverse(bestmoves.begin(), bestmoves.end());
      const auto [_, curdepth, m_best] = *std::max_element(std::begin(bestmoves), std::end(bestmoves));
      str::pdebug("depth:", curdepth, "pline:", _line_str(pline[m_best].full(), true), "size:", pline[m_best].full().size(), "eval", eval_best);
      if(should_stop_iddfs)break;
    }
    if(bestmoves.empty())return std::make_tuple(-MATERIAL_KING, board::nomove);
    const auto [eval, _, best_m] = bestmoves.front();
    return std::make_tuple(eval, best_m);
  }

  void reset_planning() {
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    ++tt_age;
  }

  decltype(auto) get_zobrist_alphabeta_scope() {
    return zobrist::make_store_object_scope<tt_ab_entry>(ab_ttable);
  }

  zobrist::ttable_ptr<tt_ab_entry> ab_ttable = nullptr;

  decltype(auto) make_callback_f() {
    return [&](int16_t depth, move_t m, double eval, const MoveLine &pline, move_t ponder_m) mutable -> bool {
      return true;
    };
  }

  template <typename F>
  move_t get_fixed_depth_move_iddfs(int16_t depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    debug_depth = depth;
    auto [_, m] = iterative_deepening_dfs(depth, searchmoves, store_scope.get_object(), std::forward<F>(callback_f));
    evaluation = _;
    return m;
  }

  move_t get_fixed_depth_move_iddfs(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move_iddfs(depth, make_callback_f(), searchmoves);
  }

  size_t nodes_searched = 0;
  double evaluation = MATERIAL_KING;
  ply_index_t tt_age = 0;
  size_t zb_hit = 0, zb_miss = 0, zb_occupied = 0;
  template <typename F>
  move_t get_fixed_depth_move(int16_t depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    MoveLine pline;
    debug_depth = depth;
    evaluation = alpha_beta(-MATERIAL_KING, MATERIAL_KING, depth, pline, store_scope.get_object());
    move_t m = get_random_move();
    if(!pline.full().empty()) {
      m = pline.full().front();
    }
    str::pdebug("pvline:", _line_str(pline, true), "size:", pline.size(), "eval:", evaluation);
    assert(check_valid_sequence(pline));
    return m;
  }

  move_t get_fixed_depth_move(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move(depth, make_callback_f(), searchmoves);
  }

  struct tt_perft_entry {
    board_info info;
    int16_t depth;
    size_t nodes;
  };

  zobrist::ttable_ptr<tt_perft_entry> perft_ttable = nullptr;
  decltype(auto) get_zobrist_perft_scope() {
    return zobrist::make_store_object_scope<tt_perft_entry>(perft_ttable);
  }
  size_t _perft(int16_t depth, std::array<tt_perft_entry, ZOBRIST_SIZE> &perft_ttable) {
    if(depth == 1 || depth == 0) {
      return count_moves(activePlayer());
    }
    // look-up:
    zobrist::key_t k = self.zb_hash();
    board_info info = get_board_info();
    if(perft_ttable[k].info == info && perft_ttable[k].depth == depth) {
      ++zb_hit;
      return perft_ttable[k].nodes;
    } else {
      if(perft_ttable[k].info.is_unset())++zb_occupied;
      ++zb_miss;
    }
    // search
    const bool overwrite = true;
    size_t nodes = 0;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      volatile auto mscope = move_scope(bitmask::_pos_pair(i, j));
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_ttable);
    });
    if(overwrite) {
      perft_ttable[k] = { .info=info, .depth=depth, .nodes=nodes };
    }
    return nodes;
  }

  inline size_t perft(int16_t depth=1) {
    decltype(auto) store_scope = get_zobrist_perft_scope();
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    return _perft(depth, store_scope.get_object());
  }
};
