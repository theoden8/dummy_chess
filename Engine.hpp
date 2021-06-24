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
    const COLOR c = self[i].color;
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

  INLINE size_t count_moves(COLOR c=NEUTRAL) const {
    int16_t no_moves = 0;
    if(c==NEUTRAL) {
      c=activePlayer();
    }
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      pos_t moves_from = bitmask::count_bits(get_moves_from(i));
      if(self[i].value == PAWN && (board::_y(i) == 2-1 || board::_y(i) == 7-1)
          && (
            (self[i].color == WHITE && 1+board::_y(i) == 7)
            || (self[i].color == BLACK && 1+board::_y(i) == 2))
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

  INLINE double h_material(COLOR c) const {
    double m = .0;
    for(PIECE p : {PAWN,KNIGHT,BISHOP,ROOK,QUEEN}) {
      piece_bitboard_t mask = bits[c];
      if(p == PAWN) {
        mask &= bits_pawns;
      } else if(p == BISHOP) {
        mask &= bits_slid_diag & ~bits_slid_orth;
      } else if(p == ROOK) {
        mask &= bits_slid_orth & ~bits_slid_diag;
      } else if(p == QUEEN) {
        mask &= bits_slid_orth & bits_slid_diag;
      } else if(p == KNIGHT) {
        mask &= get_knight_bits();
      }
      m += piece::size(mask) * material_of(p);
    }
    return m;
  }

  INLINE double h_pins(COLOR c) const {
    double h = .0;
    bitmask::foreach(state_pins[enemy_of(c)], [&](pos_t i) mutable -> void {
      h += material_of(self[i].value);
    });
    return h;
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
    h += h_material(c);
    h += h_pins(c) * 1e-4;
    h += count_moves(c) * 2e-4;
//    h += h_attack_cells(c) * 1e-4;
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
    }
    if(is_promotion_move(i, j)) {
      val += material_of(board::get_promotion_as(j)) - material_of(PAWN);
    }
    if(self[j & board::MOVEMASK].value != EMPTY) {
      val -= material_of(self[i].value)*.05;
    }
    return val;
  }

  struct ab_info {
    board_info info;
    int16_t depth;
    double eval;
    move_t m;
    MoveLine subpline;
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
//    str::print("move", _move_str(bitmask::_pos_pair(i, j)));
//    str::print(fen::export_as_string(export_as_fen()));
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
//      str::print("SEE: depth:", depth, "piece:", ""s + self[from].str(), "score:"s, gain[depth]);
      if(std::max(-gain[depth - 1], gain[depth]) < 0) {
//        str::print("SEE BREAK: max(", -gain[depth-1], ",", gain[depth], ") < 0");
        break;
      }
      attadef &= ~from_set;
      occupied &= ~from_set;
      if(from_set & may_xray) {
        attadef |= get_attacks_to(j, BOTH, occupied);
      }
      const COLOR c = self[(depth & 1) ? j : i].color;
      from_set = get_least_valuable_piece(attadef & bits[c]);
    } while(from_set);
    int maxdepth = depth;
    while(--depth) {
      gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }
//    for(int i = 0; i < maxdepth; ++i) {
//      str::print("gain[", i, "] = ", gain[i]);
//    }
    return gain[0];
  }

  INLINE decltype(auto) get_ab_store_zobrist(int16_t depth, MoveLine &pline, zobrist::hash_table<ab_info> &ab_store) {
    std::optional<double> maybe_score;
    const auto info = get_board_info();
    const zobrist::key_t k = self.zb_hash();
    //TODO fix evaluation caching
    if(ab_store[k].info == info && depth <= ab_store[k].depth) {
      ++zb_hit;
      ++nodes_searched;
      pline.replace_line(ab_store[k].subpline);
      double score = ab_store[k].eval;
      maybe_score.emplace(score);
    } else {
      ++zb_miss;
    }
    return std::make_tuple(maybe_score, k, info);
  }

  int debug_depth = 4;
  const MoveLine debug_moveline = MoveLine(std::vector<move_t>{
//    3 'r3k3/pQp5/3bp1pn/7q/8/P1N1P3/1PPP1P2/R1B1K3 b q - 0 21'
//    bitmask::_pos_pair(board::_pos(E, 8), board::_pos(E, 7)),
//    bitmask::_pos_pair(board::_pos(B, 7), board::_pos(A, 8)),
//    5 'rnq1kbnr/p1p4p/1p2pp2/3p2p1/2PP3N/4P3/PP1B1PQP/RN2K2R w KQkq - 0 11'
//    bitmask::_pos_pair(board::_pos(G, 2), board::_pos(F, 3)),
//    // this:
//    bitmask::_pos_pair(board::_pos(B, 1), board::_pos(A, 3)),
//    // or
//    bitmask::_pos_pair(board::_pos(G, 5), board::_pos(H, 4)),
//    --
//    bitmask::_pos_pair(board::_pos(C, 4), board::_pos(C, 5)),
//    bitmask::_pos_pair(board::_pos(C, 7), board::_pos(C, 6)),
//    3 '8/8/8/3p1k1p/3P1P2/8/2K5/8 b - - 2 46'
//    bitmask::_pos_pair(board::_pos(F, 5), board::_pos(F, 4)),
//    3 '8/8/8/3p3p/3PkP2/8/8/3K4 b - - 4 47'
//    bitmask::_pos_pair(board::_pos(E, 4), board::_pos(D, 4)),
//    3 'r5k1/pn1b2bp/2p3p1/6P1/P1Pp3P/3K4/1q6/RN4NR w - - 0 25'
//    bitmask::_pos_pair(board::_pos(A, 1), board::_pos(A, 3)),
  });

  INLINE decltype(auto) ab_get_quiesc_moves(MoveLine &pline, int8_t &delta, bool king_in_check) {
    std::vector<std::pair<double, move_t>> quiescmoves;
    quiescmoves.reserve(8);
    if(!king_in_check) {
      iter_quiesc_moves([&](pos_t i, pos_t j) mutable -> void {
        double val = move_heuristic(i, j);
        const move_t m = bitmask::_pos_pair(i, j);
        if(is_promotion_move(i, j) || is_enpassant_take_move(i, j) || material_of(self[i].value) <= material_of(self[j].value)) {
          ;
        } else {
          // captures and checks
          bool should_prune = true;
          if(is_naively_checking_move(i, j)) {
            if(!is_capture_move(i, j)) {
              --delta;
            }
            should_prune = false;
          }
          if(is_capture_move(i, j)) {
            double see = static_exchange_evaluation(i, j);
            if(see < 0 && !king_in_check && should_prune)return;
            val = see;
          }
        }
        if(pline.find_in_mainline(m)) {
          val += 10.;
        }
        quiescmoves.emplace_back(-val, m);
      });
    } else {
      iter_moves([&](pos_t i, pos_t j) mutable -> void {
        double val = move_heuristic(i, j);
        const move_t m = bitmask::_pos_pair(i, j);
        if(pline.find_in_mainline(m)) {
          val += 10.;
        }
        quiescmoves.emplace_back(-val, m);
      });
    }
    std::sort(quiescmoves.begin(), quiescmoves.end());
    return quiescmoves;
  }

  double alpha_beta_quiescence(double alpha, double beta, int16_t depth, MoveLine &pline,
                              zobrist::hash_table<ab_info> &ab_store, int8_t delta)
  {
    double score = evaluate();
    double bestscore = -MATERIAL_KING;
    const bool king_in_check = state_checkline[activePlayer()] != ~0ULL;
    if(!king_in_check) {
      if(score >= beta || delta <= 0) {
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
//    const auto k = zb_hash();
//    const auto info = get_board_info();
    const auto [zbscore, k, info] = get_ab_store_zobrist(depth, pline, ab_store);
    if(zbscore.has_value()) {
      return zbscore.value();
    }
    const bool overwrite = false;//(depth >= ab_store[k].depth);
    decltype(auto) quiescmoves = ab_get_quiesc_moves(pline, delta, king_in_check);
    if(quiescmoves.empty()) {
      ++nodes_searched;
      return score;
    }
    assert(pline.empty());
    move_t bestmove = board::nomove;
    for(const auto [_, m] : quiescmoves) {
      //assert(check_valid_move(m));
      MoveLine pline_alt = pline.branch_from_past();
      {
        volatile auto mscope = move_scope(m);
        pline_alt.premove(m);
        assert(pline_alt.empty());
        score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_store, delta));
        assert(check_valid_sequence(pline_alt));
        if(!debug_moveline.empty() && pline_alt.startswith(debug_moveline)) {
          std::string TAB = ""s;
          for(int i=0;i<debug_depth-depth;++i)TAB+=" "s;
          printf("%s", TAB.c_str());
          printf("depth=%d, %s, score=%.5f (%.4f, %.4f) %s\n", depth,board::_move_str(m).c_str(),score, beta, alpha,
            ("["s + str::join(_line_str(pline_alt.full())) + "]"s).c_str());
        }
        pline_alt.recall();
      }
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite && std::abs(score) > 1e-7) {
          ab_store[k] = { .info=info, .depth=depth, .eval=score, .m=m, .subpline=pline_alt.get_future() };
        }
        return score;
      } else if(score > alpha) {
        pline.replace_line(pline_alt);
        bestscore = alpha;
        bestmove = m;
        if(score > alpha) {
          alpha = score;
        }
      }
    }
    if(overwrite && std::abs(bestscore) > 1e-7) {
      ab_store[k] = { .info=info, .depth=depth, .eval=bestscore, .m=bestmove, .subpline=pline.get_future() };
    }
    return alpha;
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
                                  zobrist::hash_table<ab_info> &ab_store)
  {
    if(depth == 0) {
      const int delta = 5;
      assert(pline.empty());
      const double score = alpha_beta_quiescence(alpha, beta, depth, pline, ab_store, delta);
      return score;
    }

    // zobrist
//    const auto k = zb_hash();
//    const auto info = get_board_info();
    const auto [zbscore, k, info] = get_ab_store_zobrist(depth, pline, ab_store);
    if(zbscore.has_value()) {
      if(!debug_moveline.empty() && pline.startswith(debug_moveline)) {
        std::string TAB=""s;
        for(int i=debug_depth;i>depth;--i)TAB+=" ";
        std::string actinfo = "memoized"s;
        printf("%s", TAB.c_str());
        const move_t m = ab_store[k].m;
        const double score = zbscore.value();
        const fen::FEN f = export_as_fen();
        printf("depth=%d, %s, score=%.5f (%.4f, %.4f) %s %s fen=%s\n", depth,board::_move_str(m).c_str(),score, beta, alpha, actinfo.c_str(),
          ("["s + str::join(_line_str(pline.full())) + "]"s).c_str(),
          fen::export_as_string(f).c_str());
        if(fen::export_as_string(f) == "rnq1kbnr/p1p4p/1p2pp2/3p4/2PP3p/N3PQ2/PP1B1P1P/R3K2R b KQkq - 1 2"s) {
          str::print(TAB, "  bits:", bits[WHITE], bits[BLACK], bits_pawns, bits_slid_diag, bits_slid_orth, pos_king[WHITE], pos_king[BLACK]);
        }
      }
      return zbscore.value();
    }
    const bool overwrite = false;//(depth >= ab_store[k].depth);
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
        score = -score_decay(alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_store));
//        if(!check_valid_sequence(pline_alt)) {
//          str::print("pline not playable", _line_str(pline_alt.full()));
//          str::print("sequence not valid: [", _line_str(pline_alt), "]");
//          str::print("FEN: ", fen::export_as_string(self.export_as_fen()));
//        }
        assert(check_valid_sequence(pline_alt));
        if(!debug_moveline.empty() && pline_alt.startswith(debug_moveline)) {
          std::string TAB=""s;
          for(int i=debug_depth;i>depth;--i)TAB+=" ";

          std::string actinfo = ""s;
          if(score >= beta) {
            actinfo = "beta-curoff"s;
          } else if(score > bestscore) {
            actinfo = "new best"s;
          }
          printf("%s", TAB.c_str());
          printf("depth=%d, %s, score=%.5f (%.4f, %.4f) %s %s\n", depth,board::_move_str(m).c_str(),score, beta, alpha, actinfo.c_str(),
            ("["s + str::join(_line_str(pline_alt.full())) + "]"s).c_str());
        }
        pline_alt.recall();
      }
      if(score >= beta) {
        if(pline_alt.size() < size_t(depth)) {
//          if(!check_line_terminates(pline_alt)) {
//            str::print("A: PLINE_ALT NOT ENOUGH LENGTH", _line_str(pline_alt), "depth", depth, "full", _line_str(pline_alt.full()));
//          }
          assert(check_line_terminates(pline_alt));
        }
        pline.replace_line(pline_alt);
        if(overwrite && std::abs(score) > 1e-7) {
          ab_store[k] = { .info=info, .depth=depth, .eval=score, .m=m, .subpline=pline.get_future() };
        }
        return score;
      } else if(score > bestscore) {
        if(pline_alt.size() < size_t(depth)) {
//          if(!check_line_terminates(pline_alt)) {
//            str::print("B: PLINE_ALT NOT ENOUGH LENGTH", _line_str(pline_alt), "depth", depth, "full", _line_str(pline_alt.full()));
//          }
          assert(check_line_terminates(pline_alt));
        }
        m_best = m;
        pline.replace_line(pline_alt);
        bestscore = score;
        if(score > alpha) {
          alpha = score;
        }
      }
    };
    if(overwrite && std::abs(bestscore) > 1e-7) {
      ab_store[k] = { .info=info, .depth=depth, .eval=bestscore, .m=m_best, .subpline=pline.get_future() };
    }
    return bestscore;
  }

  std::pair<double, double> init_aspiration_window(double eval) {
//    return {eval - .25, eval + .25};
    return {-MATERIAL_KING, MATERIAL_KING};
  }

  void set_aspiration_window_re_search(double &aw_alpha, double &aw_beta) {
//    if(aw_alpha < -3) {
//      aw_alpha = -MATERIAL_KING;
//      aw_beta = MATERIAL_KING;
//    }
//    aw_alpha -= 1.;
//    aw_beta += 1.;
  }

  template <typename F>
  decltype(auto) iterative_deepening_dfs(int16_t depth, const std::unordered_set<move_t> &searchmoves,
                                                    zobrist::hash_table<ab_info> &ab_store, F &&callback_f)
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
      double alpha = -MATERIAL_KING;
      for(size_t i = 0; i < bestmoves.size(); ++i) {
        auto &[eval, curdepth, m] = bestmoves[i];
        if(d == 0) {
          pline[m].premove(m);
        }
        str::pdebug("IDDFS:", d, "move:", _move_str(m), "pre-eval:", eval);
        {
          volatile auto mscope = move_scope(m);
          auto [aw_alpha, aw_beta] = init_aspiration_window(eval);
          while(1) {
            MoveLine mline = pline[m];
            eval = -alpha_beta(-aw_beta, -aw_alpha, d, mline, ab_store);
            ++curdepth;
            if(mline.size() >= (size_t)d || check_line_terminates(mline)) {
              alpha = std::max(eval, alpha);
              pline[m].replace_line(mline);
              assert(pline[m].full().front() == m);
              break;
            }
            set_aspiration_window_re_search(aw_alpha, aw_beta);
            str::perror("RE-SEARCH", aw_alpha);
          }
//          str::print("line:", _line_str(mline), "depth:", d);
        }
        {
          const auto [eval_best, d_best, m_best] = *std::max_element(std::begin(bestmoves), std::end(bestmoves));
          if(!callback_f(d_best, m_best, eval_best, pline[m_best].full(), m)) {
            should_stop_iddfs = true;
            break;
          }
        }
        //if(d+1==depth)str::print("eval:", eval, "move:", _move_str(m), "pvline", _line_str(pline[m]));
      }
      std::sort(bestmoves.begin(), bestmoves.end());
      std::reverse(bestmoves.begin(), bestmoves.end());
      const auto [eval_best, curdepth, m_best] = *std::max_element(std::begin(bestmoves), std::end(bestmoves));
      str::pdebug("depth:", curdepth, "pline:", _line_str(pline[m_best].full(), true), "size:", pline[m_best].full().size(), "eval", alpha);
      if(should_stop_iddfs)break;
    }
    if(bestmoves.empty())return std::make_tuple(-MATERIAL_KING, board::nomove);
    const auto [eval, _, best_m] = bestmoves.front();
    return std::make_tuple(eval, best_m);
  }

  template <typename F>
  decltype(auto) iterative_deepening_astar(int16_t depth, const std::unordered_set<move_t> &searchmoves,
                                                      std::array<ab_info, ZOBRIST_SIZE> &ab_store, F &&callback_f)
  {
    if(depth == 0)return std::make_tuple(MATERIAL_KING, board::nomove);
    std::vector<std::tuple<double, int, move_t>> bestmoves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      if(searchmoves.empty() || searchmoves.find(m) != std::end(searchmoves)) {
        bestmoves.emplace_back(-val, 0, m);
      }
    });
    std::sort(bestmoves.begin(), bestmoves.end());
    std::map<move_t, MoveLine> pline;
    std::vector<double> alphas(depth, -MATERIAL_KING);
    while(1) {
      if(bestmoves.empty())break;
      size_t max_ind = 0;
      auto mx = bestmoves.front();
      bool changed = false;
      // find max
      for(size_t i = 0; i < bestmoves.size(); ++i) {
        const auto [eval, d, m] = bestmoves[i];
        const auto [mxeval, mxd, mxm] = mx;
        if(d == depth + 1)continue;
        changed = true;
        // encourage lower depth
        if(eval > mxeval || mxd == depth + 1) {
          max_ind = i, mx = bestmoves[i];
        }
      }
      if(!changed)break;
      // increase depth of max
      auto [eval, d, m] = mx;
      if(d == 0)pline[m].premove(m);
      str::pdebug("IDA*:", d, "move:", _move_str(m), "pre-eval:", eval, "pline", _line_str(pline[m].full()));
      volatile auto mscope = move_scope(m);
      auto [aw_alpha, aw_beta] = init_aspiration_window(eval);
      while(1) {
        MoveLine mline = pline[m];
        eval = -alpha_beta(-aw_beta, -aw_alpha, d, mline, ab_store);
        if(mline.size() >= (size_t)d || check_line_terminates(mline)) {
          pline[m] = mline;
          bestmoves[max_ind] = {eval, d + 1, m};
          if(eval > alphas[d]) {
            alphas[d] = eval;
          }
          break;
        }
        set_aspiration_window_re_search(aw_alpha, aw_beta);
        str::print("RE-SEARCH", aw_alpha);
      }
      //printf("increase %.5f %d %s\n", score, d, _move_str(m).c_str());
      const auto [eval_best, depth_best, m_best] = *std::max_element(std::begin(bestmoves), std::end(bestmoves));
      if(!callback_f(depth_best, m_best, eval_best, pline[m_best], m_best)) {
        break;
      }
      str::pdebug("depth:", d, "pline:", _line_str(pline[m_best].full(), true), "size:", pline[m_best].full().size(), "eval", eval_best);
    }
    if(bestmoves.empty())return std::make_tuple(MATERIAL_KING, board::nomove);
    const auto [eval, _, m] = *std::max_element(bestmoves.begin(), bestmoves.end());
    return std::make_tuple(eval, m);
  }

  void reset_planning() {
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0;
  }

  decltype(auto) get_zobrist_alphabeta_scope() {
    return zobrist::make_store_object_scope<ab_info>(ab_store);
  }

  zobrist::hash_table_ptr<ab_info> ab_store = nullptr;

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

  template <typename F>
  move_t get_fixed_depth_move_idastar(int16_t depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves={}) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    debug_depth = depth;
    auto [_, m] = iterative_deepening_astar(depth, searchmoves, store_scope.get_object(), std::forward<F>(callback_f));
    evaluation = _;
    return m;
  }

  template <typename F>
  move_t get_fixed_depth_move_idastar(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move_idastar(depth, make_callback_f(), searchmoves);
  }

  size_t nodes_searched = 0;
  double evaluation = MATERIAL_KING;
  size_t zb_hit = 0, zb_miss = 0;
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
    if(!check_valid_sequence(pline)) {
      str::pdebug("pvline not playable");
    }
    return m;
  }

  move_t get_fixed_depth_move(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move(depth, make_callback_f(), searchmoves);
  }

  struct perft_info {
    board_info info;
    int16_t depth;
    size_t nodes;
  };

  zobrist::hash_table_ptr<perft_info> perft_store = nullptr;
  decltype(auto) get_zobrist_perft_scope() {
    return zobrist::make_store_object_scope<perft_info>(perft_store);
  }
  size_t _perft(int16_t depth, std::array<perft_info, ZOBRIST_SIZE> &perft_store) {
    if(depth == 1 || depth == 0) {
      return count_moves();
    }
    zobrist::key_t k = self.zb_hash();
    board_info info = get_board_info();
    if(perft_store[k].info == info && perft_store[k].depth == depth) {
      ++zb_hit;
      return perft_store[k].nodes;
    }
    bool overwrite = true;
    ++zb_miss;
    size_t nodes = 0;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      volatile auto mscope = move_scope(bitmask::_pos_pair(i, j));
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_store);
    });
    if(overwrite) {
      perft_store[k] = { .info=info, .depth=depth, .nodes=nodes };
    }
    return nodes;
  }

  inline size_t perft(int16_t depth=1) {
    decltype(auto) store_scope = get_zobrist_perft_scope();
    zb_hit = 0, zb_miss = 0;
    return _perft(depth, store_scope.get_object());
  }
};
