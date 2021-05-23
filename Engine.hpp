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
  {}

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
  INLINE void iter_capture_moves_from(pos_t i, F &&func) const {
    const COLOR c = self[i].color;
    const piece_bitboard_t foes = bits[enemy_of(c)];
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      if(is_promotion_move(i, j)
          || is_enpassant_take_move(i, j)
          || self[j & board::MOVEMASK].color == enemy_of(c))
      {
        func(i, j);
      }
    });
  }

  template <typename F>
  INLINE void iter_check_moves_from(pos_t i, F &&func) const {
    const COLOR c = self[i].color;
    const piece_bitboard_t foes = bits[enemy_of(c)];
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      if(is_promotion_move(i, j)
          || is_enpassant_take_move(i, j)
          || self[j & board::MOVEMASK].color == enemy_of(c))
      {
        func(i, j);
      }
    });
  }

  piece_bitboard_t get_capture_moves_from(pos_t i) const {
    piece_bitboard_t pb = 0x00;
    iter_capture_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
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
  INLINE void iter_capture_moves(F &&func) const {
    bitmask::foreach(bits[activePlayer()], [&](pos_t i) mutable -> void {
      iter_capture_moves_from(i, func);
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
            (self[i].color == WHITE && board::_y(i) == 7-1)
            || (self[i].color == BLACK && board::_y(i) == 2-1)
        ))
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
               MATERIAL_KING = 1e9;

  inline double material_of(PIECE p) const {
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

  double heuristic_of(COLOR c) const {
    double h = .0;
    h += h_material(c);
    h += h_pins(c) * 1e-4;
    h += count_moves(c) * 2e-4;
//    h += h_attack_cells(c) * 1e-4;
    return h;
  }

  double evaluate() const {
    if(self.is_draw()){
      return 0;
    } else if(is_checkmate()) {
      return -1e7;
    }
    double score;
    score = heuristic_of(play_as) - heuristic_of(enemy_of(play_as));
    return (play_as == activePlayer()) ? score : -score;
  }

  double move_heuristic(pos_t i, pos_t j) const {
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
    COLOR play_as;
    int16_t depth;
    double eval;
    move_t m;
    MoveLine subpline;
  };

  double score_decay(double score) const {
    if(std::abs(score) > 1e6) {
      score *= .999;
    }
    return score;
  }

  piece_bitboard_t get_least_valuable_piece(piece_bitboard_t mask) const {
    // find material heuristic ordering
    const std::vector<PIECE> piece_types = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    std::vector<std::pair<double, PIECE>> pieces;
    for(PIECE p : piece_types) {
      pieces.emplace_back(material_of(p), p);
    }
    std::sort(pieces.begin(), pieces.end());
    // short-hands for bits
    const piece_bitboard_t diag = self.bits_slid_diag, orth = self.bits_slid_orth;
    const piece_bitboard_t kings = piece::pos_mask(pos_king[WHITE]) | piece::pos_mask(pos_king[BLACK]);
    // get some piece that is minimal
    piece_bitboard_t found = 0x00ULL;
    for(const auto &[_, p] : pieces) {
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

  decltype(auto) get_ab_store_zobrist(int16_t depth, MoveLine &pline, zobrist::hash_table<ab_info> &ab_store) {
    std::optional<double> maybe_score;
    const auto info = get_board_info();
    const zobrist::key_t k = zb_hash();
    if(ab_store[k].info == info && depth <= ab_store[k].depth && ab_store[k].play_as == play_as) {
      ++zb_hit;
      ++nodes_searched;
      pline.replace_line(ab_store[k].subpline);
      double score = (ab_store[k].play_as == play_as) ? ab_store[k].eval : -ab_store[k].eval;
      maybe_score.emplace(score);
    } else {
      ++zb_miss;
    }
    return std::make_tuple(maybe_score, k, info);
  }

  double alpha_beta_quiescence(double alpha, double beta, int16_t depth, MoveLine &pline,
                              zobrist::hash_table<ab_info> &ab_store, int8_t delta)
  {
    double score = evaluate();
    double bestscore = -DBL_MAX;
    const bool king_in_check = state_checkline[activePlayer()] != ~0ULL;
    if(king_in_check) {
      if(score >= beta || delta <= 0) {
        ++nodes_searched;
        return score;
      } else if(score > bestscore) {
        bestscore = score;
        if(score > alpha) {
          alpha = score;
        }
      }
      --delta;
    }

    // zobrist
    const auto [zbscore, k, info] = get_ab_store_zobrist(depth, pline, ab_store);
    if(zbscore.has_value()) {
      return zbscore.value();
    }
    bool overwrite = (depth + 1 >= ab_store[k].depth);

    std::vector<std::pair<double, move_t>> quiescmoves;
    quiescmoves.reserve(8);
    if(!king_in_check) {
      iter_capture_moves([&](pos_t i, pos_t j) mutable -> void {
        double val = move_heuristic(i, j);
        const move_t m = bitmask::_pos_pair(i, j);
        if(is_promotion_move(i, j) || is_enpassant_take_move(i, j) || material_of(self[i].value) <= material_of(self[j].value)) {
          ;
        } else {
          double see = static_exchange_evaluation(i, j);
          if(see < 0 && !king_in_check)return;
          val += see;
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
    if(quiescmoves.empty()) {
      ++nodes_searched;
      return score;
    }
    std::sort(quiescmoves.begin(), quiescmoves.end());
    assert(pline.empty());
    move_t bestmove = board::nomove;
    for(const auto [_, m] : quiescmoves) {
      //assert(check_valid_move(m));
      MoveLine pline_alt = pline.branch_from_past();
      {
        volatile auto mscope = move_scope(m);
        pline_alt.premove(m);
        score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_store, delta));
        assert(check_valid_sequence(pline_alt));
        pline_alt.recall();
      }
      //for(int i=0;i>depth;--i)printf(" ");
      //printf("depth=%d, score=%.5f, %s\n", depth,score,board::_move_str(m).c_str());
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite) {
          ab_store[k] = { .info=info, .play_as = play_as, .depth=depth, .eval=score, .m=m, .subpline=pline_alt.get_future() };
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
//      str::print(TAB, "score", score);
    }
    if(overwrite) {
      ab_store[k] = { .info=info, .play_as = play_as, .depth=depth, .eval=bestscore, .m=bestmove, .subpline=pline.get_future() };
    }
    return alpha;
  }

  double alpha_beta(double alpha, double beta, int16_t depth, MoveLine &pline,
                                  zobrist::hash_table<ab_info> &ab_store)
  {
    if(depth == 0) {
      const int delta = 5;
      const double score = alpha_beta_quiescence(alpha, beta, depth, pline, ab_store, delta);
      return score;
    }

    // zobrist
    const auto [zbscore, k, info] = get_ab_store_zobrist(depth, pline, ab_store);
    if(zbscore.has_value()) {
      return zbscore.value();
    }
    bool overwrite = (depth + 1 >= ab_store[k].depth);
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
    if(moves.empty()) {
      return evaluate();
    }
    std::random_shuffle(moves.begin(), moves.end());
    std::sort(moves.begin(), moves.end());
    move_t m_best = board::nomove;
    double bestscore = -DBL_MAX;
    for(const auto [_, m] : moves) {
      MoveLine pline_alt = pline.branch_from_past();
      double score;
      {
        volatile auto mscope = move_scope(m);
        pline_alt.premove(m);
        score = -score_decay(alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_store));
//        if(!check_valid_sequence(pline_alt)) {
//          str::print("pline not playable", _line_str(pline_alt.full()));
//          str::print("sequence not valid: [", _line_str(pline_alt), "]");
//          str::print("FEN: ", fen::export_as_string(self.export_as_fen()));
//        }
        assert(check_valid_sequence(pline_alt));
        pline_alt.recall();
      }
      //printf("depth=%d, score=%.5f, %s\n", depth,score,board::_move_str(m).c_str());
      if(score >= beta) {
        if(pline_alt.size() < size_t(depth)) {
//          if(!check_line_terminates(pline_alt)) {
//            str::print("A: PLINE_ALT NOT ENOUGH LENGTH", _line_str(pline_alt), "depth", depth, "full", _line_str(pline_alt.full()));
//          }
          assert(check_line_terminates(pline_alt));
        }
        pline.replace_line(pline_alt);
        if(overwrite) {
          ab_store[k] = { .info=info, .play_as = play_as, .depth=depth, .eval=score, .m=m, .subpline=pline.get_future() };
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
    if(overwrite) {
      ab_store[k] = { .info=info, .play_as = play_as, .depth=depth, .eval=bestscore, .m=m_best, .subpline=pline.get_future() };
    }
    return bestscore;
  }

  std::pair<double, double> init_aspiration_window(double eval) {
    return {eval - .75, eval + .75};
  }

  void set_aspiration_window_re_search(double &aw_alpha, double &aw_beta) {
    if(aw_alpha < -3) {
      aw_alpha = -DBL_MAX;
      aw_beta = DBL_MAX;
    }
    aw_alpha -= 1.;
    aw_beta += 1.;
  }

  template <typename F>
  std::pair<double, move_t> iterative_deepening_dfs(int16_t depth, const std::unordered_set<move_t> &searchmoves,
                                                    zobrist::hash_table<ab_info> &ab_store, F &&callback_f)
  {
    if(depth == 0)return {DBL_MAX, board::nomove};
    std::vector<std::pair<double, move_t>> bestmoves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      double val = move_heuristic(i, j);
      const move_t m = bitmask::_pos_pair(i, j);
      if(searchmoves.empty() || searchmoves.find(m) != std::end(searchmoves)) {
        bestmoves.emplace_back(-val,m);
      }
    });
    std::sort(bestmoves.begin(), bestmoves.end());
    std::map<move_t, MoveLine> pline;
    for(int16_t d = 0; d < depth; ++d) {
      double alpha = -DBL_MAX;
      for(size_t i = 0; i < bestmoves.size(); ++i) {
        auto &[eval, m] = bestmoves[i];
        if(d == 0) {
          pline[m].premove(m);
        }
        str::print("IDDFS:", d, "move:", _move_str(m), "pre-eval:", eval);
        {
          volatile auto mscope = move_scope(m);
          auto [aw_alpha, aw_beta] = init_aspiration_window(eval);
          while(1) {
            MoveLine mline = pline[m];
            eval = -alpha_beta(-aw_beta, -aw_alpha, d, mline, ab_store);
            if(mline.size() >= (size_t)d || check_line_terminates(mline)) {
              alpha = std::max(eval, alpha);
              pline[m].replace_line(mline);
              assert(pline[m].full()[0] == m);
              break;
            }
            set_aspiration_window_re_search(aw_alpha, aw_beta);
            str::print("RE-SEARCH", aw_alpha);
          }
//          str::print("line:", _line_str(mline), "depth:", d);
        }
        //if(d+1==depth)str::print("eval:", eval, "move:", _move_str(m), "pvline", _line_str(pline[m]));
      }
      std::sort(bestmoves.begin(), bestmoves.end());
      std::reverse(bestmoves.begin(), bestmoves.end());
      const move_t m_best = bestmoves.front().second;
      if(!callback_f(m_best, pline[m_best])) {
        break;
      }
      str::print("depth:", d, "pline:", _line_str(pline[m_best].full(), true), "size:", pline[m_best].full().size(), "eval", alpha);
    }
    if(bestmoves.empty())return {DBL_MAX, board::nomove};
//    const move_t m_best = bestmoves.front().second;
//    str::print("principal variation:"s, _line_str(pline[m_best]), "size"s, pline.size());
    return bestmoves.front();
  }

  template <typename F>
  std::pair<double, move_t> iterative_deepening_astar(int16_t depth, const std::unordered_set<move_t> &searchmoves,
                                                      std::array<ab_info, ZOBRIST_SIZE> &ab_store, F &&callback_f)
  {
    if(depth == 0)return {DBL_MAX, board::nomove};
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
    std::vector<double> alphas(depth, -DBL_MAX);
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
      str::print("IDA*:", d, "move:", _move_str(m), "pre-eval:", eval, "pline", _line_str(pline[m].full()));
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
//      const move_t m_best = bestmoves.front().second;
//      if(!callback_f(m_best, pline[m_best])) {
//        break;
//      }
//      str::print("depth:", d, "pline:", _line_str(pline[m_best].full(), true), "size:", pline[m_best].full().size(), "eval", alpha);
    }
    if(bestmoves.empty())return {DBL_MAX, board::nomove};
    auto [eval, d, m] = *std::max_element(bestmoves.begin(), bestmoves.end());
    return {eval, m};
  }

  void reset_planning() {
    play_as = activePlayer();
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0;
  }

  decltype(auto) get_zobrist_alphabeta_scope() {
    return zobrist::make_store_object_scope<ab_info>(ab_store);
  }

  zobrist::hash_table_ptr<ab_info> ab_store = nullptr;
  move_t get_fixed_depth_move_iddfs(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    auto [_, m] = iterative_deepening_dfs(depth, searchmoves, store_scope.get_object(),
                                          [&](const move_t m, const MoveLine &pline) mutable -> bool { return true; });
    evaluation = _;
    return m;
  }

  move_t get_fixed_depth_move_idastar(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    auto [_, m] = iterative_deepening_astar(depth, searchmoves, store_scope.get_object(),
                                            [&](const move_t m, const MoveLine &pline) mutable -> bool { return true; });
    evaluation = _;
    return m;
  }

  COLOR play_as = WHITE;
  size_t nodes_searched = 0;
  double evaluation = DBL_MAX;
  size_t zb_hit = 0, zb_miss = 0;
  template <typename F>
  move_t get_fixed_depth_move(int16_t depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    reset_planning();
    decltype(auto) store_scope = get_zobrist_alphabeta_scope();
    MoveLine pline;
    evaluation = alpha_beta(-1e9, 1e9, depth, pline, store_scope.get_object());
    move_t m = get_random_move();
    if(!pline.full().empty()) {
      m = pline.full().front();
    }
    str::print("pvline:", _line_str(pline, true), "size:", pline.size(), "eval:", evaluation);
    if(!check_valid_sequence(pline)) {
      str::print("pvline not playable");
    }
    return m;
  }

  decltype(auto) get_fixed_depth_move(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move(depth, [](const move_t m, const MoveLine &pline) mutable -> bool {return true;}, searchmoves);
  }
//
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
    zobrist::key_t k = zb_hash();
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
