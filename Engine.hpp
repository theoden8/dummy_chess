#pragma once


#include <cfloat>
#include <algorithm>
#include <valarray>
#include <map>

#include <FEN.hpp>
#include <PGN.hpp>
#include <Board.hpp>
#include <DebugTracer.hpp>


class Engine : public Board {
public:
  static constexpr std::array<pos_t, 4> PROMOTION_PIECES = {
    board::PROMOTE_KNIGHT, board::PROMOTE_BISHOP,
    board::PROMOTE_ROOK, board::PROMOTE_QUEEN
  };

  template <typename F>
  INLINE void iter_moves_from(pos_t i, F &&func) const {
    bitmask::foreach(state.moves[i], [&](pos_t j) mutable -> void {
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
  INLINE void iter_moves(F &&func) const {
    bitmask::foreach(bits[activePlayer()], [&](pos_t i) mutable -> void {
      iter_moves_from(i, func);
    });
  }

  INLINE size_t count_moves(COLOR c) const {
    assert(c == WHITE || c == BLACK);
    int16_t no_moves = 0;
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      pos_t moves_from = bitmask::count_bits(state.moves[i]);
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
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  // for MC-style testing
  move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  static constexpr float MATERIAL_PAWN = 1,
                         MATERIAL_KNIGHT = 3.25,
                         MATERIAL_BISHOP = 3.25,
                         MATERIAL_ROOK = 5,
                         MATERIAL_QUEEN = 10,
                         MATERIAL_KING = 1e4;
  std::vector<PIECE> MATERIAL_PIECE_ORDERING;

  static constexpr float MATERIALS[] = {
    MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP,
    MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING, .0};
  INLINE constexpr float material_of(PIECE p) const {
    return MATERIALS[int(p)];
  }

  INLINE float count_material(piece_bitboard_t mask) const {
    float m = .0;
    m += piece::size(mask & bits_pawns) * material_of(PAWN);
    m += piece::size(mask & bits_slid_diag & bits_slid_orth) * material_of(QUEEN);
    m += piece::size(mask & bits_slid_diag & ~bits_slid_orth) * material_of(BISHOP);
    m += piece::size(mask & ~bits_slid_diag & bits_slid_orth) * material_of(ROOK);
    m += piece::size(mask & get_knight_bits()) * material_of(KNIGHT);
    return m;
  }

  struct h_state {
  };
  h_state hstate;
  std::vector<h_state> hstate_hist;
  void _init_hstate() {}
  void _backup_on_event() {
    hstate_hist.emplace_back(hstate);
  }
  void _update_pos_change(pos_t i, pos_t j) {}
  void _update_change() {}
  void _restore_on_event() {
    hstate = hstate_hist.back();
    hstate_hist.pop_back();
  }

  INLINE float h_material(COLOR c) const {
    float h = count_material(bits[c]);
    const piece_bitboard_t bishops = bits[c] & bits_slid_diag & ~bits_slid_orth;
    // bishop pair
    if((bishops & bitmask::wcheckers) && (bishops & bitmask::bcheckers)) {
      h += .1;
    }
    if(bits_pawns & bits[c]) {
      h += .1;
    } else {
      const piece_bitboard_t side = piece::file_mask(A) | piece::file_mask(H);
      h -= .02 * piece::size(bits[c] & bits_pawns & side);
      h += .01 * piece::size(bits[c] & bits_pawns & ~side);
    }
    return h;
  }

  INLINE float h_attack_cells(COLOR c) const {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    float h = .0;
    decltype(auto) h_add = [&](pos_t i, float mat) mutable -> void {
      auto a = state.attacks[i];
      h += float(piece::size(a & occupied) + piece::size(a)) / mat;
    };
    bitmask::foreach(bits[c] & bits_pawns, [&](pos_t i) mutable -> void {
      h_add(i, MATERIAL_PAWN);
    });
    bitmask::foreach(bits[c] & bits_slid_diag & ~bits_slid_orth, [&](pos_t i) mutable -> void {
      h_add(i, MATERIAL_BISHOP);
    });
    bitmask::foreach(bits[c] & get_knight_bits(), [&](pos_t i) mutable -> void {
      h_add(i, MATERIAL_KNIGHT);
    });
    bitmask::foreach(bits[c] & ~bits_slid_diag & bits_slid_orth, [&](pos_t i) mutable -> void {
      h_add(i, MATERIAL_ROOK);
    });
    bitmask::foreach(bits[c] & bits_slid_diag & bits_slid_orth, [&](pos_t i) mutable -> void {
      h_add(i, MATERIAL_QUEEN);
    });
    h_add(pos_king[c], MATERIAL_KING);
    return h;
  }

  INLINE float h_mobility(COLOR c) const {
    float h = .0;
    bitmask::foreach(bits[c] & ~bits_pawns, [&](pos_t i) mutable -> void {
      h += std::sqrt(piece::size(state.moves[i]));
    });
    return h;
  }

  INLINE float h_pawn_structure(COLOR c) const {
    float h = .0;
    for(pos_t f = 0; f < board::LEN; ++f) {
      const piece_bitboard_t pawns_file = piece::file_mask(f) & bits_pawns & bits[c];
      if(!pawns_file) {
        h -= .02;
      } else {
        h += .01 * (2. - piece::size(pawns_file));
      }
    }
    for(pos_t f = B; f <= G; ++f) {
      const piece_bitboard_t pawns_file = piece::file_mask(f) & bits_pawns & bits[c];
      if(!pawns_file)continue;
      const piece_bitboard_t furthest_pawn = (c == WHITE) ? bitmask::highest_bit(pawns_file) : bitmask::lowest_bit(pawns_file);
      const pos_t pawnr = (c == WHITE) ? board::_y(bitmask::log2(pawns_file))
                                       : board::_y(bitmask::log2_msb(pawns_file));
      const piece_bitboard_t adjacent_files = (piece::file_mask(f - 1) | piece::file_mask(f) | piece::file_mask(f + 1));
      const piece_bitboard_t ahead = (c == WHITE) ? adjacent_files << (board::LEN * (1+pawnr))
                                                  : adjacent_files >> (board::LEN * (8-pawnr));
      assert(~ahead & furthest_pawn);
      if((ahead & bits_pawns & ~bits[c]))continue;
      const float pawnr_abs = ((c == WHITE) ? pawnr : 7 - pawnr);
      h += .001 * pawnr_abs * pawnr_abs;
    }
    h += .005 * piece::size(piece::get_pawn_attacks(bits_pawns & bits[c], c) & bits[c]);
    return h;
  }

  INLINE float heuristic_of(COLOR c) const {
    float h = .0;
    h += h_material(c);
    h += h_pawn_structure(c);
//    h -= count_material(state.pins[c]) * 1e-4;
//    h += h_mobility(c) * 1e-4;
    h += h_attack_cells(c) * 1e-4;
    return h;
  }

  INLINE float evaluate() const {
    if(self.is_draw() || self.can_draw_repetition()){
      return -1e-4;
    } else if(self.is_checkmate()) {
      return -MATERIAL_KING;
    }
    const COLOR c = activePlayer();
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  INLINE float move_heuristic_extra_material(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(is_promotion_move(i, _j)) {
      return material_of(board::get_promotion_as(j)) - material_of(PAWN);
    } else if(is_enpassant_take_move(i, _j)) {
      return material_of(PAWN);
    }
    return 0.;
  }

  INLINE float move_heuristic_check(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(is_naively_checking_move(i, _j)) {
      return .5;
    }
    return .0;
  }

  INLINE float move_heuristic_see(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    float val = .0;
    val += move_heuristic_extra_material(i, j);
    if(val > 0 || is_castling_move(i, _j)) {
      return val;
    }
    const float mat_i = material_of(self[i].value),
                mat_j = material_of(self[_j].value);
    if(mat_i < mat_j) {
      return mat_j - mat_i;
    }
    val += static_exchange_evaluation(i, _j);
    return val;
  }

  INLINE float move_heuristic_mvv_lva(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    float val = material_of(self[_j].value);
    if(is_castling_move(i, _j)) {
      val += .5;
    } else {
      const PIECE frompiece = self[i].value;
      val -= 1.05*(frompiece == KING ? 50. : material_of(frompiece));
    }
    val += move_heuristic_extra_material(i, j);
    val += move_heuristic_check(i, j);
    return val;
  }

  INLINE float move_heuristic_pv(move_t m, const MoveLine &pline, move_t firstmove=board::nullmove) const {
    if(m == firstmove) {
      return 1000.;
    } else if(m == pline.front_in_mainline()) {
      return 100.;
    }
    return .0;
  }

  INLINE float move_heuristic_cmt(move_t m, const MoveLine &pline, const std::vector<double> &countermove_table) const {
    const move_t p_m = pline.get_previous_move();
    if(p_m != board::nullmove) {
      const pos_t i = bitmask::first(m), _j = bitmask::second(m);
      const size_t cmt_outer = board::NO_PIECE_INDICES * board::SIZE;
      const pos_t p_j = bitmask::second(p_m) & board::MOVEMASK;
      return countermove_table[cmt_outer * (self[p_j].piece_index * board::SIZE + p_j) + self[i].piece_index * board::SIZE + _j];
    }
    return .0;
  }

  INLINE bool score_is_mate(float score) const {
    return std::abs(score) > MATERIAL_KING - 2000;
  }

  INLINE float score_decay(float score) const {
    if(score_is_mate(score)) {
      score -= score / std::abs(score);
    }
    return score;
  }

  INLINE int score_mate_in(float score) const {
    assert(score_is_mate(score));
    const int mate_in = int(MATERIAL_KING - std::abs(score));
    return (score < 0) ? -mate_in : mate_in;
  }

  INLINE float get_pvline_score(const MoveLine &pline) {
    assert(check_valid_sequence(pline));
    auto rec = self.recursive_move_scope();
    for(const move_t &m : pline) {
      rec.scope(m);
    }
    float pvscore = evaluate();
    for(size_t i = 0; i < pline.size(); ++i) {
      pvscore = -score_decay(pvscore);
    }
    return pvscore;
  }

  INLINE bool check_pvline_score(const MoveLine &pline, float score) {
    return std::abs(score - get_pvline_score(pline)) < 1e-6;
  }

  decltype(auto) get_least_valuable_piece(piece_bitboard_t mask) const {
    // short-hands for bits
    const piece_bitboard_t diag = self.bits_slid_diag,
                           orth = self.bits_slid_orth;
    // get some piece that is minimal
    piece_bitboard_t found = 0x00ULL;
    for(const auto p : MATERIAL_PIECE_ORDERING) {
      switch(p) {
        case PAWN: if(mask & self.bits_pawns) found = mask & self.bits_pawns; break;
        case KNIGHT: if(mask & self.get_knight_bits()) found = mask & self.get_knight_bits(); break;
        case BISHOP: if(mask & diag & ~orth) found = mask & diag & ~orth; break;
        case ROOK: if(mask & ~diag & orth) found = mask & ~diag & orth; break;
        case QUEEN: if(mask & diag & orth) found = mask & diag & orth; break;
        case KING: if(mask & get_king_bits()) found = mask & get_king_bits(); break;
        default: break;
      }
      if(found)return std::make_pair(bitmask::highest_bit(found), p);
    }
    return std::make_pair(UINT64_C(0), EMPTY);
  }

  float static_exchange_evaluation(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
//    str::pdebug("move", pgn::_move_str(self, bitmask::_pos_pair(i, j)));
//    str::pdebug(fen::export_as_string(export_as_fen()));
    std::array<float, 37> gain = {0.};
    int16_t depth = 0;
    const piece_bitboard_t may_xray = bits_slid_diag | bits_slid_orth | bits_pawns;
    piece_bitboard_t from_set = piece::pos_mask(i);
    piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    piece_bitboard_t attadef = get_attacks_to(j, BOTH, occupied);
    assert(self[j].value != KING);
    gain[depth] = material_of(self[j].value);

    PIECE curpiece = self[i].value;
    do {
//      const pos_t from = bitmask::log2_of_exp2(from_set);
      ++depth;
      gain[depth] = (curpiece == KING ? 50. : material_of(curpiece)) - gain[depth - 1];
//      str::pdebug("SEE: depth:", depth, "piece:", board::_pos_str(from) + " "s + self[from].str(), "score:"s, gain[depth]);
      if(std::max(-gain[depth - 1], gain[depth]) < -1e-7) {
//        str::pdebug("SEE BREAK: max(", -gain[depth-1], ",", gain[depth], ") < 0");
        break;
      }
      attadef &= ~from_set;
      occupied &= ~from_set;
      if(from_set & may_xray) {
        if(from_set & bits_slid_diag) {
          attadef |= get_sliding_diag_attacks_to(j, occupied);
        }
        if(from_set & bits_slid_orth) {
          attadef |= get_sliding_orth_attacks_to(j, occupied);
        }
      }
      const COLOR c = self.color_at_pos((depth & 1) ? j : i);
      std::tie(from_set, curpiece) = get_least_valuable_piece(attadef & bits[c]);
    } while(from_set);
//    int16_t maxdepth = depth;
    while(--depth) {
      gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }
//    for(int16_t i = 0; i < maxdepth; ++i) {
//      str::pdebug("gain[", i, "] = ", gain[i]);
//    }
    return gain[0];
  }

  struct tt_eval_entry {
    board_info info;
    float eval;
  };

  INLINE float e_ttable_probe(zobrist::ttable<tt_eval_entry> &e_ttable) {
    return evaluate();
//    const zobrist::key_t k = zb_hash();
//    if(e_ttable[k].info == state.info) {
//      return e_ttable[k].eval;
//    }
//    const float eval = evaluate();
//    const bool overwrite = true;
//    if(overwrite) {
//      e_ttable[k] = { .info=state.info, .eval=eval };
//    }
//    return eval;
  }

  struct tt_ab_entry {
    board_info info;
    int16_t depth;
    float lowerbound, upperbound, exactscore;
    move_t m;
    MoveLine subpline;
    ply_index_t age;

    INLINE bool can_apply(const board_info &_info, int16_t _depth, ply_index_t _age) const {
      return !is_inactive(_age) && depth >= _depth && info == _info;
    }

    INLINE bool can_use_move(const board_info &_info, int16_t _depth, ply_index_t _age) const {
      return !is_inactive(_age) && ((depth >= (_depth - 1) && _depth > 3) || (depth <= 0)) && info == _info;
    }

    INLINE bool is_inactive(ply_index_t cur_age) const {
      return info.is_unset() || cur_age != age;
    }
  };

  INLINE decltype(auto) ab_ttable_probe(int16_t depth, const zobrist::ttable<tt_ab_entry> &ab_ttable) {
    const zobrist::key_t k = zb_hash();
    const bool tt_has_entry = ab_ttable[k].can_apply(state.info, depth, tt_age);
    if(tt_has_entry) {
      ++zb_hit;
      ++nodes_searched;
    } else {
      ++zb_miss;
    }
    return std::make_tuple(tt_has_entry, k);
  }

  INLINE bool is_knight_fork_move(pos_t i, pos_t j) const {
    if(piece::pos_mask(i) & ~get_knight_bits())return false;
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t attack_mask = get_attack_mask(ec);
    const piece_bitboard_t targets = (bits_slid_orth | ((bits_pawns | bits_slid_orth) & ~attack_mask) | get_king_bits()) & bits[ec];
    return piece::size(piece::get_knight_attack(j) & targets) >= 2;
  }

  INLINE bool is_pawn_fork_move(pos_t i, pos_t j) const {
    if(piece::pos_mask(i) & ~bits_pawns)return false;
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const pos_t jbit = piece::pos_mask(j);
    const piece_bitboard_t notpinned = bits[ec] & ~state.pins[ec];
    return (~(
            piece::get_knight_attacks(get_knight_bits() & notpinned)
            | piece::get_pawn_attacks(bits_pawns & notpinned, ec)
           ) & jbit)
           && (~piece::get_sliding_diag_attacks(bits_slid_diag & ~bits_slid_orth & notpinned, bits[c] | bits[ec]) & jbit)
           && piece::size(piece::get_pawn_attack(j, c) & bits[ec] & ~bits_pawns) == 2;
  }

  decltype(auto) ab_get_quiesc_moves(int16_t depth, MoveLine &pline, const std::vector<double> &countermove_table, int8_t nchecks, bool king_in_check, move_t firstmove=board::nullmove) const
  {
    std::vector<std::tuple<float, move_t, bool>> quiescmoves;
    quiescmoves.reserve(8);
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t pawn_attacks = piece::get_pawn_attacks(bits_pawns & bits[ec], ec);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      float val = .0;
      const move_t m = bitmask::_pos_pair(i, j);
      j &= board::MOVEMASK;
      bool reduce_nchecks = false;
      const bool allow_checking = (nchecks > 0 && -depth < 3);
      if(king_in_check) {
        val = move_heuristic_mvv_lva(i, j);
      } else if(is_promotion_move(i, j) || is_enpassant_take_move(i, j)) {
        val = move_heuristic_mvv_lva(i, j);
      } else if(is_naively_capture_move(i, j)) {
        if(material_of(self[i].value) + 1e-9 < material_of(self[j].value)) {
          val = move_heuristic_mvv_lva(i, j);
        } else {
          val = move_heuristic_see(i, j);
          if(val < -1e-9)return;
        }
      } else if((~pawn_attacks & piece::pos_mask(j)) && (is_knight_fork_move(i, j) || is_pawn_fork_move(i, j))) {
        const float see = static_exchange_evaluation(i, j);
        if(see < -1e-9)return;
        val = 0.;
      } else if((allow_checking && is_naively_checking_move(i, j))) {
        const float see = static_exchange_evaluation(i, j);
        if(see < -1e-9)return;
        reduce_nchecks = (m != firstmove);
        val = .5;
      } else {
        return;
      }
      if(val > -.75) {
        val += move_heuristic_pv(m, pline, firstmove);
      }
      val += move_heuristic_cmt(m, pline, countermove_table);
      quiescmoves.emplace_back(-val, m, reduce_nchecks);
    });
    std::sort(quiescmoves.begin(), quiescmoves.end());
    // no need for reverse sort: values are already negated
    return quiescmoves;
  }

  float alpha_beta_quiescence(float alpha, float beta, int16_t depth, MoveLine &pline,
                               zobrist::ttable<tt_ab_entry> &ab_ttable,
                               zobrist::ttable<tt_eval_entry> &e_ttable,
                               const std::vector<double> &countermove_table,
                               int8_t nchecks)
  {
    float score = -MATERIAL_KING;
    float bestscore = -MATERIAL_KING;

    const bool king_in_check = state.checkline[activePlayer()] != bitmask::full;
    if(!king_in_check) {
      score = e_ttable_probe(e_ttable);
      debug.update_standpat(depth, alpha, beta, score, pline, nchecks);
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

    move_t m_best = board::nullmove;
    const auto [tt_has_entry, k] = ab_ttable_probe(depth, ab_ttable);
    debug.update_line(depth, alpha, beta, pline);
    if(tt_has_entry) {
      const auto &zb = ab_ttable[k];
      debug.update_mem(depth, alpha, beta, zb, pline);
      if(zb.lowerbound >= beta || zb.upperbound <= alpha) {
        ++nodes_searched;
        pline.replace_line(zb.subpline);
        return zb.exactscore;
      }
      alpha = std::max(alpha, zb.lowerbound);
      beta = std::min(beta, zb.upperbound);
      m_best = zb.m;
    } else if(ab_ttable[k].can_use_move(state.info, depth, tt_age)) {
      const auto &zb = ab_ttable[k];
      m_best = zb.m;
    }

    constexpr bool overwrite = true;
    const bool tt_inactive_entry = ab_ttable[k].is_inactive(tt_age);
    const bool tt_replace_depth = tt_inactive_entry || depth >= ab_ttable[k].depth;
    const bool tt_inexact_entry = tt_inactive_entry || ab_ttable[k].lowerbound != ab_ttable[k].upperbound;

    decltype(auto) quiescmoves = ab_get_quiesc_moves(depth, pline, countermove_table, nchecks, king_in_check, m_best);
    if(quiescmoves.empty()) {
      ++nodes_searched;
      if(king_in_check) {
        score = e_ttable_probe(e_ttable);
      }
      return score;
    }
    assert(pline.empty());
    bool repetitions = false;
    for(size_t move_index = 0; move_index < quiescmoves.size(); ++move_index) {
      const auto &[_, m, reduce_nchecks] = quiescmoves[move_index];
      MoveLine pline_alt = pline.branch_from_past();
      {
        auto mscope = mline_scope(m, pline_alt);
        assert(pline_alt.empty());
        repetitions = can_draw_repetition();
        if(repetitions) {
          ++nodes_searched;
          score = 1e-4;
        } else {
          score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_ttable, e_ttable, countermove_table, reduce_nchecks ? nchecks - 1 : nchecks));
        }
      }
      debug.update(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
      debug.check_score(depth, score, pline_alt);
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite && tt_replace_depth && tt_inexact_entry && !repetitions) {
          if(tt_inactive_entry)++zb_occupied;
          ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=beta, .upperbound=FLT_MAX, .exactscore=score,
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
    if(overwrite && m_best != board::nullmove && !repetitions) {
      if((tt_replace_depth) && bestscore <= alpha) {
        if(tt_inactive_entry)++zb_occupied;
        ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=-FLT_MAX, .upperbound=alpha, .exactscore=bestscore,
                                      .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      } else if((tt_replace_depth || tt_inexact_entry) && bestscore > alpha) {
        if(tt_inactive_entry)++zb_occupied;
        ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=bestscore, .upperbound=bestscore, .exactscore=bestscore,
                                      .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      }
    }
    return bestscore;
  }

  decltype(auto) ab_get_ordered_moves(const MoveLine &pline, move_t firstmove, const std::vector<double> &countermove_table) {
    std::vector<std::pair<float, move_t>> moves;
    moves.reserve(32);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      float val = .0;
      val += move_heuristic_see(i, j);
      if(val > -.75 || m == firstmove) {
        val += move_heuristic_pv(m, pline, firstmove);
      }
      val += move_heuristic_cmt(m, pline, countermove_table);
      moves.emplace_back(-val, m);
    });
    std::sort(moves.begin(), moves.end());
    return moves;
  }

  template <typename F>
  float alpha_beta(float alpha, float beta, int16_t depth, MoveLine &pline,
                   zobrist::ttable<tt_ab_entry> &ab_ttable,
                   zobrist::ttable<tt_eval_entry> &e_ttable,
                   std::vector<double> &countermove_table,
                   int16_t initdepth, F &&callback_f,
                   bool allow_nullmoves=true,
                   const std::unordered_set<move_t> &searchmoves={})
  {
    if(depth <= 0) {
      assert(pline.empty());
      const int nchecks = 0;
      return alpha_beta_quiescence(alpha, beta, 0, pline, ab_ttable, e_ttable, countermove_table, nchecks);
    }
    move_t m_best = board::nullmove;
    const auto [tt_has_entry, k] = ab_ttable_probe(depth, ab_ttable);
    debug.update_line(depth, alpha, beta, pline);
    if(tt_has_entry) {
      const auto &zb = ab_ttable[k];
      debug.update_mem(depth, alpha, beta, zb, pline);
      if(zb.lowerbound >= beta || zb.upperbound <= alpha) {
        pline.replace_line(zb.subpline);
        ++nodes_searched;
        return zb.exactscore;
      }
      alpha = std::max(alpha, zb.lowerbound);
      beta = std::min(beta, zb.upperbound);
      m_best = zb.m;
    } else if(ab_ttable[k].can_use_move(state.info, depth, tt_age)) {
      const auto &zb = ab_ttable[k];
      m_best = zb.m;
    }

    constexpr bool overwrite = true;
    const bool tt_inactive_entry = ab_ttable[k].is_inactive(tt_age);
    const bool tt_replace_depth = tt_inactive_entry || depth >= ab_ttable[k].depth;
    const bool tt_inexact_entry = tt_inactive_entry || ab_ttable[k].lowerbound != ab_ttable[k].upperbound;

    // nullmove reduction
    const bool nullmove_allowed = (allow_nullmoves && state.checkline[activePlayer()] != bitmask::full
                                   && ((bits_pawns | get_king_bits()) & bits[activePlayer()] != bits[activePlayer()]));
    if(nullmove_allowed && depth >= 3) {
      const int16_t R = (depth > 6) ? 4 : 3;
      MoveLine pline_alt = pline.branch_from_past();
      float score = .0;
      {
        auto mscope = mline_scope(board::nullmove, pline_alt);
        score = -score_decay(alpha_beta(-alpha-1e-7, -alpha, depth-R-1, pline_alt, ab_ttable, e_ttable, countermove_table, initdepth, callback_f, false));
      }
      if(score >= beta) {
        depth -= 4;
        if(depth <= 0) {
          const int nchecks = 0;
          return alpha_beta_quiescence(alpha, beta, 0, pline, ab_ttable, e_ttable, countermove_table, nchecks);
        }
      }
    }

    decltype(auto) moves = ab_get_ordered_moves(pline, m_best, countermove_table);
    if(moves.empty()) {
      ++nodes_searched;
      return e_ttable_probe(e_ttable);
    }
    float bestscore = -FLT_MAX;
    bool repetitions = false;
    bool should_stop = false;
    for(size_t move_index = 0; move_index < moves.size(); ++move_index) {
      const auto [m_val, m] = moves[move_index];
      if(!searchmoves.empty() && searchmoves.find(m) == searchmoves.end()) {
        continue;
      } else if(initdepth == depth) {
        if(!callback_f(depth, bestscore)) {
          should_stop = true;
          str::pdebug("should stop");
          break;
        }
      }
      MoveLine pline_alt = pline.branch_from_past();
      float score;
      if(move_index == 0) {
        { // move scope
          auto mscope = mline_scope(m, pline_alt);
          assert(pline_alt.empty());
          repetitions = can_draw_repetition();
          if(repetitions) {
            ++nodes_searched;
            score = -1e-4;
          } else {
            score = -score_decay(alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_ttable, e_ttable, countermove_table, initdepth, callback_f, allow_nullmoves));
          }
        } // end move scope
        debug.update(depth, alpha, beta, bestscore, score, m, pline, pline_alt, "firstmove"s);
        debug.check_score(depth, score, pline_alt);
        bestscore = score;
        pline.replace_line(pline_alt);
        if(bestscore > alpha) {
          if(bestscore >= beta) {
            if(overwrite && tt_replace_depth && tt_inexact_entry && !repetitions) {
              if(tt_inactive_entry)++zb_occupied;
              ab_ttable[k] = { .info=state.info, .depth=depth, .lowerbound=beta, .upperbound=FLT_MAX, .exactscore=score,
                               .m=m, .subpline=pline.get_future(), .age=tt_age };
            }
            const pos_t i = bitmask::first(m), j = bitmask::second(m) & board::MOVEMASK;
            const move_t p_m = pline.get_previous_move();
            if(p_m != board::nullmove) {
              const size_t cmt_outer = board::NO_PIECE_INDICES * board::SIZE;
              const pos_t p_j = bitmask::second(p_m) & board::MOVEMASK;
              const size_t cmt_index = cmt_outer * (self[p_j].piece_index * board::SIZE + p_j) + (self[i].piece_index * board::SIZE + j);
              countermove_table[cmt_index] += (depth * depth * (double(move_index + 1) / double(moves.size()))) * 1e-8;
              if(countermove_table[cmt_index] > .5) {
                const double mx = *std::max_element(countermove_table.begin(), countermove_table.end());
                //str::print("renormalizing countermove table");
                for(auto &cm : countermove_table) {
                  cm /= (mx * 1e8);
                }
              }
            }
            return bestscore;
          }
          alpha = bestscore;
        }
      } else {
        { // move scope
          auto mscope = mline_scope(m, pline_alt);
          assert(pline_alt.empty());
          if(repetitions) {
            ++nodes_searched;
            score = -1e-4;
            if(score > alpha) {
              alpha = score;
            }
          } else {
//            const bool lmr_allowed = move_index >= 4 && depth >= 3 && (state.checkline[activePlayer()] != bitmask::full && m_val < 0);
            score = -score_decay(alpha_beta(-alpha-1e-7, -alpha, depth - 1, pline_alt, ab_ttable, e_ttable, countermove_table, initdepth, callback_f, allow_nullmoves));
            if(score > alpha && score < beta) {
              pline_alt.clear();
              score = -score_decay(alpha_beta(-beta, -alpha, depth - 1, pline_alt, ab_ttable, e_ttable, countermove_table, initdepth, callback_f, allow_nullmoves));
              if(score > alpha) {
                alpha = score;
              }
            }
          }
        } // end move scope
        debug.update(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
        debug.check_score(depth, score, pline_alt);
        if(score > bestscore) {
          pline.replace_line(pline_alt);
          if(score >= beta) {
            if(overwrite && tt_replace_depth && tt_inexact_entry && !repetitions) {
              if(tt_inactive_entry)++zb_occupied;
              ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=beta, .upperbound=FLT_MAX, .exactscore=score,
                                            .m=m, .subpline=pline.get_future(), .age=tt_age };
            }
            const pos_t i = bitmask::first(m), j = bitmask::second(m) & board::MOVEMASK;
            const move_t p_m = pline.get_previous_move();
            if(p_m != board::nullmove) {
              const size_t cmt_outer = board::NO_PIECE_INDICES * board::SIZE;
              const pos_t p_j = bitmask::second(p_m) & board::MOVEMASK;
              const size_t cmt_index = cmt_outer * (self[p_j].piece_index * board::SIZE + p_j) + (self[i].piece_index * board::SIZE + j);
              countermove_table[cmt_index] += (depth * depth * (double(move_index + 1) / double(moves.size()))) * 1e-8;
              if(countermove_table[cmt_index] > .5) {
                const double mx = *std::max_element(countermove_table.begin(), countermove_table.end());
                //str::print("renormalizing countermove table");
                for(auto &cm : countermove_table) {
                  cm /= (mx * 1e8);
                }
              }
            }
            return score;
          }
          bestscore = score;
        }
      }
    };
    if(should_stop) {
      return bestscore;
    }
    if(overwrite && m_best != board::nullmove && !repetitions) {
      if(bestscore <= alpha) {
        if(tt_inactive_entry)++zb_occupied;
        ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=-FLT_MAX, .upperbound=alpha, .exactscore=bestscore,
                                       .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      } else if(bestscore > alpha) {
        if(tt_inactive_entry)++zb_occupied;
        ab_ttable[k] = (tt_ab_entry){ .info=state.info, .depth=depth, .lowerbound=bestscore, .upperbound=bestscore, .exactscore=bestscore,
                                      .m=m_best, .subpline=pline.get_future(), .age=tt_age };
      }
    }
    return bestscore;
  }

  struct ab_storage_t {
    zobrist::StoreScope<tt_ab_entry> ab_ttable_scope;
    zobrist::StoreScope<tt_eval_entry> e_ttable_scope;
    std::vector<double> cmh_table;

    void reset() {
      ab_ttable_scope.reset();
      e_ttable_scope.reset();
    }
  };

  template <typename F>
  decltype(auto) iterative_deepening_dfs(int16_t depth, ab_storage_t &ab_storage, const std::unordered_set<move_t> &searchmoves, F &&callback_f) {
    auto &ab_ttable = ab_storage.ab_ttable_scope.get_object();
    auto &e_ttable = ab_storage.e_ttable_scope.get_object();
    auto &cmh_table = ab_storage.cmh_table;
    std::fill(cmh_table.begin(), cmh_table.end(), .0);
    MoveLine pline;
    float eval = .0;
    move_t m = board::nullmove;
    bool should_stop = false;
    for(int16_t d = 1; d <= depth; ++d) {
      if(!pline.empty() && check_line_terminates(pline) && int(pline.size()) < d) {
        eval = get_pvline_score(pline);
        break;
      }
      const float eval_prev = eval;
      const std::valarray<float> aspiration_window = {.01, .15, 1.25, MATERIAL_QUEEN - 1};
      pos_t aw_index_alpha = 0, aw_index_beta = 0;
      MoveLine new_pline = pline;
      do {
        float aw_alpha = -MATERIAL_KING, aw_beta = MATERIAL_KING;
        if(aw_index_alpha < aspiration_window.size()) {
          aw_alpha = eval_prev - aspiration_window[aw_index_alpha];
        }
        if(aw_index_beta < aspiration_window.size()) {
          aw_beta = eval_prev + aspiration_window[aw_index_beta];
        }
        const bool final_window = (aw_index_alpha == aspiration_window.size() && aw_index_beta == aspiration_window.size());
        str::pdebug("depth:", d, "aw", aw_alpha, aw_beta);
        int16_t new_depth = d - 1;
        const float new_eval = alpha_beta(aw_alpha, aw_beta, d, new_pline, ab_ttable, e_ttable, cmh_table, d,
          [&](int16_t _depth, float _eval) mutable -> bool {
            const move_t _m = new_pline.front();
            if(_depth == d && final_window && eval < _eval && _m != board::nullmove) {
              eval=_eval, m=_m, pline=new_pline, new_depth=d;
              str::pdebug("IDDFS:", d, "pline:", pgn::_line_str(self, pline), "size:", pline.size(), "eval", eval);
              debug.check_score(d, eval, pline);
            }
            return !(should_stop = !callback_f(new_depth, m, eval, pline, board::nullmove));
          }, true, searchmoves);
        if(new_eval < aw_alpha) {
          ++aw_index_alpha;
        } else if(new_eval > aw_beta) {
          ++aw_index_beta;
        } else {
          eval=new_eval, m=new_pline.front(), pline=new_pline, new_depth=d;
          debug.check_score(d, eval, pline);
          if(!callback_f(d, m, eval, pline, board::nullmove) || (score_is_mate(eval) && d > 0 && int(pline.size()) < d)) {
            should_stop = true;
          }
          break;
        }
        if(should_stop)break;
      } while(1);
      str::pdebug("IDDFS:", d, "pline:", pgn::_line_str(self, pline), "size:", pline.size(), "eval", eval);
      if(should_stop)break;
    }
    return std::make_pair(eval, m);
  }

  size_t nodes_searched = 0;
  float evaluation = MATERIAL_KING;
  ply_index_t tt_age = 0;
  size_t zb_hit = 0, zb_miss = 0, zb_occupied = 0;
  size_t ezb_hit = 0, ezb_miss = 0, ezb_occupied = 0;

  void reset_planning() {
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    ezb_hit = 0, ezb_miss = 0, ezb_occupied = 0;
    ++tt_age;
  }

  zobrist::ttable_ptr<tt_ab_entry> ab_ttable = nullptr;
  zobrist::ttable_ptr<tt_eval_entry> e_ttable = nullptr;

  decltype(auto) get_zobrist_alphabeta_scope() {
    constexpr size_t size_ab = ZOBRIST_SIZE;
    constexpr size_t mem_ab = size_ab * (sizeof(tt_ab_entry) + 16 * sizeof(move_t));
    constexpr size_t size_e = 0;
    constexpr size_t mem_e = size_e * sizeof(tt_eval_entry);
    constexpr size_t size_cmh = size_t(board::NO_PIECE_INDICES) * size_t(board::SIZE);
    constexpr size_t size_cmh_twice = size_cmh * size_cmh;
    constexpr size_t mem_cmh = size_cmh_twice * sizeof(double);
    constexpr size_t mem_total = mem_ab+mem_e+mem_cmh;
    str::pdebug("alphabeta scope", "ab:", mem_ab, "e:", mem_e, "cmh:", mem_cmh, "total:", mem_total);
    _printf("MEM: %luMB %luKB %luB\n", mem_total>>20, (mem_total>>10)&((1<<10)-1), mem_total&((1<<10)-1));
    return (ab_storage_t){
      .ab_ttable_scope=zobrist::make_store_object_scope<tt_ab_entry>(ab_ttable, size_ab),
      .e_ttable_scope=zobrist::make_store_object_scope<tt_eval_entry>(e_ttable, size_e),
      .cmh_table=std::vector<double>(size_cmh_twice, .0)
    };
  }

  INLINE decltype(auto) make_callback_f() {
    return [&](int16_t depth, move_t m, float eval, const MoveLine &pline, move_t ponder_m) -> bool {
      return true;
    };
  }

  template <typename F>
  move_t get_fixed_depth_move_iddfs(int16_t depth, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    reset_planning();
    decltype(auto) ab_storage = get_zobrist_alphabeta_scope();
    assert(ab_ttable != nullptr && e_ttable != nullptr);
    debug.set_depth(depth);
    const auto [_, m] = iterative_deepening_dfs(depth, ab_storage, searchmoves, std::forward<F>(callback_f));
    evaluation = _;
    return m;
  }

  INLINE move_t get_fixed_depth_move_iddfs(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    return get_fixed_depth_move_iddfs(depth, make_callback_f(), searchmoves);
  }

  struct tt_perft_entry {
    board_info info;
    int16_t depth;
    size_t nodes;
  };

  zobrist::ttable_ptr<tt_perft_entry> perft_ttable = nullptr;
  decltype(auto) get_zobrist_perft_scope() {
    constexpr size_t size_perft = ZOBRIST_SIZE;
    constexpr size_t mem_perft = size_perft * sizeof(tt_perft_entry);
    return zobrist::make_store_object_scope<tt_perft_entry>(perft_ttable, size_perft);
  }
  size_t _perft(int16_t depth, std::vector<tt_perft_entry> &perft_ttable) {
    if(depth == 1 || depth == 0) {
      return count_moves(activePlayer());
    }
    // look-up:
    const zobrist::key_t k = zb_hash();
    if(perft_ttable[k].info == state.info && perft_ttable[k].depth == depth) {
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
      auto mscope = move_scope(bitmask::_pos_pair(i, j));
      ++nodes_searched;
      nodes += _perft(depth - 1, perft_ttable);
    });
    if(overwrite) {
      perft_ttable[k] = { .info=state.info, .depth=depth, .nodes=nodes };
    }
    return nodes;
  }

  inline size_t perft(int16_t depth=1) {
    decltype(auto) store_scope = get_zobrist_perft_scope();
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    return _perft(depth, store_scope.get_object());
  }

  DebugTracer<Engine> debug;
  Engine(const fen::FEN fen=fen::starting_pos):
    Board(fen), debug(*this)
  {
    const std::vector<PIECE> piece_types = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    std::vector<std::pair<float, PIECE>> pieces;
    for(PIECE p : piece_types) {
      pieces.emplace_back(material_of(p), p);
    }
    std::sort(pieces.begin(), pieces.end());
    for(const auto &[_, p] : pieces) {
      MATERIAL_PIECE_ORDERING.emplace_back(p);
    }
    hstate_hist.reserve(100);
    _init_hstate();
  }

  virtual ~Engine() {}
};
