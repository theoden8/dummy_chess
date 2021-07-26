#pragma once


#include <algorithm>
#include <valarray>
#include <map>

#include <FEN.hpp>
#include <PGN.hpp>
#include <Board.hpp>
#include <DebugTracer.hpp>


class Engine : public Board {
public:
  static constexpr bool ENABLE_TT = true;
  static constexpr bool ENABLE_SELECTIVITY = true;

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
  INLINE void iter_drop_moves(F &&func) const {
    if(crazyhouse) {
      if(is_draw_halfmoves()||is_draw_material())return;
      const COLOR c = activePlayer();
      const piece_bitboard_t ch_drop_locations = ~(bits[WHITE]|bits[BLACK]) & state.checkline[c];
      {
        if(n_subs[Piece::get_piece_index(PAWN, c)]) {
          bitmask::foreach(ch_drop_locations & board::PAWN_RANKS, [&](pos_t j) mutable -> void {
            func(board::DROP_PAWN, j);
          });
        }
      }
      for(PIECE p : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        if(n_subs[Piece::get_piece_index(p, c)]) {
          bitmask::foreach(ch_drop_locations, [&](pos_t j) mutable -> void {
            func(pos_t(p) | board::CRAZYHOUSE_DROP, j);
          });
        }
      }
    }
  }

  template <typename F>
  INLINE void iter_moves(F &&func) const {
    const COLOR c = activePlayer();
    bitmask::foreach(bits[c], [&](pos_t i) mutable -> void {
      iter_moves_from(i, std::forward<F>(func));
    });
    iter_drop_moves(std::forward<F>(func));
  }

  INLINE size_t count_moves(COLOR c) const {
    assert(c < NO_COLORS);
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
    if(crazyhouse) {
      const COLOR c = activePlayer();
      const piece_bitboard_t ch_drop_locations = ~(bits[WHITE]|bits[BLACK]) & state.checkline[c];
      if(n_subs[Piece::get_piece_index(PAWN, c)]) {
        no_moves += piece::size(ch_drop_locations & board::PAWN_RANKS);
      }
      pos_t npieces = 0;
      for(PIECE p : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        if(n_subs[Piece::get_piece_index(p, c)]) {
          ++npieces;
        }
      }
      no_moves += npieces * piece::size(ch_drop_locations);
    }
    return no_moves;
  }

  // for MC-style testing
  INLINE move_t get_random_move() const {
    std::vector<move_t> moves;
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  INLINE move_t get_random_move_from(pos_t i) const {
    std::vector<move_t> moves;
    iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      moves.emplace_back(bitmask::_pos_pair(i, j));
    });
    if(moves.empty())return board::nullmove;
    return moves[rand() % moves.size()];
  }

  static constexpr int32_t CENTIPAWN = 256,
                           MATERIAL_PAWN = 100*CENTIPAWN,
                           MATERIAL_KNIGHT = 325*CENTIPAWN,
                           MATERIAL_BISHOP = 325*CENTIPAWN,
                           MATERIAL_ROOK = 500*CENTIPAWN,
                           MATERIAL_QUEEN = 1000*CENTIPAWN,
                           MATERIAL_KING = 1000*MATERIAL_PAWN;
  std::vector<PIECE> MATERIAL_PIECE_ORDERING;

  static constexpr int32_t MATERIALS[] = {
    MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP,
    MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING, 0};
  INLINE constexpr int32_t material_of(PIECE p) const {
    return MATERIALS[int(p)];
  }

  INLINE int32_t count_material(piece_bitboard_t mask) const {
    int32_t m = 0;
    m += piece::size(mask & bits_pawns) * material_of(PAWN);
    m += piece::size(mask & bits_slid_diag & ~bits_slid_orth) * material_of(BISHOP);
    m += piece::size(mask & ~bits_slid_diag & bits_slid_orth) * material_of(ROOK);
    m += piece::size(mask & bits_slid_diag & bits_slid_orth) * material_of(QUEEN);
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

  INLINE int32_t h_material(COLOR c) const {
    int32_t h = count_material(bits[c]);
    const piece_bitboard_t bishops = bits[c] & bits_slid_diag & ~bits_slid_orth;
    // bishop pair
    if((bishops & bitmask::wcheckers) && (bishops & bitmask::bcheckers)) {
      h += 10*CENTIPAWN;
    }
    if(bits_pawns & bits[c]) {
      h += 10*CENTIPAWN;
    } else {
      const piece_bitboard_t side = piece::file_mask(A) | piece::file_mask(H);
      h -= 2*CENTIPAWN * piece::size(bits[c] & bits_pawns & side);
      h += 1*CENTIPAWN * piece::size(bits[c] & bits_pawns & ~side);
    }
    if(crazyhouse) {
      h += MATERIAL_PAWN * n_subs[Piece::get_piece_index(PAWN, c)];
      h += (MATERIAL_KNIGHT + 50*CENTIPAWN) * n_subs[Piece::get_piece_index(KNIGHT, c)];
      h += MATERIAL_BISHOP * n_subs[Piece::get_piece_index(BISHOP, c)];
      h += MATERIAL_ROOK * n_subs[Piece::get_piece_index(ROOK, c)];
      h += MATERIAL_QUEEN * n_subs[Piece::get_piece_index(QUEEN, c)];
    }
    return h;
  }

  INLINE int32_t h_attack_cells(COLOR c) const {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    int32_t h = .0;

    decltype(auto) h_add = [&](pos_t i, int32_t mat) mutable -> void {
      auto a = state.attacks[i];
      h += (piece::size(a & occupied) + piece::size(a)) * MATERIAL_PAWN / mat;
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

  INLINE int32_t h_mobility(COLOR c) const {
    int32_t h = 0;
    bitmask::foreach(bits[c] & ~bits_pawns, [&](pos_t i) mutable -> void {
      h += std::sqrt(piece::size(state.moves[i]));
    });
    return h;
  }

  INLINE int32_t h_pawn_structure(COLOR c) const {
    int32_t h = 0;
    // zero/doubled/tripple pawns
    for(pos_t f = 0; f < board::LEN; ++f) {
      const piece_bitboard_t pawns_file = piece::file_mask(f) & bits_pawns & bits[c];
      if(!pawns_file) {
        h -= 2*CENTIPAWN;
      } else {
        h += 1*CENTIPAWN * (3 - piece::size(pawns_file) * 2);
      }
    }
    for(pos_t f = A; f <= H; ++f) {
      const piece_bitboard_t pawns_file = piece::file_mask(f) & bits_pawns & bits[c];
      if(!pawns_file)continue;
      const piece_bitboard_t furthest_pawn = (c == WHITE) ? bitmask::highest_bit(pawns_file) : bitmask::lowest_bit(pawns_file);
      const pos_t pawnr = (c == WHITE) ? board::_y(bitmask::log2(pawns_file))
                                       : board::_y(bitmask::log2_msb(pawns_file));
      piece_bitboard_t adjacent_files = piece::file_mask(f);
      int32_t sidepawn_decay = 1;
      if(f != A)adjacent_files |= piece::file_mask(f - 1);
      if(f != H)adjacent_files |= piece::file_mask(f + 1);
      if(f == A || f == H) {
        sidepawn_decay = 2;
      }
      // isolated pawns
      if(!(adjacent_files & ~pawns_file & bits_pawns & bits[c])) {
        h -= CENTIPAWN / 4 / sidepawn_decay;
      } else {
        h += CENTIPAWN / 4 / sidepawn_decay;
      }
      // pass-pawns
      const piece_bitboard_t ahead = (c == WHITE) ? adjacent_files << (board::LEN * (1+pawnr))
                                                  : adjacent_files >> (board::LEN * (8-pawnr));
      assert(~ahead & furthest_pawn);
      if((ahead & bits_pawns & ~bits[c]))continue;
      const int32_t pawnr_abs = ((c == WHITE) ? pawnr : 7 - pawnr);
      h += (CENTIPAWN / 10) * pawnr_abs * pawnr_abs / sidepawn_decay;
    }
    // pawn strucure coverage
    h += (CENTIPAWN / 2) * piece::size(piece::get_pawn_attacks(bits_pawns & bits[c], c) & bits[c]);
    return h;
  }

  INLINE int32_t heuristic_of(COLOR c) const {
    int32_t h = 0;
    h += h_material(c);
    h += h_pawn_structure(c);
//    h -= count_material(state.pins[c]);
//    h += h_mobility(c);
    h += h_attack_cells(c);
    return h;
  }

  INLINE int32_t evaluate() const {
    if(self.is_draw()){
      return 0;
    } else if(self.is_checkmate()) {
      return -MATERIAL_KING;
    }
    const COLOR c = activePlayer();
    return heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  INLINE float move_heuristic_extra_material(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(crazyhouse && is_drop_move(i, j)) {
    } else if(is_promotion_move(i, _j)) {
      return float(material_of(board::get_promotion_as(j)) - material_of(PAWN)) / MATERIAL_PAWN;
    } else if(is_enpassant_take_move(i, _j)) {
      return float(material_of(PAWN)) / MATERIAL_PAWN;
    }
    return .0;
  }

  INLINE float move_heuristic_check(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(crazyhouse && is_drop_move(i, j)) {
    } else if(is_naively_checking_move(i, _j)) {
      return .5;
    }
    return .0;
  }

  INLINE float move_heuristic_attacker_decay(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    float val = .0;
    const PIECE frompiece = self[i].value;
    const float mat_from = (frompiece == KING ? 50. : float(material_of(frompiece)) / MATERIAL_PAWN);
    const pos_t _j = j & board::MOVEMASK;
    if(self.empty_at_pos(_j) || (edefendmap & piece::pos_mask(_j))) {
      const float decay_i = (edefendmap & piece::pos_mask(_j)) ? .02 : .05;
      val -= mat_from * decay_i;
    } else {
      val -= mat_from * .01;
    }
    return val;
  }

  INLINE float move_heuristic_see(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    const pos_t _j = j & board::MOVEMASK;
    float val = .0;
    const float extra_val = move_heuristic_extra_material(i, j);
    val += extra_val;
    if(crazyhouse && is_drop_move(i, _j)) {
      return .0;
    } else if(val > 0 || is_castling_move(i, _j)) {
      return val;
    }
    const float mat_i = float(material_of(self[i].value)) / MATERIAL_PAWN,
                mat_j = float(material_of(self[_j].value)) / MATERIAL_PAWN;
    if(~edefendmap & piece::pos_mask(_j)) {
      return mat_j;
    } else if(mat_i < mat_j) {
      return mat_j - mat_i - extra_val;
    }
    const float see = float(static_exchange_evaluation(i, _j)) / MATERIAL_PAWN;
    val += see;
    if(see < 0) {
      val -= extra_val;
    }
    return val;
  }

  INLINE float move_heuristic_mvv_lva(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    const pos_t _j = j & board::MOVEMASK;
    float val = .0;
    if(is_drop_move(i, _j)) {
      ;
    } else if(is_castling_move(i, _j)) {
      val += .5;
    } else if(is_naively_capture_move(i, _j)) {
      const float mat_to = float(material_of(self[_j].value)) / MATERIAL_PAWN;
      val += mat_to;
      const PIECE frompiece = self[i].value;
      const float mat_from = (frompiece == KING ? 50. : float(material_of(frompiece)) / MATERIAL_PAWN);
      if(edefendmap & piece::pos_mask(_j)) {
        val -= mat_from;
      }
      val += move_heuristic_attacker_decay(i, j, edefendmap);
    }
    val += move_heuristic_extra_material(i, j);
    val += move_heuristic_check(i, j);
    return val;
  }

  INLINE float move_heuristic_pv(move_t m, const MoveLine &pline, move_t firstmove=board::nullmove) const {
    if(pline.is_mainline() && m == pline.front_in_mainline()) {
      return 2000.;
    } else if(m == firstmove) {
      return 1000.;
    }
    return .0;
  }

  INLINE size_t get_cmt_index(const MoveLine &pline, move_t m) const {
    constexpr size_t NO_INDEX = SIZE_MAX;
    const move_t p_m = pline.get_previous_move();
    if(m != board::nullmove && p_m != board::nullmove && !(crazyhouse && is_drop_move(bitmask::first(m), bitmask::second(m)))) {
      const pos_t i = bitmask::first(m), _j = bitmask::second(m);
      const pos_t p_i = bitmask::first(p_m) & board::MOVEMASK;
      const pos_t p_j = bitmask::second(p_m) & board::MOVEMASK;
      // castling
      if(self.empty_at_pos(p_j) || is_castling_move(p_i, p_j)) {
        return NO_INDEX;
      }
      // crazyhouse, i is invalid
      const size_t i_piecevalue = size_t(is_drop_move(i, _j) ? board::get_drop_as(i) : self[i].value);
      const size_t cmt_outer = size_t(NO_PIECES) * size_t(board::SIZE);
      const size_t cmt_index = cmt_outer * cmt_outer * size_t(activePlayer() == WHITE ? 0 : 1)
                                         + cmt_outer * (size_t(self[p_j].value) * size_t(board::SIZE) + p_j)
                                                     + (i_piecevalue * size_t(board::SIZE) + _j);
      return cmt_index;
    }
    return NO_INDEX;
  }

  INLINE float move_heuristic_cmt(move_t m, const MoveLine &pline, const std::vector<double> &cmh_table) const {
    const size_t cmt_index = get_cmt_index(pline, m);
    if(cmt_index != SIZE_MAX) {
      return cmh_table[cmt_index];
    }
    return .0;
  }

  static INLINE float score_float(int32_t score) {
    return float(score)  / MATERIAL_PAWN;
  }

  static INLINE bool score_is_mate(int32_t score) {
    return std::abs(score) > MATERIAL_KING - 2000;
  }

  static INLINE int32_t score_decay(int32_t score) {
    if(score_is_mate(score)) {
      score -= score / std::abs(score);
    }
    return score;
  }

  static INLINE int16_t score_mate_in(int32_t score) {
    assert(score_is_mate(score));
    const int mate_in = int(MATERIAL_KING - std::abs(score));
    return (score < 0) ? -mate_in : mate_in;
  }

  INLINE int32_t get_pvline_score(const MoveLine &pline) {
    assert(check_valid_sequence(pline));
    auto rec = self.recursive_move_scope();
    for(const move_t &m : pline) {
      rec.scope(m);
    }
    int32_t pvscore = evaluate();
    for(size_t i = 0; i < pline.size(); ++i) {
      pvscore = -score_decay(pvscore);
    }
    return pvscore;
  }

  INLINE bool check_pvline_score(const MoveLine &pline, int32_t score) {
    return score == get_pvline_score(pline);
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

  int32_t static_exchange_evaluation(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
//    str::pdebug("move", pgn::_move_str(self, bitmask::_pos_pair(i, j)));
//    str::pdebug(fen::export_as_string(export_as_fen()));
    std::array<int32_t, 37> gain;
    int16_t depth = 0;
    const piece_bitboard_t may_xray = bits_slid_diag | bits_slid_orth | bits_pawns;
    piece_bitboard_t from_set = piece::pos_mask(i);
    piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    piece_bitboard_t attadef = get_attacks_to(j, BOTH, occupied);
    // optional: remove pinned pieces (inaccurately)
    if(!(attadef & get_king_bits())) {
      attadef &= ~(state.pins[WHITE] | state.pins[BLACK]);
    }
    assert(self[j].value != KING);
    gain[depth] = material_of(self[j].value);

    PIECE curpiece = self[i].value;
    do {
//      const pos_t from = bitmask::log2_of_exp2(from_set);
      ++depth;
      gain[depth] = (curpiece == KING ? 50*MATERIAL_PAWN : material_of(curpiece)) - gain[depth - 1];
//      str::pdebug("SEE: depth:", depth, "piece:", board::_pos_str(from) + " "s + self[from].str(), "score:"s, gain[depth]);
      if(std::max(-gain[depth-1], gain[depth]) < 0) {
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
      gain[depth-1] = -std::max(-gain[depth-1], gain[depth]);
    }
//    for(int16_t i = 0; i < maxdepth; ++i) {
//      str::pdebug("gain[", i, "] = ", gain[i]);
//    }
    return gain[0];
  }

  struct tt_eval_entry {
    board_info info;
    int32_t eval;
  };

  INLINE int32_t e_ttable_probe(zobrist::ttable<tt_eval_entry> &e_ttable) {
    return evaluate();
//    const zobrist::key_t k = zb_hash();
//    if(e_ttable[k].info == state.info) {
//      return e_ttable[k].eval;
//    }
//    const int32_t eval = evaluate();
//    const bool overwrite = true;
//    if(overwrite) {
//      e_ttable[k] = { .info=state.info, .eval=eval };
//    }
//    return eval;
  }

  typedef enum { TT_EXACT, TT_ALLNODE, TT_CUTOFF } TT_NODE_TYPE;
  struct tt_ab_entry {
    board_info info;
    int16_t depth;
    int32_t score;
    move_t m;
    ply_index_t age;
    TT_NODE_TYPE ndtype;
    MoveLine subpline;

    INLINE bool can_apply(const board_info &_info, int16_t _depth, ply_index_t _age) const {
      return !is_inactive(_age) && depth >= _depth && info == _info;
    }

    INLINE bool can_use_move(const board_info &_info, int16_t _depth, ply_index_t _age) const {
      if(is_inactive(_age))return false;
      if(depth <= 0) {
        return info == _info;
      }
      return (depth >= _depth - 1 || ndtype == TT_EXACT || (ndtype == TT_CUTOFF && depth >= depth - 3))
        && info == _info;
    }

    INLINE bool is_inactive(ply_index_t cur_age) const {
      return info.is_unset() || cur_age != age;
    }

    INLINE void write(board_info &_info, int32_t _score, int16_t _depth, move_t _m, ply_index_t _age, TT_NODE_TYPE _ndtype, const MoveLine &pline) {
      info=_info, depth=_depth, score=_score, m=_m, age=_age, ndtype=_ndtype;
      subpline=pline.get_future();
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

  INLINE bool has_promoting_pawns() const {
    const COLOR c = activePlayer();
    const piece_bitboard_t pawnrank = piece::rank_mask(c == WHITE ? -1+1 : -1+8);
    const piece_bitboard_t promoting_pawns = bits[c] & bits_pawns & pawnrank;
    bool has_moves = false;
    bitmask::foreach_early_stop(promoting_pawns, [&](pos_t i) mutable -> bool {
      has_moves = state.moves[i];
      return !has_moves;
    });
    return has_moves;
  }

  decltype(auto) ab_get_quiesc_moves(int16_t depth, MoveLine &pline, const std::vector<double> &cmh_table, int8_t nchecks, bool king_in_check, move_t firstmove=board::nullmove) const
  {
    std::vector<std::tuple<float, move_t, bool>> quiescmoves;
    quiescmoves.reserve(8);
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t edefendmap = get_attack_mask(ec) & bits[ec];
    const piece_bitboard_t pawn_attacks = piece::get_pawn_attacks(bits_pawns & bits[ec], ec);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      float val = .0;
      const move_t m = bitmask::_pos_pair(i, j);
      j &= board::MOVEMASK;
      bool reduce_nchecks = false;
      const bool allow_checking = (nchecks > 0 && -depth < 3);
      if(king_in_check) {
        val = move_heuristic_mvv_lva(i, j, edefendmap);
      } else if(crazyhouse && is_drop_move(i, j)) {
        return;
      } else if(is_promotion_move(i, j) || is_enpassant_take_move(i, j)) {
        val = move_heuristic_see(i, j, edefendmap);
        if(val < -1e-9)return;
      } else if(is_naively_capture_move(i, j)) {
        if(material_of(self[i].value) + 1e-9 < material_of(self[j].value) && (~edefendmap & piece::pos_mask(j))) {
          val = move_heuristic_mvv_lva(i, j, edefendmap) + .01;
        } else {
          val = move_heuristic_see(i, j, edefendmap);
          if(val < -1e-9)return;
        }
      } else if((~pawn_attacks & piece::pos_mask(j)) && (is_knight_fork_move(i, j) || is_pawn_fork_move(i, j))) {
        const int32_t see = static_exchange_evaluation(i, j);
        if(see < 0)return;
        val = .0;
      } else if((allow_checking && is_naively_checking_move(i, j))) {
        const int32_t see = static_exchange_evaluation(i, j);
        if(see < 0)return;
        reduce_nchecks = (m != firstmove);
        val = .5;
      } else {
        return;
      }
      if(val > -.75) {
        val += move_heuristic_pv(m, pline, firstmove);
      }
      val += move_heuristic_cmt(m, pline, cmh_table);
      quiescmoves.emplace_back(val, m, reduce_nchecks);
    });
    std::sort(quiescmoves.begin(), quiescmoves.end(), std::greater<>());
    return quiescmoves;
  }

  struct alpha_beta_state {
    zobrist::ttable<tt_ab_entry> &ab_ttable;
    zobrist::ttable<tt_eval_entry> &e_ttable;
    std::vector<double> &cmh_table;
    const int16_t initdepth;
    const std::unordered_set<move_t> &searchmoves;

    void normalize_cmh_table(size_t cmt_index) {
      if(cmh_table[cmt_index] > .5) {
        const double mx = *std::max_element(cmh_table.begin(), cmh_table.end());
        str::pdebug("renormalizing countermove table");
        for(auto &cm : cmh_table) {
          cm /= (mx * 1e8);
        }
      }
    }
  };

  int32_t alpha_beta_quiescence(int32_t alpha, int32_t beta, int16_t depth, MoveLine &pline, alpha_beta_state &ab_state, int8_t nchecks) {
    if(is_draw()) {
      pline.clear();
      return 0;
    } else if(is_checkmate()) {
      return -MATERIAL_KING;
    }

    int32_t score = -MATERIAL_KING;
    int32_t bestscore = -MATERIAL_KING;

    const COLOR c = activePlayer();
    const bool king_in_check = state.checkline[c] != bitmask::full;
    if(!king_in_check) {
      score = e_ttable_probe(ab_state.e_ttable);
      debug.update_standpat(depth, alpha, beta, score, pline, nchecks);
      if(score >= beta) {
        ++nodes_searched;
        return score;
      } else if(score > bestscore) {
        bestscore = score;
        const int32_t BIG_DELTA = MATERIAL_QUEEN + (has_promoting_pawns() ? MATERIAL_QUEEN - MATERIAL_PAWN : 0);
        if(score > alpha) {
          alpha = score;
        } else if(ENABLE_SELECTIVITY && score < alpha - BIG_DELTA) {
          return score;
        }
      }
    }

    move_t m_best = board::nullmove;
    const auto [tt_has_entry, k] = ab_ttable_probe(depth, ab_state.ab_ttable);
    debug.update_line(depth, alpha, beta, pline);
    auto &zb = ab_state.ab_ttable[k];
    if(tt_has_entry) {
      debug.update_mem(depth, alpha, beta, zb, pline);
      if(zb.ndtype == TT_EXACT || (zb.score < alpha && zb.ndtype == TT_ALLNODE) || (zb.score >= beta && zb.ndtype == TT_CUTOFF)) {
        ++nodes_searched;
        pline.replace_line(zb.subpline);
        return zb.score;
      } else if(alpha < zb.score && zb.score < beta) {
        if(zb.ndtype == TT_CUTOFF) {
          alpha = zb.score;
        } else if(zb.ndtype == TT_ALLNODE) {
          beta = zb.score;
        } else {
          abort();
        }
      }
      m_best = zb.m;
    } else if(zb.can_use_move(state.info, depth, tt_age)) {
      m_best = zb.m;
    }

    constexpr bool overwrite = ENABLE_TT;
    const bool tt_inactive_entry = zb.is_inactive(tt_age);
    const bool tt_replace_depth = tt_inactive_entry || depth >= zb.depth;
    const bool tt_inexact_entry = tt_inactive_entry || zb.ndtype != TT_EXACT;

    decltype(auto) quiescmoves = ab_get_quiesc_moves(depth, pline, ab_state.cmh_table, nchecks, king_in_check, m_best);
    if(quiescmoves.empty()) {
      ++nodes_searched;
      if(king_in_check) {
        score = e_ttable_probe(ab_state.e_ttable);
      }
      return score;
    }
    bool isdraw_pathdep = false;
    constexpr int32_t DELTA = 300 * CENTIPAWN;
    const bool delta_prune_allowed = ENABLE_SELECTIVITY && !king_in_check && piece::size(bits[c] & ~bits_pawns) > 5;
    TT_NODE_TYPE ndtype = TT_ALLNODE;
    for(size_t move_index = 0; move_index < quiescmoves.size(); ++move_index) {
      const auto [m_val, m, reduce_nchecks] = quiescmoves[move_index];
      MoveLine pline_alt = pline.branch_from_past();
      if(delta_prune_allowed && score + std::round(m_val * MATERIAL_PAWN) < alpha - DELTA) {
        continue;
      }
      {
        auto mscope = mline_scope(m, pline_alt);
        isdraw_pathdep = is_draw_halfmoves() || is_draw_repetition();
        score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_state, reduce_nchecks ? nchecks - 1 : nchecks));
      }
      debug.update_q(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
      debug.check_score(depth, score, pline_alt);
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite && tt_replace_depth && tt_inexact_entry && !isdraw_pathdep && !pline.find(board::nullmove)) {
          if(tt_inactive_entry)++zb_occupied;
          zb.write(state.info, score, depth, m, tt_age, TT_CUTOFF, pline);
        }
        return score;
      } else if(score > bestscore) {
        pline.replace_line(pline_alt);
        bestscore = score;
        m_best = m;
        if(score > alpha) {
          alpha = score;
          ndtype = TT_EXACT;
        }
      }
    }
    if(overwrite && m_best != board::nullmove && !isdraw_pathdep && !pline.find(board::nullmove)) {
      if(tt_replace_depth && ndtype == TT_ALLNODE) {
        if(tt_inactive_entry)++zb_occupied;
        zb.write(state.info, bestscore, depth, m_best, tt_age, TT_ALLNODE, pline);
      } else if((tt_replace_depth || tt_inexact_entry) && ndtype == TT_EXACT) {
        if(tt_inactive_entry)++zb_occupied;
        zb.write(state.info, bestscore, depth, m_best, tt_age, TT_EXACT, pline);
      }
    }
    return bestscore;
  }

  decltype(auto) ab_get_ordered_moves(const MoveLine &pline, move_t firstmove, const std::vector<double> &cmh_table) {
    std::vector<std::pair<float, move_t>> moves;
    moves.reserve(32);
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(activePlayer());
    const piece_bitboard_t edefendmap = get_attack_mask(ec) & bits[ec];
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      float val = .0;
      val += move_heuristic_pv(m, pline, firstmove);
      if(val < 1000.) {
        val += move_heuristic_see(i, j, edefendmap);
        val += move_heuristic_cmt(m, pline, cmh_table);
      }
      moves.emplace_back(val, m);
    });
    std::sort(moves.begin(), moves.end(), std::greater<>());
    return moves;
  }

  template <typename F>
  int32_t alpha_beta_pv(int32_t alpha, int32_t beta, int16_t depth, MoveLine &pline,
                   alpha_beta_state &ab_state,
                   bool allow_nullmoves, bool scoutsearch,
                   F &&callback_f)
  {
    if(is_draw()) {
      pline.clear();
      return .0;
    } else if(is_checkmate()) {
      return -MATERIAL_KING;
    }
    if(depth <= 0) {
      const int nchecks = 0;
      return alpha_beta_quiescence(alpha, beta, 0, pline, ab_state, nchecks);
    }
    move_t m_best = board::nullmove;
    const auto [tt_has_entry, k] = ab_ttable_probe(depth, ab_state.ab_ttable);
    debug.update_line(depth, alpha, beta, pline);
    auto &zb = ab_state.ab_ttable[k];
    if(tt_has_entry) {
      debug.update_mem(depth, alpha, beta, zb, pline);
      debug.check_score(depth, zb.score, zb.subpline);
      if(zb.ndtype == TT_EXACT || (zb.score < alpha && zb.ndtype == TT_ALLNODE) || (zb.score >= beta && zb.ndtype == TT_CUTOFF)) {
        ++nodes_searched;
        pline.replace_line(zb.subpline);
        return zb.score;
      } else if(alpha < zb.score && zb.score < beta) {
        if(zb.ndtype == TT_CUTOFF) {
          alpha = zb.score;
        } else if(zb.ndtype == TT_ALLNODE) {
          beta = zb.score;
        } else {
          abort();
        }
      }
      m_best = zb.m;
    } else if(zb.can_use_move(state.info, depth, tt_age)) {
      m_best = zb.m;
    }

    constexpr bool overwrite = ENABLE_TT;
    const bool tt_inactive_entry = zb.is_inactive(tt_age);
    const bool tt_replace_depth = tt_inactive_entry || depth >= zb.depth;
    const bool tt_inexact_entry = tt_inactive_entry || zb.ndtype != TT_EXACT;

    // nullmove pruning
    // perform this inside a scout
    const COLOR c = activePlayer();
    const bool nullmove_allowed = ENABLE_SELECTIVITY && scoutsearch && allow_nullmoves && (state.checkline[c] == bitmask::full)
                                && piece::size((bits_slid_diag | get_knight_bits() | bits_slid_orth) & bits[c]) > 0;
    if(nullmove_allowed) {
      const int16_t R = (depth > 6) ? 4 : 3;
      const int16_t DR = 4;
      MoveLine pline_alt = pline.branch_from_past();
      int32_t score = 0;
      {
        auto mscope = mline_scope(board::nullmove, pline_alt);
        score = -score_decay(alpha_beta_pv(-beta-1, -beta, std::max(0, depth-R-1), pline_alt, ab_state, false, true, callback_f));
      }
      if(score >= beta) {
        depth -= DR;
        if(depth <= 0) {
          const int nchecks = 0;
          // TODO zug-zwang will fail low
          return alpha_beta_quiescence(beta-1, beta, 0, pline, ab_state, nchecks);
        } else {
          {
            pline_alt.premove(board::nullmove);
            const move_t nextmove = pline_alt.get_next_move();
            const size_t cmt_index = get_cmt_index(pline_alt, nextmove);
            if(cmt_index != SIZE_MAX) {
              assert(cmt_index < ab_state.cmh_table.size());
              ab_state.cmh_table[cmt_index] += ((depth + 2) * (depth + 2)) * 1e-8;
              ab_state.normalize_cmh_table(cmt_index);
            }
            pline_alt.recall();
          }
        }
      }
    }

    decltype(auto) moves = ab_get_ordered_moves(pline, m_best, ab_state.cmh_table);
    // this is nonempty - draw and checkmate checks are above
    assert(!moves.empty());
    int32_t bestscore = -MATERIAL_KING;
    bool isdraw_pathdep = false;
    TT_NODE_TYPE ndtype = TT_ALLNODE;
    for(size_t move_index = 0; move_index < moves.size(); ++move_index) {
      const auto [m_val, m] = moves[move_index];
      if(ab_state.initdepth == depth && !ab_state.searchmoves.empty() && ab_state.searchmoves.find(m) == ab_state.searchmoves.end()) {
        continue;
      }
      MoveLine pline_alt = pline.branch_from_past(m);
      int32_t score;
      // first move should be the pv search
      // TODO IID when no PV move
      if(m_val > 999. && !scoutsearch) {
        { // move scope
          auto mscope = mline_scope(m, pline_alt);
          isdraw_pathdep = is_draw_halfmoves() || is_draw_repetition();
          score = -score_decay(alpha_beta_pv(-beta, -alpha, depth - 1, pline_alt, ab_state, allow_nullmoves, scoutsearch, callback_f));
        } // end move scope
        debug.update_pv(depth, alpha, beta, bestscore, score, m, pline, pline_alt, "firstmove"s);
        debug.check_score(depth, score, pline_alt);
//        debug.log_pv(depth, pline, "move "s + pgn::_move_str(self, m) + " score "s + std::to_string(score_float(score)));
        if(score > bestscore) {
          pline.replace_line(pline_alt);
          bestscore = score;
          if(score > alpha) {
            if(score >= beta) {
              if(overwrite && tt_replace_depth && tt_inexact_entry && !isdraw_pathdep) {
                if(tt_inactive_entry)++zb_occupied;
                zb.write(state.info, score, depth, m, tt_age, TT_CUTOFF, pline);
              }
              const size_t cmt_index = get_cmt_index(pline, m);
              if(cmt_index != SIZE_MAX) {
                ab_state.cmh_table[cmt_index] += (depth * depth * (double(move_index + 1) / double(moves.size()))) * 1e-8;
                ab_state.normalize_cmh_table(cmt_index);
              }
              return bestscore;
            }
            alpha = bestscore;
            ndtype = TT_EXACT;
          } else if(ab_state.initdepth == depth) {
            // fail low on aspiration. re-do the search
            return bestscore;
          }
        }
      } else {
        { // move scope
          auto mscope = mline_scope(m, pline_alt);
          isdraw_pathdep = is_draw_halfmoves() || is_draw_repetition();
          pos_t i = bitmask::first(m), j = bitmask::second(m) & board::MOVEMASK;
          const bool interesting_move = (is_drop_move(i, j) || is_castling_move(i, j) || is_promotion_move(i, j) || (is_naively_capture_move(i, j) && m_val > -3.5) || (is_naively_checking_move(i, j) && m_val > -9.));
          const bool lmr_allowed = false && ENABLE_SELECTIVITY && !scoutsearch && move_index >= (pline.is_mainline() ? 15 : 4) && depth >= 3 && state.checkline[c] == bitmask::full && m_val < .1 && !interesting_move;
          MoveLine pline_alt2 = pline_alt;
          if(lmr_allowed) {
            score = -score_decay(alpha_beta_pv(-alpha-1, -alpha, depth - 2, pline_alt2, ab_state, allow_nullmoves, true, callback_f));
          } else {
            score = -score_decay(alpha_beta_pv(-alpha-1, -alpha, depth - 1, pline_alt2, ab_state, allow_nullmoves, true, callback_f));
          }
          if(!scoutsearch && score > alpha) {
            score = -score_decay(alpha_beta_pv(-beta, -alpha, depth - 1, pline_alt, ab_state, allow_nullmoves, scoutsearch, callback_f));
            if(score > alpha) {
              alpha = score;
              ndtype = TT_EXACT;
            }
          } else {
            pline_alt = pline_alt2;
          }
        } // end move scope
        debug.update_pv(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
        debug.check_score(depth, score, pline_alt);
//        debug.log_pv(depth, pline_alt, "move "s + pgn::_move_str(self, m) + " score "s + std::to_string(score_float(score)));
        if(score > bestscore) {
          pline.replace_line(pline_alt);
          if(score >= beta) {
            if(overwrite && tt_replace_depth && tt_inexact_entry && !isdraw_pathdep) {
              if(tt_inactive_entry)++zb_occupied;
              zb.write(state.info, score, depth, m, tt_age, TT_CUTOFF, pline);
            }
            const size_t cmt_index = get_cmt_index(pline, m);
            if(cmt_index != SIZE_MAX) {
              ab_state.cmh_table[cmt_index] += (depth * depth * (double(move_index + 1) / double(moves.size()))) * 1e-8;
              ab_state.normalize_cmh_table(cmt_index);
            }
            return score;
          }
          bestscore = score;
        }
      }
      if(ab_state.initdepth <= depth + 1 + (int16_t(pline.line.size()) / 7) && move_index + 1 != moves.size()) {
        if(!callback_f(depth, bestscore)) {
//          str::pdebug("should stop");
          return bestscore;
        }
      }
    }
    if(overwrite && m_best != board::nullmove && !isdraw_pathdep && !pline.find(board::nullmove)) {
      if(tt_replace_depth && ndtype == TT_ALLNODE) {
        if(tt_inactive_entry)++zb_occupied;
        zb.write(state.info, bestscore, depth, m_best, tt_age, TT_ALLNODE, pline);
      } else if((tt_replace_depth || tt_inexact_entry) && ndtype == TT_EXACT && !scoutsearch) {
        if(tt_inactive_entry)++zb_occupied;
        zb.write(state.info, bestscore, depth, m_best, tt_age, TT_EXACT, pline);
      }
    }
//    debug.log_pv(depth, pline, "returning all-node pv-bestscore "s + std::to_string(score_float(bestscore)));
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

    void end_scope() {
      ab_ttable_scope.end_scope();
      e_ttable_scope.end_scope();
    }
  };

  typedef struct _iddfs_state {
    int16_t curdepth = 1;
    int32_t eval = 0;
    MoveLine pline;

    INLINE void reset() {
      curdepth = 1;
      eval = .0;
      pline = MoveLine();
    }

    INLINE move_t currmove() const {
      return pline.front();
    }

    INLINE move_t pondermove() const {
      if(pline.size() > 1) {
        return pline[1];
      }
      return board::nullmove;
    }

    INLINE move_t ponderhit() {
      move_t m = pline.front();
      if(m != board::nullmove) {
        pline.shift_start();
        --curdepth;
      }
      return m;
    }
  } iddfs_state;

  template <typename F>
  decltype(auto) iterative_deepening_dfs(int16_t depth, ab_storage_t &ab_storage, iddfs_state &idstate, const std::unordered_set<move_t> &searchmoves, F &&callback_f) {
    auto &ab_ttable = ab_storage.ab_ttable_scope.get_object();
    auto &e_ttable = ab_storage.e_ttable_scope.get_object();
    auto &cmh_table = ab_storage.cmh_table;
    if(idstate.curdepth <= 1) {
      if(idstate.curdepth == 0) {
        idstate.reset();
      }
      std::fill(cmh_table.begin(), cmh_table.end(), .0);
    }
    bool should_stop = false;
    for(int16_t d = idstate.curdepth; d <= depth; ++d) {
      if(!idstate.pline.empty() && check_line_terminates(idstate.pline) && int(idstate.pline.size()) < d) {
        idstate.eval = get_pvline_score(idstate.pline);
        break;
      }
      const int32_t eval_prev = idstate.eval;
      const std::valarray<int32_t> aspiration_window = {CENTIPAWN, 15*CENTIPAWN, 125*CENTIPAWN, MATERIAL_QUEEN - MATERIAL_PAWN};
      pos_t aw_index_alpha = 0, aw_index_beta = 0;
      MoveLine new_pline = idstate.pline;
      do {
        int32_t aw_alpha = -MATERIAL_KING, aw_beta = MATERIAL_KING;
        if(aw_index_alpha < aspiration_window.size()) {
          aw_alpha = eval_prev - aspiration_window[aw_index_alpha];
        }
        if(aw_index_beta < aspiration_window.size()) {
          aw_beta = eval_prev + aspiration_window[aw_index_beta];
        }
        const bool final_window = (aw_index_alpha == aspiration_window.size() && aw_index_beta == aspiration_window.size());
        str::pdebug("depth:", d, "aw", score_float(aw_alpha), score_float(aw_beta));
        bool inner_break = false;
        alpha_beta_state ab_state = (alpha_beta_state){ .ab_ttable=ab_ttable, .e_ttable=e_ttable, .cmh_table=cmh_table,
                                                        .initdepth=d, .searchmoves=searchmoves };
        const int32_t new_eval = alpha_beta_pv(aw_alpha, aw_beta, d, new_pline, ab_state, true, false,
          [&](int16_t _depth, int32_t _eval) mutable -> bool {
            const move_t _m = new_pline.front();
            bool verbose = false;
            if(!inner_break && _depth == d && final_window && idstate.eval < _eval && _m != board::nullmove) {
              idstate.eval=_eval, idstate.pline=new_pline, idstate.curdepth=d;
              verbose = true;
              str::pdebug("IDDFS:", d, "pline:", idstate.pline.pgn(self), "size:", idstate.pline.size(), "eval", score_float(idstate.eval));
              debug.check_score(d, idstate.eval, idstate.pline);
            }
            should_stop = !callback_f(verbose);
            if(should_stop && d != _depth) {
              inner_break = true;
            }
            return !should_stop;
          });
        if(inner_break)break;
        if(new_eval <= aw_alpha) {
          ++aw_index_alpha;
        } else if(new_eval >= aw_beta) {
          ++aw_index_beta;
        } else {
          if(!should_stop || new_eval >= idstate.eval) {
            idstate.eval=new_eval, idstate.pline=new_pline, idstate.curdepth=d;
            should_stop = !callback_f(true);
          }
          debug.check_score(d, idstate.eval, idstate.pline);
          if(should_stop || (score_is_mate(idstate.eval) && d > 0 && int(idstate.pline.size()) < d)) {
            should_stop = true;
          }
          break;
        }
        if(should_stop)break;
      } while(1);
      str::pdebug("IDDFS:", d, "pline:", idstate.pline.pgn(self), "size:", idstate.pline.size(), "eval", score_float(idstate.eval));
      if(should_stop)break;
    }
    if(idstate.pline.empty() || idstate.pline.find(board::nullmove)) {
      const move_t m = get_random_move();
      if(m != board::nullmove) {
        str::pdebug("replace move with random move", _move_str(m));
        idstate.curdepth = 0;
        idstate.pline = MoveLine(std::vector<move_t>(m));
      }
    }
    return idstate.eval;
  }

  size_t nodes_searched = 0;
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
    const size_t size_ab = zobrist_size;
    const size_t mem_ab = size_ab * (sizeof(tt_ab_entry) + 16 * sizeof(move_t));
    const size_t size_e = 0;
    const size_t mem_e = size_e * sizeof(tt_eval_entry);
    const size_t size_cmh = size_t(NO_PIECES) * size_t(board::SIZE);
    const size_t size_cmt = size_t(NO_COLORS) * size_cmh * size_cmh;
    const size_t mem_cmh = size_cmt  * sizeof(double);
    const size_t mem_total = mem_ab+mem_e+mem_cmh;
    str::pdebug("alphabeta scope", "ab:", mem_ab, "e:", mem_e, "cmh:", mem_cmh, "total:", mem_total);
    _printf("MEM: %luMB %luKB %luB\n", mem_total>>20, (mem_total>>10)&((1<<10)-1), mem_total&((1<<10)-1));
    return (ab_storage_t){
      .ab_ttable_scope=zobrist::make_store_object_scope<tt_ab_entry>(ab_ttable, size_ab),
      .e_ttable_scope=zobrist::make_store_object_scope<tt_eval_entry>(e_ttable, size_e),
      .cmh_table=std::vector<double>(size_cmt, .0)
    };
  }

  INLINE decltype(auto) make_callback_f() {
    return [&](bool verbose) -> bool { return true; };
  }

  template <typename F>
  move_t start_thinking(int16_t depth, iddfs_state &idstate, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    reset_planning();
    decltype(auto) ab_storage = get_zobrist_alphabeta_scope();
    assert(ab_ttable != nullptr && e_ttable != nullptr);
    debug.set_depth(depth);
    iterative_deepening_dfs(depth, ab_storage, idstate, searchmoves, std::forward<F>(callback_f));
    return idstate.currmove();
  }

  INLINE move_t start_thinking(int16_t depth, iddfs_state &idstate, const std::unordered_set<move_t> &searchmoves={}) {
    return start_thinking(depth, idstate, make_callback_f(), searchmoves);
  }

  INLINE move_t start_thinking(int16_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    iddfs_state idstate;
    return start_thinking(depth, idstate, searchmoves);
  }

  struct tt_perft_entry {
    board_info info;
    int16_t depth;
    size_t nodes;
  };

  zobrist::ttable_ptr<tt_perft_entry> perft_ttable = nullptr;
  decltype(auto) get_zobrist_perft_scope() {
    const size_t size_perft = zobrist_size;
    const size_t mem_perft = size_perft * sizeof(tt_perft_entry);
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
    constexpr bool overwrite = true;
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
  Engine(const fen::FEN fen=fen::starting_pos, size_t zbsize=ZOBRIST_SIZE):
    Board(fen, zbsize), debug(*this)
  {
    const std::vector<PIECE> piece_types = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    std::vector<std::pair<int32_t, PIECE>> pieces;
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
