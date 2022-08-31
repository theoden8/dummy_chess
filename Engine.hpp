#pragma once


#include "Piece.hpp"
#include <memory>
#include <algorithm>
#include <valarray>
#include <unordered_set>
#include <optional>

#include <FEN.hpp>
#include <PGN.hpp>
#include <Board.hpp>
#include <Perft.hpp>
#include <DebugTracer.hpp>

#include <tbconfig.h>

class Engine : public Perft {
public:
  static constexpr bool ENABLE_TT = true;
  static constexpr bool ENABLE_TT_RETURN = true && ENABLE_TT;
  static constexpr bool ENABLE_SELECTIVITY = true;
  static constexpr bool ENABLE_SEL_BIGDELTA = true && ENABLE_SELECTIVITY,
                        ENABLE_SEL_DELTA = true && ENABLE_SELECTIVITY,
                        ENABLE_SEL_NMR = true && ENABLE_SELECTIVITY,
                        ENABLE_SEL_LMR = true && ENABLE_SELECTIVITY;
  static constexpr bool ENABLE_IID = true;
  static constexpr bool ENABLE_SYZYGY = true;

  inline Board &as_board() {
    return (Board &)self;
  }

  inline const Board &as_board() const {
    return (const Board &)self;
  }

  // callbacks
  INLINE void make_move(pos_t i, pos_t j) {
    Perft::make_move(i, j);
  }

  INLINE void make_move(move_t m) {
    return make_move(bitmask::first(m), bitmask::second(m));
  }

  INLINE void retract_move() {
    Perft::retract_move();
  }

  INLINE Board &as_board() { return (Board &)self; }
  INLINE const Board &as_board() const { return (const Board &)self; }

  // provide affected methods
  INLINE decltype(auto) engine_move_scope(move_t m) {
    return make_move_scope(self, m);
  }

  INLINE decltype(auto) engine_mline_scope(move_t m, MoveLine &mline) {
    return make_mline_scope(self, m, mline);
  }

  INLINE decltype(auto) engine_recursive_move_scope() {
    return make_recursive_move_scope(self);
  }

  INLINE decltype(auto) engine_recursive_mline_scope(MoveLine &mline) {
    return make_recursive_mline_scope(self, mline);
  }

  // definitions and such
  using score_t = int32_t;
  using mval_t = float;
  constexpr static score_t NOSCORE = INT32_MIN;

  static constexpr score_t CENTIPAWN = 256,
                           MATERIAL_PAWN = 100*CENTIPAWN,
                           MATERIAL_KNIGHT = 325*CENTIPAWN,
                           MATERIAL_BISHOP = 325*CENTIPAWN,
                           MATERIAL_ROOK = 500*CENTIPAWN,
                           MATERIAL_QUEEN = 1000*CENTIPAWN,
                           MATERIAL_KING = 1000*MATERIAL_PAWN;
  std::vector<PIECE> MATERIAL_PIECE_ORDERING;

  static constexpr score_t MATERIALS[] = {
    MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP,
    MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING, 0};

  INLINE constexpr score_t material_of(PIECE p) const {
    return MATERIALS[int(p)];
  }
  INLINE constexpr score_t material_of_capped(PIECE p) const {
    if(p == KING)return 50*MATERIAL_PAWN;
    return MATERIALS[int(p)];
  }

  INLINE score_t count_material(piece_bitboard_t mask) const {
    score_t m = 0;
    m += piece::size(mask & bits_pawns) * material_of(PAWN);
    m += piece::size(mask & bits_slid_diag & ~bits_slid_orth) * material_of(BISHOP);
    m += piece::size(mask & ~bits_slid_diag & bits_slid_orth) * material_of(ROOK);
    m += piece::size(mask & bits_slid_diag & bits_slid_orth) * material_of(QUEEN);
    m += piece::size(mask & get_knight_bits()) * material_of(KNIGHT);
    return m;
  }

  INLINE bool tb_can_probe() const {
    return ENABLE_SYZYGY && tb::can_probe(self);
  }

  INLINE score_t tb_probe_wdl() {
    int8_t wdl = tb::probe_wdl(self);
    ++tb_probes;
    if(wdl == -1) {
      return NOSCORE;
    }
    ++tb_hit;
    switch(wdl) {
      case TB_WIN:
      return MATERIAL_KING/2;
      case TB_CURSED_WIN:
      return 0;//MATERIAL_KING/4;
      case TB_DRAW:
      return 0;
      case TB_BLESSED_LOSS:
      return 0;//-MATERIAL_KING/4;
      case TB_LOSS:
      return -MATERIAL_KING/2;
    }
    abort();
  }

  void list_probe_root_wdl() {
    tb::probe_root_wdl(self, [&](move_t m, score_t approx_score, int32_t tb_rank) mutable -> void {
      assert(check_valid_move(m));
      str::print("WDL", "move:", pgn::_move_str(self, m),
                 "rank:", tb_rank, "score:", score_string(approx_score));
    });
  }

  void list_probe_root_dtz() {
    tb::probe_root_dtz(self, [&](move_t m, score_t approx_score, int32_t tb_rank) mutable -> void {
      assert(check_valid_move(m));
      str::print("DTZ", "move:", pgn::_move_str(self, m), "rank:", tb_rank, "score:", score_string(approx_score),
                 "SCORE", approx_score, "SCORE-MATE", approx_score-MATERIAL_KING, "SCORE_IS_MATE", score_is_mate(approx_score));
    });
  }

  void list_probe_root() {
    tb::probe_root(self, [&](move_t m, unsigned wdl, unsigned dtz) mutable -> void {
      assert(check_valid_move(m));
      str::print("DTZ", "move:", board::_move_str(m), "wdl:", tb::wdl_to_str(self, wdl), "dtz:", dtz);
    });
  }

  INLINE score_t h_material(COLOR c) const {
    score_t h = count_material(bits[c]);
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

  INLINE score_t h_attack_cells(COLOR c) const {
    const piece_bitboard_t occupied = bits[WHITE] | bits[BLACK];
    score_t h = 0;

    decltype(auto) h_add = [&](pos_t i, score_t mat) mutable -> void {
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

  INLINE score_t h_mobility(COLOR c) const {
    score_t h = 0;
    bitmask::foreach(bits[c] & ~bits_pawns, [&](pos_t i) mutable -> void {
      h += std::sqrt(piece::size(state.moves[i]));
    });
    return h;
  }

  INLINE score_t h_pawn_structure(COLOR c) const {
    score_t h = 0;
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
                                       : board::_y(bitmask::log2_lsb(pawns_file));
      piece_bitboard_t adjacent_files = piece::file_mask(f);
      score_t sidepawn_decay = 1;
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
      const score_t pawnr_abs = ((c == WHITE) ? pawnr : 7 - pawnr);
      h += (CENTIPAWN / 10) * pawnr_abs * pawnr_abs / sidepawn_decay;
    }
    // pawn strucure coverage
    h += (CENTIPAWN / 2) * piece::size(piece::get_pawn_attacks(bits_pawns & bits[c], c) & bits[c]);
    return h;
  }

  INLINE score_t heuristic_of(COLOR c) const {
    score_t h = 0;
    h += h_material(c);
    h += h_pawn_structure(c);
//    h -= count_material(state.pins[c]);
//    h += h_mobility(c);
    h += h_attack_cells(c);
    return h;
  }

  INLINE score_t evaluate(score_t wdl_score=NOSCORE) {
    if(self.is_draw()){
      return 0;
    } else if(self.is_checkmate()) {
      return -MATERIAL_KING;
    }
    const COLOR c = activePlayer();
    if(self.tb_can_probe()) {
      if(wdl_score == NOSCORE) {
        wdl_score = self.tb_probe_wdl();
      }
      if(wdl_score == NOSCORE) {
        wdl_score = 0;
      } else if(wdl_score == 0) {
        return 0;
      }
    } else {
      wdl_score = 0;
    }
    return wdl_score + heuristic_of(c) - heuristic_of(enemy_of(c));
  }

  INLINE bool is_repeated_thought(const MoveLine &pline) {
    assert(check_valid_sequence(pline));
    const size_t thought_moves = pline.start;
    const size_t no_iter = !crazyhouse ? std::min<size_t>(self.get_halfmoves(), thought_moves) : thought_moves;
    if(no_iter < 3) {
      return false;
    }
    for(size_t i = NO_COLORS - 1; i < no_iter; i += NO_COLORS) {
      const auto &state_iter = state_hist[state_hist.size() - i - 1];
      if(state.info == state_iter.info) {
        return true;
      }
    }
    return false;
  }

  INLINE bool is_path_dependent(const MoveLine &pline) {
    return is_draw_halfmoves() || is_draw_repetition() || is_repeated_thought(pline);
  }

  INLINE mval_t move_heuristic_extra_material(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(crazyhouse && is_drop_move(i, j)) {
    } else if(is_promotion_move(i, _j)) {
      return mval_t(material_of(board::get_promotion_as(j)) - material_of(PAWN)) / MATERIAL_PAWN;
    } else if(is_enpassant_take_move(i, _j)) {
      return mval_t(material_of(PAWN)) / MATERIAL_PAWN;
    }
    return .0;
  }

  INLINE mval_t move_heuristic_check(pos_t i, pos_t j) const {
    const pos_t _j = j & board::MOVEMASK;
    if(crazyhouse && is_drop_move(i, j)) {
    } else if(is_naively_checking_move(i, _j)) {
      return .5;
    }
    return .0;
  }

  INLINE mval_t move_heuristic_recapture(pos_t i, pos_t j, move_t prevmove) const {
    if(prevmove != board::nullmove && (bitmask::second(prevmove) & board::MOVEMASK) == (j & board::MOVEMASK)) {
      return .1;
    }
    return .0;
  }

  INLINE mval_t move_heuristic_attacker_decay(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    mval_t val = .0;
    const PIECE frompiece = self[i].value;
    const mval_t mat_from = mval_t(material_of_capped(frompiece)) / MATERIAL_PAWN;
    const pos_t _j = j & board::MOVEMASK;
    if(self.empty_at_pos(_j) || (edefendmap & piece::pos_mask(_j))) {
      const mval_t decay_i = (edefendmap & piece::pos_mask(_j)) ? .02 : .05;
      val -= mat_from * decay_i;
    } else {
      val -= mat_from * .01;
    }
    return val;
  }

  INLINE mval_t move_heuristic_see(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    const pos_t _j = j & board::MOVEMASK;
    mval_t val = .0;
    const mval_t extra_val = move_heuristic_extra_material(i, j);
    val += extra_val;
    if(crazyhouse && is_drop_move(i, _j)) {
      return .0;
    } else if(val > 0 || is_castling_move(i, _j)) {
      return val;
    }
    const mval_t mat_i = mval_t(material_of(self[i].value)) / MATERIAL_PAWN,
                mat_j = mval_t(material_of(self[_j].value)) / MATERIAL_PAWN;
    if(~edefendmap & piece::pos_mask(_j)) {
      return mat_j;
    } else if(mat_i < mat_j) {
      return mat_j - mat_i - extra_val;
    }
    mval_t see = mval_t(static_exchange_evaluation(i, _j)) / MATERIAL_PAWN;
    if(std::abs(see) > 100) {
      see = (see < 0) ? -100 : 100;
    }
    val += see;
    if(see < 0) {
      val -= extra_val;
    }
    return val;
  }

  INLINE mval_t move_heuristic_mvv_lva(pos_t i, pos_t j, piece_bitboard_t edefendmap) const {
    const pos_t _j = j & board::MOVEMASK;
    mval_t val = .0;
    if(is_drop_move(i, _j)) {
      ;
    } else if(is_castling_move(i, _j)) {
      ;
    } else if(is_naively_capture_move(i, _j)) {
      const mval_t mat_to = mval_t(material_of_capped(self[_j].value)) / MATERIAL_PAWN;
      val += mat_to;
      const PIECE frompiece = self[i].value;
      const mval_t mat_from = mval_t(material_of_capped(frompiece)) / MATERIAL_PAWN;
      if(edefendmap & piece::pos_mask(_j)) {
        val -= mat_from;
      }
      val += move_heuristic_attacker_decay(i, j, edefendmap);
    }
    val += move_heuristic_extra_material(i, j);
    val += move_heuristic_check(i, j);
    return val;
  }

  INLINE mval_t move_heuristic_pv(move_t m, const MoveLine &pline, move_t hashmove=board::nullmove, move_t my_threatmove=board::nullmove) const {
    if(pline.is_mainline() && m == pline.front()) {
      return 2000.;
    } else if(m == hashmove || m == my_threatmove || pline.front() == m) {
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

  INLINE mval_t move_heuristic_cmt(move_t m, const MoveLine &pline, const std::vector<float> &cmh_table) const {
    const size_t cmt_index = get_cmt_index(pline, m);
    if(cmt_index != SIZE_MAX) {
      return cmh_table[cmt_index];
    }
    return .0;
  }

  static INLINE float score_float(score_t score) {
    return float(score)  / MATERIAL_PAWN;
  }

  static INLINE bool score_is_mate(score_t score) {
    return std::abs(score) > MATERIAL_KING - 16000;
  }

  static INLINE bool score_is_tb(score_t score) {
    return std::abs(score) > MATERIAL_KING / 4 - 1000 && !score_is_mate(score);
  }

  static INLINE score_t score_material(score_t score) {
    if(score_is_mate(score)) {
      return score;
    } else if(score_is_tb(score)) {
      return score > 0 ? score - 500 : score + 500;
    }
    return score;
  }

  static INLINE score_t score_decay(score_t score) {
    if(score_is_mate(score)) {
      score += (score > 0) ? -1 : 1;
    }
    return score;
  }

  static INLINE score_t score_decay(score_t score, depth_t depth) {
#ifndef NDEBUG
    score_t sscore = score;
    for(depth_t i = 0; i < depth; ++i) {
      sscore = -score_decay(sscore);
    }
#endif
    if(score_is_mate(score)) {
      score += (score > 0) ? -depth : depth;
    }
    score = (depth & 1) ? -score : score;
    assert(score == sscore);
    return score;
  }

  static INLINE depth_t score_mate_in(score_t score) {
    assert(score_is_mate(score));
    const int mate_in = int(MATERIAL_KING - std::abs(score));
    return (score < 0) ? -mate_in : mate_in;
  }

  static INLINE std::string score_string(score_t score) {
    char s[256];
    if(score == NOSCORE) {
      snprintf(s, sizeof(s), "noscore");
    } else if(score_is_mate(score)) {
      snprintf(s, sizeof(s), "MATE%s%d", score > 0 ? "+":(score_mate_in(score)==0?"-":""), (int)score_mate_in(score));
    } else {
      snprintf(s, sizeof(s), "%s%.05f", score>0?"+":(score==0?" ":""), score_float(score));
    }
    return s;
  }

  static INLINE bool mval_is_primary(mval_t mval) {
    return mval > 999.;
  }

  static INLINE bool mval_is_tbwin(mval_t mval) {
    return !mval_is_primary(mval) && mval > 800.;
  }

  static INLINE bool mval_is_tb_cursedwin(mval_t mval) {
    return !mval_is_primary(mval) && !mval_is_tbwin(mval) && mval > 350.;
  }

  static INLINE bool mval_is_tb_windraw(mval_t mval) {
    return mval_is_tbwin(mval) || mval_is_tb_cursedwin(mval);
  }

  INLINE score_t get_pvline_score(const MoveLine &pline) {
    assert(check_valid_sequence(pline));
    score_t score = NOSCORE;
    walk_end(pline, [&](const ply_index_t d) mutable -> void {
      score = score_decay(evaluate(), d);
    });
    return score;
  }

  INLINE bool check_pvline_score(const MoveLine &pline, score_t score) {
    const score_t pvscore = get_pvline_score(pline);
    return score == pvscore && (!pline.tb || (pvscore == 0 || score_is_tb(pvscore)));
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

  score_t static_exchange_evaluation(pos_t i, pos_t j) const {
    assert(j <= board::MOVEMASK);
//    str::pdebug("move", pgn::_move_str(self, bitmask::_pos_pair(i, j)));
//    str::pdebug(fen::export_as_string(export_as_fen()));
    std::array<score_t, 37> gain;
    int8_t depth = 0;
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
      gain[depth] = material_of(curpiece) - gain[depth - 1];
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
//    uint8_t maxdepth = depth;
    while(--depth) {
      gain[depth-1] = -std::max(-gain[depth-1], gain[depth]);
    }
//    for(uint8_t i = 0; i < maxdepth; ++i) {
//      str::pdebug("gain[", i, "] = ", gain[i]);
//    }
    return gain[0];
  }

  struct tt_eval_entry {
    board_info info;
    score_t eval;
  };

  INLINE score_t e_ttable_probe(zobrist::ttable<tt_eval_entry> &e_ttable) {
    return evaluate();
//    const zobrist::key_t k = zb_hash();
//    if(e_ttable[k].info == state.info) {
//      return e_ttable[k].eval;
//    }
//    const score_t eval = evaluate();
//    const bool overwrite = true;
//    if(overwrite) {
//      e_ttable[k] = { .info=state.info, .eval=eval };
//    }
//    return eval;
  }

  typedef enum { TT_EXACT, TT_ALLNODE, TT_CUTOFF } TT_NODE_TYPE;
  struct tt_ab_entry {
    TT_NODE_TYPE ndtype : 2;
    depth_t depth : 15;
    ply_index_t age : 15;
    score_t score;
    board_info info;
  #ifndef NDEBUG
    std::array<move_t, 3> m_hint;
    MoveLine subpline;
  #else
    std::array<move_t, 20> m_hint;
  #endif

    INLINE bool can_apply(const board_info &_info, depth_t _depth, ply_index_t _age) const {
      return !is_inactive(_age) && depth >= _depth && info == _info;
    }

    INLINE bool can_use_move(const board_info &_info, depth_t _depth, ply_index_t _age) const {
      if(is_inactive(_age))return false;
      if(depth <= 0) {
        return info == _info;
      }
      return ((depth >= _depth - 1 && ndtype == TT_ALLNODE)
              || ndtype == TT_EXACT
              || (ndtype == TT_CUTOFF && depth >= depth - 3))
        && info == _info;
    }

    INLINE move_t front() const {
      assert(m_hint.size() > 0);
      return m_hint.front();
    }

    INLINE move_t back() const {
      assert(m_hint.size() > 0);
      return m_hint.back();
    }

    INLINE bool should_replace(depth_t _depth, TT_NODE_TYPE _ndtype, int16_t _age) const {
      if(is_inactive(_age))return true;
      switch(_ndtype) {
        case TT_CUTOFF: return _depth >= depth && ndtype != TT_EXACT;
        case TT_ALLNODE:return _depth >= depth && ndtype != TT_EXACT;
        case TT_EXACT:  return _depth >= depth;
        default: break;
      }
      abort();
    }

    INLINE bool is_inactive(ply_index_t cur_age) const {
      return info.is_unset() || cur_age != age;
    }

    INLINE void write(const board_info &_info, score_t _score, depth_t _depth, ply_index_t _age, TT_NODE_TYPE _ndtype, const MoveLine &pline) {
      assert(!pline.tb);
      info=_info, depth=_depth, score=_score, age=_age, ndtype=_ndtype;
      for(size_t i = 0; i < m_hint.size(); ++i) {
        m_hint[i] = (pline.size() > i) ? pline[i] : board::nullmove;
      }
      #ifndef NDEBUG
      subpline=pline.get_future();
      #endif
    }
  };

  struct alpha_beta_state {
    zobrist::ttable<tt_ab_entry> &ab_ttable;
    zobrist::ttable<tt_eval_entry> &e_ttable;
    std::vector<float> &cmh_table;
    const depth_t initdepth;
    const std::unordered_set<move_t> &searchmoves;

    void normalize_cmh_table(size_t cmt_index) {
      if(cmh_table[cmt_index] > .5) {
        const float mx = *std::max_element(cmh_table.begin(), cmh_table.end());
        _printf("renormalizing countermove table\n");
        for(auto &cm : cmh_table) {
          cm /= (mx * 1e7);
        }
      }
    }
  };

  INLINE decltype(auto) make_callback_f() {
    return [&](...) -> bool { return true; };
  }

  decltype(auto) ab_ttable_probe(score_t &alpha, score_t &beta, depth_t depth, move_t &hashmove, move_t &your_threatmove,
                                 MoveLine &pline, alpha_beta_state &ab_state, bool allow_nullmoves, int8_t nchecks=0)
  {
    const zobrist::key_t k = zb_hash();
    const bool tt_has_entry = ab_state.ab_ttable[k].can_apply(state.info, depth, tt_age);
    if(tt_has_entry) {
      ++zb_hit;
    } else {
      ++zb_miss;
    }

    const bool scoutsearch = (alpha + 1 == beta);

    auto &zb = ab_state.ab_ttable[k];
    debug.update_line(depth, alpha, beta, pline);
    score_t maybe_result = NOSCORE;
    if(tt_has_entry) {
      debug.update_mem(depth, alpha, beta, zb, pline);
      if(zb.ndtype == TT_EXACT || (zb.score < alpha && zb.ndtype == TT_ALLNODE) || (zb.score >= beta && zb.ndtype == TT_CUTOFF)) {
        bool draw_pathdep = false;
        if(ENABLE_TT_RETURN && zb.back() == board::nullmove) {
          ++nodes_searched;
          walk_early_stop(zb.m_hint,
            [&](const move_t m, auto &&do_step_f) mutable -> bool {
              if(m == board::nullmove) {
                return false;
              }
              assert(check_valid_move(m));
              do_step_f();
              if(is_draw_halfmoves() || is_draw_repetition()) {
                draw_pathdep = true;
                return false;
              }
              return true;
            }
          );
          if(!draw_pathdep) {
            pline.clear();
            pline.draft_line(zb.m_hint);
            maybe_result = zb.score;
            debug.check_score(zb.depth, maybe_result, pline);
            return std::make_pair(k, maybe_result);
          }
        }
        if(!draw_pathdep) {
  //        pline.draft_line(zb.m_hint);
          bool can_return_score = false;
          depth_t R = 0;
          MoveLine pline_alt = pline.branch_from_past();
          walk_early_stop(zb.m_hint,
            // foreach
            [&](const move_t m, auto &&do_step_f) mutable -> bool {
              if(m == board::nullmove) {
                return false;
              }
              assert(check_valid_move(m));
              pline_alt.premove(m);
              do_step_f();
              if(is_draw_halfmoves() || is_draw_repetition()) {
                draw_pathdep = true;
                return false;
              }
              return true;
            },
            // endfunc
            [&](const ply_index_t nsteps) mutable -> void {
              R = nsteps;
              if(!draw_pathdep) {
                draw_pathdep = is_draw_repetition() || is_draw_halfmoves();
              }
              // doesn't lead to path-dependent draw
              // changed halfmove clock or pvline at least zb.depth - depth
              const depth_t subdepth = zb.depth - R;
              can_return_score = !draw_pathdep && (R < (ssize_t)zb.m_hint.size() || subdepth < depth || get_halfmoves() < R) && (state.info != zb.info);
              if(can_return_score) {
                score_t _alpha=alpha, _beta=beta;
                if(R & 1)_alpha=-beta,_beta=-alpha;
                score_t score = NOSCORE;
                if(subdepth >= 0) {
                  if(!scoutsearch) {
                    score = alpha_beta_pv(_alpha,_beta,subdepth,pline_alt,ab_state,allow_nullmoves,make_callback_f());
                    debug.check_score(depth, score, pline_alt);
                  } else {
                    assert(_alpha + 1 == _beta);
                    score = alpha_beta_scout(_beta,subdepth,pline_alt,ab_state,allow_nullmoves);
                    debug.check_score(depth, score, pline_alt);
                  }
                } else {
                  score = alpha_beta_quiescence(_alpha,_beta,subdepth,pline_alt,ab_state,nchecks);
                  debug.check_score(depth, score, pline_alt);
                }
                debug.check_score(subdepth, score, pline_alt);
                if((score < alpha && zb.ndtype == TT_ALLNODE) || (score >= beta && zb.ndtype == TT_CUTOFF)) {
                  // note: current position rolls back only after return statement
                  maybe_result = score_decay(score, R);
                } else {
                  can_return_score = false;
                }
                assert(check_valid_sequence(pline_alt));
              }
            }
          );
          pline_alt.recall_n(R);
          assert(check_valid_sequence(pline_alt));
          if(can_return_score) {
            assert(check_valid_sequence(pline));
            pline.replace_line(pline_alt);
            debug.check_score(zb.depth, maybe_result, pline);
            return std::make_pair(k, maybe_result);
          }
        }
      } else if(alpha < zb.score && zb.score < beta) {
        if(zb.ndtype == TT_CUTOFF) {
          alpha = zb.score;
        } else if(zb.ndtype == TT_ALLNODE) {
          beta = zb.score;
        } else {
          str::perror("error: unknown node type");
          abort();
        }
      }
      hashmove = zb.front();
    } else if(zb.can_use_move(state.info, depth, tt_age)) {
      hashmove = zb.front();
//      if(zb.m_hint.size() >= 2) {
//        your_threatmove = zb.m_hint[1];
//      }
    }
    return std::make_pair(k, maybe_result);
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
    const pos_t j_mask = piece::pos_mask(j);
    const piece_bitboard_t notpinned = bits[ec] & ~state.pins[ec];
    return (~(
              piece::get_knight_attacks(get_knight_bits() & notpinned)
              | piece::get_pawn_attacks(bits_pawns & notpinned, ec)
             ) & j_mask)
           && (~piece::get_sliding_diag_attacks(bits_slid_diag & ~bits_slid_orth & notpinned, bits[c] | bits[ec]) & j_mask)
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

  decltype(auto) abq_get_ordered_moves(depth_t depth, const MoveLine &pline, const std::vector<float> &cmh_table,
                                       int8_t nchecks, bool king_in_check, move_t hashmove=board::nullmove) const
  {
    std::vector<std::tuple<mval_t, move_t, bool>> quiescmoves;
    quiescmoves.reserve(8);
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t edefendmap = get_attack_mask(ec) & bits[ec];
    const piece_bitboard_t pawn_attacks = piece::get_pawn_attacks(bits_pawns & bits[ec], ec);
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      mval_t val = .0;
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
        const score_t see = static_exchange_evaluation(i, j);
        if(see < 0)return;
        val = .0;
      } else if((allow_checking && is_naively_checking_move(i, j))) {
        const score_t see = static_exchange_evaluation(i, j);
        if(see < 0)return;
        reduce_nchecks = (m != hashmove);
        val += .1;
      } else {
        return;
      }
      if(val > -.75) {
        val += move_heuristic_pv(m, pline, hashmove);
      }
      val += move_heuristic_cmt(m, pline, cmh_table);
      quiescmoves.emplace_back(val, m, reduce_nchecks);
    });
    std::sort(quiescmoves.begin(), quiescmoves.end(), std::greater<>());
    return quiescmoves;
  }

  score_t alpha_beta_quiescence(score_t alpha, score_t beta, depth_t depth, MoveLine &pline, alpha_beta_state &ab_state, int8_t nchecks) {
    assert(alpha < beta);
    assert(check_valid_sequence(pline));
    // check draw and checkmate
    if(self.is_draw()) {
      pline.clear();
      return 0;
    } else if(self.is_checkmate()) {
      pline.clear();
      return -MATERIAL_KING;
    }

    score_t score = -MATERIAL_KING;
    score_t bestscore = -MATERIAL_KING;
    const COLOR c = activePlayer();

    // stand pat score
    const bool king_in_check = state.checkline[c] != bitmask::full;
    if(!king_in_check) {
      score = e_ttable_probe(ab_state.e_ttable);
      debug.update_standpat(depth, alpha, beta, score, pline, nchecks);
      if(score >= beta) {
        ++nodes_searched;
        pline.clear();
        return score;
      } else if(score > bestscore) {
        bestscore = score;
        // big delta pruning: the position is irrecoverable
        const score_t BIG_DELTA = MATERIAL_QUEEN + (has_promoting_pawns() ? MATERIAL_QUEEN - MATERIAL_PAWN : 0);
        if(score > alpha) {
          alpha = score;
        } else if(ENABLE_SEL_BIGDELTA && score < alpha - BIG_DELTA) {
          pline.clear();
          return score;
        }
      }
    }

    // ttable probe
    move_t hashmove = board::nullmove;
    move_t your_threatmove = board::nullmove;
    const auto &&[k, maybe_score] = ab_ttable_probe(alpha, beta, depth, hashmove, your_threatmove, pline, ab_state, false, nchecks);
    auto &zb = ab_state.ab_ttable[k];
    if(maybe_score != NOSCORE) {
      return maybe_score;
    }
    assert(check_valid_move(hashmove, false));

    constexpr bool overwrite = true && ENABLE_TT;

    // filter out and order quiescent moves
    decltype(auto) quiescmoves = abq_get_ordered_moves(depth, pline, ab_state.cmh_table, nchecks, king_in_check, hashmove);
    if(quiescmoves.empty()) {
      ++nodes_searched;
      if(king_in_check) {
        score = e_ttable_probe(ab_state.e_ttable);
      }
      pline.clear();
      return score;
    }

    // main alpha-beta loop
    bool ispathdep = false;
    TT_NODE_TYPE ndtype = TT_ALLNODE;
    move_t m_best = board::nullmove;
    for(size_t move_index = 0; move_index < quiescmoves.size(); ++move_index) {
      const auto [m_val, m, reduce_nchecks] = quiescmoves[move_index];
      MoveLine pline_alt = pline.branch_from_past(m);
      // delta pruning: if a move won't come as close as 3 pawns to alpha, ignore
      constexpr score_t DELTA = 300 * CENTIPAWN;
      const bool delta_prune_allowed = ENABLE_SEL_DELTA && !king_in_check && piece::size(bits[c] & ~bits_pawns) > 5;
      if(delta_prune_allowed && score + std::round(m_val * MATERIAL_PAWN) < alpha - DELTA) {
        continue;
      }
      // recurse
      bool new_ispathdep = false;
      { // quescence sub-search move scope
        decltype(auto) mscope = self.engine_mline_scope(m, pline_alt);
        new_ispathdep = is_path_dependent(pline_alt);
        score = -score_decay(alpha_beta_quiescence(-beta, -alpha, depth - 1, pline_alt, ab_state, reduce_nchecks ? nchecks - 1 : nchecks));
        debug.check_pathdep(new_ispathdep, depth, score, pline_alt);
      } // end move scope
      debug.update_q(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
      debug.check_score(depth, score, pline_alt);
      if(score >= beta) {
        pline.replace_line(pline_alt);
        if(overwrite && zb.should_replace(depth, TT_CUTOFF, tt_age) && !ispathdep && !pline.tb && !pline.find(board::nullmove)) {
          if(zb.is_inactive(tt_age))++zb_occupied;
          zb.write(state.info, score, depth, tt_age, TT_CUTOFF, pline);
        }
        return score;
      } else if(score > bestscore) {
        pline.replace_line(pline_alt);
        ispathdep = new_ispathdep;
        bestscore = score;
        m_best = m;
        if(score > alpha) {
          alpha = score;
          ndtype = TT_EXACT;
        }
      }
    }
    if(overwrite && m_best != board::nullmove && !ispathdep && !pline.find(board::nullmove)) {
      // ALLNODE or EXACT
      if(zb.should_replace(depth, ndtype, tt_age) && !pline.tb) {
        if(zb.is_inactive(tt_age))++zb_occupied;
        zb.write(state.info, bestscore, depth, tt_age, ndtype, pline);
      }
    }
    if(m_best == board::nullmove) {
      pline.clear();
    }
    return bestscore;
  }

  decltype(auto) tb_get_ordered_moves(const MoveLine &pline, move_t hashmove, move_t my_threatmove, const std::vector<float> &cmh_table, bool prune=true) {
    assert(tb_can_probe());
    ++tb_probes;
    decltype(auto) moves = tb::get_ranked_moves(self, prune);
    ++tb_hit;
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(c);
    const piece_bitboard_t edefendmap = get_attack_mask(ec) & bits[ec];
    for(auto &[val, m] : moves) {
      const pos_t i = bitmask::first(m), j = bitmask::second(m);
      val += move_heuristic_pv(m, pline, hashmove, my_threatmove);
      if(moves.size() > 3) {
        val += move_heuristic_see(i, j, edefendmap);
        val += move_heuristic_cmt(m, pline, cmh_table);
      }
    }
    std::sort(moves.begin(), moves.end(), std::greater<>());
    return moves;
  }

  decltype(auto) ab_get_ordered_moves(const MoveLine &pline, move_t hashmove, move_t my_threatmove, const std::vector<float> &cmh_table) {
    std::vector<std::pair<mval_t, move_t>> moves;
    moves.reserve(32);
    const COLOR c = activePlayer();
    const COLOR ec = enemy_of(activePlayer());
    const piece_bitboard_t edefendmap = get_attack_mask(ec) & bits[ec];
    iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      mval_t val = .0;
      val += move_heuristic_pv(m, pline, hashmove, my_threatmove);
      if(!mval_is_primary(val)) {
        val += move_heuristic_see(i, j, edefendmap);
        val += move_heuristic_cmt(m, pline, cmh_table);
      }
      moves.emplace_back(val, m);
    });
    std::sort(moves.begin(), moves.end(), std::greater<>());
    return moves;
  }

  score_t ab_nullmove_probe(score_t alpha, score_t beta, depth_t &depth,
                                           MoveLine &pline, alpha_beta_state &ab_state,
                                           move_t my_threatmove, move_t &your_threatmove, bool &mate_threat)
  {
    const depth_t R = (depth > 6) ? 4 : 3;
    const depth_t DR = 4;
    MoveLine pline_alt = pline.branch_from_past();
    score_t score = 0;
    {
      decltype(auto) mscope = self.engine_mline_scope(board::nullmove, pline_alt);
      score = -score_decay(alpha_beta_scout(-beta, std::max<depth_t>(0, depth-R-1), pline_alt, ab_state, false));
      mate_threat = ((score_is_mate(score) || score_is_tb(score)) && score < 0);
    }
    if(score >= beta) {
      your_threatmove = pline_alt.get_next_move();
      {
        pline_alt.premove(board::nullmove);
        const move_t nextmove = pline_alt.front();
        const size_t cmt_index = get_cmt_index(pline_alt, nextmove);
        if(cmt_index != SIZE_MAX) {
          assert(cmt_index < ab_state.cmh_table.size());
          const depth_t cmh_depth = std::max<depth_t>(0, depth - R + 2);
          ab_state.cmh_table[cmt_index] += (cmh_depth*cmh_depth) * 1e-7;
          ab_state.normalize_cmh_table(cmt_index);
        }
        pline_alt.recall();
      }
      depth -= DR;
      if(depth <= 0) {
        const int nchecks = 0;
        // TODO zug-zwang will fail low
        return alpha_beta_quiescence(beta-1, beta, 0, pline, ab_state, nchecks);
      }
    }
    return NOSCORE;
  }

  score_t alpha_beta_scout(score_t zw_beta, depth_t depth, MoveLine &pline, alpha_beta_state &ab_state,
                           bool allow_nullmoves, move_t my_threatmove=board::nullmove)
  {
    score_t alpha = zw_beta - 1, beta = zw_beta;
    assert(alpha + 1 == beta);
    assert(check_valid_sequence(pline));
    // check draw and checkmate
    if(self.is_draw()) {
      ++nodes_searched;
      pline.clear();
      return 0;
    } else if(self.is_checkmate()) {
      ++nodes_searched;
      pline.clear();
      return -MATERIAL_KING;
    } else if(tb_can_probe()) {
      assert(!is_draw_repetition());
      score_t wdl_score = tb_probe_wdl();
      ++tb_probes;
      // prune if WDL score certainly improves current position
      // WDL_WIN: win or draw  (0 or WDL+eval) real score >= 0
      // WDL_LOSE: lose or draw (0 or -WDL+eval) real score <= 0
      if(wdl_score != NOSCORE) {
        if((wdl_score <= 0 && alpha > 0) || (wdl_score >= 0 && beta < 0)) {
          ++tb_hit;
          pline.clear();
          pline.tb = true;
          ++nodes_searched;
          return evaluate(wdl_score);
        }
      }
    }
    // drop to quiescence search
    if(depth <= 0) {
      const int nchecks = 0;
      return alpha_beta_quiescence(alpha, beta, 0, pline, ab_state, nchecks);
    }
    // ttable probe
    move_t hashmove = my_threatmove;
    move_t your_threatmove = board::nullmove;
    const auto &&[k, maybe_score] = ab_ttable_probe(alpha, beta, depth, hashmove, your_threatmove, pline, ab_state, allow_nullmoves, 0);
    auto &zb = ab_state.ab_ttable[k];
    if(maybe_score != NOSCORE) {
      return maybe_score;
    }

    constexpr bool overwrite = true && ENABLE_TT;

    // nullmove pruning
    const COLOR c = activePlayer();
    const bool nullmove_allowed = ENABLE_SEL_NMR && allow_nullmoves && (state.checkline[c] == bitmask::full)
                                  && piece::size((bits_slid_diag | get_knight_bits() | bits_slid_orth) & bits[c]) > 0;
    bool mate_threat = false;
    if(nullmove_allowed) {
      score_t maybe_score_nm = ab_nullmove_probe(alpha, beta, depth, pline, ab_state, my_threatmove, your_threatmove, mate_threat);
      if(maybe_score_nm != NOSCORE) {
        return maybe_score_nm;
      }
    }

    std::vector<std::pair<mval_t, move_t>> moves;
//    ablated away
//    if(tb_can_probe()) {
//      moves = tb_get_ordered_moves(pline, hashmove, my_threatmove, ab_state.cmh_table, true);
//    }
    if(moves.empty()) {
      // move ordering: PV | hashmove | SEE | CMH
      // | means breaks ties
      moves = ab_get_ordered_moves(pline, hashmove, my_threatmove, ab_state.cmh_table);
    }
    // this is nonempty - draw and checkmate checks are above
    assert(!moves.empty());

    // pvsearch main loop
    score_t bestscore = -MATERIAL_KING;
    bool ispathdep = false;
    TT_NODE_TYPE ndtype = TT_ALLNODE;
    for(size_t move_index = 0; move_index < moves.size(); ++move_index) {
      const auto [m_val, m] = moves[move_index];
      // searchmoves filtering (UCI)
      if(ab_state.initdepth == depth && !ab_state.searchmoves.empty() && !ab_state.searchmoves.contains(m)) {
        continue;
      }
      MoveLine pline_alt = pline.branch_from_past(m);
      score_t score = 0;
      bool new_ispathdep = false;
      { // move scope
        decltype(auto) mscope = self.engine_mline_scope(m, pline_alt);
        new_ispathdep = is_path_dependent(pline_alt);
        pos_t i = bitmask::first(m), j = bitmask::second(m) & board::MOVEMASK;
        const bool interesting_move = (is_drop_move(i, j) || is_castling_move(i, j) || is_promotion_move(i, j) || (is_naively_capture_move(i, j) && m_val > -3.5) || (is_naively_checking_move(i, j) && m_val > -9.));
        const bool lmr_allowed = ENABLE_SEL_LMR && move_index >= 4 && depth >= 3
                                 && state.checkline[c] == bitmask::full && m_val < .1 && !interesting_move;
        if(lmr_allowed) {
          score = -score_decay(alpha_beta_scout(-alpha, depth - 2, pline_alt, ab_state, allow_nullmoves, your_threatmove));
        } else {
          score = -score_decay(alpha_beta_scout(-alpha, depth - 1, pline_alt, ab_state, allow_nullmoves, your_threatmove));
        }
        debug.check_pathdep(new_ispathdep, depth, score, pline_alt);
      } // end move scope
      debug.update_zw(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
      debug.check_score(depth, score, pline_alt);
//      debug.log_pv(depth, pline_alt, "move "s + pgn::_move_str(self, m) + " score "s + score_string(score));
      if(score > bestscore) {
        pline.replace_line(pline_alt);
        ispathdep = new_ispathdep;
        bestscore = score;
        if(score >= beta) {
          if(overwrite && zb.should_replace(depth, TT_CUTOFF, tt_age) && !ispathdep && !pline.tb && !pline.find(board::nullmove)) {
            if(zb.is_inactive(tt_age))++zb_occupied;
            zb.write(state.info, score, depth, tt_age, TT_CUTOFF, pline);
          }
          const size_t cmt_index = get_cmt_index(pline, m);
          if(cmt_index != SIZE_MAX) {
            ab_state.cmh_table[cmt_index] += (depth * depth * (float(move_index + 1) / float(moves.size()))) * 1e-7;
            ab_state.normalize_cmh_table(cmt_index);
          }
          return score;
        }
      }
    }
    if(overwrite && !ispathdep && !pline.tb && !pline.find(board::nullmove)) {
      if(ndtype == TT_ALLNODE && zb.should_replace(depth, TT_ALLNODE, tt_age)) {
        if(zb.is_inactive(tt_age))++zb_occupied;
        zb.write(state.info, bestscore, depth, tt_age, TT_ALLNODE, pline);
      }
    }
//    debug.log_pv(depth, pline, "returning all-node pv-bestscore "s + std::to_string(score_float(bestscore)));
    return bestscore;
  }

  template <typename F>
  score_t alpha_beta_pv(score_t alpha, score_t beta, depth_t depth, MoveLine &pline, alpha_beta_state &ab_state,
                        bool allow_nullmoves, F &&callback_f, move_t my_threatmove=board::nullmove)
  {
    assert(alpha < beta);
    assert(check_valid_sequence(pline));
    // check draw and checkmate
    if(self.is_draw()) {
      ++nodes_searched;
      pline.clear();
      return 0;
    } else if(self.is_checkmate()) {
      ++nodes_searched;
      pline.clear();
      return -MATERIAL_KING;
    } else if(tb_can_probe()) {
      assert(!is_draw_repetition());
      score_t wdl_score = tb_probe_wdl();
      ++tb_probes;
      // prune if WDL score is too low
      if(wdl_score != NOSCORE) {
        if((wdl_score <= 0 && alpha > 0) || (wdl_score >= 0 && beta < 0)) {
          ++tb_hit;
          pline.tb = true;
          pline.clear();
          ++nodes_searched;
          return evaluate(wdl_score);
        } else if(alpha < 0 && wdl_score >= 0 && beta != 0) {
          assert(beta > 0);
          alpha = 0;
        }
      }
    }
    // drop to quiescence search
    if(depth <= 0) {
      const int nchecks = 0;
      return alpha_beta_quiescence(alpha, beta, 0, pline, ab_state, nchecks);
    }
    // ttable probe
    move_t hashmove = my_threatmove;
    move_t your_threatmove = board::nullmove;
    const auto &&[k, maybe_score] = ab_ttable_probe(alpha, beta, depth, hashmove, your_threatmove, pline, ab_state, allow_nullmoves, 0);
    auto &zb = ab_state.ab_ttable[k];
    if(maybe_score != NOSCORE) {
      return maybe_score;
    }

    constexpr bool overwrite = true && ENABLE_TT;

    std::vector<std::pair<mval_t, move_t>> moves;
    if(tb_can_probe()) {
      // get moves according to their WDL Rank
      // empty if error; prunes moves with worse wdl score
      moves = tb_get_ordered_moves(pline, hashmove, my_threatmove, ab_state.cmh_table, true);
    }
    if(moves.empty()) {
      // move ordering: PV | hashmove | SEE | CMH
      // | means breaks ties
      moves = ab_get_ordered_moves(pline, hashmove, my_threatmove, ab_state.cmh_table);
    }
    // this is nonempty - draw and checkmate checks are above
    assert(!moves.empty());

    // internal iterative deepening
    const mval_t _max_mval = moves.front().first;
    const bool iid_allow = true && ENABLE_IID
                           && !(mval_is_primary(_max_mval) || mval_is_tb_windraw(_max_mval))
                           && depth >= 9;
    if(iid_allow) {
      assert(tb_can_probe() || pline.empty());
      alpha_beta_pv(alpha, beta, depth / 2, pline, ab_state, true, make_callback_f(), your_threatmove);
      const move_t m = pline.front();
      // find move index
      size_t m_ind = 0;
      for(m_ind = 0; moves[m_ind].second != m && m_ind < moves.size(); ++m_ind)
        ;
      assert(m_ind != moves.size());
      moves[m_ind].first = move_heuristic_pv(moves[m_ind].second, pline, board::nullmove, board::nullmove);
      std::sort(moves.begin(), moves.end(), std::greater<>());
      //_perror("PV iid allowed %d\n", (int)depth);
    }

    // pvsearch main loop
    score_t bestscore = -MATERIAL_KING;
    bool ispathdep = false;
    TT_NODE_TYPE ndtype = TT_ALLNODE;
    for(size_t move_index = 0; move_index < moves.size(); ++move_index) {
      const auto [m_val, m] = moves[move_index];
      // searchmoves filtering (UCI)
      if(ab_state.initdepth == depth && !ab_state.searchmoves.empty() && !ab_state.searchmoves.contains(m)) {
        continue;
      }
      // pv for full-window score
      MoveLine pline_alt = pline.branch_from_past(m);
      score_t score = 0;
      bool new_ispathdep = false;
      // PV or hash move, full window
      if(mval_is_primary(m_val) || (mval_is_tb_windraw(m_val) && move_index == 0)) {
        { // PV move scope
          decltype(auto) mscope = self.engine_mline_scope(m, pline_alt);
          new_ispathdep = is_path_dependent(pline_alt);
          score = -score_decay(alpha_beta_pv(-beta, -alpha, depth - 1, pline_alt, ab_state, allow_nullmoves, callback_f, your_threatmove));
          debug.check_pathdep(new_ispathdep, depth, score, pline_alt);
        } // end move scope
        debug.update_pv(depth, alpha, beta, bestscore, score, m, pline, pline_alt, "hashmove"s);
        debug.check_score(depth, score, pline_alt);
//        debug.log_pv(depth, pline, "move "s + pgn::_move_str(self, m) + " score "s + std::to_string(score_float(score)));
        if(m == pline.front() && score < alpha && ab_state.initdepth == depth && !ispathdep) {
          // fail low on aspiration. re-do the search
          return score;
        }
        if(score > alpha) {
          alpha = score;
          ndtype = TT_EXACT;
        }
      // other moves, zero window first
      } else {
        { // scout move scope
          const COLOR c = activePlayer();
          decltype(auto) mscope = self.engine_mline_scope(m, pline_alt);
          new_ispathdep = is_path_dependent(pline_alt);
          const pos_t i = bitmask::first(m), j = bitmask::second(m) & board::MOVEMASK;
          const bool interesting_move = (is_drop_move(i, j) || is_castling_move(i, j) || is_promotion_move(i, j) || (is_naively_capture_move(i, j) && m_val > -3.5) || (is_naively_checking_move(i, j) && m_val > -9.));
          const bool lmr_allowed = ENABLE_SEL_LMR && move_index >= (pline.is_mainline() ? 15 : 4)
                                   && depth >= 3 && state.checkline[c] == bitmask::full && m_val < .1 && !interesting_move;
          MoveLine pline_alt2 = pline_alt;
          score_t subscore = NOSCORE;
          if(lmr_allowed) {
            subscore = alpha_beta_scout(-alpha, depth - 2, pline_alt2, ab_state, allow_nullmoves, your_threatmove);
          } else {
            subscore = alpha_beta_scout(-alpha, depth - 1, pline_alt2, ab_state, allow_nullmoves, your_threatmove);
          }
          score = -score_decay(subscore);
          debug.check_score(depth, subscore, pline_alt2);
          debug.check_pathdep(new_ispathdep, depth, score, pline_alt);
          if(score > alpha && score < beta) {
            // zw search successful, perform full-window search
            pline_alt.replace_line(pline_alt2);
            new_ispathdep = is_path_dependent(pline_alt);
            score = -score_decay(alpha_beta_pv(-beta, -alpha, depth - 1, pline_alt, ab_state, allow_nullmoves, callback_f, your_threatmove));
            debug.check_pathdep(new_ispathdep, depth, score, pline_alt);
            // score > beta included
            if(score > alpha) {
              alpha = score;
              ndtype = TT_EXACT;
            }
          } else {
            pline_alt.replace_line(pline_alt2);
          }
        } // end move scope
        debug.update_pv(depth, alpha, beta, bestscore, score, m, pline, pline_alt);
        debug.check_score(depth, score, pline_alt);
//        debug.log_pv(depth, pline_alt, "move "s + pgn::_move_str(self, m) + " score "s + std::to_string(score_float(score)));
      }
      debug.check_score(depth, score, pline_alt);
      if(score > bestscore) {
        pline.replace_line(pline_alt);
        ispathdep = new_ispathdep;
        your_threatmove = pline.get_next_move();
        bestscore = score;
        if(score >= beta) {
          if(overwrite && zb.should_replace(depth, TT_CUTOFF, tt_age) && !ispathdep && !pline.tb && !pline.find(board::nullmove)) {
            if(zb.is_inactive(tt_age))++zb_occupied;
            zb.write(state.info, score, depth, tt_age, TT_CUTOFF, pline);
          }
          const size_t cmt_index = get_cmt_index(pline, m);
          if(cmt_index != SIZE_MAX) {
            ab_state.cmh_table[cmt_index] += (depth * depth * (float(move_index + 1) / float(moves.size()))) * 1e-7;
            ab_state.normalize_cmh_table(cmt_index);
          }
          return score;
        }
      }
      if(ab_state.initdepth <= depth + 1 + (depth_t(pline.line.size()) / 7) && move_index + 1 != moves.size()) {
        if(!callback_f(depth, bestscore)) {
          return bestscore;
        }
      }
    }
    if(overwrite && !ispathdep && !pline.tb && !pline.find(board::nullmove)) {
      if(ndtype == TT_ALLNODE && zb.should_replace(depth, TT_ALLNODE, tt_age)) {
        if(zb.is_inactive(tt_age))++zb_occupied;
        zb.write(state.info, bestscore, depth, tt_age, TT_ALLNODE, pline);
      } else if(ndtype == TT_EXACT && zb.should_replace(depth, TT_EXACT, tt_age)) {
        if(zb.is_inactive(tt_age))++zb_occupied;
        zb.write(state.info, bestscore, depth, tt_age, TT_EXACT, pline);
      }
    }
//    debug.log_pv(depth, pline, "returning all-node pv-bestscore "s + std::to_string(score_float(bestscore)));
    return bestscore;
  }


  struct ab_storage_t {
    zobrist::StoreScope<tt_ab_entry> ab_ttable_scope;
    zobrist::StoreScope<tt_eval_entry> e_ttable_scope;
    std::vector<float> cmh_table;

    void reset() {
      ab_ttable_scope.reset();
      e_ttable_scope.reset();
    }
  };

  typedef struct _iddfs_state {
    depth_t curdepth = 1;
    score_t eval = 0;
    MoveLine pline;

    INLINE void reset() {
      curdepth = 1;
      eval = 0;
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
  decltype(auto) iterative_deepening_dfs(depth_t depth, ab_storage_t &ab_storage, iddfs_state &idstate, const std::unordered_set<move_t> &searchmoves, F &&callback_f) {
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
    str::pdebug("iterative deepening");
    for(depth_t d = idstate.curdepth; d <= depth; ++d) {
      if(check_line_terminates(idstate.pline) && ssize_t(idstate.pline.size()) < d && !idstate.pline.tb) {
        idstate.eval = get_pvline_score(idstate.pline);
//        fprintf(stderr, "pv: (%s) %s\n", score_string(idstate.eval).c_str(), idstate.pline.pgn_full(self).c_str());
//        fflush(stderr);
        break;
      }
      const score_t eval_prev = idstate.eval;
      const std::valarray<score_t> aspiration_window = {CENTIPAWN, 15*CENTIPAWN, 125*CENTIPAWN, MATERIAL_QUEEN - MATERIAL_PAWN};
      pos_t aw_index_alpha = 0, aw_index_beta = 0;
      MoveLine new_pline = idstate.pline;
      do {
        score_t aw_alpha = -MATERIAL_KING, aw_beta = MATERIAL_KING;
        if(aw_index_alpha < aspiration_window.size()) {
          aw_alpha = eval_prev - aspiration_window[aw_index_alpha];
        }
        if(aw_index_beta < aspiration_window.size()) {
          aw_beta = eval_prev + aspiration_window[aw_index_beta];
        }
        const bool final_window = (aw_index_alpha == aspiration_window.size() && aw_index_beta == aspiration_window.size());
        str::pdebug("depth:", d, "aw", score_string(aw_alpha), score_string(aw_beta));
        bool inner_break = false;
        alpha_beta_state ab_state = (alpha_beta_state){ .ab_ttable=ab_ttable, .e_ttable=e_ttable, .cmh_table=cmh_table,
                                                        .initdepth=d, .searchmoves=searchmoves };
        const score_t new_eval = alpha_beta_pv(aw_alpha, aw_beta, d, new_pline, ab_state, true,
          [&](depth_t _depth, score_t _eval) mutable -> bool {
            const move_t _m = new_pline.front();
            should_stop = !callback_f(false);
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
          if(!should_stop) {
            idstate.eval=new_eval, idstate.pline=new_pline, idstate.curdepth=d;
            should_stop = !callback_f(true);
          }
          debug.check_score(d, idstate.eval, idstate.pline);
          if(should_stop || (score_is_mate(idstate.eval) && d > 0 && ssize_t(idstate.pline.size()) < d)) {
            should_stop = true;
          }
          break;
        }
        if(should_stop)break;
      } while(1);
      str::pdebug("IDDFS:", d, "pline:", idstate.pline.pgn(self), "size:", idstate.pline.size(), "eval", score_float(idstate.eval), "nodes", nodes_searched);
      if(should_stop)break;
    }
    if(idstate.pline.empty() || idstate.pline.front() == board::nullmove) {
      const move_t m = get_random_move();
      if(m != board::nullmove) {
        str::pdebug("replace move with random move", _move_str(m), m);
        idstate.curdepth = 0;
        idstate.pline = MoveLine(std::vector<move_t>{m});
        str::pdebug("pline", _line_str(idstate.pline));
      }
    } else if(idstate.pline.find(board::nullmove)) {
      while(idstate.pline.find(board::nullmove)) {
        idstate.pline.pop_back();
      }
    }
    str::pdebug(_line_str(idstate.pline));
    return idstate.eval;
  }

  ply_index_t tt_age = 0;
  size_t ezb_hit = 0, ezb_miss = 0, ezb_occupied = 0;
  size_t tb_hit = 0, tb_probes = 0;

  std::shared_ptr<zobrist::ttable<tt_ab_entry>> ab_ttable;
  std::shared_ptr<zobrist::ttable<tt_eval_entry>> e_ttable;

  decltype(auto) get_zobrist_alphabeta_scope() {
    const size_t size_ab = zobrist_size;
    // approximate
    const size_t mem_ab = size_ab * (sizeof(tt_ab_entry));
    const size_t size_e = 0;
    const size_t mem_e = size_e * sizeof(tt_eval_entry);
    const size_t size_cmh = size_t(NO_PIECES) * size_t(board::SIZE);
    const size_t size_cmt = size_t(NO_COLORS) * size_cmh * size_cmh;
    const size_t mem_cmh = sizeof(std::vector<float>) + size_cmt  * sizeof(float);
    const size_t mem_total = mem_ab+mem_e+mem_cmh;
    str::pdebug("alphabeta scope", "ab:", mem_ab, "e:", mem_e, "cmh:", mem_cmh, "total:", mem_total);
    _printf("MEM: %zuMB %zuKB %zuB\n", mem_total>>20, (mem_total>>10)&((1<<10)-1), mem_total&((1<<10)-1));
    return (ab_storage_t){
      .ab_ttable_scope=zobrist::make_store_object_scope<tt_ab_entry>(ab_ttable, size_ab),
      .e_ttable_scope=zobrist::make_store_object_scope<tt_eval_entry>(e_ttable, size_e),
      .cmh_table=std::vector<float>(size_cmt, .0)
    };
  }

  template <typename F>
  move_t start_thinking(depth_t depth, iddfs_state &idstate, F &&callback_f, const std::unordered_set<move_t> &searchmoves) {
    nodes_searched = 0;
    zb_hit = 0, zb_miss = 0, zb_occupied = 0;
    ezb_hit = 0, ezb_miss = 0, ezb_occupied = 0;
    tb_hit = 0, tb_probes = 0;
    ++tt_age;
    decltype(auto) ab_storage = get_zobrist_alphabeta_scope();
    assert(ab_ttable != nullptr && e_ttable != nullptr);
    debug.set_depth(depth);
    iterative_deepening_dfs(depth, ab_storage, idstate, searchmoves, std::forward<F>(callback_f));
    return idstate.currmove();
  }

  INLINE move_t start_thinking(depth_t depth, iddfs_state &idstate, const std::unordered_set<move_t> &searchmoves={}) {
    return start_thinking(depth, idstate, make_callback_f(), searchmoves);
  }

  INLINE move_t start_thinking(depth_t depth, const std::unordered_set<move_t> &searchmoves={}) {
    iddfs_state idstate;
    return start_thinking(depth, idstate, searchmoves);
  }

  // constructor
  DebugTracer<Engine> debug;
  Engine(const fen::FEN &fen=fen::starting_pos, size_t zbsize=ZOBRIST_SIZE):
    Perft(fen, zbsize), debug(self)
  {
    const std::vector<PIECE> piece_types = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
    std::vector<std::pair<score_t, PIECE>> pieces;
    for(PIECE p : piece_types) {
      pieces.emplace_back(material_of(p), p);
    }
    std::sort(pieces.begin(), pieces.end());
    for(const auto &[_, p] : pieces) {
      MATERIAL_PIECE_ORDERING.emplace_back(p);
    }
  }

  virtual ~Engine() {}
};

const int32_t TB_VALUE_FPAWN = Engine::MATERIAL_PAWN;
const int32_t TB_VALUE_MATE = Engine::MATERIAL_KING;
const int32_t TB_VALUE_INFINITE = Engine::MATERIAL_KING * 2;
const int32_t TB_MAX_MATE_PLY = INT16_MAX >> 1;
