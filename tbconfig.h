/*
 * tbconfig.h
 * (C) 2015 basil, all rights reserved,
 * Modifications Copyright 2016-2017 Jon Dart
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef TBCONFIG_H
#define TBCONFIG_H

#include <cstdint>

/****************************************************************************/
/* BUILD CONFIG:                                                            */
/****************************************************************************/

/*
 * Define TB_CUSTOM_POP_COUNT to override the internal popcount
 * implementation. To do this supply a macro or function definition
 * here:
 */
#define TB_CUSTOM_POP_COUNT(x) bitmask::count_bits(x)

/*
 * Define TB_CUSTOM_LSB to override the internal lsb
 * implementation. To do this supply a macro or function definition
 * here:
 */
#define TB_CUSTOM_LSB(x) bitmask::log2_lsb(x)

/*
 * Define TB_NO_STDINT if you do not want to use <stdint.h> or it is not
 * available.
 */
/* #define TB_NO_STDINT */

/*
 * Define TB_NO_STDBOOL if you do not want to use <stdbool.h> or it is not
 * available or unnecessary (e.g. C++).
 */
/* #define TB_NO_STDBOOL */

/*
 * Define TB_NO_THREADS if your program is not multi-threaded.
 */
#define TB_NO_THREADS

/*
 * Define TB_NO_HELPER_API if you do not need the helper API.
 */
#define TB_NO_HELPER_API

/*
 * Define TB_NO_HW_POP_COUNT if there is no hardware popcount instruction.
 *
 * Note: if defined, TB_CUSTOM_POP_COUNT is always used in preference
 * to any built-in popcount functions.
 *
 * If no custom popcount function is defined, and if the following
 * define is not set, the code will attempt to use an available hardware
 * popcnt (currently supported on x86_64 architecture only) and otherwise
 * will fall back to a software implementation.
 */
/* #define TB_NO_HW_POP_COUNT */

/***************************************************************************/
/* SCORING CONSTANTS                                                       */
/***************************************************************************/
/*
 * Fathom can produce scores for tablebase moves. These depend on the
 * value of a pawn, and the magnitude of mate scores. The following
 * constants are representative values but will likely need
 * modification to adapt to an engine's own internal score values.
 */

//#define TB_VALUE_FPAWN 100  /* value of pawn in endgame */
//#define TB_VALUE_MATE 32000
//#define TB_VALUE_INFINITE 32767 /* value above all normal score values */
#define TB_VALUE_DRAW 0
//#define TB_MAX_MATE_PLY 255

extern const int32_t TB_VALUE_FPAWN;
extern const int32_t TB_VALUE_MATE;
extern const int32_t TB_VALUE_INFINITE;
extern const int32_t TB_MAX_MATE_PLY;

/***************************************************************************/
/* ENGINE INTEGRATION CONFIG                                               */
/***************************************************************************/

/*
 * If you are integrating tbprobe into an engine, you can replace some of
 * tbprobe's built-in functionality with that already provided by the engine.
 * This is OPTIONAL.  If no definition are provided then tbprobe will use its
 * own internal defaults.  That said, for engines it is generally a good idea
 * to avoid redundancy.
 */

#include <Piece.hpp>

/*
 * Define TB_KING_ATTACKS(square) to return the king attacks bitboard for a
 * king at `square'.
 */
#define TB_KING_ATTACKS(square) piece::get_king_attack((pos_t)(square))

/*
 * Define TB_KNIGHT_ATTACKS(square) to return the knight attacks bitboard for
 * a knight at `square'.
 */
#define TB_KNIGHT_ATTACKS(square) piece::get_knight_attack((pos_t)(square))

/*
 * Define TB_ROOK_ATTACKS(square, occ) to return the rook attacks bitboard
 * for a rook at `square' assuming the given `occ' occupancy bitboard.
 */
#define TB_ROOK_ATTACKS(square, occ) piece::get_sliding_orth_attack((pos_t)(square), (piece_bitboard_t)(occ))

/*
 * Define TB_BISHOP_ATTACKS(square, occ) to return the bishop attacks bitboard
 * for a bishop at `square' assuming the given `occ' occupancy bitboard.
 */
#define TB_BISHOP_ATTACKS(square, occ) piece::get_sliding_diag_attack((pos_t)(square), (piece_bitboard_t)(occ))

/*
 * Define TB_QUEEN_ATTACKS(square, occ) to return the queen attacks bitboard
 * for a queen at `square' assuming the given `occ' occupancy bitboard.
 * NOTE: If no definition is provided then tbprobe will use:
 *       TB_ROOK_ATTACKS(square, occ) | TB_BISHOP_ATTACKS(square, occ)
 */
/* #define TB_QUEEN_ATTACKS(square, occ)    <DEFINITION> */

/*
 * Define TB_PAWN_ATTACKS(square, color) to return the pawn attacks bitboard
 * for a `color' pawn at `square'.
 * NOTE: This definition must work for pawns on ranks 1 and 8.  For example,
 *       a white pawn on e1 attacks d2 and f2.  A black pawn on e1 attacks
 *       nothing.  Etc.
 * NOTE: This definition must not include en passant captures.
 */
#define TB_PAWN_ATTACKS(square, color) piece::get_pawn_attack((pos_t)(square), color ? WHITE : BLACK)

#include <external/fathom/src/tbprobe.c>

#include <Board.hpp>

namespace tb {
  INLINE bool init(const std::string &syzygy_path="external/syzygy/src"s) {
    return tb_init(syzygy_path.c_str());
  }

  INLINE void free() {
    tb_free();
  }

  INLINE move_t u16_to_move_t(uint16_t tb_move) {
    const pos_t i = TB_MOVE_FROM(tb_move);
    const pos_t _j = TB_MOVE_TO(tb_move);
    pos_t promote = 0x00;
    switch(TB_MOVE_PROMOTES(tb_move)) {
      case TB_PROMOTES_FKNIGHT:promote=board::PROMOTE_KNIGHT;break;
      case TB_PROMOTES_FBISHOP:promote=board::PROMOTE_BISHOP;break;
      case TB_PROMOTES_FROOK:  promote=board::PROMOTE_ROOK;break;
      case TB_PROMOTES_FQUEEN: promote=board::PROMOTE_QUEEN;break;
    }
    move_t m = bitmask::_pos_pair(i, _j | promote);
    return m;
  }

  INLINE move_t u_to_move_t(unsigned result) {
    const pos_t i = TB_GET_FROM(result);
    const pos_t _j = TB_GET_TO(result);
    pos_t promote = 0x00;
    switch(TB_GET_PROMOTES(result)) {
      case TB_PROMOTES_FKNIGHT:promote=board::PROMOTE_KNIGHT;break;
      case TB_PROMOTES_FBISHOP:promote=board::PROMOTE_BISHOP;break;
      case TB_PROMOTES_FROOK:  promote=board::PROMOTE_ROOK;break;
      case TB_PROMOTES_FQUEEN: promote=board::PROMOTE_QUEEN;break;
    }
    move_t m = bitmask::_pos_pair(i, _j | promote);
    return m;
  }

  // above preprocessor definitions should produce a correct
  // material score (no heuristics)
  INLINE auto score_of(const TbRootMove &tb_root_move) {
    auto sc = tb_root_move.tbScore;
    if(std::abs(sc) >= TB_VALUE_MATE - 16000) {
      const decltype(sc) sc_mate = TB_VALUE_MATE - tb_root_move.pvSize;
      sc = (sc < 0) ? -sc_mate : sc_mate;
    }
    return sc;
  }

  INLINE bool can_probe(const Board &b) {
    return (TB_LARGEST > 0) && !(
        b.crazyhouse || (
          b.chess960
          && b.get_castlings_rook_mask() != bitmask::_pos_pair(0x00, 0x00))
      || piece::size(b.bits[WHITE] | b.bits[BLACK]) > TB_LARGEST
      // anything except castlings == 0x00 is rejected
      //|| b.get_castlings_rook_mask() != bitmask::_pos_pair(0x00, 0x00)
    );
  }

  namespace internal {
  INLINE unsigned _get_enpassant(const Board &b) {
    return (b.enpassant_trace() == board::nopos) ? 0 : b.enpassant_trace();
  }

  INLINE bool _get_turn(const Board &b) {
    return (b.activePlayer() == WHITE);
  }

  INLINE unsigned _get_castlings(const Board &b) {
    if(b.chess960) {
      assert(b.get_castlings_rook_mask() == bitmask::_pos_pair(0x00, 0x00));
      return 0x00;
    }
#if 1
    // anything except castlings == 0x00 is rejected
    assert(b.get_castlings_rook_mask() == bitmask::_pos_pair(0x00, 0x00));
    return 0x00;
#else
    unsigned castlings = 0x00;
    if(b.is_castling(WHITE, KING_SIDE)) {
      castlings |= TB_CASTLING_K;
    }
    if(b.is_castling(WHITE, QUEEN_SIDE)) {
      castlings |= TB_CASTLING_Q;
    }
    if(b.is_castling(BLACK, KING_SIDE)) {
      castlings |= TB_CASTLING_k;
    }
    if(b.is_castling(BLACK, QUEEN_SIDE)) {
      castlings |= TB_CASTLING_q;
    }
    return castlings;
#endif
  }

  MoveLine _get_rootmove_pline(TbRootMove &tb_root_move) {
    MoveLine mline;
    for(size_t i = 0; i < tb_root_move.pvSize; ++i) {
      mline.put(tb::u16_to_move_t(tb_root_move.pv[i]));
    }
    return mline;
  }

  const char *_wdl_to_name_str[5] =
  {
      "loss",
      "blessed_loss",
      "draw",
      "cursed_win",
      "win"
  };

  template <typename F>
  void _foreach_tbrootmoves(TbRootMoves &tb_root_moves, F &&func) {
    for(size_t m_index = 0; m_index < tb_root_moves.size; ++m_index) {
      TbRootMove tb_root_move = tb_root_moves.moves[m_index];
      const move_t m = tb::u16_to_move_t(tb_root_move.move);
//      func(m, tb::score_of(tb_root_move), tb_root_move.tbRank, [=]() mutable -> MoveLine {
//        return tb::internal::_get_rootmove_pline(tb_root_move);;
//      });
      func(m, tb::score_of(tb_root_move), tb_root_move.tbRank);
    }
  }
  } // namespace internal

  std::string wdl_to_str(const Board &b, unsigned wdl) {
    return tb::internal::_wdl_to_name_str[(tb::internal::_get_turn(b)? wdl: 4-wdl)];
  }

  int8_t probe_wdl(const Board &b) {
    assert(tb::can_probe(b));
    const ply_index_t halfmoves = 0;
    unsigned result = tb_probe_wdl(
      b.bits[WHITE], b.bits[BLACK], b.get_king_bits(),
      b.bits_slid_diag & b.bits_slid_orth,
      b.bits_slid_orth & ~b.bits_slid_diag,
      b.bits_slid_diag & ~b.bits_slid_orth,
      b.get_knight_bits(), b.bits_pawns,
      0, // anything except halfmoves == 0 is rejected
      tb::internal::_get_castlings(b),
      tb::internal::_get_enpassant(b),
      tb::internal::_get_turn(b)
    );
    if(result == TB_RESULT_FAILED) {
      return -1;
    }
    return TB_GET_WDL(result);
  }

  std::string probe_wdl_s(const Board &b) {
    int8_t res = tb::probe_wdl(b);
    if(res == -1) {
      return "failed"s;
    }
    return tb::wdl_to_str(b, res);
  }

  template <typename F>
  INLINE int8_t probe_root_dtz(const Board &b, F &&func) {
    assert(tb::can_probe(b));
    //assert(tb::probe_wdl(b) != -1);
    TbRootMoves result = {
      .size = 0,
    };
    // TODO?
    const bool has_repeated = true;
    // score will be corrected by the engine, so better not lose hope
    const bool use_rule50 = true;
    const int ret = tb_probe_root_dtz(
      b.bits[WHITE], b.bits[BLACK], b.get_king_bits(),
      b.bits_slid_diag & b.bits_slid_orth,
      b.bits_slid_orth & ~b.bits_slid_diag,
      b.bits_slid_diag & ~b.bits_slid_orth,
      b.get_knight_bits(), b.bits_pawns,
      b.get_halfmoves(), // anything except halfmoves == 0 is rejected
      tb::internal::_get_castlings(b),
      tb::internal::_get_enpassant(b),
      tb::internal::_get_turn(b),
      has_repeated, use_rule50,
      &result
    );
    if(ret == 0)return -1;
    tb::internal::_foreach_tbrootmoves(result, std::forward<F>(func));
    return 0;
  }

  template <typename F>
  INLINE int8_t probe_root_wdl(const Board &b, F &&func) {
    assert(tb::can_probe(b));
    bool use_rule50 = true;
    TbRootMoves moveresults;
    unsigned ret = tb_probe_root_wdl(
      b.bits[WHITE], b.bits[BLACK], b.get_king_bits(),
      b.bits_slid_diag & b.bits_slid_orth,
      b.bits_slid_orth & ~b.bits_slid_diag,
      b.bits_slid_diag & ~b.bits_slid_orth,
      b.get_knight_bits(), b.bits_pawns,
      b.get_halfmoves(), // anything except halfmoves == 0 is rejected
      tb::internal::_get_castlings(b),
      tb::internal::_get_enpassant(b),
      tb::internal::_get_turn(b),
      use_rule50,
      &moveresults
    );
    if(ret == TB_RESULT_FAILED) {
      return -1;
    }
    tb::internal::_foreach_tbrootmoves(moveresults, std::forward<F>(func));
    return 0;
  }

  INLINE decltype(auto) get_ranked_moves(Board &b, bool prune=false) {
    assert(tb::can_probe(b));
    std::vector<std::pair<float, move_t>> tbmoves;
    tbmoves.reserve(8);
    int32_t min_tbrank = INT32_MIN, min_tbscore = INT32_MIN;
    auto &&func = [&](move_t m, int32_t tbRank, int32_t tbScore) mutable -> void {
//      if(prune && (min_tbrank < tbRank || (min_tbrank == tbRank && min_tbscore < tbScore))) {
      if(prune && min_tbrank < tbRank) {
        tbmoves.clear();
        min_tbrank=tbRank, min_tbscore=tbScore;
      }
      if(!prune || min_tbrank == tbRank) {
        float val = float(tbRank) / TB_VALUE_FPAWN + float(tbScore) / 1000.;
        tbmoves.emplace_back(val / 2, m);
      }
    };
    int8_t ret = probe_root_dtz(b, func);
    if(ret == -1) {
      ret = probe_root_wdl(b, func);
    }
    return tbmoves;
  }

  template <typename F>
  INLINE void probe_root(Board &b, F &&func) {
    if(!tb::can_probe(b)) {
      return;
    }
    unsigned moveresults[TB_MAX_MOVES];
    unsigned result = tb_probe_root(
      b.bits[WHITE], b.bits[BLACK], b.get_king_bits(),
      b.bits_slid_diag & b.bits_slid_orth,
      b.bits_slid_orth & ~b.bits_slid_diag,
      b.bits_slid_diag & ~b.bits_slid_orth,
      b.get_knight_bits(), b.bits_pawns,
      b.get_halfmoves(),
      tb::internal::_get_castlings(b),
      tb::internal::_get_enpassant(b),
      tb::internal::_get_turn(b),
      moveresults
    );
    assert(result != TB_RESULT_FAILED);
    if(result == TB_RESULT_CHECKMATE || result == TB_RESULT_STALEMATE) {
      return;
    }
//    func(tb::u_to_move_t(result), TB_GET_WDL(result), TB_GET_DTZ(result));
    for(size_t i = 0; moveresults[i] != TB_RESULT_FAILED; ++i) {
      move_t m = tb::u_to_move_t(moveresults[i]);
      func(m, TB_GET_WDL(moveresults[i]), TB_GET_DTZ(moveresults[i]));
    }
  }

  INLINE move_t probe_root(const Board &b) {
    assert(tb::can_probe(b));
    unsigned result = tb_probe_root(
      b.bits[WHITE], b.bits[BLACK], b.get_king_bits(),
      b.bits_slid_diag & b.bits_slid_orth,
      b.bits_slid_orth & ~b.bits_slid_diag,
      b.bits_slid_diag & ~b.bits_slid_orth,
      b.get_knight_bits(), b.bits_pawns,
      b.get_halfmoves(), // anything except halfmoves == 0 is rejected
      tb::internal::_get_castlings(b),
      tb::internal::_get_enpassant(b),
      tb::internal::_get_turn(b),
      NULL
    );
    if(result == TB_RESULT_FAILED) {
      return board::nullmove;
    } else if(result == TB_RESULT_CHECKMATE || result == TB_RESULT_STALEMATE) {
      return board::nullmove;
    }
    return tb::u_to_move_t(result);
  }
} // namespace tb

#endif
