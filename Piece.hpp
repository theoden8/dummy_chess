#pragma once


#include <iostream>

#include <String.hpp>
#include <Bitmask.hpp>

#include <m42.h>

#define self (*this)

// pieces
typedef enum : pos_t { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, EMPTY, NO_PIECES=EMPTY } PIECE;
using piece_index_t = pos_t;

// two bytes, one for each move; second byte may contain information about promotion (2 bits)
typedef pos_pair_t move_t;
typedef int16_t ply_index_t;

// type alias: bitboard (as a mask)
typedef uint64_t piece_bitboard_t;

// player color, neutral when neither/both
typedef enum : pos_t { WHITE, BLACK, NEUTRAL, NO_COLORS=NEUTRAL, BOTH } COLOR;

inline constexpr COLOR enemy_of(COLOR c) {
  if(c == NEUTRAL)return NEUTRAL;
  return (c == WHITE) ? BLACK : WHITE;
}

typedef enum : pos_t { KING_SIDE, QUEEN_SIDE } CASTLING_SIDE;

enum { A,B,C,D,E,F,G,H } BOARD_FILE;

typedef enum : pos_t {
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8,
} BOARD_CELL;

namespace board {
  constexpr pos_t LEN = 8;
  constexpr pos_t SIZE = LEN*LEN;
  constexpr pos_t MOVEMASK = 0x3f;
  constexpr pos_t PROMOTE_KNIGHT = 0<<6,
                  PROMOTE_BISHOP = 1<<6,
                  PROMOTE_ROOK = 2<<6,
                  PROMOTE_QUEEN = 3<<6;
  constexpr pos_t nopos = 0xff;
  constexpr move_t nullmove = bitmask::_pos_pair(nopos, nopos);
  constexpr ply_index_t nocastlings = INT16_MAX;
  constexpr pos_t CASTLING_K_WHITE = 0,
                  CASTLING_Q_WHITE = 1,
                  CASTLING_K_BLACK = 2,
                  CASTLING_Q_BLACK = 3;
  constexpr pos_t NO_PIECE_INDICES = int(NO_PIECES)*int(NO_COLORS);
  constexpr pos_t CRAZYHOUSE_DROP = 1 << 7;
  constexpr pos_t NO_DROPPIECE_INDICES = (int(NO_PIECES) - 1)*int(NO_COLORS);
  constexpr pos_t DROP_PAWN = pos_t(PAWN) | CRAZYHOUSE_DROP,
                  DROP_KNIGHT = pos_t(KNIGHT) | CRAZYHOUSE_DROP,
                  DROP_BISHOP = pos_t(BISHOP) | CRAZYHOUSE_DROP,
                  DROP_ROOK = pos_t(ROOK) | CRAZYHOUSE_DROP,
                  DROP_QUEEN = pos_t(QUEEN) | CRAZYHOUSE_DROP;

  // j has 8 bits: 2 for promotion info, 6 for destination
  INLINE PIECE get_promotion_as(pos_t j) {
    switch(j & ~board::MOVEMASK) {
      case board::PROMOTE_KNIGHT:return KNIGHT;
      case board::PROMOTE_BISHOP:return BISHOP;
      case board::PROMOTE_ROOK:return ROOK;
      case board::PROMOTE_QUEEN:return QUEEN;
    }
    return PAWN;
  }

  // if a piece is dropped, CRAZYHOUSE_DROP indicates that it is a drop-move,
  // and 3 bits are used to indicate piece type. e.g. [1]0000[101]b
  INLINE PIECE get_drop_as(pos_t i) {
    return (PIECE)(i & ~board::CRAZYHOUSE_DROP);
  }

  INLINE constexpr pos_t _castling_index(COLOR c, CASTLING_SIDE side) {
    if(c==WHITE && side==KING_SIDE)return CASTLING_K_WHITE;
    if(c==WHITE && side==QUEEN_SIDE)return CASTLING_Q_WHITE;
    if(c==BLACK && side==KING_SIDE)return CASTLING_K_BLACK;
    if(c==BLACK && side==QUEEN_SIDE)return CASTLING_Q_BLACK;
    return 0xff;
  }

  ALWAYS_INLINE constexpr pos_t _x(pos_t i) {
    assert(i < SIZE);
    return i % LEN;
  }

  ALWAYS_INLINE constexpr pos_t _y(pos_t i) {
    assert(i < SIZE);
    return i / LEN;
  }

  ALWAYS_INLINE constexpr pos_t _pos(pos_t i, pos_t j) {
    return i + (j - 1) * LEN;
  }

  ALWAYS_INLINE constexpr pos_t file_mask(pos_t x) {
    return bitmask::vline << x;
  }

  ALWAYS_INLINE constexpr pos_t rank_mask(pos_t y) {
    return bitmask::hline << (y * board::LEN);
  }

  std::string _pos_str(pos_t i) {
    std::string p;
    i &= board::MOVEMASK;
    p += 'a' + board::_x(i);
    p += '1' + board::_y(i);
    return p;
  }

  std::string _move_str(move_t m, bool ispawn=false) {
    if(m == board::nullmove) {
      return "0000"s;
    }
    const pos_t i = bitmask::first(m),
                j = bitmask::second(m) & board::MOVEMASK;
    if(i & board::CRAZYHOUSE_DROP) {
      std::string ps = "";
      const PIECE p = board::get_drop_as(i);
      switch(p) {
        case PAWN:ps="P";break;
        case KNIGHT:ps="N";break;
        case BISHOP:ps="B";break;
        case ROOK:ps="R";break;
        case QUEEN:ps="Q";break;
        default:abort();
      }
      return ps + "@"s + _pos_str(j);
    }
    std::string sp;
    if(ispawn && (board::_y(j) == -1+1 || board::_y(j) == -1+8)) {
      switch(board::get_promotion_as(bitmask::second(m))) {
        case KNIGHT:sp='n';break;
        case BISHOP:sp='b';break;
        case ROOK:sp='r';break;
        case QUEEN:sp='q';break;
        default:break;
      }
    }
    return _pos_str(i) + _pos_str(j) + sp;
  }
} // namespace board


namespace piece {
  INLINE constexpr piece_bitboard_t pos_mask(pos_t k) {
    assert(k < 64);
    return 1ULL << k;
  }

  INLINE constexpr piece_bitboard_t rank_mask(pos_t r) {
    return bitmask::hline << (r * board::LEN);
  }

  INLINE constexpr piece_bitboard_t file_mask(pos_t f) {
    assert(f <= H);
    return bitmask::vline << f;
  }

  INLINE constexpr bool is_set(piece_bitboard_t mask, pos_t b) {
    return mask & pos_mask(b);
  }

  INLINE void set_pos(piece_bitboard_t &mask, pos_t b) {
//    assert(!is_set(mask, b));
    mask |= pos_mask(b);
  }

  INLINE void unset_pos(piece_bitboard_t &mask, pos_t b) {
//    assert(is_set(mask, b));
    mask &= ~pos_mask(b);
  }

  INLINE void move_pos(piece_bitboard_t &mask, pos_t i, pos_t j) {
    unset_pos(mask, i);
    set_pos(mask, j);
  }

  INLINE constexpr size_t size(piece_bitboard_t mask) {
    return bitmask::count_bits(mask);
  }


  INLINE piece_bitboard_t get_pawn_attack(pos_t pos, COLOR c) {
    return M42::pawn_attacks(c, pos);
  }

  INLINE piece_bitboard_t get_pawn_attacks(piece_bitboard_t mask, COLOR c) {
    constexpr piece_bitboard_t left = piece::file_mask(A);
    constexpr piece_bitboard_t right = piece::file_mask(H);
    constexpr piece_bitboard_t mid = ~(left | right);
    if(c == WHITE) {
      return ((mask & left ) << (board::LEN + 1))
           | ((mask & mid  ) << (board::LEN + 1))
           | ((mask & mid  ) << (board::LEN - 1))
           | ((mask & right) << (board::LEN - 1));
    } else {
      return ((mask & left ) >> (board::LEN - 1))
           | ((mask & mid  ) >> (board::LEN - 1))
           | ((mask & mid  ) >> (board::LEN + 1))
           | ((mask & right) >> (board::LEN + 1));
    }
  }

  INLINE piece_bitboard_t get_pawn_push_moves(COLOR c, pos_t i, piece_bitboard_t occupied) {
    if(c == WHITE) {
      const piece_bitboard_t pushes = (piece::pos_mask(i) << board::LEN) & ~occupied;
      return (pushes | ((pushes & piece::rank_mask(-1+3)) << board::LEN)) & ~occupied;
    } else {
      const piece_bitboard_t pushes = (piece::pos_mask(i) >> board::LEN) & ~occupied;
      return (pushes | ((pushes & piece::rank_mask(-1+6)) >> board::LEN)) & ~occupied;
    }
  }

  INLINE bool is_pawn_double_push(COLOR c, pos_t i, pos_t j) {
    return (c == WHITE ? j - i : i - j) == 2*board::LEN;
  }

  INLINE bool is_pawn_promotion_move(COLOR c, pos_t i, pos_t j) {
    return board::_y(j) == -1 + ((c == WHITE) ? 8 : 1);
  }

  INLINE pos_t get_pawn_enpassant_trace(COLOR c, pos_t i, pos_t j) {
    assert(is_pawn_double_push(c, i, j));
    return j + ((c == WHITE) ? -board::LEN : board::LEN);
  }


  INLINE piece_bitboard_t get_sliding_diag_attack(pos_t pos, piece_bitboard_t occupied) {
    return M42::bishop_attacks(pos, occupied);
  }

  INLINE piece_bitboard_t get_sliding_diag_xray_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_diag_attack(pos, friends|foes) & foes;
    return get_sliding_diag_attack(pos, friends | (foes ^ blockers));
  }

  INLINE piece_bitboard_t get_sliding_diag_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    assert(i < board::SIZE && j < board::SIZE);
    const piece_bitboard_t attacked_bit = piece::pos_mask(j);
    const pos_t x_i=board::_x(i), y_i=board::_y(i),
                x_j=board::_x(j), y_j=board::_y(j);
    const piece_bitboard_t diags = M42::diag_attacks(i, occupied),
                           adiags = M42::adiag_attacks(i, occupied);
    // shifting uint64_t by 64 is undefined behavior
    constexpr pos_t maxshift = 8 * sizeof(bitmask::full);
    const pos_t _shift_bottom = board::LEN * (board::LEN - y_i);
    const piece_bitboard_t bottomhalf = (_shift_bottom >= maxshift) ? 0x00 : (bitmask::full >> _shift_bottom);
    const pos_t _shift_top = (board::LEN * (y_i + 1));
    const piece_bitboard_t tophalf = (_shift_top >= maxshift) ? 0x00 : (bitmask::full << _shift_top);
    piece_bitboard_t r = 0x00;
    if(x_j < x_i && y_i < y_j) {
      // top-left
      r = adiags & tophalf;
    } else if(x_i < x_j && y_i < y_j) {
      // top-right
      r = diags & tophalf;
    } else if(x_j < x_i && y_j < y_i) {
      // bottom-left
      r = diags & bottomhalf;
    } else if(x_i < x_j && y_j < y_i) {
      // bottom-right
      r = adiags & bottomhalf;
    }
    return (r & attacked_bit) ? r : 0x00;
  }

  INLINE piece_bitboard_t get_sliding_diag_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_diag_attack(i, friends|foes) & foes;
    return get_sliding_diag_attacking_ray(i, j, friends | (foes ^ blockers));
  }

  INLINE piece_bitboard_t get_sliding_diag_attacks(piece_bitboard_t mask, piece_bitboard_t occupied) {
    piece_bitboard_t ret = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      ret |= get_sliding_diag_attack(pos, occupied);
    });
    return ret;
  }


  INLINE piece_bitboard_t get_sliding_orth_attack(pos_t pos, piece_bitboard_t occupied) {
    return M42::rook_attacks(pos, occupied);
  }

  INLINE piece_bitboard_t get_sliding_orth_xray_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_orth_attack(pos, friends|foes) & foes;
    return get_sliding_orth_attack(pos, friends | (foes ^ blockers));
  }

  INLINE piece_bitboard_t get_sliding_orth_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    const piece_bitboard_t attacked_bit = piece::pos_mask(j);
    const pos_t x_i=board::_x(i), y_i=board::_y(i),
                x_j=board::_x(j), y_j=board::_y(j);
    if(x_i != x_j && y_i != y_j)return 0x00;
    piece_bitboard_t r = 0x00;;
    if(y_i == y_j) {
      const piece_bitboard_t rankattacks = M42::rank_attacks(i, occupied);
      const piece_bitboard_t shift = board::LEN * y_i;
      if(x_j < x_i) {
        const piece_bitboard_t left = (bitmask::hline >> (board::LEN - x_i)) << shift;
        r = rankattacks & left;
      } else if(x_i < x_j) {
        const piece_bitboard_t right = ((bitmask::hline << (x_i + 1)) & bitmask::hline) << shift;
        r = rankattacks & right;
      }
    } else if(x_i == x_j) {
      const piece_bitboard_t fileattacks = M42::file_attacks(i, occupied);
      if(y_i < y_j) {
        const piece_bitboard_t tophalf = bitmask::full << (board::LEN * (y_i + 1));
        r = fileattacks & tophalf;
      } else if(y_j < y_i) {
        const piece_bitboard_t bottomhalf = bitmask::full >> (board::LEN * (board::LEN - y_i));
        r = fileattacks & bottomhalf;
      }
    }
    return (r & attacked_bit) ? r : 0x00;
  }

  INLINE piece_bitboard_t get_sliding_orth_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_orth_attack(i, friends|foes) & foes;
    return get_sliding_orth_attacking_ray(i, j, friends | (foes ^ blockers));
  }

  INLINE piece_bitboard_t get_sliding_orth_attacks(piece_bitboard_t mask, piece_bitboard_t occupied) {
    piece_bitboard_t ret = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      ret |= get_sliding_orth_attack(pos, occupied);
    });
    return ret;
  }


  INLINE piece_bitboard_t get_knight_attack(pos_t pos) {
    return M42::knight_attacks(pos);
  }

  INLINE piece_bitboard_t get_knight_attacks(piece_bitboard_t mask) {
    return M42::calc_knight_attacks(mask);
  }


  INLINE piece_bitboard_t get_king_attack(pos_t pos) {
    return M42::king_attacks(pos);
  }

  INLINE piece_bitboard_t get_king_castling_moves(COLOR c, pos_t kingpos, piece_bitboard_t occupied, piece_bitboard_t attack_mask,
                                                  bool queen_side, bool king_side, pos_t qcastlrook, pos_t kcastlrook, bool chess960)
  {
    assert(qcastlrook != 0xff || !queen_side);
    assert(kcastlrook != 0xff || !king_side);
    piece_bitboard_t castlemoves = 0x00;
    if((queen_side || king_side) && !piece::is_set(attack_mask, kingpos)) {
      // can't castle when checked
      const pos_t castlrank = (c == WHITE) ? 1 : 8;
      const pos_t castleleft = board::_pos(C, castlrank);
      const pos_t castleright = board::_pos(G, castlrank);
      if(queen_side) {
        const pos_t qrook = board::_pos(qcastlrook, castlrank);
        const piece_bitboard_t castleleftcheck = bitmask::ones_between_eq_symm(castleleft, kingpos);
        const piece_bitboard_t castleleftcheckocc = castleleftcheck | bitmask::ones_between_eq_symm(qrook, board::_pos(D, castlrank));
        const piece_bitboard_t exclude = piece::pos_mask(kingpos) | piece::pos_mask(qrook);
        if(!(attack_mask & castleleftcheck) && !(occupied & ~exclude & castleleftcheckocc)) {
          if(!chess960) {
            castlemoves |= piece::pos_mask(castleleft);
          } else {
            castlemoves |= piece::pos_mask(qrook);
          }
        }
      }
      if(king_side) {
        const pos_t krook = board::_pos(kcastlrook, castlrank);
        const piece_bitboard_t castlerightcheck = bitmask::ones_between_eq_symm(kingpos, castleright);
        const piece_bitboard_t castlerightcheckocc = castlerightcheck | bitmask::ones_between_eq_symm(board::_pos(F, castlrank), krook);
        const piece_bitboard_t exclude = piece::pos_mask(kingpos) | piece::pos_mask(krook);
        if(!(attack_mask & castlerightcheck) && !(occupied & ~exclude & castlerightcheckocc)) {
          if(!chess960) {
            castlemoves |= piece::pos_mask(castleright);
          } else {
            castlemoves |= piece::pos_mask(krook);
          }
        }
      }
    }
    return castlemoves;
  }

  INLINE pos_pair_t get_king_castle_rook_move(COLOR c, pos_t i, pos_t k_j, pos_t qcastlrook, pos_t kcastlrook) {
    const pos_t castlrank = (c == WHITE) ? 1 : 8;
    if(k_j == board::_pos(C, castlrank))return bitmask::_pos_pair(board::_pos(qcastlrook, castlrank), board::_pos(D, castlrank));
    if(k_j == board::_pos(G, castlrank))return bitmask::_pos_pair(board::_pos(kcastlrook, castlrank), board::_pos(F, castlrank));
    abort();
    return board::nullmove;
  }

  void print(piece_bitboard_t mask) {
    std::cout << piece::size(mask) << std::endl;
    bitmask::print(mask);
  }
} // namespace piece

namespace board {
  constexpr piece_bitboard_t PAWN_RANKS =
          piece::rank_mask(1) | piece::rank_mask(2) | piece::rank_mask(3)
        | piece::rank_mask(4) | piece::rank_mask(5) | piece::rank_mask(6);
} // namespace board


// interface to piece bitboard
struct Piece {
  PIECE value;
  COLOR color;
  pos_t piece_index;

  static INLINE constexpr pos_t get_piece_index(PIECE p, COLOR c) {
    return (p==EMPTY) ? int(NO_PIECES)*int(NO_COLORS) : int(p)*int(NO_COLORS)+c;
  }

  constexpr INLINE Piece(PIECE p, COLOR c):
    value(p), color(c),
    piece_index(get_piece_index(p, c))
  {}

  INLINE constexpr char str() const {
    char c = '*';
    switch(value) {
      case EMPTY: return c;
      case PAWN: c = 'p'; break;
      case KNIGHT: c = 'n'; break;
      case BISHOP: c = 'b'; break;
      case ROOK: c = 'r'; break;
      case QUEEN: c = 'q'; break;
      case KING: c = 'k'; break;
    }
    return (color == WHITE) ? toupper(c) : c;
  }
};
