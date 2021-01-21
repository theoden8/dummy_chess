#pragma once


#include <string>

#include <Bitboard.hpp>


// pieces
typedef enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, EMPTY, NO_PIECES=EMPTY } PIECE;

// two bytes, one for each move; second byte may contain information about promotion (2 bits)
typedef pos_pair_t move_t;
typedef uint16_t move_index_t;

// type alias: bitboard (as a mask)
typedef uint64_t piece_bitboard_t;

// player color, neutral when neither/both
typedef enum { WHITE, BLACK, NEUTRAL, NO_COLORS=NEUTRAL, BOTH } COLOR;

inline constexpr COLOR enemy_of(COLOR c) {
  if(c == NEUTRAL)return NEUTRAL;
  return (c == WHITE) ? BLACK : WHITE;
}

typedef enum { KING_SIDE, QUEEN_SIDE } CASTLING_SIDE;

enum {
  A,B,C,D,E,F,G,H
} ENUM_ROW;

namespace board {
  constexpr pos_t LEN = 8;
  constexpr pos_t SIZE = LEN*LEN;
  constexpr pos_t MOVEMASK = 0x3f;
  constexpr pos_t PROMOTE_KNIGHT = 0<<6,
                  PROMOTE_BISHOP = 1<<6,
                  PROMOTE_ROOK = 2<<6,
                  PROMOTE_QUEEN = 3<<6;
  constexpr move_t nomove = bitmask::_pos_pair(0xff, 0xff);
  constexpr move_index_t nocastlings = ~move_index_t(0);
  constexpr pos_t CASTLING_K_WHITE = 0,
                  CASTLING_Q_WHITE = 1,
                  CASTLING_K_BLACK = 2,
                  CASTLING_Q_BLACK = 3;
  constexpr pos_t NO_PIECE_INDICES = int(NO_PIECES)*int(NO_COLORS) + 1;

  INLINE PIECE get_promotion_as(pos_t j) {
    switch(j & ~board::MOVEMASK) {
      case board::PROMOTE_KNIGHT:return KNIGHT;
      case board::PROMOTE_BISHOP:return BISHOP;
      case board::PROMOTE_ROOK:return ROOK;
      case board::PROMOTE_QUEEN:return QUEEN;
    }
    return PAWN;
  }

  INLINE constexpr pos_t _castling_index(COLOR c, CASTLING_SIDE side) {
    if (c==WHITE && side==KING_SIDE)return CASTLING_K_WHITE;
    if (c==WHITE && side==QUEEN_SIDE)return CASTLING_Q_WHITE;
    if (c==BLACK && side==KING_SIDE)return CASTLING_K_BLACK;
    if (c==BLACK && side==QUEEN_SIDE)return CASTLING_Q_BLACK;
    return 0xff;
  }

  ALWAYS_INLINE constexpr pos_t _x(pos_t i) {
    return i % LEN;
  }

  ALWAYS_INLINE constexpr pos_t _y(pos_t i) {
    return i / LEN;
  }

  ALWAYS_INLINE constexpr pos_t _pos(pos_t i, pos_t j) {
    return i + (j - 1) * LEN;
  }

  std::string _pos_str(pos_t i) {
    std::string p;
    i &= board::MOVEMASK;
    p += 'a' + board::_x(i);
    p += '1' + board::_y(i);
    return p;
  }

  std::string _move_str(move_t m, bool ispawn=false) {
    const pos_t i = bitmask::first(m) & board::MOVEMASK,
                j = bitmask::second(m) & board::MOVEMASK;
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
