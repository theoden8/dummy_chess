#pragma once


#include <Bitboard.hpp>


// pieces
typedef enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, EMPTY, NO_PIECES=EMPTY } PIECE;

// two bytes, one for each move; second byte may contain information about promotion (2 bits)
typedef pos_pair_t move_t;

// type alias: bitboard (as a mask)
typedef uint64_t piece_bitboard_t;

// player color, neutral when neither/both
typedef enum { WHITE, BLACK, NEUTRAL, NO_COLORS=NEUTRAL } COLOR;

inline constexpr COLOR enemy_of(COLOR c) {
  if(c == NEUTRAL)return NEUTRAL;
  return (c == WHITE) ? BLACK : WHITE;
}


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

  inline constexpr pos_t _x(pos_t i) {
    return i % LEN;
  }

  inline constexpr pos_t _y(pos_t i) {
    return i / LEN;
  }

  inline constexpr pos_t _pos(pos_t i, pos_t j) {
    return i + (j - 1) * LEN;
  }
} // namespace board
