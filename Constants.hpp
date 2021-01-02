#pragma once


#include <Bitboard.hpp>


// pieces
typedef enum { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING } PIECE;

// type alias: bitboard (as a mask)
typedef uint64_t piece_bitboard_t;

// player color, neutral when neither/both
typedef enum { NEUTRAL = 0, WHITE, BLACK } COLOR;

COLOR enemy_of(COLOR c) {
  if(c == NEUTRAL)return NEUTRAL;
  return (c == WHITE) ? BLACK : WHITE;
}


enum {
  A,B,C,D,E,F,G,H
} ENUM_ROW;

namespace board {
  constexpr pos_t LEN = 8;
  constexpr pos_t SIZE = LEN*LEN;

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
