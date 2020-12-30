#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <climits>

#include <iostream>
#include <functional>

// pieces
typedef enum { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING } PIECE;

// type alias: bitboard (as a mask)
typedef uint64_t piece_loc_t;

// player color, neutral when neither/both
typedef enum { NEUTRAL = 0, WHITE, BLACK } COLOR;

// type alias: position index on bitboard
typedef uint8_t pos_t;

// event sequance that can be reverted
typedef enum { NOEVENT, DEATH, KILL, SPAWN } EVENT;
struct event {
  EVENT type;
  pos_t position;
  constexpr event():
    type(NOEVENT), position(UCHAR_MAX)
  {}
};

// https://graphics.stanford.edu/~seander/bithacks.html
// bit-hacks from stanford graphics, to be placed here
namespace bitmask {
  // get the number of set bits
  constexpr pos_t size(piece_loc_t mask) {
    if(!mask)
      return 0;
    if((mask & (mask - 1)) == 0)
      return 1;
    using i64 = piece_loc_t;
    piece_loc_t x = mask;
    x = x - ((x >> 1) & (i64)~(i64)0/3);
    x = (x & (i64)~(i64)0/15*3) + ((x >> 2) & (i64)~(i64)0/15*3);
    x = (x + (x >> 4)) & (i64)~(i64)0/255*15;
    return (i64)(x * ((i64)~(i64)0/255)) >> (sizeof(i64) - 1) * CHAR_BIT;
  }

  // iterate set bits with a function F
  template <typename F>
  constexpr void foreach(piece_loc_t mask, F &&func) {
    if(!mask)
      return;
    piece_loc_t x = mask;
    if((mask & (mask - 1)) == 0) {
      #define S(k) if (x >= (UINT64_C(1) << k)) { r += k; x >>= k; }
      pos_t r = 0x00; S(32); S(16); S(8); S(4); S(2); S(1);
      func(r);
      return;
      #undef S
    }
    while(x) {
      piece_loc_t t = x;
      pos_t shift = 0;
      pos_t r = 0x00;
      r = (t > 0xFFFFFFFF) << 5; t >>= r;
      shift = (t > 0xFFFF) << 4; t >>= shift; r |= shift;
      shift = (t > 0xFF) << 3; t >>= shift; r |= shift;
      shift = (t > 0xF) << 2; t >>= shift; r |= shift;
      shift = (t > 0x3) << 1; t >>= shift; r |= shift;
      r |= (t >> 1);
      func(r);
      x &= ~(1 << r);
    }
  }

  // print locations of each set bit
  void print(piece_loc_t mask) {
    /* static auto pr = [](pos_t p) { std::cout << p << std::endl; }; */
    static const auto pr = [](pos_t p) { printf("(%c, %d)\n", 'A' + p / 8, p % 8 + 1); };
    foreach(mask, pr);
  }

  void print_mask(piece_loc_t mask, int markspot=-1) {
    char s[256];
    int j = 0;
    piece_loc_t I = 1;
    for(int i = 0; i < 64; ++i) {
      if(i == markspot) {
        s[j++] = 'x';
      } else {
        s[j++] = (mask & (I << i)) ? '*' : '.';
      }
      s[j++] = ' ';
      if(i % 8 == 7) {
        s[j++] = '\n';
      }
    }
    s[j] = '\0';
    puts(s);
  }
} // namespace bitmask

// generic piece wrapper for higher order code
struct Piece {
  const PIECE value;
  const COLOR color;
  piece_loc_t mask;
  event last_event;

  constexpr Piece(PIECE p, COLOR c, piece_loc_t loc = 0x00):
    value(p), color(c), mask(loc), last_event()
  {}

  constexpr bool is_set(pos_t i) {
    return mask & (1 << i);
  }

  constexpr bool is_empty() {
    return value == EMPTY;
  }

  constexpr void set_event(pos_t i, EVENT e = NOEVENT) {
    if(e != NOEVENT) {
      last_event.type = e;
      last_event.position = i;
    }
  }

  constexpr void set(pos_t i) {
    assert(!is_set(i));
    mask |= 1 << i;
  }

  constexpr void unset(pos_t i) {
    assert(is_set(i));
    mask &= ~(1 << i);
  }

  constexpr void move(pos_t i, pos_t j) {
    unset(i);
    set(j);
  }

  constexpr pos_t size() const {
    return bitmask::size(mask);
  }


  constexpr void foreach(std::function<void(pos_t)>&&func) {
    bitmask::foreach(mask, func);
  }

  void print() {
    std::cout << int(size()) << std::endl;
    bitmask::print(mask);
  }

  constexpr char str() const {
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
    if(color == WHITE)
      c = toupper(c);
    return c;
  }

  static constexpr Piece &get(PIECE P = EMPTY, COLOR C = NEUTRAL) {
    if(P == EMPTY) {
      return pieces[13-1];
    }
    return pieces[(P - PAWN) * 2 + C - WHITE];
  }

  static constexpr Piece get(Piece &p) {
    return get(p.value, p.color);
  }

  static Piece pieces[];
};

// shouldn't be here. We want to load them from
// FEN file in the future when initializing board
Piece Piece::pieces[] = {
  Piece(PAWN, WHITE),
  Piece(PAWN, BLACK),
  Piece(KNIGHT, WHITE),
  Piece(KNIGHT, BLACK),
  Piece(BISHOP, WHITE),
  Piece(BISHOP, BLACK),
  Piece(ROOK, WHITE),
  Piece(ROOK, BLACK),
  Piece(QUEEN, WHITE),
  Piece(QUEEN, BLACK),
  Piece(KING, WHITE),
  Piece(KING, BLACK),
  Piece(EMPTY, NEUTRAL, 0xff)
};
