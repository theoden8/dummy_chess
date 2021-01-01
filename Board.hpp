#pragma once

#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <climits>

#include <array>
#include <iostream>
#include <functional>

// pieces
typedef enum { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING } PIECE;

// type alias: bitboard (as a mask)
typedef uint64_t piece_bitboard_t;

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
  template <typename T>
  inline constexpr bool is_exp2(T v) {
    return !(v & (v - 1));
  }

  template <typename T>
  inline constexpr pos_t log2_of_exp2(T v) {
    pos_t r = 0x00;
         if (v >= (UINT64_C(1) << 32)) { r = 32; v >>= 32; }
    else if (v >= (UINT64_C(1) << 16)) { r = 16; v >>= 16; }
    else if (v >= (UINT64_C(1) << 8 )) { r = 8 ; v >>= 8 ; }
    else if (v >= (UINT64_C(1) << 4 )) { r = 4 ; v >>= 4 ; }
    else if (v >= (UINT64_C(1) << 2 )) { r = 2 ; v >>= 2 ; }
    else if (v >= (UINT64_C(1) << 1 )) { r = 1 ; v >>= 1 ; }
    return r;
  }

  template <typename T>
  inline constexpr pos_t count_bits(T v) {
    if(!v)return 0;
    if(bitmask::is_exp2(v))return 1;
    v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
    v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
    return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; // count
  }

  template <typename T>
  constexpr pos_t log2(T t) {
    pos_t shift = 0;
    pos_t r = 0;
    r =     (t > 0xFFFFFFFFLLU)?1<<5:0; t >>= r;
    shift = (t > 0xFFFF)       ?1<<4:0; t >>= shift; r |= shift;
    shift = (t > 0xFF)         ?1<<3:0; t >>= shift; r |= shift;
    shift = (t > 0xF)          ?1<<2:0; t >>= shift; r |= shift;
    shift = (t > 0x3)          ?1<<1:0; t >>= shift; r |= shift;
                                                     r |= (t >> 1);
    return r;
  }

  // iterate set bits with a function F
  template <typename F>
  constexpr void foreach(piece_bitboard_t mask, F &&func) {
    if(!mask)return;
    if(is_exp2(mask)) {
      func(bitmask::log2_of_exp2(mask));
      return;
    }
    piece_bitboard_t x = mask;
    while(x) {
      pos_t r = bitmask::log2(x);
      func(r);
      // unset msb
      x &= ~(1LLU << r);
    }
  }

  // print locations of each set bit
  void print(piece_bitboard_t mask) {
    /* static auto pr = [](pos_t p) { std::cout << p << std::endl; }; */
    foreach(mask, [](pos_t p) mutable { printf("(%c, %d)\n", 'A' + p / 8, p % 8 + 1); });
  }

  void print_mask(piece_bitboard_t mask, int markspot=-1) {
    char s[256];
    int j = 0;
    for(int i = 0; i < 64; ++i) {
      if(i == markspot) {
        s[j++] = 'x';
      } else {
        s[j++] = (mask & (1LLU << i)) ? '*' : '.';
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


// piece-view of the game
struct Piece {
  const PIECE value;
  const COLOR color;
  piece_bitboard_t mask;
  event last_event;

  constexpr Piece(PIECE p, COLOR c, piece_bitboard_t loc = 0x00):
    value(p), color(c), mask(loc), last_event()
  {}

  constexpr bool is_set(pos_t i) const {
    return mask & (1LLU << i);
  }

  constexpr bool empty() const {
    return value == EMPTY;
  }

  constexpr void set_event(pos_t i, EVENT e = NOEVENT) {
    if(e != NOEVENT) {
      last_event.type = e;
      last_event.position = i;
    }
  }

  constexpr void set_pos(pos_t i) {
    assert(!is_set(i));
    mask |= 1LLU << i;
  }

  constexpr void unset_pos(pos_t i) {
    assert(is_set(i));
    mask &= ~(1LLU << i);
  }

  constexpr void move(pos_t i, pos_t j) {
    this->unset_pos(i);
    this->set_pos(j);
  }

  constexpr pos_t size() const {
    return bitmask::count_bits(mask);
  }


  void foreach(std::function<void(pos_t)>&&func) {
    bitmask::foreach(mask, func);
  }

  void foreach(std::function<void(pos_t)>&&func) const {
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
};

struct ChangeEvent {
  pos_t from, to;
  event ev_from, ev_to;
  constexpr ChangeEvent(pos_t from, pos_t to, event ev_from, event ev_to):
    from(from), to(to), ev_from(ev_from), ev_to(ev_to)
  {
    if(ev_from.type == KILL)
      assert(ev_to.type == DEATH);
  }
};

enum {
  A,B,C,D,E,F,G,H
} ENUM_ROW;


// board view of the game
class Board {
public:
  static constexpr const pos_t LENGTH = 8;
  static constexpr const pos_t SIZE = 64;
private:
  Board &self = *this;
  std::array <Piece *, SIZE> board_;
public:
  std::array<Piece, 2*6+1>  pieces = {
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
  Board()
  {
    for(pos_t i = 0; i < SIZE; ++i) {
      self.board_[i] = &self.get_piece(EMPTY);
    }
  }

  inline Piece &operator[](pos_t i) {
    return *board_[i];
  }

  inline const Piece &operator[](pos_t i) const {
    return *board_[i];
  }

  static constexpr pos_t _x(pos_t i) {
    return i % LENGTH;
  }

  static constexpr pos_t _y(pos_t i) {
    return i / LENGTH;
  }

  static constexpr pos_t _pos(pos_t i, pos_t j) {
    return i + (j - 1) * LENGTH;
  }

  const Piece &get_piece(PIECE P = EMPTY, COLOR C = NEUTRAL) const {
    if(P == EMPTY) {
      return pieces[13-1];
    }
    return pieces[(P - PAWN) * 2 + C - WHITE];
  }

  constexpr Piece &get_piece(PIECE P = EMPTY, COLOR C = NEUTRAL) {
    if(P == EMPTY) {
      return pieces[13-1];
    }
    return pieces[(P - PAWN) * 2 + C - WHITE];
  }

  const Piece get_piece(const Piece &p) const {
    return get_piece(p.value, p.color);
  }

  constexpr Piece get_piece(Piece &p) {
    return get_piece(p.value, p.color);
  }

  void unset_pos(pos_t i) {
    self[i].unset_pos(i);
    set_pos(i, self.get_piece(EMPTY));
  }

  void set_pos(pos_t i, Piece &p) {
    p.set_pos(i);
    self.board_[i] = &p;
  }

  Piece &put_pos(pos_t i, Piece &p) {
    Piece &target = self[i];
    if(!target.empty()) {
      p.set_event(KILL);
      target.set_event(DEATH);
    }
    set_pos(i, p);
    return target;
  }

  ChangeEvent move(pos_t i, pos_t j) {
    assert(!self[i].empty());
    event ev = put_pos(j, self[i]).last_event;
    self.unset_pos(i);
    return ChangeEvent(i, j, self[j].last_event, ev);
  }

  piece_bitboard_t get_piece_positions(COLOR c) const {
    piece_bitboard_t mask = UINT64_C(0);
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})mask |= self.get_piece(p, c).mask;
    return mask;
  }

  void print() {
    for(pos_t i = LENGTH; i > 0; --i) {
      for(pos_t j = 0; j < LENGTH; ++j) {
        Piece &p = self[(i-1) * LENGTH + j];
        std::cout << p.str() << " ";
      }
      std::cout << std::endl;
    }
  }
};
