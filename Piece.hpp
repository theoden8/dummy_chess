#pragma once


#include <iostream>

#include <Constants.hpp>
#include <Event.hpp>
#include <Bitboard.hpp>
#include <Attacks.hpp>


// interface to piece bitboard
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

  // attack from specific position by this type of piece
  inline constexpr piece_bitboard_t get_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) const {
    if(value==PAWN  &&color==WHITE)return Attacks<PAWN  ,WHITE>::get_attacks(pos,friends,foes);
    if(value==KNIGHT&&color==WHITE)return Attacks<KNIGHT,WHITE>::get_attacks(pos,friends,foes);
    if(value==BISHOP&&color==WHITE)return Attacks<BISHOP,WHITE>::get_attacks(pos,friends,foes);
    if(value==ROOK  &&color==WHITE)return Attacks<ROOK  ,WHITE>::get_attacks(pos,friends,foes);
    if(value==QUEEN &&color==WHITE)return Attacks<QUEEN ,WHITE>::get_attacks(pos,friends,foes);
    if(value==KING  &&color==WHITE)return Attacks<KING  ,WHITE>::get_attacks(pos,friends,foes);
    if(value==PAWN  &&color==BLACK)return Attacks<PAWN  ,BLACK>::get_attacks(pos,friends,foes);
    if(value==KNIGHT&&color==BLACK)return Attacks<KNIGHT,BLACK>::get_attacks(pos,friends,foes);
    if(value==BISHOP&&color==BLACK)return Attacks<BISHOP,BLACK>::get_attacks(pos,friends,foes);
    if(value==ROOK  &&color==BLACK)return Attacks<ROOK  ,BLACK>::get_attacks(pos,friends,foes);
    if(value==QUEEN &&color==BLACK)return Attacks<QUEEN ,BLACK>::get_attacks(pos,friends,foes);
    if(value==KING  &&color==BLACK)return Attacks<KING  ,BLACK>::get_attacks(pos,friends,foes);
    return 0x00ULL;
  }

  // multi-attack
  inline constexpr piece_bitboard_t get_attacks(piece_bitboard_t friends, piece_bitboard_t foes) const {
    if(value==PAWN  &&color==WHITE)return MultiAttacks<PAWN  ,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==KNIGHT&&color==WHITE)return MultiAttacks<KNIGHT,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==BISHOP&&color==WHITE)return MultiAttacks<BISHOP,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==ROOK  &&color==WHITE)return MultiAttacks<ROOK  ,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==QUEEN &&color==WHITE)return MultiAttacks<QUEEN ,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==KING  &&color==WHITE)return MultiAttacks<KING  ,WHITE>::get_attacks(this->mask,friends,foes);
    if(value==PAWN  &&color==BLACK)return MultiAttacks<PAWN  ,BLACK>::get_attacks(this->mask,friends,foes);
    if(value==KNIGHT&&color==BLACK)return MultiAttacks<KNIGHT,BLACK>::get_attacks(this->mask,friends,foes);
    if(value==BISHOP&&color==BLACK)return MultiAttacks<BISHOP,BLACK>::get_attacks(this->mask,friends,foes);
    if(value==ROOK  &&color==BLACK)return MultiAttacks<ROOK  ,BLACK>::get_attacks(this->mask,friends,foes);
    if(value==QUEEN &&color==BLACK)return MultiAttacks<QUEEN ,BLACK>::get_attacks(this->mask,friends,foes);
    if(value==KING  &&color==BLACK)return MultiAttacks<KING  ,BLACK>::get_attacks(this->mask,friends,foes);
    return 0x00ULL;
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
