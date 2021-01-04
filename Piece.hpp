#pragma once


#include <iostream>

#include <Constants.hpp>
#include <Bitboard.hpp>
#include <Attacks.hpp>


// interface to piece bitboard
struct Piece {
  const PIECE value;
  const COLOR color;
  const pos_t piece_index;
  piece_bitboard_t mask;

  constexpr Piece(PIECE p, COLOR c, piece_bitboard_t loc=0x00):
    value(p), color(c),
    piece_index((p==EMPTY) ? int(NO_PIECES)*int(NO_COLORS) : int(p)*(int)NO_COLORS+c),
    mask(loc)
  {}

  constexpr bool is_set(pos_t i) const {
    return mask & (1LLU << i);
  }

  constexpr bool is_empty() const {
    return value == EMPTY;
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

  // xray-attack from specific position by this type of piece
  inline constexpr piece_bitboard_t get_xray_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) const {
    if(value==PAWN  &&color==WHITE)return xRayAttacks<PAWN  ,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==KNIGHT&&color==WHITE)return xRayAttacks<KNIGHT,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==BISHOP&&color==WHITE)return xRayAttacks<BISHOP,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==ROOK  &&color==WHITE)return xRayAttacks<ROOK  ,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==QUEEN &&color==WHITE)return xRayAttacks<QUEEN ,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==KING  &&color==WHITE)return xRayAttacks<KING  ,WHITE>::get_xray_attacks(pos,friends,foes);
    if(value==PAWN  &&color==BLACK)return xRayAttacks<PAWN  ,BLACK>::get_xray_attacks(pos,friends,foes);
    if(value==KNIGHT&&color==BLACK)return xRayAttacks<KNIGHT,BLACK>::get_xray_attacks(pos,friends,foes);
    if(value==BISHOP&&color==BLACK)return xRayAttacks<BISHOP,BLACK>::get_xray_attacks(pos,friends,foes);
    if(value==ROOK  &&color==BLACK)return xRayAttacks<ROOK  ,BLACK>::get_xray_attacks(pos,friends,foes);
    if(value==QUEEN &&color==BLACK)return xRayAttacks<QUEEN ,BLACK>::get_xray_attacks(pos,friends,foes);
    if(value==KING  &&color==BLACK)return xRayAttacks<KING  ,BLACK>::get_xray_attacks(pos,friends,foes);
    return 0x00ULL;
  }

  // attack from specific position by this type of piece
  inline constexpr piece_bitboard_t get_moves(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes,
                                              piece_bitboard_t attack_mask=0x00, piece_bitboard_t castlings=0x00,
                                              pos_t enpassant=event::enpassantnotrace) const
  {
    if(value==PAWN  &&color==WHITE)return Moves<PAWN  ,WHITE>::get_moves(pos,friends,foes,enpassant);
    if(value==KNIGHT&&color==WHITE)return Moves<KNIGHT,WHITE>::get_moves(pos,friends,foes);
    if(value==BISHOP&&color==WHITE)return Moves<BISHOP,WHITE>::get_moves(pos,friends,foes);
    if(value==ROOK  &&color==WHITE)return Moves<ROOK  ,WHITE>::get_moves(pos,friends,foes);
    if(value==QUEEN &&color==WHITE)return Moves<QUEEN ,WHITE>::get_moves(pos,friends,foes);
    if(value==KING  &&color==WHITE)return Moves<KING  ,WHITE>::get_moves(pos,friends,foes,attack_mask,castlings);
    if(value==PAWN  &&color==BLACK)return Moves<PAWN  ,BLACK>::get_moves(pos,friends,foes,enpassant);
    if(value==KNIGHT&&color==BLACK)return Moves<KNIGHT,BLACK>::get_moves(pos,friends,foes);
    if(value==BISHOP&&color==BLACK)return Moves<BISHOP,BLACK>::get_moves(pos,friends,foes);
    if(value==ROOK  &&color==BLACK)return Moves<ROOK  ,BLACK>::get_moves(pos,friends,foes);
    if(value==QUEEN &&color==BLACK)return Moves<QUEEN ,BLACK>::get_moves(pos,friends,foes);
    if(value==KING  &&color==BLACK)return Moves<KING  ,BLACK>::get_moves(pos,friends,foes,attack_mask,castlings);
    return 0x00ULL;
  }

  // multi-attacks
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

  // multi xray-attacks
  inline constexpr piece_bitboard_t get_xray_attacks(piece_bitboard_t friends, piece_bitboard_t foes) const {
    if(value==PAWN  &&color==WHITE)return MultixRayAttacks<PAWN  ,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==KNIGHT&&color==WHITE)return MultixRayAttacks<KNIGHT,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==BISHOP&&color==WHITE)return MultixRayAttacks<BISHOP,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==ROOK  &&color==WHITE)return MultixRayAttacks<ROOK  ,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==QUEEN &&color==WHITE)return MultixRayAttacks<QUEEN ,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==KING  &&color==WHITE)return MultixRayAttacks<KING  ,WHITE>::get_xray_attacks(this->mask,friends,foes);
    if(value==PAWN  &&color==BLACK)return MultixRayAttacks<PAWN  ,BLACK>::get_xray_attacks(this->mask,friends,foes);
    if(value==KNIGHT&&color==BLACK)return MultixRayAttacks<KNIGHT,BLACK>::get_xray_attacks(this->mask,friends,foes);
    if(value==BISHOP&&color==BLACK)return MultixRayAttacks<BISHOP,BLACK>::get_xray_attacks(this->mask,friends,foes);
    if(value==ROOK  &&color==BLACK)return MultixRayAttacks<ROOK  ,BLACK>::get_xray_attacks(this->mask,friends,foes);
    if(value==QUEEN &&color==BLACK)return MultixRayAttacks<QUEEN ,BLACK>::get_xray_attacks(this->mask,friends,foes);
    if(value==KING  &&color==BLACK)return MultixRayAttacks<KING  ,BLACK>::get_xray_attacks(this->mask,friends,foes);
    return 0x00ULL;
  }

  inline void foreach(std::function<void(pos_t)>&&func) {
    bitmask::foreach(mask, func);
  }

  inline void foreach(std::function<void(pos_t)>&&func) const {
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
    if(color == WHITE)c = toupper(c);
    return c;
  }
};
