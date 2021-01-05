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
    const MPIECE mp = get_mpiece_value(value, color);
    if(mp==WPAWNM) return Attacks<WPAWNM>::get_attacks(pos,friends,foes);
    if(mp==BPAWNM) return Attacks<BPAWNM>::get_attacks(pos,friends,foes);
    if(mp==KNIGHTM)return Attacks<KNIGHTM>::get_attacks(pos,friends,foes);
    if(mp==BISHOPM)return Attacks<BISHOPM>::get_attacks(pos,friends,foes);
    if(mp==ROOKM)  return Attacks<ROOKM>::get_attacks(pos,friends,foes);
    if(mp==QUEENM) return Attacks<QUEENM>::get_attacks(pos,friends,foes);
    if(mp==KINGM)  return Attacks<KINGM>::get_attacks(pos,friends,foes);
    return 0x00ULL;
  }

  inline constexpr piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) const {
    const MPIECE mp = get_mpiece_value(value, color);
    if(mp==BISHOPM)return Attacks<BISHOPM>::get_attacking_ray(i,j,occupied);
    if(mp==ROOKM)  return Attacks<ROOKM>::get_attacking_ray(i,j,occupied);
    if(mp==QUEENM) return Attacks<QUEENM>::get_attacking_ray(i,j,occupied);
    return 0x00ULL;
  }

  inline constexpr piece_bitboard_t get_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) const {
    const MPIECE mp = get_mpiece_value(value, color);
    if(mp==BISHOPM)return xRayAttacks<BISHOPM>::get_attacking_xray(i,j,friends,foes);
    if(mp==ROOKM)  return xRayAttacks<ROOKM>::get_attacking_xray(i,j,friends,foes);
    if(mp==QUEENM) return xRayAttacks<QUEENM>::get_attacking_xray(i,j,friends,foes);
    return 0x00ULL;
  }

  // attack from specific position by this type of piece
  inline constexpr piece_bitboard_t get_moves(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes,
                                              piece_bitboard_t attack_mask=0x00, piece_bitboard_t castlings=0x00,
                                              pos_t enpassant=event::enpassantnotrace) const
  {
    const MPIECE mp = get_mpiece_value(value, color);
    if(mp==WPAWNM) return Moves<WPAWNM>::get_moves(pos,friends,foes,enpassant);
    if(mp==BPAWNM) return Moves<BPAWNM>::get_moves(pos,friends,foes,enpassant);
    if(mp==KNIGHTM)return Moves<KNIGHTM>::get_moves(pos,friends,foes);
    if(mp==BISHOPM)return Moves<BISHOPM>::get_moves(pos,friends,foes);
    if(mp==ROOKM)  return Moves<ROOKM>::get_moves(pos,friends,foes);
    if(mp==QUEENM) return Moves<QUEENM>::get_moves(pos,friends,foes);
    if(mp==KINGM&&color==WHITE)return Moves<KINGM>::get_moves<WHITE>(pos,friends,foes,attack_mask,castlings);
    if(mp==KINGM&&color==BLACK)return Moves<KINGM>::get_moves<BLACK>(pos,friends,foes,attack_mask,castlings);
    return 0x00ULL;
  }

  // multi-attacks
  inline constexpr piece_bitboard_t get_attacks(piece_bitboard_t friends, piece_bitboard_t foes) const {
    const MPIECE mp = get_mpiece_value(value, color);
    if(mp==WPAWNM) return MultiAttacks<WPAWNM>::get_attacks(mask,friends,foes);
    if(mp==BPAWNM) return MultiAttacks<BPAWNM>::get_attacks(mask,friends,foes);
    if(mp==KNIGHTM)return MultiAttacks<KNIGHTM>::get_attacks(mask,friends,foes);
    if(mp==BISHOPM)return MultiAttacks<BISHOPM>::get_attacks(mask,friends,foes);
    if(mp==ROOKM)  return MultiAttacks<ROOKM>::get_attacks(mask,friends,foes);
    if(mp==QUEENM) return MultiAttacks<QUEENM>::get_attacks(mask,friends,foes);
    if(mp==KINGM)  return MultiAttacks<KINGM>::get_attacks(mask,friends,foes);
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
