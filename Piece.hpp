#pragma once


#include <iostream>

#include <Bitmask.hpp>
#include <Bitboard.hpp>
#include <Attacks.hpp>


namespace piece {
  inline constexpr bool is_set(piece_bitboard_t mask, pos_t b) {
    return mask & (1LLU << b);
  }

  inline void set_pos(piece_bitboard_t &mask, pos_t b) {
    assert(!is_set(mask, b));
    mask |= 1LLU << b;
  }

  inline void unset_pos(piece_bitboard_t &mask, pos_t b) {
    assert(is_set(mask, b));
    mask &= ~(1LLU << b);
  }

  inline void move(piece_bitboard_t &mask, pos_t i, pos_t j) {
    unset_pos(mask, i);
    set_pos(mask, j);
  }

  inline constexpr size_t size(piece_bitboard_t mask) {
    return bitmask::count_bits(mask);
  }

  inline void set_king_pos_white(pos_pair_t &kings, pos_t i) {
    kings = bitmask::_pos_pair(i, bitmask::second(kings));
  }

  inline void set_king_pos_black(pos_pair_t &kings, pos_t i) {
    kings = bitmask::_pos_pair(bitmask::first(kings), i);
  }

  constexpr pos_t uninitialized_king = 0xff;
  constexpr pos_pair_t uninitialized_kings = bitmask::_pos_pair(uninitialized_king, uninitialized_king);
  inline void unset_king_pos_white(pos_pair_t &kings) {
    set_king_pos_white(kings, uninitialized_king);
  }

  inline void unset_king_pos_black(pos_pair_t &kings) {
    set_king_pos_black(kings, uninitialized_king);
  }

  inline piece_bitboard_t pos_mask(pos_t k) {
    return 1ULL << k;
  }

  inline constexpr piece_bitboard_t get_pawn_attack(pos_t pos, COLOR c) {
    if(c==WHITE) return Attacks<WPAWNM>::get_attacks(pos);
    if(c==BLACK) return Attacks<BPAWNM>::get_attacks(pos);
    return 0x00ULL;
  }

  inline piece_bitboard_t get_pawn_attacks(piece_bitboard_t mask, COLOR c) {
    if(c == WHITE) {
      return MultiAttacks<WPAWNM>::get_attacks(mask);
    } else {
      return MultiAttacks<BPAWNM>::get_attacks(mask);
    }
  }

  inline piece_bitboard_t get_sliding_diag_attack(pos_t pos, piece_bitboard_t occupied) {
    return Attacks<BISHOPM>::get_attacks(pos,occupied);
  }

  inline piece_bitboard_t get_sliding_diag_xray_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_diag_attack(pos, friends|foes) & foes;
    return get_sliding_diag_attack(pos, friends | (foes ^ blockers));
  }

  inline piece_bitboard_t get_sliding_diag_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    return Attacks<BISHOPM>::get_attacking_ray(i,j,occupied);
  }

  inline piece_bitboard_t get_sliding_diag_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_diag_attack(i, friends|foes) & foes;
    return get_sliding_diag_attacking_ray(i, j, friends | (foes ^ blockers));
  }

  inline piece_bitboard_t get_sliding_diag_attacks(piece_bitboard_t mask, piece_bitboard_t occupied) {
    return MultiAttacks<BISHOPM>::get_attacks(mask,occupied);
  }


  inline piece_bitboard_t get_sliding_orth_attack(pos_t pos, piece_bitboard_t occupied) {
    return Attacks<ROOKM>::get_attacks(pos,occupied);
  }

  inline piece_bitboard_t get_sliding_orth_xray_attack(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_orth_attack(pos, friends|foes) & foes;
    return get_sliding_orth_attack(pos, friends | (foes ^ blockers));
  }

  inline piece_bitboard_t get_sliding_orth_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    return Attacks<ROOKM>::get_attacking_ray(i,j,occupied);
  }

  inline piece_bitboard_t get_sliding_orth_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = get_sliding_orth_attack(i, friends|foes) & foes;
    return get_sliding_orth_attacking_ray(i, j, friends | (foes ^ blockers));
  }

  inline piece_bitboard_t get_sliding_orth_attacks(piece_bitboard_t mask, piece_bitboard_t occupied) {
    return MultiAttacks<ROOKM>::get_attacks(mask,occupied);
  }


  inline piece_bitboard_t get_knight_attack(pos_t pos) {
    return Attacks<KNIGHTM>::get_attacks(pos);
  }

  inline piece_bitboard_t get_knight_attacks(piece_bitboard_t mask) {
    return MultiAttacks<KNIGHTM>::get_attacks(mask);
  }

  inline piece_bitboard_t get_king_attack(pos_t pos) {
    return Attacks<KINGM>::get_attacks(pos);
  }

//  inline piece_bitboard_t get_king_attacks(piece_bitboard_t mask, piece_bitboard_t occupied) {
//    return MultiAttacks<KINGM>::get_attacks(mask, occupied);
//  }

  void print(piece_bitboard_t mask) {
    std::cout << piece::size(mask) << std::endl;
    bitmask::print(mask);
  }
} // namespace piece


// interface to piece bitboard
struct Piece {
  PIECE value;
  COLOR color;
  pos_t piece_index;

  static inline constexpr pos_t get_piece_index(PIECE p, COLOR c) {
    return (p==EMPTY) ? int(NO_PIECES)*int(NO_COLORS) : int(p)*(int)NO_COLORS+c;
  }

  constexpr Piece(PIECE p, COLOR c):
    value(p), color(c),
    piece_index(get_piece_index(p, c))
  {}

  piece_bitboard_t get_attack(pos_t pos, piece_bitboard_t occupied) {
    switch(value) {
      case EMPTY: return 0x00;
      case PAWN: return piece::get_pawn_attack(pos, color);
      case KNIGHT: return piece::get_knight_attack(pos);
      case BISHOP: return piece::get_sliding_diag_attack(pos, occupied);
      case ROOK: return piece::get_sliding_orth_attack(pos, occupied);
      case QUEEN: return piece::get_sliding_diag_attack(pos, occupied) | piece::get_sliding_orth_attack(pos, occupied);
      case KING: return piece::get_king_attack(pos);
    }
    return 0x00;
  }

  inline constexpr bool is_empty() const {
    return value == EMPTY;
  }

  inline constexpr char str() const {
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
