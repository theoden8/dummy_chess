#pragma once


#include <Board.hpp>

#include <array>


template <PIECE P, COLOR C> struct Attacks;

// attack mask from multiple pieces of the kind at once
template <PIECE P, COLOR C> struct MultiAttacks {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    piece_bitboard_t res = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      res |= Attacks<P, C>::get_attacks(pos, friends, foes);
    });
    return res;
  }
};

// pawn attacks
template <COLOR C> struct Attacks<PAWN, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<PAWN,C>::get(i);
  }

  static constexpr piece_bitboard_t get(pos_t i) {
    pos_t offset = 0;
    piece_bitboard_t smask = 0x00 | (0x07LLU<<8);
    piece_bitboard_t lmask = 0x00;
    piece_bitboard_t mask = 0x00;
    if constexpr(C == WHITE) {
      offset = 8*2+2-1;
      lmask = smask | (0x20LLU);
      mask = (Board::_y(i) == 1) ? lmask : smask;
    } else if constexpr(C == BLACK) {
      offset = 2-1;
      lmask = smask | (0x20LLU<<16);
      mask = (Board::_y(i) == Board::LENGTH-2) ? lmask : smask;
    }
    if(i <= offset)
      return mask >> (offset - i);
    return mask << (i - offset);
  }
};

// knight attacks
// https://www.chessprogramming.org/Knight_Pattern
std::array<piece_bitboard_t, Board::SIZE> knightattacks = {};
template <COLOR C> struct Attacks <KNIGHT, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<KNIGHT,C>::get(i);
  }

  static constexpr piece_bitboard_t get(pos_t i) {
    // memoized attacks
    if(knightattacks[i] != 0x00) {
      return knightattacks[i];
    }
    piece_bitboard_t I = piece_bitboard_t(1) << i;
    bool not_a = Board::_x(i) != A;
    bool not_ab = not_a && Board::_x(i) != B;
    bool not_h = Board::_x(i) != H;
    bool not_gh = not_h && Board::_x(i) != G;
    piece_bitboard_t mask = 0x00;
    if(not_h) mask|=I<<17; // bitmask::print_mask(mask, i);
    if(not_gh)mask|=I<<10; // bitmask::print_mask(mask, i);
    if(not_gh)mask|=I>>6;  // bitmask::print_mask(mask, i);
    if(not_h) mask|=I>>15; // bitmask::print_mask(mask, i);
    if(not_a) mask|=I<<15; // bitmask::print_mask(mask, i);
    if(not_ab)mask|=I<<6;  // bitmask::print_mask(mask, i);
    if(not_ab)mask|=I>>10; // bitmask::print_mask(mask, i);
    if(not_a) mask|=I>>17; // bitmask::print_mask(mask, i);
    knightattacks[i] = mask;
    return mask;
  }
};

// bishop attacks
template <COLOR C> struct Attacks<BISHOP, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // TODO change according to piece positions
    return Attacks<BISHOP,C>::get(i);
  }

  static constexpr piece_bitboard_t cut(pos_t i, piece_bitboard_t mask, int8_t tblr) {
    pos_t shift = 0x00;
    bool top = tblr & 0x08;
    bool bottom = tblr & 0x04;
    bool left = tblr & 0x02;
    bool right = tblr & 0x01;
    if(top) {
    }
    return 0x00;
  }
  static constexpr piece_bitboard_t get(pos_t i) {
    piece_bitboard_t
      mask1 = 9241421688590303745ULL,
      mask2 = 72624976668147840ULL;
    // to fix
    // *******
    //  *******
    // * ******
    // ** *****
    // *** ****
    // **** ***
    // *****B**
    // ********
    return mask1 << i
      | ((i <= 7) ? mask2 >> (7 - i) : mask2 << (i - 7))
      | ((i <= 56) ? mask2 >> (56 - i) : mask2 << (i - 56))
      | ((i <= 63) ? mask1 >> (63 - i) : mask1 << (i - 63));
  }
};

// rook attacks
template <COLOR C> struct Attacks<ROOK, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // TODO change according to piece positions
    return Attacks<ROOK,C>::get(i);
  }

  static constexpr piece_bitboard_t get(pos_t i) {
    return piece_bitboard_t(9259542123273814144LLU) >> Board::_x(i)
      | piece_bitboard_t(0xFF) >> (Board::LENGTH * Board::_y(i));
  }
};

// queen attacks
template <COLOR C> struct Attacks<QUEEN, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<BISHOP, C>::get_attacks(i, friends, foes) | Attacks<ROOK, C>::get_attacks(i, friends, foes);
  }

  static constexpr piece_bitboard_t get(pos_t i) {
    return Attacks<BISHOP, C>::get(i) | Attacks<ROOK, C>::get(i);
  }
};

// king attacks, generic
template <> struct Attacks<KING, NEUTRAL> {
  static constexpr piece_bitboard_t get(pos_t i) {
    constexpr piece_bitboard_t mask = (0x70LLU<<0) | (0x50LLU<<8) | (0x70LLU<<16);
    constexpr piece_bitboard_t offset = 10-1;
    if(i <= offset)return mask >> (offset - i);
    return mask << (i - offset);
  }
};

// king attacks, by a player
template <COLOR C> struct Attacks<KING, C> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<KING,C>::get(i);
  }

  static constexpr piece_bitboard_t get(pos_t i) {
    piece_bitboard_t mask = Attacks<KING, NEUTRAL>::get(i);
    if (C == WHITE && i == Board::_pos(E, 1))
      mask |= 0x24;
    else if (C == BLACK && i == Board::_pos(E, 8))
      mask |= 0x24LLU << (Board::SIZE - Board::LENGTH);
    return mask;
  }
};


constexpr piece_bitboard_t get_piece_attacks(PIECE p, COLOR c, pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
  if(p==PAWN  &&c==WHITE)return Attacks<PAWN  ,WHITE>::get_attacks(pos,friends,foes);
  if(p==KNIGHT&&c==WHITE)return Attacks<KNIGHT,WHITE>::get_attacks(pos,friends,foes);
  if(p==BISHOP&&c==WHITE)return Attacks<BISHOP,WHITE>::get_attacks(pos,friends,foes);
  if(p==ROOK  &&c==WHITE)return Attacks<ROOK  ,WHITE>::get_attacks(pos,friends,foes);
  if(p==QUEEN &&c==WHITE)return Attacks<QUEEN ,WHITE>::get_attacks(pos,friends,foes);
  if(p==KING  &&c==WHITE)return Attacks<KING  ,WHITE>::get_attacks(pos,friends,foes);
  if(p==PAWN  &&c==BLACK)return Attacks<PAWN  ,BLACK>::get_attacks(pos,friends,foes);
  if(p==KNIGHT&&c==BLACK)return Attacks<KNIGHT,BLACK>::get_attacks(pos,friends,foes);
  if(p==BISHOP&&c==BLACK)return Attacks<BISHOP,BLACK>::get_attacks(pos,friends,foes);
  if(p==ROOK  &&c==BLACK)return Attacks<ROOK  ,BLACK>::get_attacks(pos,friends,foes);
  if(p==QUEEN &&c==BLACK)return Attacks<QUEEN ,BLACK>::get_attacks(pos,friends,foes);
  if(p==KING  &&c==BLACK)return Attacks<KING  ,BLACK>::get_attacks(pos,friends,foes);
  abort();
}
