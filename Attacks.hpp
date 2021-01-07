#pragma once


#include <array>
#include <utility>
#include <type_traits>

#include <m42.h>

#include <Bitboard.hpp>
#include <Constants.hpp>
#include <Event.hpp>


typedef enum { WPAWNM, BPAWNM, KNIGHTM, BISHOPM, ROOKM, QUEENM, KINGM, NO_MPIECES } MPIECE;
namespace {

template <PIECE P, COLOR CC> struct piece_to_mpiece   {static constexpr MPIECE mp = NO_MPIECES;};
template <> struct piece_to_mpiece<PAWN,WHITE>        {static constexpr MPIECE mp = WPAWNM;};
template <> struct piece_to_mpiece<PAWN,BLACK>        {static constexpr MPIECE mp = BPAWNM;};
template <COLOR CC> struct piece_to_mpiece<KNIGHT,CC> {static constexpr MPIECE mp = KNIGHTM;};
template <COLOR CC> struct piece_to_mpiece<BISHOP,CC> {static constexpr MPIECE mp = BISHOPM;};
template <COLOR CC> struct piece_to_mpiece<ROOK,CC>   {static constexpr MPIECE mp = ROOKM;};
template <COLOR CC> struct piece_to_mpiece<QUEEN,CC>  {static constexpr MPIECE mp = QUEENM;};
template <COLOR CC> struct piece_to_mpiece<KING,CC>   {static constexpr MPIECE mp = KINGM;};

} // namespace

template <PIECE p, COLOR c>
constexpr MPIECE get_mpiece = ::piece_to_mpiece<p,c>::mp;
inline constexpr MPIECE get_mpiece_value(PIECE p, COLOR c) {
  if(p==PAWN && c==WHITE)return WPAWNM;
  if(p==PAWN && c==BLACK)return BPAWNM;
  if(p==KNIGHT)return KNIGHTM;
  if(p==BISHOP)return BISHOPM;
  if(p==ROOK)return ROOKM;
  if(p==QUEEN)return QUEENM;
  if(p==KING)return KINGM;
  return NO_MPIECES;
}


template <MPIECE MP> struct Attacks;

template <MPIECE MP> struct xRayAttacks {
  static inline piece_bitboard_t get_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = Attacks<MP>::get_attacks(i, friends, foes) & foes;
    const piece_bitboard_t occupied = friends | (foes ^ blockers);
    return Attacks<MP>::get_attacking_ray(i, j, occupied);
  }
};

// attack mask from multiple pieces of the kind at once
template <MPIECE MP> struct MultiAttacks {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    piece_bitboard_t res = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      res |= Attacks<MP>::get_attacks(pos, friends, foes);
    });
    return res;
  }
};

template <MPIECE MP> struct Moves {
  static constexpr piece_bitboard_t get_moves(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    // knights, rooks, bishops and queens, no pins/checks considerations
    return Attacks<MP>::get_attacks(mask, friends, foes) & ~friends;
  }
};

// pawn attacks
template <> struct Attacks<WPAWNM> {
  static piece_bitboard_t get_attacks(pos_t i) {
    return M42::pawn_attacks(WHITE, i);
  }
};

template <> struct Attacks<BPAWNM> {
  static piece_bitboard_t get_attacks(pos_t i) {
    return M42::pawn_attacks(BLACK, i);
  }
};

// attack mask from multiple pieces of the kind at once
template <> struct MultiAttacks<WPAWNM> {
  static piece_bitboard_t get_attacks(piece_bitboard_t mask) {
    constexpr piece_bitboard_t left = bitmask::vline;
    constexpr piece_bitboard_t right = bitmask::vline << 7;
    constexpr piece_bitboard_t mid = ~UINT64_C(0) ^ left ^ right;
    return ((mask & left ) << (board::LEN + 1))
         | ((mask & mid  ) << (board::LEN + 1))
         | ((mask & mid  ) << (board::LEN - 1))
         | ((mask & right) << (board::LEN - 1));
  }
};
// attack mask from multiple pieces of the kind at once
template <> struct MultiAttacks<BPAWNM> {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask) {
    constexpr piece_bitboard_t left = bitmask::vline;
    constexpr piece_bitboard_t right = bitmask::vline << 7;
    constexpr piece_bitboard_t mid = ~UINT64_C(0) ^ left ^ right;
    return ((mask & left) >> (board::LEN - 1))
           | ((mask & mid) >> (board::LEN - 1))
           | ((mask & mid) >> (board::LEN + 1))
           | ((mask & right) >> (board::LEN + 1));
  }
};


template <MPIECE MP>
static constexpr std::enable_if_t<MP == WPAWNM || MP == BPAWNM, piece_bitboard_t>
get_pawn_moves(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes, pos_t enpassant) {
  const piece_bitboard_t enpassant_mask = (enpassant == event::enpassantnotrace) ? 0x00 : 1ULL << enpassant;
  const piece_bitboard_t attacks = Attacks<MP>::get_attacks(i) & (foes|enpassant_mask);
  const piece_bitboard_t moves = Moves<MP>::get_basic_move(i) & ~(friends | foes);
  if(bitmask::count_bits(moves) != 1) {
    return attacks|moves;
  }
  const int pos_to = bitmask::log2_of_exp2(moves),
            pos_from = i;
  if(std::abs(pos_from - pos_to) == board::LEN) {
    return attacks|moves;
  }
  return attacks;
}

template <>
struct Moves<WPAWNM> {
  static constexpr piece_bitboard_t get_moves(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes, pos_t enpassant) {
    return get_pawn_moves<WPAWNM>(i, friends, foes, enpassant);
  }

  static constexpr inline bool is_enpassant_move(pos_t i, pos_t j) {
    return (j - i) == 2*board::LEN;
  }

  static constexpr inline bool is_promotion_move(pos_t i, pos_t j) {
    return board::_y(j) == 8-1;
  }

  static constexpr inline pos_t get_enpassant_trace(pos_t i, pos_t j) {
    assert(is_enpassant_move(i, j));
    return j-board::LEN;
  }

  static constexpr piece_bitboard_t get_basic_move(pos_t i) {
    const piece_bitboard_t maskpos = 1ULL << i;
    constexpr pos_t step = board::LEN;
    piece_bitboard_t mask = 0x00;
    mask |= maskpos<<step;
    if(1+board::_y(i) == 2)mask|=maskpos<<(2*step);
    return mask;
  }
};

template <>
struct Moves<BPAWNM> {
  static constexpr piece_bitboard_t get_moves(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes, pos_t enpassant) {
    return get_pawn_moves<BPAWNM>(i, friends, foes, enpassant);
  }

  static constexpr inline bool is_enpassant_move(pos_t i, pos_t j) {
    return (i - j) == 2*board::LEN;
  }

  static constexpr inline pos_t get_enpassant_trace(pos_t i, pos_t j) {
    assert(is_enpassant_move(i, j));
    return j+board::LEN;
  }

  static constexpr inline bool is_promotion_move(pos_t i, pos_t j) {
    return board::_y(j) == 1-1;
  }

  static constexpr piece_bitboard_t get_basic_move(pos_t i) {
    constexpr pos_t step = board::LEN;
    const piece_bitboard_t maskpos = 1ULL << i;
    piece_bitboard_t mask = 0x00;
    mask |= maskpos>>step;
    if(1+board::_y(i) == 7)mask|=maskpos>>(2*step);
    return mask;
  }
};


// knight attacks
// https://www.chessprogramming.org/Knight_Pattern
template <> struct Attacks <KNIGHTM> {
  static inline piece_bitboard_t get_attacks(pos_t i) {
    return M42::knight_attacks(i);
  }
};

template <> struct MultiAttacks<KNIGHTM> {
  static inline piece_bitboard_t get_attacks(piece_bitboard_t knights, piece_bitboard_t friends, piece_bitboard_t foes) {
    return M42::calc_knight_attacks(knights);
  }
};

template <> struct Moves<KNIGHTM> {
  static piece_bitboard_t get_moves(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    // knights, rooks, bishops and queens, no pins/checks considerations
    return Attacks<KNIGHTM>::get_attacks(mask) & ~friends;
  }
};

// bishop attacks
std::array<piece_bitboard_t, board::LEN - 1> leftquadrants = {0x0ULL};
std::array<piece_bitboard_t, board::LEN - 1> bottomquadrants = {0x0ULL};
template <> struct Attacks<BISHOPM> {
  static inline piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // enemy king's line of attack is currently handled on the board level
    const piece_bitboard_t occupied = friends | foes;
    return M42::bishop_attacks(i, occupied);
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    const piece_bitboard_t attacked_bit = 1ULL << j;
    const pos_t x_i=board::_x(i), y_i=board::_y(i),
                x_j=board::_x(j), y_j=board::_y(j);
    const piece_bitboard_t diags = M42::diag_attacks(i, occupied),
                           adiags = M42::adiag_attacks(i, occupied);
    const piece_bitboard_t bottomhalf = ~0ULL >> board::LEN * (board::LEN - y_i);
    const piece_bitboard_t tophalf = ~0ULL << board::LEN * (y_i + 1);
    piece_bitboard_t r = 0x00;
    if(x_j < x_i && y_i < y_j) {
      // top-left
      r = adiags & tophalf;
    } else if(x_i < x_j && y_i < y_j) {
      // top-right
      r = diags & tophalf;
    } else if(x_j < x_i && y_j < y_i) {
      // bottom-left
      r = diags & bottomhalf;
    } else if(x_i < x_j && y_j < y_i) {
      // bottom-right
      r = adiags & bottomhalf;
    }
    return (r & attacked_bit) ? r : 0x00;
  }
};

// seg-fault
//template <> struct MultiAttacks<BISHOPM> {
//  static piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
//    return M42::calc_bishop_attacks(mask, friends | foes);
//  }
//};

// rook attacks
template <> struct Attacks<ROOKM> {
  static inline piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // enemy king's line of attack is currently handled on the board level
    const piece_bitboard_t occupied = friends | foes;
    return M42::rook_attacks(i, occupied);
  }

  // -> -> -> -> -> -> -> ->
  // a1 b1 c1 d1 e1 f1 g1 h1
  // ..
  // a8 b8 c8 d8 e8 f8 g8 h8
  // bits shifted the "other way"
  static inline piece_bitboard_t get_left_ray(pos_t i, piece_bitboard_t occupied) {
    const pos_t y = board::_y(i), x = board::_x(i);
    constexpr pos_t axlen = board::LEN;
    piece_bitboard_t left = (bitmask::hline >> (8 - x));
    left <<= axlen*y;
    left &= bitmask::ones_after_eq_bit(bitmask::highest_bit(left & occupied));
    return left;
  }

  static inline piece_bitboard_t get_right_ray(pos_t i, piece_bitboard_t occupied) {
    const pos_t y = board::_y(i), x = board::_x(i);
    constexpr pos_t axlen = board::LEN;
    const piece_bitboard_t left = (bitmask::hline >> (8 - x));
    piece_bitboard_t right = ~(1ULL << x) & (~left & bitmask::hline);
    right <<= axlen*y;
    right &= bitmask::ones_before_eq_bit(bitmask::lowest_bit(right & occupied));
    return right;
  }

  static inline piece_bitboard_t get_top_ray(pos_t i, piece_bitboard_t occupied) {
    const pos_t y = board::_y(i), x = board::_x(i);
    constexpr pos_t axlen = board::LEN;
    const piece_bitboard_t vertical = bitmask::vline << x;
    piece_bitboard_t up = ~(1ULL << i) & (vertical << y*axlen);
    up &= bitmask::ones_before_eq_bit(bitmask::lowest_bit(up & occupied));
    return up;
  }

  static inline piece_bitboard_t get_bottom_ray(pos_t i, piece_bitboard_t occupied) {
    const pos_t y = board::_y(i), x = board::_x(i);
    constexpr pos_t axlen = board::LEN;
    const piece_bitboard_t vertical = bitmask::vline << x;
    const piece_bitboard_t up = ~(1ULL << i) & (vertical << y*axlen);
    piece_bitboard_t down = ~(1ULL << i) & (~up & vertical);
    down &= bitmask::ones_after_eq_bit(bitmask::highest_bit(down & occupied));
    return down;
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    const piece_bitboard_t attacked_bit = 1ULL << j;
    const pos_t x_i=board::_x(i), y_i=board::_y(i),
                x_j=board::_x(j), y_j=board::_y(j);
    if(x_i != x_j && y_i != y_j)return 0x00;
    piece_bitboard_t r = 0x00;;
    if(x_j < x_i) {
      r = get_left_ray(i, occupied);
    } else if(x_i < x_j) {
      r = get_right_ray(i, occupied);
    } else if(y_i < y_j) {
      r = get_top_ray(i, occupied);
    } else if(y_j < y_i) {
      r = get_bottom_ray(i, occupied);
    }
    return (r & attacked_bit) ? r : 0x00;
  }

  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    return (bitmask::vline << board::_x(i)
          | bitmask::hline << (board::LEN * board::_y(i)))
          & ~(1ULL << i);
  }
};

// seg-fault
//template <> struct MultiAttacks<ROOKM> {
//  static piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
//    return M42::calc_rook_attacks(mask, friends | foes);
//  }
//};

// queen attacks
template <> struct Attacks<QUEENM> {
  static inline piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<BISHOPM>::get_attacks(i, friends, foes)
         | Attacks<ROOKM>::get_attacks(i, friends, foes);
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    const piece_bitboard_t diag = Attacks<BISHOPM>::get_attacking_ray(i, j, occupied);
    if(diag)return diag;
    const piece_bitboard_t axes = Attacks<ROOKM>::get_attacking_ray(i, j, occupied);
    if(axes)return axes;
    return 0x00ULL;
  }
};

template <> struct MultiAttacks<QUEENM> {
  // make use of bishops'/rooks' optimizations
  static piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    return MultiAttacks<BISHOPM>::get_attacks(mask, friends, foes)
         | MultiAttacks<ROOKM>::get_attacks(mask, friends, foes);
  }
};

// king attacks
template <> struct Attacks<KINGM> {
  static inline piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return M42::king_attacks(i);
  }

  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    constexpr piece_bitboard_t left = (0x1ULL<<0) | (0x1ULL<<8) | (0x1ULL<<16);
    constexpr piece_bitboard_t right =  (0x4ULL<<0) | (0x4ULL<<8) | (0x4ULL<<16);
    constexpr piece_bitboard_t mid = (0x2ULL<<0) | (0x2ULL<<16);
    piece_bitboard_t mask = mid;
    if(board::_x(i)!=A)mask|=left;
    if(board::_x(i)!=H)mask|=right;
    constexpr pos_t offset = 8+2-1;
    if(i <= offset)return mask >> (offset - i);
    return mask << (i - offset);
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied){return 0x00;}
};

// king moves
template <> struct Moves<KINGM> {
  template <COLOR CC>
  static inline constexpr piece_bitboard_t get_moves(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes,
                                                     piece_bitboard_t attack_mask, piece_bitboard_t castlings)
  {
    const piece_bitboard_t occupied = friends | foes;
    constexpr pos_t shift = (CC == WHITE) ? 0 : (board::SIZE-board::LEN);
    castlings &= bitmask::hline << shift;
    piece_bitboard_t castlemoves = 0x00;
    if(castlings) {
      // can't castle when checked
      if(attack_mask & (1ULL << i))castlings=0x00;
      constexpr piece_bitboard_t castleleft = 0x40ULL << shift;
      constexpr piece_bitboard_t castleleftcheck = 0x20ULL << shift;
      constexpr piece_bitboard_t castleleftcheckocc = 0x60ULL << shift;
      constexpr piece_bitboard_t castleright = 0x04ULL << shift;
      constexpr piece_bitboard_t castlerightcheck = 0x08ULL << shift;
      constexpr piece_bitboard_t castlerightcheckocc = 0x0EULL << shift;
      if((castlings & castleleft)
          && !(attack_mask & castleleftcheck)
          && !(occupied & castleleftcheckocc))
        castlemoves|=castleleft;
      if((castlings & castleright)
          && !(attack_mask & castlerightcheck)
          && !(occupied & castlerightcheckocc))
        castlemoves|=castleright;
    }
    return (Attacks<KINGM>::get_basic(i) & ~friends & ~attack_mask) | castlemoves;
  }

  template <COLOR CC>
  static inline constexpr bool is_castling_move(pos_t i, pos_t j) {
    constexpr pos_t shift = (CC == WHITE) ? 0 : (board::SIZE-board::LEN);
    constexpr piece_bitboard_t castlings = 0x44ULL << shift;
    constexpr pos_t kingpos = (CC == WHITE) ? board::_pos(E, 1) : board::_pos(E, 8);
    return (i == kingpos) && ((1ULL << j) & castlings);
  }

  template <COLOR CC>
  static inline constexpr pos_pair_t castle_rook_move(pos_t i, pos_t j) {
    constexpr pos_t shift = (CC == WHITE) ? 0 : board::SIZE - board::LEN;
    constexpr piece_bitboard_t castleleft = board::_pos(C, 1) + shift;
    constexpr piece_bitboard_t castleright = board::_pos(G, 1) + shift;
    if(j == castleleft) return bitmask::_pos_pair(board::_pos(A, 1) + shift, board::_pos(D, 1) + shift);
    if(j == castleright)return bitmask::_pos_pair(board::_pos(H, 1) + shift, board::_pos(F, 1) + shift);
    abort();
    return 0x00;
  }
};
