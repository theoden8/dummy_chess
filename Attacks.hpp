#pragma once


#include <array>

#include <Bitboard.hpp>
#include <Constants.hpp>
#include <Event.hpp>


template <PIECE P, COLOR CC> struct Attacks;

template <PIECE P, COLOR CC> struct xRayAttacks {
  static constexpr piece_bitboard_t get_xray_attacks(pos_t pos, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = Attacks<P, CC>::get_attacks(pos, friends, foes) & foes;
    return Attacks<P, CC>::get_attacks(pos, friends, foes ^ blockers);
  }

  static inline piece_bitboard_t get_attacking_xray(pos_t i, pos_t j, piece_bitboard_t friends, piece_bitboard_t foes) {
    const piece_bitboard_t blockers = Attacks<P, CC>::get_attacks(i, friends, foes) & foes;
    const piece_bitboard_t occupied = friends | (foes ^ blockers);
    return Attacks<P, CC>::get_attacking_ray(i, j, occupied);
  }
};

// attack mask from multiple pieces of the kind at once
template <PIECE P, COLOR CC> struct MultiAttacks {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    piece_bitboard_t res = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      res |= Attacks<P, CC>::get_attacks(pos, friends, foes);
    });
    return res;
  }
};

// attack mask from multiple pieces of the kind at once
template <PIECE P, COLOR CC> struct MultixRayAttacks {
  static constexpr piece_bitboard_t get_xray_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    piece_bitboard_t res = 0x00;
    bitmask::foreach(mask, [&](pos_t pos) mutable noexcept -> void {
      res |= xRayAttacks<P, CC>::get_xray_attacks(pos, friends, foes);
    });
    return res;
  }
};

template <PIECE P, COLOR CC> struct Moves {
  // TODO never the case when piece is pinned
  static constexpr piece_bitboard_t get_moves(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    // knights, rooks, bishops and queens which are not pinned:
    return Attacks<P,CC>::get_attacks(mask, friends, foes) & ~friends;
  }
};

// pawn attacks
template <COLOR CC> struct Attacks<PAWN, CC> {
  static constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<PAWN,CC>::get_basic_attack(i);
  }

  static constexpr piece_bitboard_t get_basic_attack(pos_t i) {
    pos_t offset = 0;
    piece_bitboard_t smask = 0x00 | (0x05ULL<<8);
    piece_bitboard_t lmask = 0x00;
    piece_bitboard_t mask = 0x00;
    piece_bitboard_t forwmask = 0x00;
    if constexpr(CC == WHITE) {
      offset = 2-1;
      lmask = smask | (0x20ULL<<16);
      forwmask = UINT64_C(0xFF) << (board::LEN * (board::_y(i) + 1));
      mask = (board::_y(i) == board::LEN-2) ? lmask : smask;
    } else if constexpr(CC == BLACK) {
      offset = board::LEN*2+2-1;
      lmask = smask | (0x20ULL);
      forwmask = UINT64_C(0xFF) << (board::LEN * (board::_y(i) - 1));
      mask = (board::_y(i) == 1) ? lmask : smask;
    }
    if(i <= offset)return mask >> (offset - i);
    return (mask << (i - offset)) & forwmask;
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied){return 0x00;}
};

// attack mask from multiple pieces of the kind at once
template <> struct MultiAttacks<PAWN, WHITE> {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    constexpr piece_bitboard_t left = bitmask::vline;
    constexpr piece_bitboard_t right = bitmask::vline << 7;
    constexpr piece_bitboard_t mid = ~UINT64_C(0) ^ left ^ right;
    piece_bitboard_t res = 0x00;
    res |= (mask & left ) << (board::LEN + 1);
    res |= (mask & mid  ) << (board::LEN + 1);
    res |= (mask & mid  ) << (board::LEN - 1);
    res |= (mask & right) << (board::LEN - 1);
    return res;
  }
};
// attack mask from multiple pieces of the kind at once
template <> struct MultiAttacks<PAWN, BLACK> {
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    constexpr piece_bitboard_t left = bitmask::vline;
    constexpr piece_bitboard_t right = bitmask::vline << 7;
    constexpr piece_bitboard_t mid = ~UINT64_C(0) ^ left ^ right;
    piece_bitboard_t res = 0x00;
    res |= (mask & left ) >> (board::LEN - 1);
    res |= (mask & mid  ) >> (board::LEN - 1);
    res |= (mask & mid  ) >> (board::LEN + 1);
    res |= (mask & right) >> (board::LEN + 1);
    return res;
  }
};

template <COLOR CC> struct Moves<PAWN, CC> {
  static constexpr piece_bitboard_t get_moves(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes, pos_t enpassant) {
    const piece_bitboard_t enpassant_mask = (enpassant == event::enpassantnotrace) ? 0x00 : 1ULL << enpassant;
    const piece_bitboard_t attacks = Attacks<PAWN, CC>::get_attacks(i, friends, foes) & (foes|enpassant_mask);
    piece_bitboard_t moves = Moves<PAWN, CC>::get_basic_move(i) & ~(friends | foes);
    if(bitmask::count_bits(moves) != 1)return attacks|moves;
    int pos_to=bitmask::log2_of_exp2(moves), pos_from=i;
    if(std::abs(pos_from - pos_to) == board::LEN)return attacks|moves;
    return attacks;
  }

  static constexpr bool is_enpassant_move(pos_t i, pos_t j) {
    return std::abs(i - j) == 2*board::LEN;
  }

  static constexpr pos_t get_enpassant_trace(pos_t i, pos_t j) {
    assert(is_enpassant_move(i, j));
    if constexpr(CC == WHITE)return j-board::LEN;
    else if constexpr(CC == BLACK)return j+board::LEN;
    assert(false);
    return 0x00ULL;
  }

  static constexpr piece_bitboard_t get_basic_move(pos_t i) {
    piece_bitboard_t mask = 0x00;
    piece_bitboard_t maskpos = UINT64_C(1) << i;
    pos_t step = board::LEN;
    if constexpr(CC == WHITE) {
      mask |= maskpos<<step;
      if(1+board::_y(i) == 2)mask|=maskpos<<(2*step);
    } else if constexpr(CC == BLACK) {
      mask |= maskpos>>step;
      if(1+board::_y(i) == 7)mask|=maskpos>>(2*step);
    }
    return mask;
  }
};


// knight attacks
// https://www.chessprogramming.org/Knight_Pattern
std::array<piece_bitboard_t, board::SIZE> knightattacks = {0x0ULL};
template <COLOR CC> struct Attacks <KNIGHT, CC> {
  static inline constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<KNIGHT,CC>::get_basic(i);
  }

  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    // memoized attacks
    if(knightattacks[i] != 0x00)return knightattacks[i];
    piece_bitboard_t I = 1ULL << i;
    bool not_a = board::_x(i) != A;
    bool not_ab = not_a && board::_x(i) != B;
    bool not_h = board::_x(i) != H;
    bool not_gh = not_h && board::_x(i) != G;
    piece_bitboard_t mask = 0x00;
    if(not_h) mask|=I<<17;
    if(not_gh)mask|=I<<10;
    if(not_gh)mask|=I>>6;
    if(not_h) mask|=I>>15;
    if(not_a) mask|=I<<15;
    if(not_ab)mask|=I<<6;
    if(not_ab)mask|=I>>10;
    if(not_a) mask|=I>>17;
    knightattacks[i] = mask;
    return mask;
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied){return 0x00;}
};

template <COLOR CC> struct MultiAttacks<KNIGHT, CC> {
  static inline constexpr piece_bitboard_t get_attacks(piece_bitboard_t knights, piece_bitboard_t friends, piece_bitboard_t foes) {
    using U64 = piece_bitboard_t;
    U64 l1 = (knights >> 1) & UINT64_C(0x7f7f7f7f7f7f7f7f);
    U64 l2 = (knights >> 2) & UINT64_C(0x3f3f3f3f3f3f3f3f);
    U64 r1 = (knights << 1) & UINT64_C(0xfefefefefefefefe);
    U64 r2 = (knights << 2) & UINT64_C(0xfcfcfcfcfcfcfcfc);
    U64 h1 = l1 | r1;
    U64 h2 = l2 | r2;
    return (h1<<16) | (h1>>16) | (h2<<8) | (h2>>8);
  }
};

// bishop attacks
std::array<piece_bitboard_t, board::SIZE> bishopattacks = {0x0ULL};
std::array<piece_bitboard_t, board::LEN - 1> leftquadrants = {0x0ULL};
std::array<piece_bitboard_t, board::LEN - 1> bottomquadrants = {0x0ULL};
template <COLOR CC> struct Attacks<BISHOP, CC> {
  static inline constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // enemy king's line of attack is currently handled on the board level
    const piece_bitboard_t occupied = friends | foes;
    return get_topleft_ray(i, occupied) | get_topright_ray(i, occupied)
           | get_bottomleft_ray(i, occupied) | get_bottomright_ray(i, occupied);
  }

  static inline piece_bitboard_t get_topleft_ray(int i, piece_bitboard_t occupied) {
    const pos_t x = board::_x(i), y = board::_y(i);
    const piece_bitboard_t diags = Attacks<BISHOP, CC>::get_basic(i);
    piece_bitboard_t top_left = diags & get_top_quadrant(y) & get_left_quadrant(x);
    top_left &= bitmask::ones_before_eq_bit(bitmask::lowest_bit(top_left & occupied));
    return top_left;
  }

  static inline piece_bitboard_t get_topright_ray(int i, piece_bitboard_t occupied) {
    const pos_t x = board::_x(i), y = board::_y(i);
    const piece_bitboard_t diags = Attacks<BISHOP, CC>::get_basic(i);
    piece_bitboard_t top_right = diags & get_top_quadrant(y) & get_right_quadrant(x);
    top_right &= bitmask::ones_before_eq_bit(bitmask::lowest_bit(top_right & occupied));
    return top_right;
  }

  static inline piece_bitboard_t get_bottomleft_ray(int i, piece_bitboard_t occupied) {
    const pos_t x = board::_x(i), y = board::_y(i);
    const piece_bitboard_t diags = Attacks<BISHOP, CC>::get_basic(i);
    piece_bitboard_t bottom_left = diags & get_bottom_quadrant(y) & get_left_quadrant(x);
    bottom_left &= bitmask::ones_after_eq_bit(bitmask::highest_bit(bottom_left & occupied));
    return bottom_left;
  }

  static inline piece_bitboard_t get_bottomright_ray(int i, piece_bitboard_t occupied) {
    const pos_t x = board::_x(i), y = board::_y(i);
    const piece_bitboard_t diags = Attacks<BISHOP, CC>::get_basic(i);
    piece_bitboard_t bottom_right = diags & get_bottom_quadrant(y) & get_right_quadrant(x);
    bottom_right &= bitmask::ones_after_eq_bit(bitmask::highest_bit(bottom_right & occupied));
    return bottom_right;
  }

  static inline piece_bitboard_t get_left_quadrant(int x) {
    if(x == 0)return 0x00;
    if(leftquadrants[x-1])return leftquadrants[x-1];
    piece_bitboard_t left_quadrant = 0x00;
    for(int i=0;i<x;++i)left_quadrant|=bitmask::vline<<i;
    leftquadrants[x-1] = left_quadrant;
    return left_quadrant;
  }

  static inline piece_bitboard_t get_right_quadrant(int x) {
    return ~0ULL & ~get_left_quadrant(x) & ~(bitmask::vline << x);
  }

  static inline piece_bitboard_t get_bottom_quadrant(int y) {
    if(y == 0)return 0x00;
    if(bottomquadrants[y-1])return bottomquadrants[y-1];
    piece_bitboard_t bottom_quadrant = 0x00;
    for(int i=0;i<y;++i)bottom_quadrant|=bitmask::hline<<i*board::LEN;
    bottomquadrants[y-1] = bottom_quadrant;
    return bottom_quadrant;
  }

  static inline piece_bitboard_t get_top_quadrant(int y) {
    return ~0ULL & ~get_bottom_quadrant(y) & ~(bitmask::hline << y*board::LEN);
  }


  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    piece_bitboard_t attacked_bit = 1ULL << j;
    pos_t x_i=board::_x(i), y_i=board::_y(i),
          x_j=board::_x(j), y_j=board::_y(j);
    piece_bitboard_t r = 0x00;
    if(x_j < x_i && y_i < y_j) {
      r = get_topleft_ray(i, occupied);
    } else if(x_i < x_j && y_i < y_j) {
      r = get_topright_ray(i, occupied);
    } else if(x_j < x_i && y_j < y_i) {
      r = get_bottomleft_ray(i, occupied);
    } else if(x_i < x_j && y_j < y_i) {
      r = get_bottomright_ray(i, occupied);
    }
    return (r & attacked_bit) ? r : 0x00;
  }


  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    if(bishopattacks[i])return bishopattacks[i];
    piece_bitboard_t mask = 0x00;
    pos_t step1 = board::LEN + 1;
    pos_t step2 = board::LEN - 1;
    int d = i; pos_t x=board::_x(i), y=board::_y(i);
    while(d-step1>0){d-=step1;if(board::_x(d)>x)break;mask|=1ULL<<d;} d=i;
    while(d-step2>0){d-=step2;if(board::_x(d)<x)break;mask|=1ULL<<d;} d=i;
    while(d+step1<board::SIZE){d+=step1;if(board::_x(d)<x)break;mask|=1ULL<<d;} d=i;
    while(d+step2<board::SIZE){d+=step2;if(board::_x(d)>x)break;mask|=1ULL<<d;} d=i;
    mask &= ~(1ULL << i);
    bishopattacks[i] = mask;
    return mask;
  }
};

// rook attacks
template <COLOR CC> struct Attacks<ROOK, CC> {
  static inline constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    // enemy king's line of attack is currently handled on the board level
    piece_bitboard_t occupied = friends | foes;
    return get_left_ray(i,occupied)|get_right_ray(i,occupied)|get_top_ray(i,occupied)|get_bottom_ray(i,occupied);
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
    piece_bitboard_t attacked_bit = 1ULL << j;
    pos_t x_i=board::_x(i), y_i=board::_y(i),
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

// queen attacks
template <COLOR CC> struct Attacks<QUEEN, CC> {
  static inline constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<BISHOP, CC>::get_attacks(i, friends, foes)
         | Attacks<ROOK, CC>::get_attacks(i, friends, foes);
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied) {
    piece_bitboard_t diag = Attacks<BISHOP, CC>::get_attacking_ray(i, j, occupied);
    if(diag)return diag;
    piece_bitboard_t axes = Attacks<ROOK, CC>::get_attacking_ray(i, j, occupied);
    if(axes)return axes;
    return 0x00ULL;
  }

  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    return Attacks<BISHOP, CC>::get_basic(i)
         | Attacks<ROOK, CC>::get_basic(i);
  }
};

template <COLOR CC> struct MultiAttacks<QUEEN, CC> {
  // make use of bishops'/rooks' optimizations
  static constexpr piece_bitboard_t get_attacks(piece_bitboard_t mask, piece_bitboard_t friends, piece_bitboard_t foes) {
    return MultiAttacks<BISHOP,CC>::get_attacks(mask, friends, foes)
         | MultiAttacks<ROOK,CC>::get_attacks(mask, friends, foes);
  }
};

// king attacks
template <COLOR CC> struct Attacks<KING, CC> {
  static inline constexpr piece_bitboard_t get_attacks(pos_t i, piece_bitboard_t friends, piece_bitboard_t foes) {
    return Attacks<KING, CC>::get_basic(i);
  }

  static inline constexpr piece_bitboard_t get_basic(pos_t i) {
    piece_bitboard_t left =  (0x4ULL<<0) | (0x4ULL<<8) | (0x4ULL<<16);
    piece_bitboard_t right = (0x1ULL<<0) | (0x1ULL<<8) | (0x1ULL<<16);
    constexpr piece_bitboard_t mid = (0x2ULL<<0) | (0x2ULL<<16);
    if(board::_x(i)==A)left = 0x00;
    else if(board::_x(i)==H)right=0x00;
    const piece_bitboard_t mask = left|mid|right;
    constexpr pos_t offset = 8+2-1;
    if(i <= offset)return mask >> (offset - i);
    return mask << (i - offset);
  }

  static inline piece_bitboard_t get_attacking_ray(pos_t i, pos_t j, piece_bitboard_t occupied){return 0x00;}
};

// king moves
template <COLOR CC> struct Moves<KING, CC> {
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
    return (Attacks<KING, CC>::get_basic(i) & ~friends & ~attack_mask) | castlemoves;
  }

  static inline constexpr bool is_castling_move(pos_t i, pos_t j) {
    constexpr pos_t shift = (CC == WHITE) ? 0 : (board::SIZE-board::LEN);
    constexpr piece_bitboard_t castlings = 0x44ULL << shift;
    constexpr pos_t kingpos = (CC == WHITE) ? board::_pos(E, 1) : board::_pos(E, 8);
    return (i == kingpos) && ((1ULL << j) & castlings);
  }

  static inline constexpr pos_pair_t castle_rook_move(pos_t i, pos_t j) {
    constexpr pos_t shift = (CC == WHITE) ? 0 : board::SIZE - board::LEN;
    constexpr piece_bitboard_t castleleft = board::_pos(C, 1) + shift;
    constexpr piece_bitboard_t castleright = board::_pos(G, 1) + shift;
    if(j == castleleft) return bitmask::_pos_pair(board::_pos(A, 1) + shift, board::_pos(D, 1) + shift);
    if(j == castleright)return bitmask::_pos_pair(board::_pos(H, 1) + shift, board::_pos(F, 1) + shift);
    assert(false);
    return 0x00;
  }
};
