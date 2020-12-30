#pragma once

#include <Board.hpp>

#include <array>

template <PIECE P, COLOR C> struct Attacks;

// pawn attacks
template <COLOR C> struct Attacks<PAWN, C> {
  static constexpr piece_loc_t get(pos_t i) {
    pos_t offset = 0;
    piece_loc_t smask = 0x00 | (0x07LLU<<8);
    piece_loc_t lmask = 0x00;
    piece_loc_t mask = 0x00;
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
std::array<piece_loc_t, 64> knightattacks = {};
template <COLOR C> struct Attacks <KNIGHT, C> {
  static constexpr piece_loc_t get(pos_t i) {
    // memoized attacks
    if(knightattacks[i] != 0x00) {
      return knightattacks[i];
    }
    piece_loc_t I = piece_loc_t(1) << i;
    bool not_a = Board::_x(i) != A;
    bool not_ab = not_a && Board::_x(i) != B;
    bool not_h = Board::_x(i) != H;
    bool not_gh = not_h && Board::_x(i) != G;
    piece_loc_t mask = 0x00;
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
  static constexpr piece_loc_t cut(pos_t i, piece_loc_t mask, int8_t tblr) {
    pos_t shift = 0x00;
    bool top = tblr & 0x08;
    bool bottom = tblr & 0x04;
    bool left = tblr & 0x02;
    bool right = tblr & 0x01;
    if(top) {
    }
    return 0x00;
  }
  static constexpr piece_loc_t get(pos_t i) {
    piece_loc_t
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
  static constexpr piece_loc_t get(pos_t i) {
    return piece_loc_t(9259542123273814144LLU) >> Board::_x(i)
      | piece_loc_t(0xFF) >> (Board::LENGTH * Board::_y(i));
  }
};

// queen attacks
template <COLOR C> struct Attacks<QUEEN, C> {
  static constexpr piece_loc_t get(pos_t i) {
    return Attacks<BISHOP, C>::get(i) | Attacks<ROOK, C>::get(i);
  }
};

// king attacks, generic
template <> struct Attacks<KING, NEUTRAL> {
  static constexpr piece_loc_t get(pos_t i) {
    constexpr piece_loc_t mask = (0x70LLU<<0) | (0x50LLU<<8) | (0x70LLU<<16);
    constexpr piece_loc_t offset = 10-1;
    if(i <= offset)
      return mask >> (offset - i);
    return mask << (i - offset);
  }
};

// king attacks, by a player
template <COLOR C> struct Attacks<KING, C> {
  static constexpr piece_loc_t get(pos_t i) {
    piece_loc_t mask = Attacks<KING, NEUTRAL>::get(i);
    if (C == WHITE && i == Board::_pos(E, 1))
      mask |= 0x24;
    else if (C == BLACK && i == Board::_pos(E, 8))
      mask |= 0x24LLU << (Board::SIZE - Board::LENGTH);
    return mask;
  }
};


// current game-state
// board: current board
// reaches: currently computed attack moves by each piece
class State {
  Board b;
  std::array <piece_loc_t, Board::SIZE> reaches_;
  COLOR activePlayer_;
public:

  State():
    b(), reaches_(),
    activePlayer_(WHITE)
  {
    for(pos_t i = 0; i < Board::SIZE; ++i) {
      reaches_[i] = 0x00;
    }
    for(pos_t i = 0; i < Board::LENGTH; ++i) {
      b.set_pos(Board::_pos(A + i, 2), b.get_piece(PAWN, WHITE)),
      b.set_pos(Board::_pos(A + i, 7), b.get_piece(PAWN, BLACK));
    }
    // make initial position
    for(auto &[color, N] : {std::make_pair(WHITE, 1), std::make_pair(BLACK, 8)}) {
      b.set_pos(Board::_pos(A, N), b.get_piece(ROOK, color)),
      b.set_pos(Board::_pos(B, N), b.get_piece(KNIGHT, color)),
      b.set_pos(Board::_pos(C, N), b.get_piece(BISHOP, color)),
      b.set_pos(Board::_pos(D, N), b.get_piece(QUEEN, color)),
      b.set_pos(Board::_pos(E, N), b.get_piece(KING, color)),
      b.set_pos(Board::_pos(F, N), b.get_piece(BISHOP, color)),
      b.set_pos(Board::_pos(G, N), b.get_piece(KNIGHT, color)),
      b.set_pos(Board::_pos(H, N), b.get_piece(ROOK, color));
    }
  }

  COLOR activePlayer() {
    return activePlayer_;
  }

  decltype(auto) get_piece(PIECE p, COLOR c) {
    return b.get_piece(p, c);
  }

  decltype(auto) at_pos(pos_t ind) {
    return b[ind];
  }

  piece_loc_t get_positions(COLOR color) {
    return
      b.get_piece(PAWN, color).mask
      | b.get_piece(KNIGHT, color).mask
      | b.get_piece(BISHOP, color).mask
      | b.get_piece(ROOK, color).mask
      | b.get_piece(QUEEN, color).mask
      | b.get_piece(KING, color).mask;
  }

  COLOR enemy(COLOR c) {
    if(c == NEUTRAL)
      return NEUTRAL;
    return (c == WHITE) ? BLACK : WHITE;
  }

  piece_loc_t gen_ind_reach(pos_t i) {
    piece_loc_t free_moves = 0x00;
    piece_loc_t friends = get_positions(b[i].color);
    piece_loc_t enemies = get_positions(enemy(b[i].color));
    switch(b[i].value) {
      case EMPTY:break;
      case PAWN:
        if(b[i].color == WHITE)
          free_moves =  Attacks<PAWN, WHITE>::get(i);
        else
          free_moves =  Attacks<PAWN, BLACK>::get(i);
        break;
      case KNIGHT:
        if(b[i].color == WHITE)
          free_moves =  Attacks<KNIGHT, WHITE>::get(i);
        else
          free_moves =  Attacks<KNIGHT, BLACK>::get(i);
        free_moves &= ~friends;
        break;
      case BISHOP:
        if(b[i].color == WHITE)
          free_moves =  Attacks<BISHOP, WHITE>::get(i);
        else
          free_moves =  Attacks<BISHOP, BLACK>::get(i);
        break;
      case ROOK:
        if(b[i].color == WHITE)
          free_moves =  Attacks<ROOK, WHITE>::get(i);
        else
          free_moves =  Attacks<ROOK, BLACK>::get(i);
        break;
      case QUEEN:
        if(b[i].color == WHITE)
          free_moves =  Attacks<QUEEN, WHITE>::get(i);
        else
          free_moves =  Attacks<QUEEN, BLACK>::get(i);
        break;
      case KING:
        if(b[i].color == WHITE)
          free_moves =  Attacks<KING, WHITE>::get(i);
        else
          free_moves =  Attacks<KING, BLACK>::get(i);
        free_moves &= ~friends;
        break;
    }
    return free_moves;
  }

  void print() {
    b.print();
  }
};
