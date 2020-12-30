#pragma once

#include <Board.hpp>

template <PIECE P, COLOR C> struct Moves;

// pawn moves
template <COLOR C> struct Moves<PAWN, C> {
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

// knight moves
template <COLOR C> struct Moves <KNIGHT, C> {
  // TODO fix, this is clearly wrong
  static constexpr piece_loc_t get(pos_t i) {
    piece_loc_t mask =
      piece_loc_t
          (0xA0LLU<<0)
        | (0x11LLU<<8)
        | (0x00LLU)
        | (0x11LLU<<24)
        | (0xA0LLU<<32);
    pos_t offset = 8*2+3-1;
    if(i <= offset) {
      return mask >> (offset - i);
    }
    return mask << (i - offset);
  }
};

// bishop moves
template <COLOR C> struct Moves<BISHOP, C> {
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

// rook moves
template <COLOR C> struct Moves<ROOK, C> {
  static constexpr piece_loc_t get(pos_t i) {
    return piece_loc_t(9259542123273814144LLU) >> Board::_x(i)
      | piece_loc_t(0xFF) >> (Board::LENGTH * Board::_y(i));
  }
};

// queen moves
template <COLOR C> struct Moves<QUEEN, C> {
  static constexpr piece_loc_t get(pos_t i) {
    return Moves<BISHOP, C>::get(i) | Moves<ROOK, C>::get(i);
  }
};

// king moves, generic
template <> struct Moves<KING, NEUTRAL> {
  static constexpr piece_loc_t get(pos_t i) {
    constexpr piece_loc_t mask = (0x70LLU<<0) | (0x50LLU<<8) | (0x70LLU<<16);
    constexpr piece_loc_t offset = 10-1;
    if(i <= offset)
      return mask >> (offset - i);
    return mask << (i - offset);
  }
};

// king moves, by a player
template <COLOR C> struct Moves<KING, C> {
  static constexpr piece_loc_t get(pos_t i) {
    piece_loc_t mask = Moves<KING, NEUTRAL>::get(i);
    if (C == WHITE && i == Board::_pos(E, 1))
      mask |= 0x24;
    else if (C == BLACK && i == Board::_pos(E, 8))
      mask |= 0x24LLU << (Board::SIZE - Board::LENGTH);
    return mask;
  }
};


// current game-state
// consists of current board
// and where pieces are ought to go..? I can't remember what this reaches_ does
class State {
  Board b;
  std::array <piece_loc_t, Board::SIZE> reaches_;
public:
  State():
    b(), reaches_()
  {
    for(pos_t i = 0; i < Board::SIZE; ++i) {
      reaches_[i] = 0x00;
    }
    for(pos_t i = 0; i < Board::LENGTH; ++i) {
      b.set_pos(Board::_pos(A + i, 2), Piece::get(PAWN, WHITE)),
      b.set_pos(Board::_pos(A + i, 7), Piece::get(PAWN, BLACK));
    }
    // make initial position
    for(auto &[color, N] : {std::make_pair(WHITE, 1), std::make_pair(BLACK, 8)}) {
      b.set_pos(Board::_pos(A, N), Piece::get(ROOK, color)),
      b.set_pos(Board::_pos(B, N), Piece::get(KNIGHT, color)),
      b.set_pos(Board::_pos(C, N), Piece::get(BISHOP, color)),
      b.set_pos(Board::_pos(D, N), Piece::get(QUEEN, color)),
      b.set_pos(Board::_pos(E, N), Piece::get(KING, color)),
      b.set_pos(Board::_pos(F, N), Piece::get(BISHOP, color)),
      b.set_pos(Board::_pos(G, N), Piece::get(KNIGHT, color)),
      b.set_pos(Board::_pos(H, N), Piece::get(ROOK, color));
    }
  }

  piece_loc_t get_positions(COLOR color) {
    return
      Piece::get(PAWN, color).mask
      | Piece::get(KNIGHT, color).mask
      | Piece::get(BISHOP, color).mask
      | Piece::get(ROOK, color).mask
      | Piece::get(QUEEN, color).mask
      | Piece::get(KING, color).mask;
  }

  COLOR enemy(COLOR c) {
    if(c == NEUTRAL)
      return NEUTRAL;
    return (c == WHITE) ? BLACK : WHITE;
  }

  piece_loc_t gen_ind_reach(pos_t i) {
    piece_loc_t free_moves = 0x00;
    piece_loc_t friends = get_positions(b[i]->color);
    piece_loc_t enemies = get_positions(enemy(b[i]->color));
    switch(b[i]->value) {
      case EMPTY:break;
      case PAWN:
        if(b[i]->color == WHITE)
          free_moves =  Moves<PAWN, WHITE>::get(i);
        else
          free_moves =  Moves<PAWN, BLACK>::get(i);
        break;
      case KNIGHT:
        if(b[i]->color == WHITE)
          free_moves =  Moves<KNIGHT, WHITE>::get(i);
        else
          free_moves =  Moves<KNIGHT, BLACK>::get(i);
        free_moves &= ~friends;
        break;
      case BISHOP:
        if(b[i]->color == WHITE)
          free_moves =  Moves<BISHOP, WHITE>::get(i);
        else
          free_moves =  Moves<BISHOP, BLACK>::get(i);
        break;
      case ROOK:
        if(b[i]->color == WHITE)
          free_moves =  Moves<ROOK, WHITE>::get(i);
        else
          free_moves =  Moves<ROOK, BLACK>::get(i);
        break;
      case QUEEN:
        if(b[i]->color == WHITE)
          free_moves =  Moves<QUEEN, WHITE>::get(i);
        else
          free_moves =  Moves<QUEEN, BLACK>::get(i);
        break;
      case KING:
        if(b[i]->color == WHITE)
          free_moves =  Moves<KING, WHITE>::get(i);
        else
          free_moves =  Moves<KING, BLACK>::get(i);
        free_moves &= ~friends;
        break;
    }
    return free_moves;
  }

  void print() {
    b.print();
  }
};
