#pragma once

#include <Board.hpp>
#include <Attacks.hpp>


// current game-state
// board: current board
// attacks: currently computed attack moves by each piece
class State {
  Board b;
  COLOR activePlayer_;
public:
  State():
    b(),
    activePlayer_(WHITE)
  {
    for(pos_t i = 0; i < Board::LENGTH; ++i) {
      b.set_pos(Board::_pos(A + i, 2), b.get_piece(PAWN, WHITE)),
      b.set_pos(Board::_pos(A + i, 7), b.get_piece(PAWN, BLACK));
    }
    // make initial position
    for(const auto &[color, N] : {std::make_pair(WHITE, 1), std::make_pair(BLACK, 8)}) {
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

  decltype(auto) get_attacks() const {
    std::array <piece_bitboard_t, Board::SIZE> attacks = {UINT64_C(0x00)};
    for(auto&a:attacks)a=UINT64_C(0x00);
    piece_bitboard_t friends = b.get_piece_positions(activePlayer());
    piece_bitboard_t foes = b.get_piece_positions(enemy_of(activePlayer()));
    for(PIECE p : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      for(COLOR c : {WHITE, BLACK}) {
        b.get_piece(p,c).foreach([&](pos_t pos) mutable noexcept -> void {
          attacks[pos] |= get_piece_attacks(p,c,pos,friends,foes);
        });
      }
    }
    return attacks;
  }

  piece_bitboard_t get_attacks_from(pos_t pos) const {
    return get_attacks()[pos];
  }

  COLOR activePlayer() const {
    return activePlayer_;
  }

  decltype(auto) get_piece(PIECE p, COLOR c) noexcept {
    return b.get_piece(p, c);
  }

  decltype(auto) get_piece(PIECE p, COLOR c) const noexcept {
    return b.get_piece(p, c);
  }

  decltype(auto) at_pos(pos_t ind) noexcept {
    return b[ind];
  }

  decltype(auto) at_pos(pos_t ind) const noexcept {
    return b[ind];
  }

  decltype(auto) at_pos(pos_t i, pos_t j) noexcept {
    return at_pos(Board::_pos(i,j));
  }

  decltype(auto) at_pos(pos_t i, pos_t j) const noexcept {
    return at_pos(Board::_pos(i,j));
  }

  void print() {
    b.print();
  }
};
