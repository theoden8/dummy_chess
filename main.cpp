#include <Board.hpp>

int main() {
  Board b;
  // example, print piece position
  std::cout << "positions of black rook" << std::endl;
  b.get_piece(ROOK, BLACK).print();
  // show posibble attacks for a knight
  constexpr auto pieceT = KNIGHT;
  constexpr auto colorT = BLACK;
  b.get_piece(pieceT, colorT).foreach([&](pos_t pos) mutable -> void {
    printf("black knight pos: %hhu\n", pos);
    piece_bitboard_t attacks = Attacks<get_mpiece<pieceT, colorT>>::get_basic(pos);
    bitmask::print_mask(attacks, pos);
  });
  b.print();
}
