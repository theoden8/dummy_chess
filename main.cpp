#include <State.hpp>

int main() {
  State s;
  // example, print piece position
  std::cout << "positions of black rook" << std::endl;
  Piece::get(ROOK, BLACK).print();
  // show posibble moves for a knight
  constexpr auto pieceT = KNIGHT;
  constexpr auto colorT = WHITE;
  Piece::get(pieceT, colorT).foreach([&](pos_t pos) mutable -> void {
    printf("white knight pos: %hhu\n", pos);
    piece_loc_t attacks = Attacks<pieceT, colorT>::get(pos);
    bitmask::print_mask(attacks, pos);
  });
  s.print();
}
