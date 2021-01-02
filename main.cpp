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
    piece_bitboard_t attacks = Attacks<pieceT, colorT>::get_basic(pos);
    bitmask::print_mask(attacks, pos);
  });
  // show attacks
  std::cout << "attacks" << std::endl;
  auto attacks = b.get_attacks();
  for(pos_t y=0;y<board::LEN;++y) {
    for(pos_t x=0;x<board::LEN;++x) {
      std::cout << attacks[y*board::LEN+x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  b.print();
}
