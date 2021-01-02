#include <State.hpp>
#include <Interface.hpp>

int main() {
  State s;
  // example, print piece position
  std::cout << "positions of black rook" << std::endl;
  s.get_piece(ROOK, BLACK).print();
  // show posibble attacks for a knight
  constexpr auto pieceT = KNIGHT;
  constexpr auto colorT = BLACK;
  s.get_piece(pieceT, colorT).foreach([&](pos_t pos) mutable -> void {
    printf("black knight pos: %hhu\n", pos);
    piece_bitboard_t attacks = Attacks<pieceT, colorT>::get_basic(pos);
    bitmask::print_mask(attacks, pos);
  });
  // show attacks
  std::cout << "attacks" << std::endl;
  auto attacks = s.get_attacks();
  for(pos_t y=0;y<Board::LENGTH;++y) {
    for(pos_t x=0;x<Board::LENGTH;++x) {
      std::cout << attacks[y*Board::LENGTH+x] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  s.print();
}
