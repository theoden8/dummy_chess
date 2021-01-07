#include <Board.hpp>
#include <Engine.hpp>

int main() {
  Engine b(fen::starting_pos);
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
  {
    Engine b(fen::promotion_test_pos);
    pos_t i = board::_pos(E, 7);
    b.iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
      printf("(%s, %s)\n", board::_pos_str(i).c_str(), board::_pos_str(j).c_str());
    });
  }
  b.print();
  printf("best move: %s\n", board::_pos_str(b.get_fixed_depth_move(5)).c_str());
}
