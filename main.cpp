#include <Engine.hpp>

int main(int argc, char *argv[]) {
  int depth = 4;
  if(argc >= 2) {
    depth = atoi(argv[1]);
  }
  Engine b(fen::starting_pos);
  // example, print piece position
//  std::cout << "positions of black rook" << std::endl;
//  b.get_piece(ROOK, BLACK).print();
//  // show posibble attacks for a knight
//  constexpr auto pieceT = KNIGHT;
//  constexpr auto colorT = BLACK;
//  b.get_piece(pieceT, colorT).foreach([&](pos_t pos) mutable -> void {
//    printf("black knight pos: %hhu\n", pos);
//    piece_bitboard_t attacks = Attacks<get_mpiece<pieceT, colorT>>::get_attacks(pos);
//    bitmask::print_mask(attacks, pos);
//  });
//  {
//    Engine b(fen::promotion_test_pos);
//    pos_t i = board::_pos(E, 7);
//    b.iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
//      printf("(%s, %s)\n", board::_pos_str(i).c_str(), board::_pos_str(j).c_str());
//    });
//  }
  b.print();
  printf("best move: %s\n", board::_move_str(b.get_fixed_depth_move(depth)).c_str());
  printf("evaluation: %.5f\n", b.evaluation);
  printf("nodes searched: %lu\n", b.nodes_searched);
  printf("hit rate: %.5f\n", double(b.zb_hit) / double(1e-9+ b.zb_hit + b.zb_miss));
}
