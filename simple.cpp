#include <Engine.hpp>

int main(int argc, char *argv[]) {
  int depth = 4;
  if(argc >= 2) {
    depth = atoi(argv[1]);
  }
  const fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  printf("depth: %d\n", depth);
  Engine b(f);
//  size_t nds = b.perft(depth);
//  str::print("depth", depth, "nds", nds);
//  b.make_move(board::_pos(E,2), board::_pos(E,3));
//  b.make_move(board::_pos(E,7), board::_pos(E,6));
//  b.make_move(board::_pos(D,1), board::_pos(F,3));
//  b.make_move(board::_pos(D,8), board::_pos(H,4));
//  b.make_move(board::_pos(F,3), board::_pos(F,7));
//  b.make_move(board::_pos(D,1), board::_pos(G,4));
//  b.make_move(board::_pos(D,8), board::_pos(F,6));
//  b.make_move(board::_pos(G,1), board::_pos(F,3));
//  b.make_move(board::_pos(F,6), board::_pos(E,5));
//  {
//    Engine b(fen::promotion_test_pos);
//    pos_t i = board::_pos(E, 7);
//    b.iter_moves_from(i, [&](pos_t i, pos_t j) mutable -> void {
//      printf("(%s, %s)\n", board::_pos_str(i).c_str(), board::_pos_str(j).c_str());
//    });
//  }
  b.print();
  printf("best move: %s\n", board::_move_str(b.get_fixed_depth_move_iddfs(depth)).c_str());
  printf("evaluation: %.5f\n", b.evaluation);
  printf("nodes searched: %lu\n", b.nodes_searched);
  printf("hit rate: %.5f\n", double(b.zb_hit) / double(1e-9+ b.zb_hit + b.zb_miss));
}
