#include <FEN.hpp>
#include <Engine.hpp>


int main(int argc, char *argv[]) {
  int depth = 8;
  if(argc >= 2) {
    depth = atoi(argv[1]);
  }
  const fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  Engine b(f);
  printf("depth: %d\n", depth);
  printf("IS CHECKMATE: %s\n", b.is_checkmate() ? "TRUE" : "FALSE");
  printf("IS DRAW: %s\n", b.is_draw() ? "TRUE" : "FALSE");
  printf("CAN DRAW: %s\n", b.is_draw_repetition() ? "TRUE" : "FALSE");
  Engine::iddfs_state idstate;
  std::unordered_set<move_t> searchmoves = {
  };
  printf("best move: %s\n", pgn::_move_str(b, b.get_fixed_depth_move_iddfs(depth, idstate, searchmoves)).c_str());
  printf("evaluation: %.5f\n", Engine::score_float(idstate.eval));
  printf("nodes searched: %lu\n", b.nodes_searched);
  printf("hit rate: %.5f\n", double(b.zb_hit) / double(1e-9+ b.zb_hit + b.zb_miss));
  printf("hashfull: %.5f\n", double(b.zb_occupied) / double(ZOBRIST_SIZE));
}
