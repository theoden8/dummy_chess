#include <Engine.hpp>


int main(int argc, char *argv[]) {
  const int depth = (argc >= 2) ? atoi(argv[1]) : 4;
  const fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;

  Engine e(f);

  MoveLine mline;
  for(int d = depth; d >= 0; --d) {
    str::print("pre-moved:", e._line_str(mline.full()));
    move_t bestmove = e.get_fixed_depth_move(d);
    if(bestmove == board::nomove)break;
    printf("best move: %s\n", board::_move_str(bestmove).c_str());
    printf("evaluation: %.5f\n", (((d & 1) == (depth & 1)) ? -1 : 1) * e.evaluation);
    printf("nodes searched: %lu\n", e.nodes_searched);
    printf("hit rate: %.5f\n", double(e.zb_hit) / double(1e-9+ e.zb_hit + e.zb_miss));
    e.make_move(bestmove);
    mline.premove(bestmove);
    str::print();
  }
}
