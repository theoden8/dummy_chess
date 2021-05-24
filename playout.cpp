#include <Engine.hpp>
#include <PGN.hpp>


int main(int argc, char *argv[]) {
  const int depth = (argc >= 2) ? atoi(argv[1]) : 4;
  const fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;

  Engine e(f);

  MoveLine mline;
  pgn::PGN pgnwriter(e);
  int no_moves = 0;
  {
    decltype(auto) store_scope = e.get_zobrist_alphabeta_scope();
    while(e.can_move()) {
      move_t bestmove = e.get_fixed_depth_move(depth);
      mline.put(bestmove);
      pgnwriter.handle_move(bestmove);
      str::print(pgnwriter.str());
      if(++no_moves == 400)break;
      if(e.can_draw_repetition()) {
        str::print("draw by repetition");
        break;
      }
    }
  }
  str::print(pgnwriter.str());
}
