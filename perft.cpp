#include <Engine.hpp>
#include <FEN.hpp>


int main(int argc, char *argv[]) {
  int depth = (argc >= 2) ? atoi(argv[1]) : 7;
  fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  Engine e(f);
  printf("Perft %d\n", depth);
  size_t total = 0;
  {
    decltype(auto) store_scope = e.get_zobrist_perft_scope();
    e.iter_moves([&](pos_t i, pos_t j) mutable -> void {
      const move_t m = bitmask::_pos_pair(i, j);
      event_t ev = e.get_move_event(i, j);
      std::string sm = e._move_str(m);
      e.make_move(m);
      size_t nds = 0;
      if(depth > 1) {
        nds = e.perft(depth-1);
      } else if(depth == 1) {
        nds = 1;
      } else {
        nds = 0;
      }
      printf("%s: %lu\n", sm.c_str(), nds);
      total += nds;
      e.retract_move();
    });
  }
  printf("total: %lu\n", total);
}
