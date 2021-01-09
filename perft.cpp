#include <Engine.hpp>
#include <FEN.hpp>


int main(int argc, char *argv[]) {
  int depth = (argc >= 2) ? -1+atoi(argv[1]) : -1+5;
  fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  Engine e(f);
  printf("Perft %d\n", 1+depth);
  size_t total = 0;
  e.iter_moves([&](pos_t i, pos_t j) mutable -> void {
    event_t ev1 = e.get_move_event(i, j);
    e.act_event(ev1);
    e.get_fixed_depth_move(depth);
    printf("%s%s: %lu\n", board::_pos_str(i).c_str(), board::_pos_str(j).c_str(), e.nodes_searched);
    total += e.nodes_searched;
    e.unact_event();
  });
  printf("total: %lu\n", total);
}
