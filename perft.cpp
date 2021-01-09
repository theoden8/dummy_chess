#include <Engine.hpp>
#include <FEN.hpp>


int main(int argc, char *argv[]) {
  int depth = (argc >= 2) ? -1+atoi(argv[1]) : -1+5;
  Engine e(fen::starting_pos);
  printf("Perft %d\n", 1+depth);
  size_t total = 0;
  e.iter_moves([&](pos_t i1, pos_t j1) mutable -> void {
    event_t ev1 = e.get_move_event(i1, j1);
    e.act_event(ev1);
    e.get_fixed_depth_move(depth);
    printf("%s%s: %lu\n", board::_pos_str(i1).c_str(), board::_pos_str(j1).c_str(), e.nodes_searched);
    total += e.nodes_searched;
    e.unact_event();
  });
  printf("%lu\n", total);
}
