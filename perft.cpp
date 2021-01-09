#include <Engine.hpp>
#include <FEN.hpp>


int main(int argc, char *argv[]) {
  int depth = (argc >= 2) ? atoi(argv[1]) : 5;
  fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  Engine e(f);
  printf("Perft %d\n", depth);
  size_t total = 0;
  e.iter_moves([&](pos_t i, pos_t j) mutable -> void {
    event_t ev1 = e.get_move_event(i, j);
    e.act_event(ev1);
    size_t nds = 0;
    if(depth > 1) {
      e.get_fixed_depth_move(-1+depth);
      nds = e.nodes_searched;
    } else if(depth == 1) {
      nds = 1;
    } else {
      nds = 0;
    }
    std::string sp = "";
    switch((e.is_promotion_move(i,j)) ? e.get_promotion_as(j) : EMPTY) {
      case KNIGHT:sp='n';break;
      case BISHOP:sp='b';break;
      case ROOK:sp='r';break;
      case QUEEN:sp='q';break;
      default:break;
    }
    printf("%s: %lu\n", (board::_pos_str(i) + board::_pos_str(j) + sp).c_str(), nds);
    total += nds;
    e.unact_event();
  });
  printf("total: %lu\n", total);
}
