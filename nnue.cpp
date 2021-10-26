#include <NNUE.hpp>


int main() {
  const fen::FEN f = fen::starting_pos;
  Board board(f);
  nn::halfkp *nnue = new nn::halfkp(board);
  nnue->model.load("external/network.nnue"s);
  const float eval = nn::halfkp::value_to_centipawn(nnue->init_forward_pass());
  printf("eval %f\n", eval);
  delete nnue;
}
