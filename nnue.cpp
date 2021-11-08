#include <NNUE.hpp>


int main() {
  const fen::FEN f = fen::starting_pos;
  Board board(f);
  const std::string filename_nnue = "external/network.nnue"s;
  str::print("nnue type:", std::string(nn::NNUEFileInfo(filename_nnue)));
  std::shared_ptr<nn::halfkp> nnue(new nn::halfkp());
  nnue->init_halfkp_features(board);
  nnue->model.load(filename_nnue);
  const float eval = nn::halfkp::value_to_centipawn(nnue->init_forward_pass());
  printf("eval %f\n", eval);
}
