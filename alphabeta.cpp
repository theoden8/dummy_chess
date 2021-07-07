#include <chrono>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main(int argc, char *argv[]) {
  int depth = (argc >= 2) ? atoi(argv[1]) : 8;
  fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
  Engine e(f);
  printf("Alpha-beta\n");
  {
    decltype(auto) store_scopes = e.get_zobrist_alphabeta_scope();
    auto start = system_clock::now();
    move_t m = e.get_fixed_depth_move_iddfs(depth);
    size_t nds = e.nodes_searched;
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    double sec = 1e-9*dur;
    double eval = e.evaluation;
    double kndssec = (double(nds)/sec)*1e-3;
    double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
    printf("move=%s, depth=%d, eval=%.5f\ttime=%.3f\traw=%.3f kN/sec\tnodes=%lu\thit_rate=%.3f\n",
            board::_move_str(m).c_str(), depth, eval, sec, kndssec, nds, hit_rate);
  }
}
