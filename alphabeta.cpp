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
    const auto start = system_clock::now();
    Engine::iddfs_state idstate;
    move_t m = e.get_fixed_depth_move_iddfs(depth, idstate);
    const size_t nds = e.nodes_searched;
    const auto stop = system_clock::now();
    const long dur = duration_cast<nanoseconds>(stop-start).count();
    const double sec = 1e-9*dur;
    const float eval = Engine::score_float(idstate.eval);
    const double kndssec = (double(nds)/sec)*1e-3;
    const double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
    printf("move=%s, depth=%d, eval=%.5f\ttime=%.3f\traw=%.3f kN/sec\tnodes=%lu\thit_rate=%.3f\n",
            pgn::_move_str(e, m).c_str(), depth, eval, sec, kndssec, nds, hit_rate);
  }
}
