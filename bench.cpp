#include <chrono>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main(int argc, char *argv[]) {
  const fen::FEN f = (argc >= 2) ? fen::load_from_string(argv[1]) : fen::starting_pos;
#if 1
  constexpr uint64_t shannon_number[] = { 20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167 };
  str::print("perft benchmarks");
  str::print("position", fen::export_as_string(f));
  {
    Engine e(f);
    decltype(auto) store_scope = e.get_zobrist_perft_scope();
    for(int depth = 1; depth <= 7; ++depth) {
      store_scope.reset();
      auto start = system_clock::now();
      size_t nds = e.perft(depth);
      auto stop = system_clock::now();
      long dur = duration_cast<nanoseconds>(stop-start).count();
      double sec = 1e-9*dur;
      double kndssec = (double(nds)/sec)*1e-3;
      double kndssec_raw = (double(e.nodes_searched)/sec)*1e-3;
      double hit_rate = double(e.zb_hit) / double(1e-9 + e.zb_hit + e.zb_miss);
      double hashfull = double(e.zb_occupied) / ZOBRIST_SIZE;
      printf("depth=%d, time=%.3f\t%.3f kN/sec\traw=%.3f kN/sec\tnodes=%lu\tshannon=%lu\thit_rate=%.3f\thashfull=%.3f\n",
              depth, sec, kndssec, kndssec_raw, nds, shannon_number[depth-1], hit_rate, hashfull);
    }
  }
  printf("\n");
#endif
#if 1
  str::print("alpha-beta benchmarks\n");
  str::print("position", fen::export_as_string(f));
  {
    Engine e(f);
    auto [ab_store_scope, e_store_scope] = e.get_zobrist_alphabeta_scope();
    for(const int depth : {1,2,3,4,5,6,7,8,9}) {
      ab_store_scope.reset();
      e_store_scope.reset();
      auto start = system_clock::now();
      move_t m = e.get_fixed_depth_move_iddfs(depth);
      size_t nds = e.nodes_searched;
      auto stop = system_clock::now();
      long dur = duration_cast<nanoseconds>(stop-start).count();
      double sec = 1e-9*dur;
      double eval = e.evaluation;
      double kndssec = (double(nds)/sec)*1e-3;
      double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
      double hashfull = double(e.zb_occupied) / ZOBRIST_SIZE;
      printf("move=%s, depth=%d, eval=%.5f\ttime=%.3f\traw=%.3f kN/sec\tnodes=%lu\thit_rate=%.3f\thashfull=%.3f\n",
              pgn::_move_str(e, m).c_str(), depth, eval, sec, kndssec, nds, hit_rate, hashfull);
      fflush(stdout);
    }
  }
#endif
}
