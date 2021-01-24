#include <chrono>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main() {
  constexpr uint64_t shannon_number[] = { 20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167 };
#if 1
  printf("perft benchmarks\n");
  printf("\n");
  printf("starting position\n");
  for(pos_t depth = 1; depth <= 7; ++depth) {
    Engine e(fen::starting_pos);
    auto start = system_clock::now();
    size_t nds = e.perft(depth);
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    double sec = 1e-9*dur;
    double kndssec = (double(nds)/sec)*1e-3;
    double kndssec_raw = (double(e.nodes_searched)/sec)*1e-3;
    double hit_rate = double(e.zb_hit) / double(e.zb_hit + e.zb_miss);
    printf("depth=%d, time=%.3f\t%.3f kN/sec\traw=%.3f kN/sec\tnodes=%lu\tshannon=%lu\thit_rate=%.3f\n",
            depth, sec, kndssec, kndssec_raw, nds, shannon_number[depth-1], hit_rate);
  }
  printf("\n");
#endif
#if 1
  printf("alpha-beta benchmarks\n");
  printf("\n");
  printf("starting position\n");
  for(const pos_t depth : {1,2,3,4,5,6,7,8}) {
    Engine e(fen::starting_pos);
    auto start = system_clock::now();
    move_t m = e.get_fixed_depth_move(depth);
    size_t nds = e.nodes_searched;
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    double sec = 1e-9*dur;
    double eval = e.evaluation;
    double kndssec = (double(nds)/sec)*1e-3;
    double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
    printf("move=%s, depth=%d, eval=%.5f\ttime=%.3f\traw=%.3f kN/sec\tnodes=%lu\tshannon=%lu\thit_rate=%.3f\n",
            board::_move_str(m).c_str(), depth, eval, sec, kndssec, nds, shannon_number[depth-1], hit_rate);
    fflush(stdout);
  }
#endif
}
