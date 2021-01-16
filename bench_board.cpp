#include <chrono>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main() {
  printf("fixed depth benchmarks\n");
  printf("\n");
  printf("starting position\n");
  constexpr uint64_t shannon_number[] = { 20, 400, 8902, 197281, 4865609, 119060324, 3195901860 };
  for(int depth = 1; depth <= 6; ++depth) {
    Engine e(fen::starting_pos);
    auto start = system_clock::now();
    size_t nds = e.perft(depth);
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    double sec = 1e-9*dur;
    double kndssec = (double(nds)/sec)*1e-3;
    double hit_rate = double(e.zb_hit) / double(e.zb_hit + e.zb_miss);
    printf("depth=%d, time=%.3f\t%.3f kN/sec\tnodes=%lu\tshannon=%lu\thit_rate=%.3f\n",
            depth, sec, kndssec, nds, shannon_number[depth-1], hit_rate);
  }
}
