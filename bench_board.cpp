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
    e.perft(depth);
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    double sec = 1e-9*dur;
    double ndssec = double(e.nodes_searched)/sec;
    printf("depth=%d, time=%.3f\t%.3f kN/sec\tnodes=%lu\tshannon=%lu\n",
            depth, sec, ndssec/1e3, e.nodes_searched, shannon_number[depth-1]);
  }
}
