#include <chrono>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main() {
  printf("fixed depth benchmarks\n");
  printf("\n");
  printf("starting position\n");
  for(int depth = 1; depth <= 5; ++depth) {
    Engine e(fen::starting_pos);
    auto start = system_clock::now();
    move_t m = e.get_fixed_depth_move(depth);
    auto stop = system_clock::now();
    long dur = duration_cast<nanoseconds>(stop-start).count();
    printf("depth=%d, move=%s, time=%.3f\n", depth, board::_pos_str(m).c_str(), 1e-9*dur);
  }
  return 0;
}
