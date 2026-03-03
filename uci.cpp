#ifdef DC0_ENABLED
#include <DC0EngineUCI.hpp>
#else
#include <UCI.hpp>
#endif


int main(int argc, char *argv[]) {
#ifdef DC0_ENABLED
  UCI<DC0EngineUCI> uci;
#else
  UCI<EngineUCI> uci;
#endif
  uci.run();
}
