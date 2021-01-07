#include <Interface.hpp>


int main(int argc, char *argv[]) {
  Engine engine;
  Interface iface(engine);
  iface.run();
  return 0;
}
