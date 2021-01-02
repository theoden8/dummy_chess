#include <Interface.hpp>


int main(int argc, char *argv[]) {
  Board b;
  Interface iface(b);
  iface.run();
  return 0;
}
