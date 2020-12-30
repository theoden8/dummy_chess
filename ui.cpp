#include <Interface.hpp>


int main(int argc, char *argv[]) {
  State s;
  Interface iface(s);
  iface.run();
  return 0;
}
