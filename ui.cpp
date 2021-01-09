#include <Interface.hpp>


int main(int argc, char *argv[]) {
  const fen::FEN f = (argc >= 2) ? fen::load_from_string(argv[1]) : fen::starting_pos;
  Engine engine(f);
  Interface iface(engine);
  iface.run();
  return 0;
}
