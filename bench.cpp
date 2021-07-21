#include <chrono>
#include <valarray>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;

int main(int argc, char *argv[]) {
  const fen::FEN f = (argc >= 2) ? fen::load_from_string(argv[1]) : fen::starting_pos;
//  const fen::FEN f = fen::load_from_string("r1b1kb1r/pp2pp2/n1p1q1p1/1N1nN2p/2BP4/4BQ2/PPP2PPP/R4RK1 b kq - 1 11"s);
//  const fen::FEN f = fen::load_from_string("8/5k2/2pBp2p/6p1/pP2P3/P1R1K2P/2P5/3r4 w - - 3 49"s);
//  const fen::FEN f = fen::load_from_string("rnq1kbnr/p1p4p/1p2pp2/3p2p1/2PP3N/4P3/PP1B1PQP/RN2K2R w KQkq - 0 11"s);
//  const fen::FEN f = fen::load_from_string("2rqkb1r/p2b1pp1/2n5/1p1n2Pp/N3p2P/1PPp4/P2P1P2/R1BQKBNR w KQk - 0 15"s);
//  const fen::FEN f = fen::load_from_string("1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24"s);
//  const fen::FEN f = fen::load_from_string("1r4k1/1r3pp1/3b1q1p/3p2nP/p1pP3R/2P2PP1/PPQ3K1/1RB3N1 w - - 0 1"s);
//  const fen::FEN f = fen::load_from_string("4b3/8/PB1k4/2N5/1N4P1/8/7K/8 w - - 5 59"s);
//  const fen::FEN f = fen::load_from_string("r3kb1r/p1p1ppp1/pq2Nn1p/4N3/3P1B2/8/PPP2PPP/R2Q1RK1 b kq - 0 12");
//  const fen::FEN f = fen::load_from_string("6k1/8/5KP1/8/8/8/8 b - - 4 71");
#if 1
  const std::valarray<uint64_t> shannon_number = { 20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167 };
//  const fen::FEN f = fen::load_from_string("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "s);
//  const std::valarray<uint64_t> shannon_number = { 48, 2039, 97862, 4085603, 193690690, 8031647685 };
  str::print("perft benchmarks\n");
  str::print("position", fen::export_as_string(f));
  {
    Engine e(f);
    decltype(auto) store_scope = e.get_zobrist_perft_scope();
    for(int depth = 1; depth < (int)shannon_number.size(); ++depth) {
      store_scope.reset();
      const auto start = system_clock::now();
      size_t nds = e.perft(depth);
      const auto stop = system_clock::now();
      const long dur = duration_cast<nanoseconds>(stop-start).count();
      const double sec = 1e-9*dur;
      const double kndssec = (double(nds)/sec)*1e-3;
      const double kndssec_raw = (double(e.nodes_searched)/sec)*1e-3;
      const double hit_rate = double(e.zb_hit) / double(1e-9 + e.zb_hit + e.zb_miss);
      const double hashfull = double(e.zb_occupied) / ZOBRIST_SIZE;
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
    decltype(auto) ab_storage = e.get_zobrist_alphabeta_scope();
    for(int depth = 1; depth <= 200; ++depth) {
      ab_storage.reset();
      const auto start = system_clock::now();
      Engine::iddfs_state idstate;
      move_t m = e.get_fixed_depth_move_iddfs(depth, idstate);
      const size_t nds = e.nodes_searched;
      const auto stop = system_clock::now();
      const long dur = duration_cast<nanoseconds>(stop-start).count();
      const double sec = 1e-9*dur;
      const double eval = Engine::score_float(idstate.eval);
      const double kndssec = (double(nds)/sec)*1e-3;
      const double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
      const double hashfull = double(e.zb_occupied) / ZOBRIST_SIZE;
      printf("move=%s, depth=%d/%lu, eval=%.5f\ttime=%.3f\traw=%.3f kN/sec\tnodes=%lu\thit_rate=%.3f\thashfull=%.3f\n",
              pgn::_move_str(e, m).c_str(), depth, idstate.pline.size(), eval, sec, kndssec, nds, hit_rate, hashfull);
      fflush(stdout);
    }
  }
#endif
}
