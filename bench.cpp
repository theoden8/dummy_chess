#include <ctime>
#include <chrono>
#include <valarray>

#include <Engine.hpp>
#include <FEN.hpp>

using namespace std::chrono;


std::string str_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}


int main(int argc, char *argv[]) {
  tb::init("external/syzygy/src");
  const int maxdepth = (argc >= 2) ? atoi(argv[1]) : 20;
  const fen::FEN f = (argc >= 3) ? fen::load_from_string(argv[2]) : fen::starting_pos;
//  const fen::FEN f = fen::load_from_string("r1b1kb1r/pp2pp2/n1p1q1p1/1N1nN2p/2BP4/4BQ2/PPP2PPP/R4RK1 b kq - 1 11"s);
//  const fen::FEN f = fen::load_from_string("8/5k2/2pBp2p/6p1/pP2P3/P1R1K2P/2P5/3r4 w - - 3 49"s);
//  const fen::FEN f = fen::load_from_string("rnq1kbnr/p1p4p/1p2pp2/3p2p1/2PP3N/4P3/PP1B1PQP/RN2K2R w KQkq - 0 11"s);
//  const fen::FEN f = fen::load_from_string("2rqkb1r/p2b1pp1/2n5/1p1n2Pp/N3p2P/1PPp4/P2P1P2/R1BQKBNR w KQk - 0 15"s);
//  const fen::FEN f = fen::load_from_string("1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24"s);
//  const fen::FEN f = fen::load_from_string("1r4k1/1r3pp1/3b1q1p/3p2nP/p1pP3R/2P2PP1/PPQ3K1/1RB3N1 w - - 0 1"s);
//  const fen::FEN f = fen::load_from_string("1R6/P1n5/2p5/2k5/r7/3B4/3K4/8 w - - 1 82"s);
//  const fen::FEN f = fen::load_from_string("2k5/2p2Q2/1p2p2p/pP2r1p1/4q1P1/1P4R1/2KP1P2/8 w - - 4 47"s);
//  const fen::FEN f = fen::load_from_string("4b3/8/PB1k4/2N5/1N4P1/8/7K/8 w - - 5 59"s);
//  const fen::FEN f = fen::load_from_string("r3kb1r/p1p1ppp1/pq2Nn1p/4N3/3P1B2/8/PPP2PPP/R2Q1RK1 b kq - 0 12");
//  const fen::FEN f = fen::load_from_string("6k1/8/5KP1/8/8/8/8 b - - 4 71");
  str::print("time:", str_time());
  str::print("alpha-beta benchmarks\n");
  str::print("position", fen::export_as_string(f));
  {
    Engine e(f);
    decltype(auto) ab_storage = e.get_zobrist_alphabeta_scope();
    {
      ab_storage.reset();
      const auto start = system_clock::now();
      Engine::iddfs_state idstate;
      int16_t lastdepth = 0;
      move_t lastmove = board::nullmove;
      auto &&func = [&](Engine &e) mutable -> void {
        const size_t nds = e.nodes_searched;
        const auto stop = system_clock::now();
        const long dur = duration_cast<nanoseconds>(stop-start).count();
        const double sec = 1e-9*dur;
        const double eval = Engine::score_float(idstate.eval);
        const double kndssec = (double(nds)/sec)*1e-3;
        const double hit_rate = double(e.zb_hit) / double(1e-9+e.zb_hit + e.zb_miss);
        const double hashfull = double(e.zb_occupied) / ZOBRIST_SIZE;
        const size_t tb_hit = e.tb_hit;
        printf("move=%s depth=%d/%zu eval=%s time=%.3f raw=%.3f kN/sec\tnodes=%zu\thit_rate=%.3f hashfull=%.3f tb_hit=%zu\n",
                pgn::_move_str(e, idstate.pline.front()).c_str(), idstate.curdepth, idstate.pline.size(),
                Engine::score_string(idstate.eval).c_str(), sec, kndssec, nds, hit_rate, hashfull, tb_hit);
        fflush(stdout);
      };
      e.start_thinking(maxdepth, idstate, [&](bool verbose) mutable -> bool {
        if(!verbose)return true;
        if(idstate.curdepth == lastdepth && idstate.currmove() == lastmove)return true;
        func(e);
        lastdepth = idstate.curdepth;
        lastmove = idstate.currmove();
        return true;
      }, {});
      func(e);
    }
  }
  tb::free();
}
