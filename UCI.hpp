#pragma once


#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include <FEN.hpp>
#include <Engine.hpp>


#ifndef NDEBUG
#define _printf printf
#else
#define _printf(...)
#endif


using namespace std::chrono;


struct UCI {
  Engine *engine = nullptr;
  zobrist::StoreScope<Engine::ab_info> *engine_ttable_owner = nullptr;

  std::atomic<bool> debug_mode = false;
  std::atomic<bool> should_quit = false;
  std::atomic<bool> should_stop = false;
  std::atomic<bool> job_started = false;

  std::recursive_mutex engine_mtx;
  using lock_guard = std::lock_guard<std::recursive_mutex>;
  std::thread engine_thread;

  UCI()
  {}

  void init(const fen::FEN &f) {
    lock_guard guard(engine_mtx);
    destroy();
    _printf("init\n");
    engine = new Engine(f);
    engine_ttable_owner = new std::remove_reference_t<decltype(*engine_ttable_owner)>(engine->get_zobrist_alphabeta_scope());
  }

  void destroy() {
    lock_guard guard(engine_mtx);
    if(engine != nullptr) {
      _printf("destroy\n");
      delete engine;
      engine = nullptr;
      delete engine_ttable_owner;
      engine_ttable_owner = nullptr;
    }
  }

  // https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf
  typedef enum {
    CMD_UCI,
    CMD_DEBUG,
    CMD_ISREADY,
    CMD_SETOPTION,
    CMD_REGISTER,
    CMD_UCINEWGAME,
    CMD_POSITION,
    CMD_GO,
    CMD_STOP,
    CMD_PONDERHIT,
    CMD_QUIT,
    NO_COMMANDS
  } COMMANDS;

  const std::map<std::string, COMMANDS> cmdmap = {
    {"uci"s,       CMD_UCI},
    {"debug"s,     CMD_DEBUG},
    {"isready"s,   CMD_ISREADY},
    {"setoption"s, CMD_SETOPTION},
    {"register"s,  CMD_REGISTER},
    {"ucinewgame"s,CMD_UCINEWGAME},
    {"position"s,  CMD_POSITION},
    {"go"s,        CMD_GO},
    {"stop"s,      CMD_STOP},
    {"ponderhit"s, CMD_PONDERHIT},
    {"quit"s,      CMD_QUIT},
  };

  typedef enum {
    RESP_ID,
    RESP_UCIOK,
    RESP_READYOK,
    RESP_BESTMOVE,
    RESP_COPYPROTECTION,
    RESP_REGISTRATION,
    RESP_INFO,
    RESP_OPTION,
    NO_RESPONSES
  } RESPONSE;

  const std::map<RESPONSE, std::string> respmap = {
    {RESP_ID,             "id"s},
    {RESP_UCIOK,          "uciok"s},
    {RESP_READYOK,        "readyok"s},
    {RESP_BESTMOVE,       "bestmove"s},
    {RESP_COPYPROTECTION, "copyprotection"s},
    {RESP_REGISTRATION,   "registration"s},
    {RESP_INFO,           "info"s},
    {RESP_OPTION,         "option"s},
  };

  void run() {
    std::vector<std::string> cmd = {""s};
    char c = 0;
    while((c = fgetc(stdin)) != EOF) {
      if(c == '\n') {
        if(cmd.size() > 1 || !cmd.back().empty()) {
          process_cmd(cmd);
        }
        cmd = {""s};
        if(should_quit)break;
      }
      if(isspace(c) && cmd.empty()) {
        continue;
      } else if(!isspace(c)) {
        cmd.back() += c;
      } else if(isspace(c) && !cmd.back().empty()) {
        cmd.emplace_back(""s);
      }
    }
    if(cmd.size() > 1 || !cmd.back().empty()) {
      process_cmd(cmd);
    }
    should_stop = true;
  }

  static move_t scan_move(const std::string &s) {
    assert(s.length() == 4 || s.length() == 5);
    const pos_t i_file = tolower(s[0]) - 'a';
    const pos_t i_rank = s[1] - '0';
    const pos_t i = board::_pos(A+i_file, i_rank);
    const pos_t j_file = tolower(s[2]) - 'a';
    const pos_t j_rank = s[3] - '0';
    pos_t j = board::_pos(A+j_file, j_rank);
    if(s.length() == 5) {
      switch(tolower(s[4])) {
        case 'k':j|=board::PROMOTE_KNIGHT;break;
        case 'b':j|=board::PROMOTE_BISHOP;break;
        case 'r':j|=board::PROMOTE_ROOK;break;
        case 'q':j|=board::PROMOTE_QUEEN;break;
        default:break;
      }
    }
    return bitmask::_pos_pair(i, j);
  }

  typedef struct _go_command {
    std::vector<move_t> searchmoves = {};
    bool ponder = false;
    double wtime = DBL_MAX;
    double btime = DBL_MAX;
    double winc = 0;
    double binc = 0;
    int movestogo = 0;
    int16_t depth = INT16_MAX;
    size_t nodes = SIZE_MAX;
    size_t mate = SIZE_MAX;
    double movetime = DBL_MAX;
    bool infinite = false;
  } go_command;

  typedef struct _go_perft_command {
    int16_t depth;
  } go_perft_command;

  void join_engine_thread() {
    if(job_started == true) {
      while(!engine_thread.joinable())
        ;
      engine_thread.join();
      str::pdebug("joining engine thread"s);
      job_started = false;
    }
  }

  std::string to_string(const std::vector<std::string> &cmd) {
    std::string s = "{"s;
    for(size_t i=0;i<cmd.size();++i) {
      s += "\""s + cmd[i] + "\""s;
      if(i != cmd.size() - 1) {
        s += ", "s;
      }
    }
    s += "}"s;
    return s;
  }

  void process_cmd(std::vector<std::string> cmd) {
    str::pdebug("processing cmd", to_string(cmd));
    while(cmdmap.find(cmd.front()) == std::end(cmdmap)) {
      str::perror("error: unknown command", cmd.front());
      cmd.erase(cmd.begin());
      if(cmd.empty()) {
        return;
      }
    }
    switch(cmdmap.at(cmd.front())) {
      case CMD_UCI:
        respond(RESP_ID, "name", "dummy_chess");
        respond(RESP_ID, "author", "$USER");
        respond(RESP_UCIOK);
      return;
      case CMD_DEBUG:
        if(cmd.size() < 2 || cmd[1] == "on"s) {
          debug_mode = true;
        } else if(cmd[1] == "off"s) {
          debug_mode = false;
        } else {
          str::perror("error: debug mode unknown '"s, cmd[1], "'"s);
        }
      return;
      case CMD_ISREADY:
        respond(RESP_READYOK);
      return;
      case CMD_SETOPTION:
        // no options
      return;
      case CMD_REGISTER:return;
      case CMD_UCINEWGAME:
        {
          join_engine_thread();
          destroy();
        }
      return;
      case CMD_POSITION:
      {
        lock_guard guard(engine_mtx);
        join_engine_thread();
        fen::FEN f;
        size_t ind = 1;
        if(cmd[ind] == "startpos"s) {
          ++ind;
          f = fen::starting_pos;
        } else if(cmd[ind] == "fen"s) {
          ++ind;
          const size_t begin_ind = ind;
          for(; ind < cmd.size() && cmd[ind] != "moves"s; ++ind) {
            ;
          }
          std::string s = str::join(std::vector<std::string>(cmd.begin() + begin_ind, cmd.begin() + ind), " "s);
          f = fen::load_from_string(s);
          str::pdebug("loaded fen", fen::export_as_string(f));
        }
        init(f);
        if(ind < cmd.size() && cmd[ind++] == "moves"s) {
          MoveLine moves;
          for(; ind < cmd.size(); ++ind) {
            moves.put(scan_move(cmd[ind]));
          }
          str::pdebug("moves"s, engine->_line_str(moves, true));
          for(const auto m : moves) {
            engine->make_move(m);
          }
        }
        str::pdebug("position set");
        //engine->print();
      }
      return;
      case CMD_GO:
      {
        join_engine_thread();
        should_stop = false;
        // perft command
        size_t ind = 1;
        if(cmd[ind] == "perft"s) {
          ++ind;
          if(cmd.size() != 3) {
            str::perror("error: unknown subcommand"s, cmd[ind]);
            return;
          }
          go_perft_command g = (go_perft_command){
            .depth = int16_t(atoi(cmd[ind].c_str()))
          };
          job_started = true;
          engine_thread = std::thread([&](auto g) mutable -> void {
            perform_go_perft(g);
          }, g);
          return;
        } else {
          go_command g;
          while(ind < cmd.size()) {
            if(cmd[ind] == "searchmoves"s) {
              ++ind;
              for(; ind < cmd.size(); ++ind) {
                g.searchmoves.emplace_back(scan_move(cmd[ind]));
              }
            } else if(cmd[ind] == "ponder"s) {
              g.ponder = true;
            } else if(cmd[ind] == "wtime"s) {
              g.wtime = double(atol(cmd[++ind].c_str()))*1e-3;
            } else if(cmd[ind] == "btime"s) {
              g.btime = double(atol(cmd[++ind].c_str()))*1e-3;
            } else if(cmd[ind] == "winc"s) {
              g.winc = double(atol(cmd[++ind].c_str()))*1e-3;
            } else if(cmd[ind] == "binc"s) {
              g.binc = double(atol(cmd[++ind].c_str()))*1e-3;
            } else if(cmd[ind] == "movestogo"s) {
              g.movestogo = atoi(cmd[++ind].c_str());
            } else if(cmd[ind] == "depth"s) {
              g.depth = atoi(cmd[++ind].c_str());
            } else if(cmd[ind] == "nodes"s) {
              g.nodes = atoll(cmd[++ind].c_str());
            } else if(cmd[ind] == "mate"s) {
              g.mate = atoi(cmd[++ind].c_str());
            } else if(cmd[ind] == "movetime"s) {
              g.movetime = double(atol(cmd[++ind].c_str()))*1e-3;
            } else if(cmd[ind] == "infinite"s) {
              g.infinite = true;
            } else {
              str::perror("error: unknown subcommand"s, cmd[ind]);
              return;
            }
            ++ind;
          }
          job_started = true;
          engine_thread = std::thread([&](auto g) mutable -> void {
            perform_go(g);
          }, g);
        }
      }
      return;
      case CMD_STOP:
      {
        should_stop = true;
      }
      return;
      case CMD_PONDERHIT:
        //TODO
      return;
      case CMD_QUIT:
      {
        perform_quit();
      }
      return;
      default:
        str::perror("error: unknown command");
      return;
    }
  }

  template <typename TimeT>
  INLINE size_t update_nodes_per_second(const TimeT &start, double &time_spent, size_t &nodes_searched) {
    const double prev_time_spent = time_spent;
    const size_t prev_nodes_searched = nodes_searched;
    time_spent = 1e-9*duration_cast<nanoseconds>(system_clock::now()-start).count();
    nodes_searched = engine->nodes_searched;
    const double nps = double(nodes_searched - prev_nodes_searched) / (time_spent - prev_time_spent);
    return nps;
  }

  INLINE bool check_if_should_stop(const go_command &args, double time_spent, double time_to_use) const {
    return (
        should_stop
        || (!args.infinite && time_spent >= args.movetime)
        || engine->nodes_searched >= args.nodes
        || time_spent >= time_to_use
    );
  }

  std::string get_score_type_string(double score) const {
    std::string s = ""s;
    if(!engine->score_is_mate(score)) {
      s += "cp"s;
    } else {
      s += "mate"s;
    }
    s += " "s;
    if(!engine->score_is_mate(score)) {
      s += std::to_string(int(round(score * 1e2)));
    } else {
      s += std::to_string(engine->score_mate_in(score) / 2 + 1);
    }
    return s;
  }

  void perform_go(go_command args) {
    lock_guard guard(engine_mtx);
    _printf("GO COMMAND\n");
    _printf("ponder: %d\n", args.ponder ? 1 : 0);
    _printf("wtime: %.6f, btime: %.6f\n", args.wtime, args.btime);
    _printf("winc: %.6f, binc: %.6f\n", args.winc, args.binc);
    _printf("movestogo: %d\n", args.movestogo);
    _printf("depth: %hd\n", args.depth);
    _printf("nodes: %lu\n", args.nodes);
    _printf("mate: %lu\n", args.mate);
    _printf("movetime: %.6f\n", args.movetime);
    _printf("infinite: %d\n", args.infinite ? 1 : 0);
    std::vector<std::string> searchmoves_s;
    for(auto&s:args.searchmoves)searchmoves_s.emplace_back(engine->_move_str(s));
    std::string s = str::join(searchmoves_s, ", "s);
    _printf("searchmoves: [%s]\n", s.c_str());
    // TODO ponder, mate, movestogo
    // time management
    double inctime = (engine->activePlayer() == WHITE) ? args.winc : args.binc;
    inctime = std::max(inctime - 2., inctime * .5);
    double tottime = engine->activePlayer() == WHITE ? args.wtime : args.btime;
    tottime = std::max(std::max(tottime - 2., (tottime - .2) * .7), tottime * .5);
    double movetime = std::min(60., tottime / 40. + inctime);
    if(args.movetime!=DBL_MAX)movetime=std::max(args.movetime - 1., args.movetime * .5);
    // time/note counting
    const auto start = system_clock::now();
    MoveLine currline;
    double time_spent = 0.;
    size_t nodes_searched = 0;
    const std::unordered_set<move_t> searchmoves(args.searchmoves.begin(), args.searchmoves.end());
    move_t pondermove = board::nomove;
    const move_t bestmove = engine->get_fixed_depth_move_iddfs(args.depth,
      [&](int16_t depth, move_t currmove, double curreval, const MoveLine &pline, move_t ponder_m) mutable -> bool {
        const size_t nps = update_nodes_per_second(start, time_spent, nodes_searched);
        currline.replace_line(pline);
        pondermove = ponder_m;
        const double hit_rate = double(engine->zb_hit) / double(1e-9+engine->zb_hit + engine->zb_miss);
        respond(RESP_INFO, "depth"s, depth,
                           "seldepth"s, pline.size(),
                           "nodes"s, engine->nodes_searched,
                           "nps"s, size_t(nps),
                           "currmove"s, engine->_move_str(currmove),
                           "score"s, get_score_type_string(curreval),
                           "pv"s, engine->_line_str(pline, true),
                           "time"s, int(round(time_spent * 1e3)),
                           "hashfull"s, int(round(hit_rate * 1e3))
        );
        return !check_if_should_stop(args, time_spent, movetime);
      }, searchmoves);
    should_stop = false;
    if(bestmove != board::nomove) {
      const double hit_rate = double(engine->zb_hit) / double(1e-9+engine->zb_hit + engine->zb_miss);
      respond(RESP_INFO, "seldepth"s, currline.size(),
                         "nodes"s, engine->nodes_searched,
                         "currmove"s, engine->_move_str(bestmove),
                         "score"s, get_score_type_string(engine->evaluation),
                         "pv"s, engine->_line_str(currline, true),
                         "time"s, int(round(time_spent * 1e3)),
                         "hashfull"s, int(round(hit_rate * 1e3))
      );
      if(pondermove != board::nomove) {
        respond(RESP_BESTMOVE, engine->_move_str(bestmove), "ponder"s, engine->_move_str(pondermove));
      } else {
        respond(RESP_BESTMOVE, engine->_move_str(bestmove));
      }
    }
    str::pdebug("NOTE: search is over");
    should_stop = true;
  }

  void perform_go_perft(go_perft_command args) {
    lock_guard guard(engine_mtx);
    const int16_t depth = args.depth;
    size_t total = 0;
    {
      decltype(auto) store_scope = engine->get_zobrist_perft_scope();
      engine->iter_moves([&](pos_t i, pos_t j) mutable -> void {
        const move_t m = bitmask::_pos_pair(i, j);
        event_t ev = engine->get_move_event(i, j);
        std::string sm = engine->_move_str(m);
        engine->make_move(m);
        size_t nds = 0;
        if(depth > 1) {
          nds = engine->perft(depth-1);
        } else if(depth == 1) {
          nds = 1;
        } else {
          nds = 0;
        }
        str::print(sm + ":", nds);
        total += nds;
        engine->retract_move();
      });
    }
    str::print();
    str::print("Nodes searched:"s, total);
    str::print();
  }

  void perform_quit() {
    should_stop = true;
    join_engine_thread();
    should_quit = true;
    str::pdebug("job state", job_started);
  }

  template <typename... Str>
  void respond(RESPONSE resp, Str &&... args) const {
    str::print(respmap.at(resp), std::forward<Str>(args)...);
  }

  ~UCI() {
    str::pdebug("job state", job_started);
    join_engine_thread();
    destroy();
  }
};
