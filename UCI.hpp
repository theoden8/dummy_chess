#pragma once


#include <cfloat>

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <optional>
#include <mutex>
#include <atomic>

#include <FEN.hpp>
#include <Engine.hpp>


using namespace std::chrono;


struct UCI {
  Engine *engine = nullptr;
  typename Engine::ab_storage_t *engine_ab_storage = nullptr;
  Engine::iddfs_state engine_idstate;

  std::atomic<bool> debug_mode = false;
  std::atomic<bool> should_quit = false;
  std::atomic<bool> should_stop = false;
  std::atomic<bool> should_ponderhit = false;
  std::atomic<bool> job_started = false;

  struct Options {
    size_t hash_mb = 64;
    bool chess960 = false;
    bool crazyhouse = false;
  };
  Options engine_options;

  std::recursive_mutex engine_mtx;
  using lock_guard = std::lock_guard<std::recursive_mutex>;
  std::thread engine_thread;

  UCI()
  {}

  void init(const fen::FEN &f) {
    lock_guard guard(engine_mtx);
    destroy();
    _printf("init\n");
    const size_t zobrist_size = (engine_options.hash_mb << (20 - 7));
    engine = new Engine(f, zobrist_size);
    engine_ab_storage = new typename Engine::ab_storage_t(engine->get_zobrist_alphabeta_scope());
  }

  void destroy() {
    lock_guard guard(engine_mtx);
    if(engine != nullptr) {
      _printf("destroy\n");
      delete engine_ab_storage;
      engine_ab_storage = nullptr;
      delete engine;
      engine = nullptr;
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
    CMD_DISPLAY,
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
    {"display"s,   CMD_DISPLAY},
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
    RESP_DISPLAY,
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
    {RESP_DISPLAY,        "display"s},
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
    should_ponderhit = false;
    should_stop = true;
  }

  static move_t scan_move(const std::string &s) {
    assert(s.length() == 4 || s.length() == 5);
    pos_t i = 0x00;
    if(s[1] == '@') {
      switch(toupper(s[0])) {
        case 'P':i=board::DROP_PAWN;break;
        case 'N':i=board::DROP_KNIGHT;break;
        case 'B':i=board::DROP_BISHOP;break;
        case 'R':i=board::DROP_ROOK;break;
        case 'Q':i=board::DROP_QUEEN;break;
        default:abort();
      }
    } else {
      const pos_t i_file = tolower(s[0]) - 'a';
      const pos_t i_rank = s[1] - '0';
      i = board::_pos(A+i_file, i_rank);
    }
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
    should_stop = true;
    if(job_started == true) {
      while(!engine_thread.joinable())
        ;
      engine_thread.join();
      str::pdebug("joining engine thread"s);
      job_started = false;
    }
  }

  std::string to_string(const std::vector<std::string> &cmd) {
    return "{"s + str::join(cmd, ", "s) + "}"s;
  }

  std::map<std::string, bool> boolOptions = {
    {"UCI_Chess960"s, false},
    {"Ponder"s, false},
  };

  std::map<std::string, std::tuple<int, int, int64_t>> spinOptions = {
    {"Hash"s, std::make_tuple(4, 4096, engine_options.hash_mb)},
  };

  std::map<std::string, std::pair<std::vector<std::string>, std::string>> comboOptions = {
    {"UCI_Variant"s, std::make_pair(std::vector<std::string>{"chess"s, "crazyhouse"s}, "chess"s)},
  };

  void process_cmd(std::vector<std::string> cmd) {
    str::pdebug("processing cmd", to_string(cmd));
    while(!cmdmap.contains(cmd.front())) {
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
        for(const auto &[name, dflt] : boolOptions) {
          respond(RESP_OPTION, "name"s, name, "type"s, "check"s, "default"s, dflt ? "true"s : "false"s);
        }
        for(const auto &[name, args] : spinOptions) {
          const auto &[lo, hi, dflt] = args;
          respond(RESP_OPTION, "name"s, name, "type"s, "spin"s, "default"s, dflt, "min"s, lo, "max", hi);
        }
        for(const auto &[name, args] : comboOptions) {
          const auto &[vals, dflt] = args;
          respond(RESP_OPTION, "name"s, name, "type"s, "combo"s, "default", dflt, "var "s + str::join(vals, " var "s));
        }
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
      {
        join_engine_thread();
        respond(RESP_READYOK);
      }
      return;
      case CMD_SETOPTION:
      {
        std::optional<std::string> optname;
        std::optional<std::string> optvalue;
        for(size_t i = 1; i < cmd.size(); ++i) {
          if(cmd[i] == "name"s && cmd.size() > i + 1) {
            optname.emplace(cmd[++i]);
            str::pdebug("name:", optname.value());
          } else if(cmd[i] == "value"s && cmd.size() > i + 1) {
            optvalue.emplace(cmd[++i]);
            str::pdebug("value:", optvalue.value());
          }
        }
        if(optname.has_value() && optvalue.has_value()) {
          if(boolOptions.contains(optname.value())) {
            bool val = boolOptions.at(optname.value());
            if(optvalue.value() == "true"s) {
              val = true;
              str::pdebug("set option", optname.value(), "true"s);
            } else if(optvalue.value() == "false"s) {
              val = false;
              str::pdebug("set option", optname.value(), "false"s);
            } else {
              str::perror("error: unknown optvalue", optvalue.value(), "for option", optname.value());
            }
            if(optname.value() == "UCI_Chess960"s) {
              engine_options.chess960 = val;
            }
          } else if(spinOptions.contains(optname.value())) {
            auto &[_lo, _hi, val] = spinOptions.at(optname.value());
            int v = atoi(optvalue.value().c_str());
            val = std::min(_hi, std::max(_lo, v));
            str::pdebug("set option", optname.value(), val);
          } else if(comboOptions.contains(optname.value())) {
            auto &[_vals, val] = comboOptions.at(optname.value());
            if(std::find(_vals.begin(), _vals.end(), optvalue.value()) == std::end(_vals)) {
              str::perror("error: unknown optvalue", optvalue.value(), "for option", optname.value());
            }
            if(optvalue.value() == "chess") {
              engine_options.crazyhouse = false;
            } else if(optvalue.value() == "crazyhouse") {
              engine_options.crazyhouse = true;
            }
            str::pdebug("set option", optname.value(), val);
          }
        } else {
          str::perror("error: unknown option name or value");
        }
      }
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
        join_engine_thread();
        fen::FEN f;
        size_t ind = 1;
        if(cmd[ind] == "startpos"s) {
          ++ind;
          if(engine_options.crazyhouse) {
            f = fen::starting_pos_ch;
          } else {
            f = fen::starting_pos;
          }
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
      case CMD_DISPLAY:
      {
        join_engine_thread();
        const fen::FEN f = engine->export_as_fen();
        respond(RESP_DISPLAY, "fen:"s, fen::export_as_string(f));
        const double hashfull = double(engine->zb_occupied) / double(engine->zobrist_size);
        const double hit_rate = double(engine->zb_hit) / double(1e-9 + engine->zb_hit + engine->zb_miss);
        respond(RESP_DISPLAY, "stat_hashfull:"s, hashfull);
        respond(RESP_DISPLAY, "stat_hit_rate:"s, hit_rate);
        respond(RESP_DISPLAY, "stat_nodes_searched:"s, engine->nodes_searched);
      }
      return;
      case CMD_GO:
      {
        join_engine_thread();
        should_stop = false;
        // perft command
        size_t ind = 1;
        if(cmd.size() > ind && cmd[ind] == "perft"s) {
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
        join_engine_thread();
      }
      return;
      case CMD_PONDERHIT:
      {
        should_ponderhit = true;
      }
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

  std::string get_score_type_string(int32_t score) const {
    std::string s = ""s;
    if(!engine->score_is_mate(score)) {
      s += "cp"s;
    } else {
      s += "mate"s;
    }
    s += " "s;
    if(!engine->score_is_mate(score)) {
      s += std::to_string(score / Engine::CENTIPAWN);
    } else {
      int16_t mate_in_ply = engine->score_mate_in(score);
      mate_in_ply -= (mate_in_ply < 0) ? 1 : -1;
      s += std::to_string(mate_in_ply / 2);
    }
    return s;
  }

  double time_control_movetime(const go_command &args, bool pondering) const {
    if(args.infinite || pondering) {
      str::pdebug(args.infinite, pondering);
      return DBL_MAX;
    } else if(args.movetime != DBL_MAX) {
      return std::max(args.movetime - 1., args.movetime * .5);
    }
    const COLOR c = engine->activePlayer();
    double inctime = (c == WHITE) ? args.winc : args.binc;
    inctime = std::max(inctime - 2., inctime * .3);
    double tottime = (c == WHITE) ? args.wtime : args.btime;
    tottime = std::max(std::max(tottime - 2., (tottime - .2) * .7), tottime * .5);
    return std::min(3600., tottime / 40. + inctime);
  }

  void respond_full_iddfs(const Engine::iddfs_state &engine_idstate, size_t nps, double time_spent) {
    const double hashfull = double(engine->zb_occupied) / double(engine->zobrist_size);
    respond(RESP_INFO, "depth"s, engine_idstate.curdepth,
                       "seldepth"s, engine_idstate.pline.size(),
                       "nodes"s, engine->nodes_searched,
                       "nps"s, size_t(nps),
                       "score"s, get_score_type_string(engine_idstate.eval),
                       "pv"s, engine->_line_str(engine_idstate.pline, true),
                       "time"s, int(round(time_spent * 1e3)),
                       "hashfull"s, int(round(hashfull * 1e3)));
  }

  void respond_final_iddfs(const Engine::iddfs_state &engine_idstate, move_t bestmove, double time_spent) {
    const double hashfull = double(engine->zb_occupied) / double(engine->zobrist_size);
    respond(RESP_INFO, "depth"s, engine_idstate.curdepth,
                       "seldepth"s, engine_idstate.pline.size(),
                       "nodes"s, engine->nodes_searched,
                       "score"s, get_score_type_string(engine_idstate.eval),
                       "pv"s, engine->_line_str(engine_idstate.pline, true),
                       "time"s, int(round(time_spent * 1e3)),
                       "hashfull"s, int(round(hashfull * 1e3)));
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
    // TODO mate, movestogo
    double movetime = time_control_movetime(args, args.ponder);
    const auto start = system_clock::now();
    double time_spent = 0.;
    size_t nodes_searched = 0;
    const std::unordered_set<move_t> searchmoves(args.searchmoves.begin(), args.searchmoves.end());
    if(!args.ponder) {
      engine_idstate.reset();
    } else if(engine_idstate.pline.size() >= 2) {
      engine_idstate.ponderhit();
      engine_idstate.ponderhit();
    }
    bool pondering = args.ponder;
    bool return_from_search = false;
    const move_t bestmove = engine->start_thinking(args.depth, engine_idstate, [&](bool verbose) mutable -> bool {
      if(engine_idstate.pline.empty()) {
        return true;
      } else if(return_from_search) {
        return false;
      }
      if(pondering && movetime > 1e9 && should_ponderhit) {
        pondering = false;
        const double new_movetime = time_control_movetime(args, false);
        movetime = std::max(new_movetime / 3, new_movetime - time_spent / 4);
        str::pdebug("changed time", movetime);
      }
      return_from_search = return_from_search || check_if_should_stop(args, time_spent, movetime);
      const size_t nps = update_nodes_per_second(start, time_spent, nodes_searched);
			if(verbose && !pondering && !should_stop) {
        respond_full_iddfs(engine_idstate, nps, time_spent);
			}
      return !return_from_search;
    }, searchmoves);
		while(pondering && !should_stop) {
      if(should_ponderhit) {
        pondering = false;
      }
    }
    if(!pondering && !should_stop) {
      respond_final_iddfs(engine_idstate, bestmove, time_spent);
    }
		if(engine_idstate.pondermove() != board::nullmove) {
			respond(RESP_BESTMOVE, engine->_move_str(bestmove), "ponder"s, engine->_move_str(engine_idstate.pondermove()));
		} else {
			respond(RESP_BESTMOVE, engine->_move_str(bestmove));
		}
    str::pdebug("NOTE: search is over");
    should_stop = true;
    should_ponderhit = false;
  }

  void perform_go_perft(go_perft_command args) {
    lock_guard guard(engine_mtx);
    const int16_t depth = args.depth;
    size_t total = 0;
    {
      decltype(auto) store_scope = engine->get_zobrist_perft_scope();
      engine->iter_moves([&](pos_t i, pos_t j) mutable -> void {
        const move_t m = bitmask::_pos_pair(i, j);
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
