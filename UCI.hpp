#pragma once


#include <cfloat>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <optional>
#include <memory>
#include <filesystem>

#include <unistd.h>
#include <fcntl.h>

#include <FEN.hpp>
#include <Engine.hpp>


using namespace std::chrono;


struct UCI {
  // engine singleton
  std::unique_ptr<Engine> engine_ptr;
  // must take ownership of ttables and such so that initialization
  // doesn't slow the engine down every time "go" is called
  std::unique_ptr<typename Engine::ab_storage_t> engine_ab_storage_ptr;
  // last IDDFS state (for ponderhit)
  Engine::iddfs_state engine_idstate;

  using score_t = typename Engine::score_t;
  using depth_t = typename Engine::depth_t;

  // state-changing message passing
  bool debug_mode = false;
  bool should_quit = false;
  bool should_stop = false;
  bool should_ponderhit = false;

  // these are used to initialize engine for a new game
  struct Options {
    size_t hash_mb = 64;
    bool chess960 = false;
    bool crazyhouse = false;
    std::optional<std::string> syzygy_path;
    bool tb_initialized = false;
  };
  Options engine_options;

  UCI()
  {}

  // nice lock-safe initialization according to the currently set options
  void init(const fen::FEN &f) {
    _printf("init\n");
    const size_t zobrist_size = (engine_options.hash_mb << (20 - 7));
    engine_ptr.reset(new Engine(f, zobrist_size));
    if(engine_options.syzygy_path.has_value()) {
      bool res = tb::init(engine_options.syzygy_path.value());
      if(!res) {
        str::perror("error: tb::init failed with syzygy_path = <" + engine_options.syzygy_path.value() + ">");
        abort();
      }
      engine_options.tb_initialized = true;
    }
    engine_ab_storage_ptr.reset(new Engine::ab_storage_t(engine_ptr->get_zobrist_alphabeta_scope()));
  }

  // remove engine: no memory leaks, no nothing, ready to start again
  void destroy() {
    if(!engine_ptr) {
      _printf("destroy\n");
      engine_ab_storage_ptr.reset(nullptr);
      engine_ptr.reset(nullptr);
      if(engine_options.tb_initialized && !engine_options.syzygy_path.has_value()) {
        tb::free();
      }
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

  using cmd_t = std::vector<std::string>;
  cmd_t io_cmdbuf = {""s};
  std::queue<cmd_t> q_cmd;

  bool is_nonblocking_io() {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    return (flags & O_NONBLOCK);
  }

  void set_nonblocking_io() {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
  }

  bool continue_read_cmd(bool main_loop) {
    assert(is_nonblocking_io());
    if(should_quit) {
      return false;
    }
    char c = '\0';
    while(1) {
      int ret = read(STDIN_FILENO, &c, 1);
      if(ret == -1) {
        assert(errno == EAGAIN);
        break;
      }
      if(c == EOF) {
        auto cmd = io_cmdbuf;
        io_cmdbuf.clear();
        if(cmd.size() > 1 || !cmd.back().empty()) {
          process_cmd(cmd, main_loop);
        }
        should_ponderhit = false;
        should_stop = true;
        return false;
      }
      if(c == '\n' || c == '\r') {
        auto cmd = io_cmdbuf;
        io_cmdbuf = {""s};
        if(cmd.size() > 1 || !cmd.back().empty()) {
          process_cmd(cmd, main_loop);
        }
        if(should_quit) {
          return false;
        }
      }
      if(isspace(c) && io_cmdbuf.empty()) {
        continue;
      } else if(!isspace(c)) {
        io_cmdbuf.back() += c;
      } else if(isspace(c) && !io_cmdbuf.back().empty()) {
        io_cmdbuf.emplace_back(""s);
      }
    }
    return true;
  }

  // accept and interpret commands line by line, for all eternity
  void run() {
    set_nonblocking_io();
    char c = '\0';
    while(1) {
      while(!q_cmd.empty()) {
        auto cmd = q_cmd.front();
        q_cmd.pop();
        process_cmd(cmd, true);
      }
      if(!continue_read_cmd(true)) {
        break;
      }
      usleep(1000);
    }
  }

  // read individual algebraic notation move like
  // d3f4 or d7e8q or P@c6
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

  // fully parametrized go-command
  typedef struct _go_command {
    std::vector<move_t> searchmoves = {};
    bool ponder = false;
    double wtime = DBL_MAX;
    double btime = DBL_MAX;
    double winc = 0;
    double binc = 0;
    int movestogo = 0;
    depth_t depth = INT16_MAX;
    size_t nodes = SIZE_MAX;
    size_t mate = SIZE_MAX;
    double movetime = DBL_MAX;
    bool infinite = false;
  } go_command;

  // same but only for perft, this is not in the UCI spec
  typedef struct _go_perft_command {
    depth_t depth;
  } go_perft_command;

  std::string to_string(const cmd_t &cmd) {
    return "{"s + str::join(cmd, ", "s) + "}"s;
  }

  // bunch of options
  const std::map<std::string, bool> boolOptions = {
    {"UCI_Chess960"s, false},
    {"Ponder"s, false},
  };

  const std::map<std::string, std::tuple<int, int, int>> spinOptions = {
    {"Hash"s, std::make_tuple(4, 4096, (int)engine_options.hash_mb)},
  };

  const std::map<std::string, std::pair<cmd_t, std::string>> comboOptions = {
    {"UCI_Variant"s, std::make_pair(cmd_t{"chess"s, "crazyhouse"s}, "chess"s)},
  };

  const std::map<std::string, std::string> stringOptions {
    {"SyzygyPath"s, "<empty>"s},
  };

  // tell engine to stop; wait until it stops if it's running; clean up
  void sched_stop_engine() {
    should_stop = true;
  }

  void sched_cmd(const cmd_t &cmd) {
    q_cmd.emplace(cmd);
  }

  void process_cmd(cmd_t &cmd, bool main_loop) {
    if(!main_loop) {
      assert(engine_ptr);
    }
    assert(engine_ptr || main_loop);
    str::pdebug("info string processing cmd", to_string(cmd));
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
        respond(RESP_ID, "author", "theoden8");
        for(const auto &[name, dflt] : boolOptions) {
          respond(RESP_OPTION, "name"s, name, "type"s, "check"s, "default"s, dflt ? "true"s : "false"s);
        }
        for(const auto &[name, args] : spinOptions) {
          const auto &[lo, hi, dflt] = args;
          respond(RESP_OPTION, "name"s, name, "type"s, "spin"s, "default"s, dflt, "min"s, lo, "max", hi);
        }
        for(const auto &[name, args] : comboOptions) {
          const auto &[vals, dflt] = args;
          respond(RESP_OPTION, "name"s, name, "type"s, "combo"s, "default"s, dflt, "var "s + str::join(vals, " var "s));
        }
        for(const auto &[name, dflt] : stringOptions) {
          respond(RESP_OPTION, "name"s, name, "type"s, "string"s, "default"s, dflt);
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
        if(!main_loop) {
          sched_stop_engine();
          sched_cmd(cmd);
          return;
        }
        respond(RESP_READYOK);
      }
      return;
      case CMD_SETOPTION:
      {
        if(!main_loop) {
          sched_cmd(cmd);
          return;
        }
        std::optional<std::string> optname;
        std::optional<std::string> optvalue;
        for(size_t i = 1; i < cmd.size(); ++i) {
          if(cmd[i] == "name"s && cmd.size() > i + 1) {
            optname.emplace(cmd[++i]);
            str::pdebug("info string name:", optname.value());
          } else if(cmd[i] == "value"s && cmd.size() > i + 1) {
            cmd_t cmd_value = cmd_t(cmd.begin() + i + 1, cmd.end());
            optvalue.emplace(str::join(cmd_value, " "s));
            str::pdebug("info string value:", optvalue.value());
            break;
          }
        }
        if(optname.has_value() && optvalue.has_value()) {
          if(boolOptions.contains(optname.value())) {
            const bool &dflt = boolOptions.at(optname.value());
            bool val = dflt;
            if(optvalue.value() == "true"s) {
              val = true;
            } else if(optvalue.value() == "false"s) {
              val = false;
            } else {
              str::perror("error: unknown optvalue", optvalue.value(), "for option", optname.value());
              abort();
            }
            if(optname.value() == "UCI_Chess960"s) {
              engine_options.chess960 = val;
              str::pdebug("info string setoption UCI_Chess960 =", optvalue.value());
            } else {
              str::pdebug("info string unknown option", optname.value());
            }
          } else if(spinOptions.contains(optname.value())) {
            const auto &[_lo, _hi, _dflt] = spinOptions.at(optname.value());
            int v = atoi(optvalue.value().c_str());
            int val = std::min(_hi, std::max(_lo, v));
            if(optname.value() == "Hash"s) {
              engine_options.hash_mb = val;
              str::pdebug("info string setoption Hash =", optname.value(), val);
            } else {
              str::pdebug("info string unknown option", optname.value());
            }
          } else if(comboOptions.contains(optname.value())) {
            const auto &[_vals, _dflt] = comboOptions.at(optname.value());
            if(std::find(_vals.begin(), _vals.end(), optvalue.value()) == std::end(_vals)) {
              str::perror("error: unknown optvalue", optvalue.value(), "for option", optname.value());
            }
            if(optname.value() == "UCI_Variant"s) {
              if(optvalue.value() == "chess"s) {
                engine_options.crazyhouse = false;
              } else if(optvalue.value() == "crazyhouse"s) {
                engine_options.crazyhouse = true;
              }
              str::pdebug("info string setoption UCI_Variant =", optname.value());
            } else {
              str::pdebug("info string unknown option", optname.value());
            }
          } else if(stringOptions.contains(optname.value())) {
            const auto &_dflt = stringOptions.at(optname.value());
            if(optname.value() == "SyzygyPath"s) {
              if(optvalue.value() != _dflt) {
                std::filesystem::path syzygy_path(optvalue.value());
                if(!std::filesystem::exists(syzygy_path)) {
                  str::perror("syzygy path doesn't exist <", syzygy_path, ">");
                  abort();
                }
                engine_options.syzygy_path.emplace(optvalue.value());
              } else {
                engine_options.syzygy_path.reset();
              }
              str::pdebug("info string setoption SyzygyPath =", optvalue.value());
            } else {
              str::pdebug("info string unknown option", optname.value());
            }
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
          if(!main_loop) {
            sched_stop_engine();
            sched_cmd(cmd);
            return;
          }
          destroy();
        }
      return;
      case CMD_POSITION:
      {
        if(!main_loop) {
          sched_stop_engine();
          sched_cmd(cmd);
          return;
        }
        fen::FEN f;
        size_t ind = 1;
        assert(ind < cmd.size());
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
          std::string s = str::join(cmd_t(cmd.begin() + begin_ind, cmd.begin() + ind), " "s);
          f = fen::load_from_string(s);
          str::pdebug("info string loaded fen", fen::export_as_string(f));
        }
        f.chess960 = engine_options.chess960;
        init(f);
        if(ind < cmd.size() && cmd[ind++] == "moves"s) {
          MoveLine moves;
          for(; ind < cmd.size(); ++ind) {
            moves.put(scan_move(cmd[ind]));
          }
          str::pdebug("info string moves"s, engine_ptr->_line_str(moves, true));
          for(const auto m : moves) {
            engine_ptr->make_move(m);
          }
        }
        str::pdebug("info string position set");
        //engine_ptr->print();
      }
      return;
      case CMD_DISPLAY:
      {
        if(!main_loop) {
          sched_cmd(cmd);
          return;
        }
        if(!engine_ptr) {
          cmd_t forwarded_cmd{"position"s, "startpos"s};
          process_cmd(forwarded_cmd, main_loop);
        }
        const fen::FEN f = engine_ptr->export_as_fen();
        respond(RESP_DISPLAY, "fen:"s, fen::export_as_string(f));
        const double hashfull = double(engine_ptr->zb_occupied) / double(engine_ptr->zobrist_size);
        const double hit_rate = double(engine_ptr->zb_hit) / double(1e-9 + engine_ptr->zb_hit + engine_ptr->zb_miss);
        respond(RESP_DISPLAY, "stat_hashfull:"s, hashfull);
        respond(RESP_DISPLAY, "stat_hit_rate:"s, hit_rate);
        respond(RESP_DISPLAY, "stat_tb_hits:"s, engine_ptr->tb_hit);
        respond(RESP_DISPLAY, "stat_nodes_searched:"s, engine_ptr->nodes_searched);
      }
      return;
      case CMD_GO:
      {
        if(!main_loop) {
          sched_stop_engine();
          sched_cmd(cmd);
          return;
        }
        assert(engine_ptr);
        // perft command
        size_t ind = 1;
        if(cmd.size() > ind && cmd[ind] == "perft"s) {
          ++ind;
          if(cmd.size() != 3) {
            str::perror("error: unknown subcommand"s, cmd[ind]);
            return;
          }
          go_perft_command g = (go_perft_command){
            .depth = depth_t(atoi(cmd[ind].c_str()))
          };
          perform_go_perft(g);
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
          perform_go(g);
        }
      }
      return;
      case CMD_STOP:
      {
        sched_stop_engine();
      }
      return;
      case CMD_PONDERHIT:
      {
        if(!main_loop) {
          sched_cmd(cmd);
          return;
        }
        should_ponderhit = true;
      }
      return;
      case CMD_QUIT:
      {
        if(!main_loop) {
          sched_stop_engine();
          sched_cmd(cmd);
          return;
        }
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
    nodes_searched = engine_ptr->nodes_searched;
    const double nps = double(nodes_searched - prev_nodes_searched) / (time_spent - prev_time_spent);
    return nps;
  }

  INLINE bool check_if_should_stop(const go_command &args, double time_spent, double time_to_use) const {
    return (
        should_stop
        || (!args.infinite && time_spent >= args.movetime)
        || engine_ptr->nodes_searched >= args.nodes
        || time_spent >= time_to_use
    );
  }

  std::string get_score_type_string(score_t score) const {
    std::string s = ""s;
    if(!engine_ptr->score_is_mate(score)) {
      s += "cp"s;
    } else {
      s += "mate"s;
    }
    s += " "s;
    if(!engine_ptr->score_is_mate(score)) {
      s += std::to_string(score / Engine::CENTIPAWN);
    } else {
      depth_t mate_in_ply = engine_ptr->score_mate_in(score);
      mate_in_ply -= (mate_in_ply < 0) ? 1 : -1;
      s += std::to_string(mate_in_ply / 2);
    }
    return s;
  }

  double time_control_movetime(const go_command &args, bool pondering) const {
    if(args.infinite || pondering) {
      str::pdebug("info string", args.infinite, pondering);
      return DBL_MAX;
    } else if(args.movetime != DBL_MAX) {
      return std::max(args.movetime - 1., args.movetime * .5);
    }
    const COLOR c = engine_ptr->activePlayer();
    double inctime = (c == WHITE) ? args.winc : args.binc;
    inctime = std::max(inctime - 2., inctime * .3);
    double tottime = (c == WHITE) ? args.wtime : args.btime;
    tottime = std::max(std::max(tottime - 2., (tottime - .2) * .7), tottime * .5);
    return std::min(3600., tottime / 40. + inctime);
  }

  // intermediate responses while thinking. when pondering, this might confuse the GUI
  void respond_full_iddfs(const Engine::iddfs_state &engine_idstate, size_t nps, double time_spent) {
    const double hashfull = double(engine_ptr->zb_occupied) / double(engine_ptr->zobrist_size);
    respond(RESP_INFO, "depth"s, engine_idstate.curdepth,
                       "seldepth"s, engine_idstate.pline.size(),
                       "nodes"s, engine_ptr->nodes_searched,
                       "nps"s, size_t(nps),
                       "tb_hits", engine_ptr->tb_hit,
                       "score"s, get_score_type_string(engine_idstate.eval),
                       "pv"s, engine_ptr->_line_str(engine_idstate.pline, true),
                       "time"s, int(round(time_spent * 1e3)),
                       "hashfull"s, int(round(hashfull * 1e3)));
  }

  // when out of time, spit out this final response.
  void respond_final_iddfs(const Engine::iddfs_state &engine_idstate, move_t bestmove, double time_spent) {
    const double hashfull = double(engine_ptr->zb_occupied) / double(engine_ptr->zobrist_size);
    respond(RESP_INFO, "depth"s, engine_idstate.curdepth,
                       "seldepth"s, engine_idstate.pline.size(),
                       "nodes"s, engine_ptr->nodes_searched,
                       "tb_hits", engine_ptr->tb_hit,
                       "score"s, get_score_type_string(engine_idstate.eval),
                       "pv"s, engine_ptr->_line_str(engine_idstate.pline, true),
                       "time"s, int(round(time_spent * 1e3)),
                       "hashfull"s, int(round(hashfull * 1e3)));
  }

  void perform_go(go_command args) {
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
    for(auto&s:args.searchmoves)searchmoves_s.emplace_back(engine_ptr->_move_str(s));
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
    const move_t bestmove = engine_ptr->start_thinking(args.depth, engine_idstate, [&](bool verbose) mutable -> bool {
      continue_read_cmd( false);
      if(engine_idstate.pline.empty()) {
        return true;
      } else if(return_from_search) {
        return false;
      }
      const size_t nps = update_nodes_per_second(start, time_spent, nodes_searched);
      if(pondering && movetime > 1e9 && should_ponderhit) {
        pondering = false;
        const double new_movetime = time_control_movetime(args, false);
        movetime = std::max(new_movetime / 3, new_movetime - time_spent / 4);
        time_spent = .0;
        str::pdebug("info string changed time", movetime);
      }
      return_from_search = return_from_search || check_if_should_stop(args, time_spent, movetime);
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
      respond(RESP_BESTMOVE, engine_ptr->_move_str(bestmove), "ponder"s, engine_ptr->_move_str(engine_idstate.pondermove()));
    } else {
      respond(RESP_BESTMOVE, engine_ptr->_move_str(bestmove));
    }
    str::pdebug("info string NOTE: search is over");
    should_stop = true;
    should_ponderhit = false;
  }

  void perform_go_perft(go_perft_command args) {
    const depth_t depth = args.depth;
    size_t total = 0;
    {
      decltype(auto) store_scope = engine_ptr->get_zobrist_perft_scope();
      engine_ptr->iter_moves([&](pos_t i, pos_t j) mutable -> void {
        continue_read_cmd(false);
        const move_t m = bitmask::_pos_pair(i, j);
        std::string sm = engine_ptr->_move_str(m);
        engine_ptr->make_move(m);
        size_t nds = 0;
        if(depth > 1) {
          nds = engine_ptr->perft(depth-1);
        } else if(depth == 1) {
          nds = 1;
        } else {
          nds = 0;
        }
        str::print(sm + ":", nds);
        total += nds;
        engine_ptr->retract_move();
      });
    }
    str::print();
    str::print("Nodes searched:"s, total);
    str::print();
  }

  void perform_quit() {
    sched_stop_engine();
    should_quit = true;
    str::pdebug("info string should quit");
  }

  template <typename... Str>
  void respond(RESPONSE resp, Str &&... args) const {
    str::print(respmap.at(resp), std::forward<Str>(args)...);
  }

  ~UCI() {
    if(!should_quit) {
      perform_quit();
    }
    destroy();
  }
};
