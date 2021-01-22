#pragma once


#include <string>
#include <vector>
#include <map>

#include <FEN.hpp>
#include <Engine.hpp>


#ifndef NDEBUG
#define _printf printf
#else
#define _printf(X)
#endif


struct UCI {
  Engine *engine = nullptr;

  UCI()
  {}

  void init(const fen::FEN &f) {
    destroy();
    _printf("init\n");
    engine = new Engine(f);
  }

  void destroy() {
    if(engine != nullptr) {
      _printf("destroy\n");
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

  bool should_quit = false;
  bool should_stop = false;
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
  }

  void process_cmd(std::vector<std::string> cmd) {
    _printf("processing cmd {");
    for(size_t i=0;i<cmd.size();++i) {
      _printf("\"%s\"", cmd[i].c_str());
      if(i != cmd.size() - 1) {
        _printf(", ");
      }
    }
    _printf("}\n");
    while(cmdmap.find(cmd.front()) == std::end(cmdmap)) {
      cmd.erase(cmd.begin());
      if(cmd.empty()) {
        return;
      }
    }
    switch(cmdmap.at(cmd.front())) {
      case CMD_UCI:respond(RESP_UCIOK);return;
      case CMD_DEBUG:return;
      case CMD_ISREADY:
        init(fen::starting_pos);
        respond(RESP_READYOK);
      return;
      case CMD_SETOPTION:
        // no options
      return;
      case CMD_REGISTER:return;
      case CMD_UCINEWGAME:
        destroy();
      return;
      case CMD_POSITION:
        {
          fen::FEN f;
          size_t ind = 1;
          if(cmd[1] == "startpos"s) {
            ++ind;
            f = fen::starting_pos;
          } else {
            std::string s;
            for(; ind < cmd.size(); ++ind) {
              if(cmd[ind] == "moves")break;
              if(ind != 1)s += " ";
              s += cmd[ind];
            }
            f = fen::load_from_string(s);
          }
          init(f);
          if(ind < cmd.size() && cmd[ind] == "moves"s) {
            std::vector<move_t> moves;
            ++ind;
            for(; ind < cmd.size(); ++ind) {
              assert(cmd[ind].length() == 4 || cmd[ind].length() == 5);
              const pos_t i_file = tolower(cmd[ind][0]) - 'a';
              const pos_t i_rank = cmd[ind][1] - '0';
              const pos_t i = board::_pos(A+i_file, i_rank);
              const pos_t j_file = tolower(cmd[ind][2]) - 'a';
              const pos_t j_rank = cmd[ind][3] - '0';
              pos_t j = board::_pos(A+j_file, j_rank);
              if(cmd[ind].length() == 5) {
                switch(tolower(cmd[ind][4])) {
                  case 'k':j|=board::PROMOTE_KNIGHT;break;
                  case 'b':j|=board::PROMOTE_BISHOP;break;
                  case 'r':j|=board::PROMOTE_ROOK;break;
                  case 'q':j|=board::PROMOTE_QUEEN;break;
                  default:break;
                }
              }
              moves.emplace_back(bitmask::_pos_pair(i, j));
            }
            for(const auto m : moves) {
              engine->make_move(m);
            }
          }
          engine->print();
        }
      return;
      case CMD_GO:
        should_stop = false;
        //TODO
      return;
      case CMD_STOP:
        should_stop = true;
      return;
      case CMD_PONDERHIT:
        //TODO
      return;
      case CMD_QUIT:
        should_quit = true;
      return;
      default:return;
    }
  }

  void respond(RESPONSE resp, std::vector<std::string> args = {}) {
    _printf("%s", respmap.at(resp).c_str());
    for(auto &a:args)_printf(" %s", a.c_str());
    _printf("\n");
  }

  ~UCI() {
    destroy();
  }
};
