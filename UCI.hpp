#pragma once


#include <string>
#include <vector>
#include <map>

#include <FEN.hpp>
#include <Engine.hpp>


struct UCI {
  Engine *engine = nullptr;

  UCI()
  {}

  void init(const fen::FEN &f) {
    if(engine != nullptr)return;
    engine = new Engine(f);
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
        cmd.back() += tolower(c);
      } else if(isspace(c) && !cmd.back().empty()) {
        cmd.push_back(""s);
      }
    }
    if(cmd.size() > 1 || !cmd.back().empty()) {
      process_cmd(cmd);
    }
  }

  void process_cmd(std::vector<std::string> cmd) {
    printf("processing cmd {");
    for(size_t i=0;i<cmd.size();++i) {
      printf("\"%s\"", cmd[i].c_str());
      if(i != cmd.size() - 1) {
        printf(", ");
      }
    }
    printf("}\n");
    while(cmdmap.find(cmd.front()) == std::end(cmdmap)) {
      cmd.erase(cmd.begin());
      if(cmd.empty()) {
        return;
      }
    }
    switch(cmdmap.at(cmd.front())) {
      case CMD_UCI:return;
      case CMD_DEBUG:return;
      case CMD_ISREADY:
        init(fen::starting_pos);
        respond(RESP_READYOK);
      return;
      case CMD_SETOPTION:return;
      case CMD_REGISTER:
      return;
      case CMD_UCINEWGAME:
      return;
      case CMD_POSITION:
      return;
      case CMD_GO:
      return;
      case CMD_STOP:
      return;
      case CMD_PONDERHIT:
      return;
      case CMD_QUIT:
        should_quit = true;
      return;
      default:return;
    }
  }

  void respond(RESPONSE resp, std::vector<std::string> args = {}) {
    printf("%s", respmap.at(resp).c_str());
    for(auto &a:args)printf(" %s", a.c_str());
    printf("\n");
  }

  ~UCI() {
    if(engine != nullptr) {
      delete engine;
      engine = nullptr;
    }
  }
};
