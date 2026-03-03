#pragma once

// Include order is critical: torch headers MUST come before engine headers.
// Fathom/syzygy (included transitively via Engine.hpp) defines macros like
// 'square', 'diag', 'rank' that collide with libtorch symbols.
#include <torch/torch.h>

#include <DC0Engine.hpp>
#include <UCI.hpp>

struct DC0EngineUCI : EngineUCI {
  // DC0-specific options (not in base Options struct)
  std::string search_mode = "alphabeta";
  std::string dc0_model_path;
  int dc0_blocks = 6;
  int dc0_filters = 128;
  int dc0_simulations = 800;
  int dc0_batch_size = 64;
  std::string dc0_device;  // empty = auto

  // DC0 engine (lazily initialized)
  std::unique_ptr<dc0::DC0Engine> dc0_engine;

  bool ensure_dc0_engine() {
    if (dc0_engine && dc0_engine->is_initialized()) {
      return true;
    }
    dc0_engine = std::make_unique<dc0::DC0Engine>();
    bool ok = dc0_engine->init(
        dc0_blocks,
        dc0_filters,
        dc0_model_path,
        dc0_device
    );
    if (ok) {
      dc0_engine->set_simulations(dc0_simulations);
      dc0_engine->set_batch_size(dc0_batch_size);
    }
    return ok;
  }

  // Extended option maps (hide base class versions)
  const std::map<std::string, std::tuple<int, int, int>> spinOptions = {
    {"Hash"s, std::make_tuple(4, 4096, (int)engine_options.hash_mb)},
    {"DC0Simulations"s, std::make_tuple(1, 100000, 800)},
    {"DC0BatchSize"s, std::make_tuple(1, 1024, 64)},
    {"DC0Blocks"s, std::make_tuple(1, 40, 6)},
    {"DC0Filters"s, std::make_tuple(16, 1024, 128)},
  };

  const std::map<std::string, std::pair<cmd_t, std::string>> comboOptions = {
    {"UCI_Variant"s, std::make_pair(cmd_t{"chess"s, "crazyhouse"s}, "chess"s)},
    {"SearchMode"s, std::make_pair(cmd_t{"alphabeta"s, "dc0"s}, "alphabeta"s)},
  };

  const std::map<std::string, std::string> stringOptions {
    {"SyzygyPath"s, "<empty>"s},
    {"DC0ModelPath"s, "<empty>"s},
    {"DC0Device"s, "<empty>"s},
  };

  // Extended opt handlers (delegate non-DC0 options to base)
  bool opt_spin(const std::string &optname, const int val) {
    if (optname == "DC0Simulations"s) {
      dc0_simulations = val;
      if (dc0_engine) dc0_engine->set_simulations(val);
      str::pdebug("info string setoption DC0Simulations =", val);
      return true;
    } else if (optname == "DC0BatchSize"s) {
      dc0_batch_size = val;
      if (dc0_engine) dc0_engine->set_batch_size(val);
      str::pdebug("info string setoption DC0BatchSize =", val);
      return true;
    } else if (optname == "DC0Blocks"s) {
      dc0_blocks = val;
      dc0_engine.reset();  // force re-init with new architecture
      str::pdebug("info string setoption DC0Blocks =", val);
      return true;
    } else if (optname == "DC0Filters"s) {
      dc0_filters = val;
      dc0_engine.reset();  // force re-init with new architecture
      str::pdebug("info string setoption DC0Filters =", val);
      return true;
    }
    return EngineUCI::opt_spin(optname, val);
  }

  bool opt_combo(const std::string &optname, const std::string &optvalue) {
    if (optname == "SearchMode"s) {
      search_mode = optvalue;
      str::pdebug("info string setoption SearchMode =", optvalue);
      return true;
    }
    return EngineUCI::opt_combo(optname, optvalue);
  }

  bool opt_string(const std::string &optname, const std::string &val, const std::string &val_default) {
    if (optname == "DC0ModelPath"s) {
      dc0_model_path = (val == val_default) ? "" : val;
      dc0_engine.reset();  // force re-init with new model
      str::pdebug("info string setoption DC0ModelPath =", val);
      return true;
    } else if (optname == "DC0Device"s) {
      dc0_device = (val == val_default) ? "" : val;
      dc0_engine.reset();  // force re-init with new device
      str::pdebug("info string setoption DC0Device =", val);
      return true;
    }
    return EngineUCI::opt_string(optname, val, val_default);
  }

  void destroy() {
    EngineUCI::destroy();
    if (dc0_engine) dc0_engine->new_game();
  }

  template <typename UCI>
  void perform_go_dc0(go_command args, UCI &uci) {
    _printf("GO COMMAND (dc0)\n");
    if (!ensure_dc0_engine()) {
      str::perror("error: failed to initialize dc0 engine");
      uci.respond(uci.RESP_BESTMOVE, "0000"s);
      return;
    }

    // Set the current board position on the dc0 engine
    dc0_engine->set_position(engine_ptr->as_board());

    // Map UCI go parameters to dc0 SearchParams
    dc0::SearchParams sp;
    sp.simulations = dc0_simulations;
    sp.batch_size = dc0_batch_size;
    sp.movetime = 0.0;
    sp.infinite = args.infinite;

    // nodes -> simulations
    if (args.nodes != SIZE_MAX) {
      sp.simulations = static_cast<int>(std::min(args.nodes, size_t(100000)));
    }

    // movetime (already in seconds in go_command)
    if (args.movetime != DBL_MAX) {
      sp.movetime = std::max(args.movetime - 0.05, args.movetime * 0.5);
    }

    // Time control: compute movetime from wtime/btime/winc/binc
    if (!args.infinite && args.movetime == DBL_MAX && args.nodes == SIZE_MAX) {
      double tc_movetime = uci.time_control_movetime(args, false);
      if (tc_movetime < 1e9) {
        sp.movetime = tc_movetime;
        sp.simulations = 100000;  // effectively unlimited; time will stop us
      }
    }

    if (args.infinite) {
      sp.simulations = 100000;
    }

    // Info callback: emit UCI info lines
    auto info_cb = [&uci, this](const dc0::SearchInfo& info) {
      // Build PV string
      std::string pv_str;
      for (size_t i = 0; i < info.pv.size(); ++i) {
        if (i > 0) pv_str += ' ';
        pv_str += engine_ptr->_move_str(info.pv[i]);
      }
      uci.respond(uci.RESP_INFO, "depth"s, info.depth,
                                  "seldepth"s, static_cast<int>(info.pv.size()),
                                  "nodes"s, info.nodes,
                                  "nps"s, info.nps,
                                  "score"s, "cp "s + std::to_string(info.score_cp),
                                  "pv"s, pv_str,
                                  "time"s, info.time_ms);
    };

    // Stop check: read stdin for stop command
    auto stop_check = [&uci]() -> bool {
      uci.continue_read_cmd(false);
      return uci.should_stop;
    };

    move_t bestmove = dc0_engine->go(sp, info_cb, stop_check);

    if (bestmove == board::nullmove) {
      uci.respond(uci.RESP_BESTMOVE, "0000"s);
    } else {
      uci.respond(uci.RESP_BESTMOVE, engine_ptr->_move_str(bestmove));
    }
    str::pdebug("info string NOTE: dc0 search is over");
  }

  template <typename UCI>
  void perform_go(go_command args, UCI &uci) {
    if (search_mode == "dc0"s) {
      perform_go_dc0(args, uci);
      return;
    }
    EngineUCI::perform_go(args, uci);
  }
};
