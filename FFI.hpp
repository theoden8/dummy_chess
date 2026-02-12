#pragma once

#include <cstdio>
#include <unordered_set>
#include <string>

#include <Optimizations.hpp>
#include <Piece.hpp>
#include <MoveLine.hpp>
#include <FEN.hpp>
#include <PGN.hpp>
#include <Engine.hpp>

// FFI helpers for Rust bindings.
// These are thin wrappers that delegate to the underlying C++ API.
// The main purpose is to:
// 1. Heap-allocate objects that contain std::string (SSO issues)
// 2. Provide C-compatible interfaces for complex C++ types
// 3. Avoid returning STL types by value through FFI

struct FFI {
  // ========== Global Initialization ==========
  // Call once before any other FFI functions to initialize global state
  // This avoids race conditions in Board's lazy initialization
  EXPORT static void global_init() {
    static bool initialized = false;
    if (!initialized) {
      M42::init();
      zobrist::init(1 << 20); // Default zobrist size (must be power of 2)
      initialized = true;
    }
  }
  // ========== std::string ==========
  // Heap-allocated to avoid SSO (Small String Optimization) issues
  EXPORT static std::string* make_string_ptr(const char* s) { return new std::string(s); }
  EXPORT static void destroy_string_ptr(std::string* s) { delete s; }
  EXPORT static const char* string_c_str(const std::string* s) { return s->c_str(); }
  EXPORT static size_t string_size(const std::string* s) { return s->size(); }

  // ========== std::unordered_set<move_t> ==========
  EXPORT static std::unordered_set<move_t>* make_searchmoves_ptr() { return new std::unordered_set<move_t>(); }
  EXPORT static void destroy_searchmoves_ptr(std::unordered_set<move_t>* s) { delete s; }

  // ========== fen::FEN ==========
  // Heap-allocated because FEN contains std::string members
  EXPORT static fen::FEN* make_fen_ptr(const std::string& s) { return new fen::FEN(fen::FEN::load_from_string(s)); }
  EXPORT static void destroy_fen_ptr(fen::FEN* f) { delete f; }
  EXPORT static std::string* fen_export_as_string(const fen::FEN* f) { return new std::string(f->export_as_string()); }

  // ========== Board ==========
  EXPORT static fen::FEN* board_export_as_fen(const Board& board) { return new fen::FEN(board.export_as_fen()); }

  // ========== MoveLine ==========
  EXPORT static MoveLine* make_moveline_ptr() { return new MoveLine(); }
  EXPORT static void destroy_moveline_ptr(MoveLine* ml) { delete ml; }
  EXPORT static void destroy_moveline_inplace(MoveLine* ml) { ml->~MoveLine(); }
  // In-place construction/destruction for stack-allocated MoveLine in Rust
  // (needed because clang only exports C2 constructor, not C1 like GCC)
  EXPORT static void init_moveline_inplace(MoveLine* ml) { new (ml) MoveLine(); }
  EXPORT static size_t moveline_size(const MoveLine* ml) { return ml->size(); }
  EXPORT static void moveline_clear(MoveLine* ml) { ml->clear(); }
  EXPORT static void moveline_put(MoveLine* ml, move_t m) { ml->put(m); }
  EXPORT static void moveline_pop(MoveLine* ml) { ml->pop_back(); }
  EXPORT static move_t moveline_front(const MoveLine* ml) { return ml->front(); }
  EXPORT static move_t moveline_back(const MoveLine* ml) { return ml->back(); }
  EXPORT static move_t moveline_at(const MoveLine* ml, size_t i) { return (*ml)[i]; }
  EXPORT static std::string* moveline_pgn(const MoveLine* ml, Board& board) { return new std::string(ml->pgn(board)); }
  EXPORT static std::string* moveline_pgn_full(const MoveLine* ml, Board& board) { return new std::string(ml->pgn_full(board)); }

  // ========== pgn::PGN ==========
  // Heap-allocated because PGN contains std::string and std::vector<std::string>
  EXPORT static pgn::PGN* make_pgn_ptr(Board& board) { return new pgn::PGN(board); }
  EXPORT static void destroy_pgn_ptr(pgn::PGN* p) { delete p; }
  EXPORT static bool pgn_handle_move(pgn::PGN* p, move_t m) { return p->handle_move(m, false); }
  EXPORT static void pgn_retract_move(pgn::PGN* p) { p->retract_move(); }
  EXPORT static std::string* pgn_str(const pgn::PGN* p) { return new std::string(p->str()); }
  EXPORT static size_t pgn_size(const pgn::PGN* p) { return p->size(); }
  EXPORT static size_t pgn_ply_size(const pgn::PGN* p) { return p->ply.size(); }
  EXPORT static std::string* pgn_ply_at(const pgn::PGN* p, size_t i) { return new std::string(p->ply[i]); }
  EXPORT static std::string* pgn_ending(const pgn::PGN* p) { return new std::string(p->ending); }
  EXPORT static move_t pgn_read_move(pgn::PGN* p, const std::string& s, bool* check, bool* mate) {
    return p->read_move_with_flags(s, *check, *mate);
  }

  // ========== Engine::iddfs_state ==========
  EXPORT static Engine::iddfs_state* make_iddfs_state_ptr() { return new Engine::iddfs_state(); }
  EXPORT static void destroy_iddfs_state_ptr(Engine::iddfs_state* s) { delete s; }
  EXPORT static void iddfs_state_reset(Engine::iddfs_state* s) { s->reset(); }
  EXPORT static Engine::score_t iddfs_state_eval(const Engine::iddfs_state* s) { return s->eval; }
  EXPORT static MoveLine* iddfs_state_pline_ptr(Engine::iddfs_state* s) { return &s->pline; }

  // ========== Engine ==========
  // In-place construction/destruction for Rust Box<Engine>
  // (needed because clang only exports C2 constructor, not C1 like GCC)
  EXPORT static void init_engine_inplace(Engine* engine, const fen::FEN& fen, size_t zbsize) {
    new (engine) Engine(fen, zbsize);
  }
  EXPORT static void destroy_engine_inplace(Engine* engine) { engine->~Engine(); }

  EXPORT static move_t engine_start_thinking(
    Engine& engine,
    int depth,
    Engine::iddfs_state* state,
    const std::unordered_set<move_t>* searchmoves
  ) {
    return engine.start_thinking(depth, *state, *searchmoves);
  }
};
