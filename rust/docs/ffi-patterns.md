# C++ FFI Patterns for Rust Bindings

This document describes the patterns and pitfalls encountered when creating Rust bindings for the C++ chess engine library using `bindgen`.

## The SSO (Small String Optimization) Problem

### Background

C++ `std::string` implements Small String Optimization (SSO), where short strings (typically < 16 characters) are stored inline within the string object itself, rather than on the heap. The string object contains a pointer that, for SSO strings, points to memory **within the object itself**.

### The Problem

When `bindgen` generates Rust bindings for C++ types containing `std::string`, it treats them as opaque byte arrays:

```rust
pub type string = root::__BindgenOpaqueArray<u64, 4usize>;  // 32 bytes
```

When a C++ function returns `std::string` by value, the ABI copies the raw bytes across the FFI boundary. For SSO strings, this means:

1. The string data is stored inline (bytes 16-31 in typical implementations)
2. A pointer at offset 0 points to byte 16 of the **original** object
3. After the copy, this pointer still points to the old location
4. Accessing the string data through the pointer reads garbage or crashes

### Example of the Bug

```rust
// BAD: This crashes for short strings due to SSO
let inner = unsafe { root::FFI::make_string(c_str.as_ptr()) };  // Returns by value
CppString { inner }  // The internal pointer is now invalid!
```

### The Solution

**Never return `std::string` by value through FFI.** Instead:

1. **Allocate on the heap** and return a pointer:

```cpp
// In FFI.hpp
EXPORT static std::string* make_string_ptr(const char *s) {
  return new std::string(s);
}

EXPORT static void destroy_string_ptr(std::string *s) {
  delete s;
}
```

```rust
// In Rust
struct CppString {
  inner: *mut root::std::string,  // Pointer to heap-allocated string
}

impl CppString {
  fn new(s: &str) -> CppString {
    let c_str = std::ffi::CString::new(s).expect("CString::new failed");
    let inner = unsafe { root::FFI::make_string_ptr(c_str.as_ptr()) };
    CppString { inner }
  }
}

impl Drop for CppString {
  fn drop(&mut self) {
    unsafe { root::FFI::destroy_string_ptr(self.inner) };
  }
}
```

## Structs Containing std::string

Any C++ struct that contains `std::string` members has the same problem. For example:

```cpp
struct fen::FEN {
  std::string board;  // SSO issue
  std::string subs;   // SSO issue
  // ...
};
```

### Solution

Create FFI helper functions that allocate these structs on the heap:

```cpp
EXPORT static fen::FEN* make_fen(const std::string &s) {
  return new fen::FEN(fen::FEN::load_from_string(s));
}

EXPORT static void destroy_fen_ptr(fen::FEN *f) {
  delete f;
}
```

## Structs Containing std::vector

Unlike `std::string`, `std::vector` always allocates its data on the heap. However, the vector struct itself contains pointers to that heap data. When returned by value:

1. The pointers are copied correctly
2. BUT if the original is destroyed, the heap data is freed
3. The copy now has dangling pointers

### Solution

Same pattern - allocate on the heap and manage lifetime carefully.

## PGN Struct Pattern

The `pgn::PGN` struct contains:
- `fen::FEN startfen` - has SSO issue (contains std::string members)
- `Board &board` - reference to board
- `std::vector<std::string> ply` - vector of strings with SSO issues
- `std::string ending` - SSO issue

### Solution: Heap-Allocated PGN

Create FFI helper functions that manage PGN on the heap:

```cpp
// In FFI.hpp

// Create a new PGN on the heap, bound to the given board
EXPORT static pgn::PGN* make_pgn_ptr(Board &board) {
  return new pgn::PGN(board);
}

// Destroy a heap-allocated PGN
EXPORT static void destroy_pgn_ptr(pgn::PGN *p) {
  delete p;
}

// Handle a move in the PGN (writes move notation and advances board state)
EXPORT static void pgn_handle_move(pgn::PGN *p, move_t m) {
  p->handle_move(m);
}

// Retract a move from the PGN
EXPORT static void pgn_retract_move(pgn::PGN *p) {
  p->retract_move();
}

// Get the PGN string representation (heap allocated)
EXPORT static std::string* pgn_to_string(const pgn::PGN *p) {
  return new std::string(p->str());
}

// Get the number of plies in the PGN
EXPORT static size_t pgn_size(const pgn::PGN *p) {
  return p->size();
}

// Get a specific ply string (heap allocated to avoid SSO issues)
EXPORT static std::string* pgn_ply_at(const pgn::PGN *p, size_t index) {
  if (index < p->ply.size()) {
    return new std::string(p->ply[index]);
  }
  return new std::string("");
}

// Get the ending string (heap allocated)
EXPORT static std::string* pgn_ending(const pgn::PGN *p) {
  return new std::string(p->ending);
}

// Read a move from PGN notation using a PGN object
EXPORT static move_t pgn_read_move(pgn::PGN *p, const std::string &s) {
  bool check = false, mate = false;
  return p->read_move_with_flags(s, check, mate);
}
```

Rust wrapper:

```rust
pub struct PGN {
  inner: *mut root::pgn::PGN,
}

impl PGN {
  pub fn new(chess: &mut Chess) -> PGN {
    let inner = unsafe { root::FFI::make_pgn_ptr(chess.as_board_mut()) };
    PGN { inner }
  }

  pub fn handle_move(&mut self, m: root::move_t) {
    unsafe { root::FFI::pgn_handle_move(self.inner, m) };
  }

  pub fn retract_move(&mut self) {
    unsafe { root::FFI::pgn_retract_move(self.inner) };
  }

  pub fn to_string(&self) -> String {
    let cpp_str = CppString {
      inner: unsafe { root::FFI::pgn_to_string(self.inner) },
    };
    cpp_str.to_string()
  }

  pub fn size(&self) -> usize {
    unsafe { root::FFI::pgn_size(self.inner) }
  }

  pub fn ply_at(&self, index: usize) -> String {
    let cpp_str = CppString {
      inner: unsafe { root::FFI::pgn_ply_at(self.inner, index) },
    };
    cpp_str.to_string()
  }

  pub fn ending(&self) -> String {
    let cpp_str = CppString {
      inner: unsafe { root::FFI::pgn_ending(self.inner) },
    };
    cpp_str.to_string()
  }

  pub fn read_move(&mut self, s: &str) -> root::move_t {
    let cpp_str = CppString::new(s);
    unsafe { root::FFI::pgn_read_move(self.inner, cpp_str.as_ref()) }
  }
}

impl Drop for PGN {
  fn drop(&mut self) {
    unsafe { root::FFI::destroy_pgn_ptr(self.inner) };
  }
}
```

## Engine Search Pattern

The engine search uses `iddfs_state` which contains a `MoveLine`. We heap-allocate the state and access it through thin FFI wrappers:

```cpp
// In FFI.hpp - thin delegation wrappers
EXPORT static Engine::iddfs_state* make_iddfs_state_ptr() { return new Engine::iddfs_state(); }
EXPORT static void destroy_iddfs_state_ptr(Engine::iddfs_state* s) { delete s; }
EXPORT static void iddfs_state_reset(Engine::iddfs_state* s) { s->reset(); }
EXPORT static Engine::score_t iddfs_state_eval(const Engine::iddfs_state* s) { return s->eval; }
EXPORT static MoveLine* iddfs_state_pline_ptr(Engine::iddfs_state* s) { return &s->pline; }

EXPORT static move_t engine_start_thinking(
  Engine& engine, int depth, Engine::iddfs_state* state, const std::unordered_set<move_t>* searchmoves
) {
  return engine.start_thinking(depth, *state, *searchmoves);
}
```

Rust handles the orchestration:

```rust
pub fn start_thinking_depth(&mut self, depth: i16) -> MoveLineEval {
  unsafe {
    let state = root::FFI::make_iddfs_state_ptr();
    root::FFI::iddfs_state_reset(state);
    let searchmoves = root::FFI::make_searchmoves_ptr();

    let _best = root::FFI::engine_start_thinking(self.raw(), depth as i32, state, searchmoves);

    let eval = root::FFI::iddfs_state_eval(state);
    let state_pline = root::FFI::iddfs_state_pline_ptr(state);
    let pv_size = root::FFI::moveline_size(state_pline);
    
    let mut pline = MoveLine::new();
    for i in 0..pv_size {
      pline.put(&root::FFI::moveline_at(state_pline, i));
    }

    root::FFI::destroy_searchmoves_ptr(searchmoves);
    root::FFI::destroy_iddfs_state_ptr(state);

    MoveLineEval { mline: pline, eval }
  }
}
```

## Memory Management Checklist

When working with C++ types through FFI:

1. **Never return `std::string` by value** - use heap allocation with `new`/`delete`
2. **Never return structs containing `std::string` by value** - same solution
3. **Implement `Drop` for Rust wrappers** that call appropriate C++ destructors
4. **Be careful with move semantics** - Rust doesn't understand C++ move constructors
5. **Use `std::mem::replace` or `std::mem::take`** when extracting values from temporary C++ structs to avoid double-free

## Thread Safety

The C++ chess engine has global state (magic bitboard lookup tables in `m42.cpp`) that is initialized once. Multiple threads calling into the engine simultaneously can cause race conditions.

**Solution:** Force single-threaded tests via `.cargo/config.toml`:

```toml
[env]
# C++ chess engine has global state that isn't thread-safe
RUST_TEST_THREADS = "1"
```

## Building and Testing

### Building the C++ Library

Build **without** AddressSanitizer, **without** jemalloc, and **with** NDEBUG for stable, quiet operation:

```bash
cd /path/to/dummy_chess
make clean
FEATURE_SUPPORT_SANITIZE=disabled FEATURE_SUPPORT_JEMALLOC=disabled \
  make CXXFLAGS="-std=c++20 -I. -Wall -Wextra -fno-stack-protector -Wno-unused -Wno-parentheses -DNDEBUG" \
  libdummychess.so libdummychess.a
```

**Important:** 
- **NDEBUG** suppresses debug output from the engine (search tree, IDDFS progress, etc.)
- **ASan causes intermittent crashes** (~30% failure rate) with infinite "DEADLYSIGNAL" loops. This appears to be an incompatibility between ASan's signal handling and the Rust test harness. **Do not use ASan for testing.**
- **jemalloc** requires special handling. If the C++ library uses jemalloc, you must use `LD_PRELOAD` to ensure all allocations use the same allocator:
  ```bash
  LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 cargo test
  ```
  The safest option is to build without jemalloc: `FEATURE_SUPPORT_JEMALLOC=disabled`

### Building Rust Bindings

The `build.rs` copies the pre-built library and generates bindings:

```bash
cd rust
cargo build
```

### Running Tests

```bash
cd rust
LD_LIBRARY_PATH=/path/to/dummy_chess cargo test
```

Tests run single-threaded by default (configured in `.cargo/config.toml`).

**Note:** Do not use ASan for testing - it causes intermittent "DEADLYSIGNAL" crashes due to signal handler conflicts between ASan and the Rust runtime. The code itself is correct; ASan just doesn't work reliably in this FFI context.

### Test Coverage

The test suite (45 tests) covers:

- **FEN**: parsing, roundtrip, various positions
- **Chess**: constructors, make/retract moves
- **PGN**: creation, handle_move, retract, to_string, ply access
- **MoveLine**: creation, push/pop, iteration
- **Special moves**: castling, captures, promotions
- **Memory safety**: multiple instances, rapid create/destroy cycles
- **SSO edge cases**: short and long strings
- **Engine search**: depth 1, depth 2, search after moves
