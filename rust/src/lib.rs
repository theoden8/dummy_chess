#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate libc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn c_to_str(c_str_ptr: *const std::os::raw::c_char) -> String {
    unsafe {
        let c_str = std::ffi::CStr::from_ptr(c_str_ptr);
        c_str.to_string_lossy().into_owned()
    }
}

pub fn cpp_to_str(cpp_str: &root::std::string) -> String {
  unsafe {
    let c_str = root::FFI::to_cstring(cpp_str);
    c_to_str(c_str)
  }
}

pub fn str_to_c(s: &str) -> *const std::os::raw::c_char {
  let c_str = std::ffi::CString::new(s).expect("CString::new failed");
  let c_str_ptr = c_str.as_ptr();
  std::mem::forget(c_str);
  c_str_ptr
}

pub fn str_to_cpp(s: &str) -> root::std::string {
  let c_str = str_to_c(&s);
  unsafe {
    root::FFI::make_string(c_str)
  }
}

pub struct FEN {
  fen: root::fen::FEN,
}

impl FEN {
  pub fn new(s: &String) -> FEN {
    let cpp_str = str_to_cpp(&s);
    let fen = unsafe {
      root::fen::FEN_load_from_string(&cpp_str)
    };
    FEN { fen }
  }

  pub fn from_raw(raw_fen: &root::fen::FEN) -> FEN {
    FEN { fen: raw_fen.clone() }
  }

  pub fn raw(&self) -> &root::fen::FEN {
    &self.fen
  }

  pub fn str(&self) -> String {
    let cpp_str = unsafe { self.fen.export_as_string() };
    cpp_to_str(&cpp_str)
  }
}

pub struct MoveLine {
  mline: root::MoveLine,
}

impl MoveLine {
  pub fn new() -> MoveLine {
    unsafe {
      MoveLine { mline: root::MoveLine::new() }
    }
  }

  pub fn from_raw(mline: root::MoveLine) -> MoveLine {
    MoveLine { mline: mline }
  }

  pub fn size(&self) -> usize {
    unsafe {
      self.mline.size()
    }
  }

  pub fn clear(&mut self) {
    unsafe {
      self.mline.clear()
    }
  }

  pub fn str(&self, chess: &mut Chess) -> String {
    let cpp_str = unsafe {
      self.mline.pgn_full(chess.as_board_ptr())
    };
    cpp_to_str(&cpp_str)
  }

  pub fn put(&mut self, m: &root::move_t) {
    unsafe { self.mline.put(m.clone()); };
  }

  pub fn pop(&mut self) {
    unsafe { self.mline.pop_back(); };
  }

  pub fn back(&self) -> root::move_t {
    unsafe { self.mline.back() }
  }

  pub fn front(&self) -> root::move_t {
    unsafe { self.mline.front() }
  }
}

pub struct MoveLineEval {
  pub mline: MoveLine,
  pub eval: root::Engine_score_t,
}

pub struct Chess {
  engine: root::Engine,
  hist: MoveLine,
}

impl Chess {
  pub fn new(f: &FEN) -> Chess {
    let raw_board = unsafe { root::Engine::new(f.raw(), 1usize << 19) };
    Chess { engine: raw_board, hist: MoveLine::new() }
  }

  pub fn from_string(s: &String) -> Chess {
    let fen = FEN::new(s);
    let raw_board = unsafe { root::Engine::new(fen.raw(), 1usize << 19) };
    Chess { engine: raw_board, hist: MoveLine::new() }
  }

  fn pgn(&mut self) -> root::pgn::PGN {
    unsafe { root::pgn::PGN::new(self.as_board_mut()) }
  }

  fn as_board(&self) -> &root::Board {
    &self.engine._base._base
  }

  fn as_board_mut(&mut self) -> &mut root::Board {
    &mut self.engine._base._base
  }

  fn as_board_ptr(&mut self) -> *mut root::Board {
    &mut self.engine._base._base
  }

  pub fn fen(&self) -> FEN {
    let raw_fen: root::fen::FEN = unsafe { self.as_board().export_as_fen() };
    FEN::from_raw(&raw_fen)
  }

  pub fn str_pgn(&mut self) -> String {
    let mut move_stack: Vec<root::move_t> = Vec::new();
    // undo all moves
    loop {
      if self.hist.size() == 0 {
        break;
      }
      let m = self.hist.back();
      move_stack.push(m);
      self.retract_move();
    }
    let mut raw_pgn = self.pgn();
    // re-do moves
    loop {
      match move_stack.pop() {
        Some(m) => {
          unsafe {
            raw_pgn.handle_move(m)
          };
          self.hist.put(&m);
        }
        None => break
      };
    }
    let cpp_str = unsafe { raw_pgn.str_() };
    cpp_to_str(&cpp_str)
  }

  pub fn raw(&mut self) -> &mut root::Engine {
    &mut self.engine
  }

  pub fn make_move(&mut self, m: root::move_t) {
    unsafe {
      self.as_board_mut().make_move1(m.clone());
      self.hist.put(&m);
    }
  }

  pub fn retract_move(&mut self) {
    unsafe {
      self.hist.pop();
      self.as_board_mut().retract_move();
    }
  }

  pub fn show(&mut self) {
    unsafe {
      self.as_board_mut().print();
    };
    println!("PGN:\n{}", self.str_pgn());
  }

  pub fn read_move_pgn(&mut self, s: &String) -> root::move_t {
    let mut assert_check: bool = false;
    let mut assert_mate: bool = false;
    let assert_check_ptr: *mut bool = &mut assert_check;
    let assert_mate_ptr: *mut bool = &mut assert_mate;
    let cpp_str = str_to_cpp(&s);
    unsafe {
      self.pgn().read_move_with_flags(&cpp_str, assert_check_ptr, assert_mate_ptr)
    }
  }

  pub fn make_move_pgn(&mut self, s: &String) {
    let m = self.read_move_pgn(&s);
    self.make_move(m);
  }

  pub fn start_thinking_depth(&mut self, depth: i16) -> MoveLineEval {
    let mut iddfs_state = root::Engine__iddfs_state {
      curdepth: 1,
      eval: 0,
      pline: unsafe { root::MoveLine::new() },
    };
    let searchmoves = unsafe { root::FFI::make_searchmoves() };
    unsafe {
      let m = self.engine.start_thinking(depth as root::Perft_depth_t, &mut iddfs_state, &searchmoves);
      MoveLineEval { mline: MoveLine::from_raw(iddfs_state.pline), eval: iddfs_state.eval }
    }
  }
}

#[test]
fn test_chess() {
  let starting_pos = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  let fen: FEN = FEN::new(&starting_pos);
  println!("fen: {}", fen.str());
  let mut chess = Chess::new(&fen);
  chess.make_move_pgn(&String::from("e4"));
  chess.make_move_pgn(&String::from("e5"));
  chess.show();
  for i in 1..3 {
    let peval = chess.start_thinking_depth(i);
    println!("{} | {}", peval.eval, peval.mline.str(&mut chess));
  }
  println!("EXIT");
}
