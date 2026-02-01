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
    let c_str = root::FFIString::to_cstring(cpp_str);
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
    root::FFIString::make_string(c_str)
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

pub struct Board {
  board: root::Board,
}

impl Board {
  pub fn new(f: &FEN) -> Board {
    let raw_board = unsafe { root::Board::new(f.raw(), 1usize << 19) };
    Board { board: raw_board }
  }

  pub fn from_string(s: &String) -> Board {
    let fen = FEN::new(s);
    let raw_board = unsafe { root::Board::new(fen.raw(), 1usize << 19) };
    Board { board: raw_board }
  }

  fn pgn(&mut self) -> root::pgn::PGN {
    unsafe {root::pgn::PGN::new(&mut self.board) }
  }

  pub fn fen(&self) -> FEN {
    let raw_fen: root::fen::FEN = unsafe { self.board.export_as_fen() };
    FEN::from_raw(&raw_fen)
  }

  pub fn str_pgn(&mut self) -> String {
    let cpp_str = unsafe { self.pgn().str_() };
    cpp_to_str(&cpp_str)
  }

  pub fn raw(&mut self) -> &mut root::Board {
    &mut self.board
  }

  pub fn make_move(&mut self, m: root::move_t) {
    unsafe {
      self.board.make_move1(m);
    }
  }

  pub fn retract_move(&mut self) {
    unsafe {
      self.board.retract_move();
    }
  }

  pub fn show(&mut self) {
    unsafe {
      self.board.print();
    };
    println!("PGN: {}", self.str_pgn());
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
    let cpp_str = str_to_cpp(&s);
    unsafe {
      self.pgn().read_move(&cpp_str)
    }
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

  pub fn str(&self, board: &mut Board) -> String {
    let cpp_str = unsafe {
      let raw_board = &mut board.raw();
      let raw_board_ptr: *mut root::Board = &mut **raw_board;
      self.mline.pgn_full(raw_board_ptr)
    };
    cpp_to_str(&cpp_str)
  }
}
