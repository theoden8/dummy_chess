#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]


include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use root::*;

#[no_mangle] // Ensure the Rust compiler does not mangle the name of this function
pub fn to_str(c_str_ptr: *const std::os::raw::c_char) -> String {
    unsafe {
        // Convert the raw C string to a `CStr`
        let c_str = std::ffi::CStr::from_ptr(c_str_ptr);

        // Convert the `CStr` to a Rust `String`. If the C string is not valid UTF-8,
        // this will replace invalid sequences with � (REPLACEMENT CHARACTER).
        // To handle errors differently, use `c_str.to_str()` and match on the result.
        c_str.to_string_lossy().into_owned()
    }
}


fn main() {
  println!("MAIN");
  let mut s: String = "".to_string();
  unsafe {
//    let dummychess = dummychess::new(env!("OUT_DIR"));
//    println!("{}", dummychess.fen);
    println!("starting pos");
    let starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".as_ptr() as *const _;
    println!("starting pos fen");
    let starting_pos_fen = fen::FEN_load_from_string(starting_pos);
    println!("starting pos str");
    let starting_pos_str = starting_pos_fen.export_as_string();
    println!("starting pos str show");
    FFIString::show_cstring(&starting_pos_str);
    println!("starting pos cstr");
    let starting_pos_cstr = FFIString::to_cstring(&starting_pos_str);
    println!("to str");
    s = to_str(starting_pos_cstr);
    //let board: root::Board = Board::new(&starting_pos_fen as *const _, 1 << 19);
  }
  println!("fen: {}", s);
  println!("EXIT");
}
