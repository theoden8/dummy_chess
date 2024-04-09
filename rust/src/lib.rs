#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate libc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use root::{fen,FFIString};

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

pub struct FEN {
  fen: fen::FEN,
}

impl FEN {
  pub fn new(s: &str) -> FEN {
    let c_str = s.as_ptr() as *const _;
    unsafe {
      let cpp_str = FFIString::make_string(c_str);
      let fen = fen::FEN_load_from_string(&cpp_str);
      FEN { fen }
    }
  }

  pub fn str(&self) -> String {
    unsafe {
      let cpp_str = self.fen.export_as_string();
      let c_str = FFIString::to_cstring(&cpp_str);
      to_str(c_str)
    }
  }
}

