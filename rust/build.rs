extern crate bindgen;


fn main() {
  println!("cargo:rustc-link-search=..");
  println!("cargo:rustc-link-lib=dummychess");

  let profile = std::env::var("PROFILE").unwrap();
  let mut clang_args = Vec::from(["-std=c++20", "-I..", "-Wall", "-Wextra", "-DINLINE=", "-x", "c++"]);

  if profile == "debug" {
    clang_args.push("-g3")
  } else if profile == "release" {
    clang_args.push("-O3");
    clang_args.push("-DNDEBUG");
    clang_args.push("-DMUTE_ERRORS");
  }

  let bindings = bindgen::Builder::default()
    .clang_args(&clang_args)
    .header("../Engine.hpp")
    .blocklist_type(".*Scope")
    .blocklist_item("type_")
    .allowlist_type("(Piece|FEN|PGN|Board|Perft|MoveLine|Engine)")
    .allowlist_function("(board|bitmask|FEN|fen|PGN|pgn|MoveLine|Board|Engine)::.*")
    .opaque_type("(std::.*|.*Scope.*|DebugTracer)")
    .generate_inline_functions(true)
    .dynamic_library_name("dummychess")
    .dynamic_link_require_all(true)
    //.explicit_padding(true)
    //.disable_name_namespacing()
    .enable_cxx_namespaces()
    .enable_function_attribute_detection()
    .respect_cxx_access_specs(true)
    .vtable_generation(true)
    .translate_enum_integer_types(true)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .formatter(bindgen::Formatter::Prettyplease)
    .generate()
    .expect("Unable to generate bindings");

  let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("bindings.rs"))
    .expect("Couldn't write bindings!");
}
