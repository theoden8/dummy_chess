extern crate bindgen;

use std::process::Command;

fn main() {
  let out_dir = std::env::var("OUT_DIR").unwrap();
  let out_path = std::path::PathBuf::from(&out_dir);
  let lib_path = std::path::PathBuf::from("..");

  let profile = std::env::var("PROFILE").unwrap();

  // Feature flags for C++ build - default to disabled for FFI compatibility
  let jemalloc = std::env::var("FEATURE_SUPPORT_JEMALLOC").unwrap_or("disabled".to_string());
  let sanitize = std::env::var("FEATURE_SUPPORT_SANITIZE").unwrap_or("disabled".to_string());
  let cxx = std::env::var("DUMMY_CHESS_CXX")
    .or_else(|_| std::env::var("CXX"))
    .unwrap_or_default();

  // Build C++ library with make
  // Note: CXX must be passed as a command-line argument, not an env var,
  // because make's built-in default (CXX=g++) has higher precedence than
  // environment variables.
  let mut make_args: Vec<String> = vec![
    format!("FEATURE_SUPPORT_JEMALLOC={}", jemalloc),
    format!("FEATURE_SUPPORT_SANITIZE={}", sanitize),
  ];

  if !cxx.is_empty() {
    make_args.push(format!("CXX={}", cxx));
  }

  make_args.push("libdummychess.so".to_string());
  make_args.push("libdummychess.a".to_string());

  let status = Command::new("make")
    .current_dir(&lib_path)
    .args(&make_args)
    .status()
    .expect("Failed to run make");

  if !status.success() {
    panic!("make failed with status: {}", status);
  }

  // Copy built libraries to output directory
  let lib_extension = if cfg!(target_os = "macos") {
    "dylib"
  } else {
    "so"
  };
  let lib_file_name = format!("libdummychess.{}", lib_extension);
  std::fs::copy(lib_path.join(&lib_file_name), out_path.join(&lib_file_name))
    .expect("Failed to copy shared library");

  std::fs::copy(
    lib_path.join("libdummychess.a"),
    out_path.join("libdummychess.a"),
  )
  .expect("Failed to copy static library");

  // Rerun if source changes
  println!("cargo:rerun-if-changed=../FFI.hpp");
  println!("cargo:rerun-if-changed=../Engine.hpp");
  println!("cargo:rerun-if-changed=../Board.hpp");
  println!("cargo:rerun-if-changed=../MoveLine.hpp");
  println!("cargo:rerun-if-changed=../shared_object.cpp");
  println!("cargo:rerun-if-changed=../m42.cpp");
  println!("cargo:rerun-if-env-changed=FEATURE_SUPPORT_JEMALLOC");
  println!("cargo:rerun-if-env-changed=FEATURE_SUPPORT_SANITIZE");
  println!("cargo:rerun-if-env-changed=CXX");
  println!("cargo:rerun-if-env-changed=DUMMY_CHESS_CXX");

  // Link libraries
  println!("cargo:rustc-link-lib=stdc++");
  println!("cargo:rustc-link-search={}", out_dir);
  println!("cargo:rustc-link-lib=dummychess");

  // Link ASan if enabled
  if sanitize == "enabled" {
    println!("cargo:rustc-link-lib=asan");
  }

  // Generate bindings
  let mut clang_args = vec![
    "-std=c++20",
    "-I..",
    "-Wall",
    "-Wextra",
    "-DFLAG_EXPORT",
    "-DINLINE=",
    "-x",
    "c++",
  ];

  if profile == "debug" {
    clang_args.push("-g3");
  } else {
    clang_args.push("-O3");
    clang_args.push("-DNDEBUG");
    clang_args.push("-DMUTE_ERRORS");
  }

  let bindings_path = out_path.join("bindings.rs");

  let bindings = bindgen::Builder::default()
    .clang_args(&clang_args)
    .raw_line("#[link(name=\"dummychess\")]")
    .raw_line("#[link(name=\"stdc++\")]")
    .raw_line("extern \"C\" {}")
    .header("../Engine.hpp")
    .header("../FFI.hpp")
    .blocklist_type(".*Scope")
    .blocklist_item("(type_|__type)")
    .allowlist_type("(Piece|FEN|PGN|Board|Perft|MoveLine|Engine|FFI)")
    .allowlist_function("(board|bitmask|FEN|fen|PGN|pgn|MoveLine|Board|Engine|FFI)::.*")
    .opaque_type("(std::.*|.*Scope.*|DebugTracer|const_pointer)")
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

  bindings
    .write_to_file(&bindings_path)
    .expect("Couldn't write bindings!");
}
