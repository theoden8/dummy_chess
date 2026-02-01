extern crate bindgen;
extern crate cc;


fn main() {
  let out_dir = std::env::var("OUT_DIR").unwrap();
  let out_path = std::path::PathBuf::from(&out_dir);

  let profile = std::env::var("PROFILE").unwrap();
  let mut clang_args = Vec::from(["-std=c++20", "-I..", "-Wall", "-Wextra", "-DINLINE=", "-x", "c++"]);

  if profile == "debug" {
    clang_args.push("-g3")
  } else if profile == "release" {
    clang_args.push("-O3");
    clang_args.push("-DNDEBUG");
    clang_args.push("-DMUTE_ERRORS");
  }

  if true {
    let lib_extension = if cfg!(target_os = "macos") {
      "dylib"
    } else {
      "so"
    };
    let lib_file_name = format!("libdummychess.{}", &lib_extension);
    let lib_path = std::path::PathBuf::from("..");
    std::fs::copy(&lib_path.join(&lib_file_name), &out_path.join(&lib_file_name))
      .expect("Failed to copy dynamic lib file");
    std::fs::copy(&lib_path.join("libdummychess.a"), &out_path.join("libdummychess.a"))
      .expect("Failed to copy static lib file");
  } else {
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .include("..")
        .define("MUTE_ERRORS", None)
        .define("NDEBUG", None)
        .define("USE_INTRIN", None)
        .file("../m42.cpp")
        .file("../shared_object.cpp")
        .flag("-O3")
        .flag("-Wall").flag("-Wextra")
        .flag("-Wno-unused-parameter").flag("-Wno-unused-variable").flag("-Wno-unused-but-set-variable")
        .flag("-Wno-range-loop-construct").flag("-Wno-unknown-attributes").flag("-Wno-parentheses")
        .flag("-fno-stack-protector").flag("-flto").flag("-fno-trapping-math").flag("-fno-signed-zeros").flag("-fno-exceptions")
        .flag("-pthread")
        .static_flag(true)
        .shared_flag(true)
        .compile("dummychess");
  }

  println!("cargo:rustc-link-search={}", out_dir);
  println!("cargo:rustc-link-lib=c++");
  println!("cargo:rustc-link-lib=dummychess");

  let bindings_path = out_path.join("bindings.rs");

  let bindings = bindgen::Builder::default()
    .clang_args(&clang_args)
    .raw_line("#[link(name=\"dummychess\")]")
    .raw_line("#[link(name=\"c++\")]")
    .raw_line("extern \"C\" {}")
    .header("../Engine.hpp")
    .header("../FFI.hpp")
    .blocklist_type(".*Scope")
    .blocklist_item("type_")
    .allowlist_type("(Piece|FEN|PGN|Board|Perft|MoveLine|Engine|FFIString)")
    .allowlist_function("(board|bitmask|FEN|fen|PGN|pgn|MoveLine|Board|Engine|FFIString)::.*")
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

  bindings
    .write_to_file(&bindings_path)
    .expect("Couldn't write bindings!");
}
