use std::path::Path;
use std::process::Command;

/// Detect which C++ standard library a shared library links against
/// by inspecting its dependencies (otool on macOS, ldd on Linux/FreeBSD)
fn detect_cxx_stdlib(lib_path: &Path) -> &'static str {
  let output = if cfg!(target_os = "macos") {
    Command::new("otool")
      .args(["-L", lib_path.to_str().unwrap()])
      .output()
  } else {
    Command::new("ldd").arg(lib_path).output()
  };

  if let Ok(output) = output {
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("libc++") {
      return "c++";
    } else if stdout.contains("libstdc++") {
      return "stdc++";
    }
  }

  "c++"
}

fn main() {
  let out_dir = std::env::var("OUT_DIR").unwrap();
  let out_path = std::path::PathBuf::from(&out_dir);
  let source_dir = std::path::PathBuf::from("..").canonicalize().unwrap();
  let build_dir = out_path.join("cmake_build");

  let profile = std::env::var("PROFILE").unwrap();

  // Feature flags for C++ build - default to disabled for FFI compatibility
  let jemalloc = std::env::var("OPTION_SUPPORT_JEMALLOC").unwrap_or("disabled".to_string());
  let sanitize = std::env::var("OPTION_SUPPORT_SANITIZE").unwrap_or("disabled".to_string());
  let cxx = std::env::var("DUMMY_CHESS_CXX")
    .or_else(|_| std::env::var("CXX"))
    .unwrap_or_default();
  let cc = std::env::var("DUMMY_CHESS_CC")
    .or_else(|_| std::env::var("CC"))
    .unwrap_or_default();

  // Determine CMake build type
  let cmake_build_type = if profile == "debug" {
    "Debug"
  } else {
    "Release"
  };

  // Create build directory
  std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

  // Configure with CMake
  let mut cmake_args = vec![
    format!("-S{}", source_dir.display()),
    format!("-B{}", build_dir.display()),
    format!("-DCMAKE_BUILD_TYPE={}", cmake_build_type),
    format!("-DOPTION_SUPPORT_JEMALLOC={}", jemalloc),
    format!("-DOPTION_SUPPORT_SANITIZE={}", sanitize),
    "-DBUILD_EXECUTABLES=OFF".to_string(), // Only build libraries
  ];

  if !cxx.is_empty() {
    cmake_args.push(format!("-DCMAKE_CXX_COMPILER={}", cxx));
  }
  if !cc.is_empty() {
    cmake_args.push(format!("-DCMAKE_C_COMPILER={}", cc));
  }

  let status = Command::new("cmake")
    .args(&cmake_args)
    .status()
    .expect("Failed to run cmake configure");

  if !status.success() {
    panic!("cmake configure failed with status: {}", status);
  }

  // Build only the library targets
  let status = Command::new("cmake")
    .args([
      "--build",
      build_dir.to_str().unwrap(),
      "--target",
      "dummychess",
      "dummychess_static",
      "--parallel",
    ])
    .status()
    .expect("Failed to run cmake build");

  if !status.success() {
    panic!("cmake build failed with status: {}", status);
  }

  // Determine shared library extension based on target OS
  let shared_lib_ext = if cfg!(target_os = "macos") {
    "dylib"
  } else {
    "so"
  };
  let shared_lib_name = format!("libdummychess.{}", shared_lib_ext);

  // Copy built libraries from CMake build directory to Cargo output directory
  let cmake_shared_lib = build_dir.join(&shared_lib_name);
  let cmake_static_lib = build_dir.join("libdummychess.a");

  std::fs::copy(&cmake_shared_lib, out_path.join(&shared_lib_name))
    .expect("Failed to copy shared library");

  std::fs::copy(&cmake_static_lib, out_path.join("libdummychess.a"))
    .expect("Failed to copy static library");

  // Rerun if source changes
  println!("cargo:rerun-if-changed=./build.rs");
  println!("cargo:rerun-if-changed=../FFI.hpp");
  println!("cargo:rerun-if-changed=../Engine.hpp");
  println!("cargo:rerun-if-changed=../Board.hpp");
  println!("cargo:rerun-if-changed=../MoveLine.hpp");
  println!("cargo:rerun-if-changed=../shared_object.cpp");
  println!("cargo:rerun-if-changed=../m42.cpp");
  println!("cargo:rerun-if-changed=../CMakeLists.txt");
  println!("cargo:rerun-if-env-changed=OPTION_SUPPORT_JEMALLOC");
  println!("cargo:rerun-if-env-changed=OPTION_SUPPORT_SANITIZE");
  println!("cargo:rerun-if-env-changed=CC");
  println!("cargo:rerun-if-env-changed=CXX");
  println!("cargo:rerun-if-env-changed=DUMMY_CHESS_CC");
  println!("cargo:rerun-if-env-changed=DUMMY_CHESS_CXX");
  println!("cargo:rerun-if-env-changed=PROFILE");

  // Detect which C++ standard library the built shared library links against
  let cxx_lib = detect_cxx_stdlib(&cmake_shared_lib);
  println!("cargo:rustc-link-lib={}", cxx_lib);
  println!("cargo:rustc-link-search={}", out_dir);
  println!("cargo:rustc-link-lib=dummychess");

  // Link ASan if enabled (check for any asan variant)
  if sanitize.contains("asan") {
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
    .raw_line(&format!("#[link(name=\"{}\")]", cxx_lib))
    .raw_line("extern \"C\" {}")
    .header("../Engine.hpp")
    .header("../FFI.hpp")
    .blocklist_type(".*Scope")
    .blocklist_item("(type_|__type)")
    .allowlist_type("(Piece|FEN|PGN|Board|Perft|MoveLine|Engine|FFI)")
    .allowlist_function("(board|bitmask|FEN|fen|PGN|pgn|MoveLine|Board|Engine|FFI)::.*")
    .opaque_type("(std::.*|.*Scope.*|DebugTracer|const_pointer)")
    .generate_inline_functions(true)
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
