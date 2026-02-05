cmake_minimum_required(VERSION 3.16)
project(dummy_chess VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Options
# =============================================================================

set(OPTION_SUPPORT_SANITIZE "auto" CACHE STRING "Sanitizer support: auto, enabled, minimal, or disabled")
set(OPTION_SUPPORT_JEMALLOC "auto" CACHE STRING "jemalloc support: auto, builtin, external, or disabled")
set(OPTION_SUPPORT_PGO "auto" CACHE STRING "PGO support: auto, gcc, clang, or disabled")
option(BUILD_LIBRARIES "Build shared and static libraries" ON)
option(BUILD_EXECUTABLES "Build executable targets" ON)
option(BUILD_CURSES "Build ncurses UI" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)

# =============================================================================
# Compiler Detection
# =============================================================================

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(COMPILER_IS_GCC TRUE)
  set(COMPILER_IS_CLANG FALSE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(COMPILER_IS_GCC FALSE)
  set(COMPILER_IS_CLANG TRUE)
else()
  set(COMPILER_IS_GCC FALSE)
  set(COMPILER_IS_CLANG FALSE)
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler is GCC: ${COMPILER_IS_GCC}")
message(STATUS "Compiler is Clang: ${COMPILER_IS_CLANG}")

# =============================================================================
# Feature Detection Functions
# =============================================================================

include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)

# Detect std::ranges support
function(detect_std_ranges OUT_VAR)
  check_cxx_source_compiles("
    #include <ranges>
    #include <vector>
    int main() {
      std::vector<int> v{1,2,3};
      auto r = v | std::views::filter([](int i){ return i > 1; });
      return 0;
    }
  " _HAS_STD_RANGES)
  set(${OUT_VAR} ${_HAS_STD_RANGES} PARENT_SCOPE)
endfunction()

# Detect native BSD functions (arc4random)
function(detect_native_bsd OUT_VAR)
  check_cxx_source_compiles("
    #include <stdlib.h>
    int main() { arc4random(); return 0; }
  " _HAS_NATIVE_BSD)
  set(${OUT_VAR} ${_HAS_NATIVE_BSD} PARENT_SCOPE)
endfunction()

# Detect jemalloc type (builtin, external, or disabled)
function(detect_jemalloc OUT_VAR)
  # Check for builtin jemalloc (FreeBSD, etc.) by testing if mallctl symbol exists
  check_cxx_source_compiles("
    #include <stdlib.h>
    extern \"C\" { int mallctl(const char*, void*, size_t*, void*, size_t); }
    int main() { return 0; }
  " _HAS_MALLCTL_SYMBOL)

  # Check for external jemalloc via pkg-config
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(_JEMALLOC_PKG QUIET jemalloc)
  endif()

  # On FreeBSD/systems with builtin jemalloc, mallctl is available without -ljemalloc
  if(_HAS_MALLCTL_SYMBOL AND NOT _JEMALLOC_PKG_FOUND)
    set(${OUT_VAR} "builtin" PARENT_SCOPE)
  elseif(_JEMALLOC_PKG_FOUND)
    # Test if jemalloc headers are compatible (catches musl+clang exception spec conflicts)
    set(CMAKE_REQUIRED_INCLUDES ${_JEMALLOC_PKG_INCLUDE_DIRS})
    check_cxx_source_compiles("
      #include <jemalloc/jemalloc.h>
      #include <stdlib.h>
      int main() { void *p = malloc(1); free(p); return 0; }
    " _JEMALLOC_HEADERS_OK)
    unset(CMAKE_REQUIRED_INCLUDES)
    if(_JEMALLOC_HEADERS_OK)
      set(${OUT_VAR} "external" PARENT_SCOPE)
    else()
      message(STATUS "External jemalloc headers incompatible, disabling")
      set(${OUT_VAR} "disabled" PARENT_SCOPE)
    endif()
  else()
    set(${OUT_VAR} "disabled" PARENT_SCOPE)
  endif()
endfunction()

# Detect sanitizer support (asan-{static,dynamic}+ubsan-{static,dynamic}, minimal, or disabled)
function(detect_sanitize_support OUT_VAR)
  # Check if we're on glibc or Apple (sanitizers work well there)
  check_cxx_source_compiles("
    #include <features.h>
    #if defined(__GLIBC__) || defined(__APPLE__)
    int main() { return 0; }
    #else
    #error no glibc or apple
    #endif
  " _HAS_GLIBC_OR_APPLE)

  if(NOT _HAS_GLIBC_OR_APPLE)
    # musl and other libcs don't support full sanitizers
    set(${OUT_VAR} "disabled" PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
    # OpenBSD only supports minimal sanitizer runtime
    set(${OUT_VAR} "minimal" PARENT_SCOPE)
    return()
  endif()

  file(WRITE "${CMAKE_BINARY_DIR}/test_san.cpp" "int main() { return 0; }\n")

  # Check ASan (static first, then dynamic)
  set(_ASAN "")
  try_compile(_ASAN_STATIC_WORKS
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/test_san.cpp"
    COMPILE_DEFINITIONS "-fsanitize=address"
    LINK_OPTIONS "-fsanitize=address;-static-libasan"
  )
  if(_ASAN_STATIC_WORKS)
    set(_ASAN "asan-static")
  else()
    try_compile(_ASAN_DYNAMIC_WORKS
      "${CMAKE_BINARY_DIR}"
      "${CMAKE_BINARY_DIR}/test_san.cpp"
      COMPILE_DEFINITIONS "-fsanitize=address"
      LINK_OPTIONS "-fsanitize=address"
    )
    if(_ASAN_DYNAMIC_WORKS)
      set(_ASAN "asan-dynamic")
    endif()
  endif()

  # Check UBSan (static first, then dynamic)
  set(_UBSAN "")
  try_compile(_UBSAN_STATIC_WORKS
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/test_san.cpp"
    COMPILE_DEFINITIONS "-fsanitize=undefined"
    LINK_OPTIONS "-fsanitize=undefined;-static-libubsan"
  )
  if(_UBSAN_STATIC_WORKS)
    set(_UBSAN "ubsan-static")
  else()
    try_compile(_UBSAN_DYNAMIC_WORKS
      "${CMAKE_BINARY_DIR}"
      "${CMAKE_BINARY_DIR}/test_san.cpp"
      COMPILE_DEFINITIONS "-fsanitize=undefined"
      LINK_OPTIONS "-fsanitize=undefined"
    )
    if(_UBSAN_DYNAMIC_WORKS)
      set(_UBSAN "ubsan-dynamic")
    endif()
  endif()

  # Build result string
  if(_ASAN AND _UBSAN)
    set(${OUT_VAR} "${_ASAN}+${_UBSAN}" PARENT_SCOPE)
  elseif(_ASAN)
    set(${OUT_VAR} "${_ASAN}" PARENT_SCOPE)
  elseif(_UBSAN)
    set(${OUT_VAR} "${_UBSAN}" PARENT_SCOPE)
  else()
    set(${OUT_VAR} "disabled" PARENT_SCOPE)
  endif()
endfunction()

# Detect PGO support
function(detect_pgo_support OUT_VAR)
  if(COMPILER_IS_GCC)
    # Test GCC PGO
    set(CMAKE_REQUIRED_FLAGS "-fprofile-generate")
    check_cxx_source_compiles("int main() { return 0; }" _GCC_PGO_WORKS)
    unset(CMAKE_REQUIRED_FLAGS)
    if(_GCC_PGO_WORKS)
      set(${OUT_VAR} "gcc" PARENT_SCOPE)
      return()
    endif()
  elseif(COMPILER_IS_CLANG)
    # Test Clang PGO
    set(CMAKE_REQUIRED_FLAGS "-fprofile-generate=/tmp/test")
    check_cxx_source_compiles("int main() { return 0; }" _CLANG_PGO_WORKS)
    unset(CMAKE_REQUIRED_FLAGS)
    if(_CLANG_PGO_WORKS)
      set(${OUT_VAR} "clang" PARENT_SCOPE)
      return()
    endif()
  endif()
  set(${OUT_VAR} "disabled" PARENT_SCOPE)
endfunction()

# Find llvm-profdata for Clang PGO
function(find_llvm_profdata OUT_VAR)
  if(COMPILER_IS_CLANG)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-prog-name=llvm-profdata
      OUTPUT_VARIABLE _LLVM_PROFDATA
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_LLVM_PROFDATA AND EXISTS "${_LLVM_PROFDATA}")
      set(${OUT_VAR} "${_LLVM_PROFDATA}" PARENT_SCOPE)
    else()
      find_program(_LLVM_PROFDATA_PROG llvm-profdata)
      set(${OUT_VAR} "${_LLVM_PROFDATA_PROG}" PARENT_SCOPE)
    endif()
  else()
    set(${OUT_VAR} "" PARENT_SCOPE)
  endif()
endfunction()

# =============================================================================
# Run Feature Detection
# =============================================================================

detect_std_ranges(HAS_STD_RANGES)
detect_native_bsd(HAS_NATIVE_BSD)
detect_sanitize_support(SANITIZE_TYPE_DETECTED)
detect_pgo_support(PGO_TYPE)
find_llvm_profdata(LLVM_PROFDATA)

# Sanitizer type
if(OPTION_SUPPORT_SANITIZE STREQUAL "auto")
  set(SANITIZE_TYPE ${SANITIZE_TYPE_DETECTED})
else()
  set(SANITIZE_TYPE ${OPTION_SUPPORT_SANITIZE})
endif()

# Jemalloc type
if(OPTION_SUPPORT_JEMALLOC STREQUAL "auto")
  detect_jemalloc(JEMALLOC_TYPE)
else()
  set(JEMALLOC_TYPE ${OPTION_SUPPORT_JEMALLOC})
endif()

# If no native BSD, check for libbsd
if(NOT HAS_NATIVE_BSD)
  find_library(BSD_LIBRARY bsd)
  if(BSD_LIBRARY)
    set(NEED_LIBBSD TRUE)
  else()
    message(WARNING "libbsd not found - some features may not work")
    set(NEED_LIBBSD FALSE)
  endif()
else()
  set(NEED_LIBBSD FALSE)
endif()

# Check for pthread
find_package(Threads REQUIRED)

# Check for ncurses (optional)
if(BUILD_CURSES)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(NCURSES ncursesw)
  endif()
endif()

# Jemalloc pkg-config (for external)
if(JEMALLOC_TYPE STREQUAL "external")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(JEMALLOC jemalloc)
  endif()
endif()

# Override PGO if specified via option
if(OPTION_SUPPORT_PGO STREQUAL "disabled")
  set(PGO_TYPE "disabled")
elseif(NOT OPTION_SUPPORT_PGO STREQUAL "auto")
  set(PGO_TYPE ${OPTION_SUPPORT_PGO})
endif()

message(STATUS "PGO type: ${PGO_TYPE}")
message(STATUS "LLVM profdata: ${LLVM_PROFDATA}")
message(STATUS "jemalloc type: ${JEMALLOC_TYPE}")

# =============================================================================
# External Dependencies (downloaded if missing)
# =============================================================================

# m42 magic bitboards
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/m42.cpp" OR NOT EXISTS "${CMAKE_SOURCE_DIR}/m42.h")
  message(STATUS "Downloading m42 magic bitboards...")
  execute_process(
    COMMAND ./scripts/m42_download
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE M42_RESULT
  )
  if(NOT M42_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download m42")
  endif()
endif()

# Syzygy tablebases
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/syzygy")
  message(STATUS "Downloading Syzygy tablebases...")
  execute_process(
    COMMAND ./scripts/syzygy_download
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE SYZYGY_RESULT
  )
endif()

# Fathom tablebase probing
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/fathom")
  message(STATUS "Downloading Fathom...")
  execute_process(
    COMMAND ./scripts/fathom_download
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE FATHOM_RESULT
  )
endif()

# =============================================================================
# Common Flags
# =============================================================================

set(COMMON_FLAGS -Wall -Wextra -fno-stack-protector)

if(COMPILER_IS_GCC)
  list(APPEND COMMON_FLAGS -Wno-unused -Wno-parentheses)
elseif(COMPILER_IS_CLANG)
  list(APPEND COMMON_FLAGS
    -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable
    -Wno-range-loop-construct -Wno-unknown-attributes -Wno-parentheses)
endif()

# Debug flags (sanitize format: asan-{static,dynamic}+ubsan-{static,dynamic}, minimal, or disabled)
set(DEBUG_FLAGS -g3)
set(DEBUG_LINK_FLAGS "")

if(SANITIZE_TYPE STREQUAL "minimal")
  list(APPEND DEBUG_FLAGS -fsanitize-minimal-runtime -fno-omit-frame-pointer)
  list(APPEND DEBUG_LINK_FLAGS -fsanitize-minimal-runtime)
elseif(NOT SANITIZE_TYPE STREQUAL "disabled")
  # ASan
  if(SANITIZE_TYPE MATCHES "asan-static")
    list(APPEND DEBUG_FLAGS -static-libasan -fsanitize=address)
    list(APPEND DEBUG_LINK_FLAGS -static-libasan -fsanitize=address)
  elseif(SANITIZE_TYPE MATCHES "asan-dynamic")
    list(APPEND DEBUG_FLAGS -fsanitize=address)
    list(APPEND DEBUG_LINK_FLAGS -fsanitize=address)
  endif()
  # UBSan
  if(SANITIZE_TYPE MATCHES "ubsan-static")
    list(APPEND DEBUG_FLAGS -static-libubsan -fsanitize=undefined)
    list(APPEND DEBUG_LINK_FLAGS -static-libubsan -fsanitize=undefined)
  elseif(SANITIZE_TYPE MATCHES "ubsan-dynamic")
    list(APPEND DEBUG_FLAGS -fsanitize=undefined)
    list(APPEND DEBUG_LINK_FLAGS -fsanitize=undefined)
  endif()
  # Common flags if any sanitizer is enabled
  if(SANITIZE_TYPE MATCHES "san")
    list(APPEND DEBUG_FLAGS -fno-omit-frame-pointer)
  endif()
endif()

# Optimized flags (for libraries - no LTO)
set(OPT_LIB_FLAGS -O3 -ffast-math -DNDEBUG -fno-trapping-math -fno-signed-zeros
                  -march=native -DUSE_INTRIN -fno-exceptions)

# Add -m64 on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  list(APPEND OPT_LIB_FLAGS -m64)
endif()

# Optimized flags (for executables - with LTO)
# LTO flags must be passed to both compiler and linker
set(OPT_FLAGS ${OPT_LIB_FLAGS})
if(COMPILER_IS_GCC)
  list(APPEND OPT_FLAGS -flto=auto -fwhole-program)
elseif(COMPILER_IS_CLANG)
  list(APPEND OPT_FLAGS -flto=thin)
endif()

# Library-specific compile definitions
set(LIB_DEFINITIONS INLINE= FLAG_EXPORT)

# =============================================================================
# PGO Flags Functions
# =============================================================================

# Get PGO generate flags for the compiler
function(get_pgo_generate_flags OUT_FLAGS OUT_DIR)
  set(_PROFDATA_DIR "${CMAKE_BINARY_DIR}/profdata")
  file(MAKE_DIRECTORY "${_PROFDATA_DIR}")

  if(PGO_TYPE STREQUAL "gcc")
    set(${OUT_FLAGS} "-fprofile-generate" PARENT_SCOPE)
    set(${OUT_DIR} "${CMAKE_BINARY_DIR}" PARENT_SCOPE)  # GCC uses .gcda files in build dir
  elseif(PGO_TYPE STREQUAL "clang")
    set(${OUT_FLAGS} "-fprofile-generate=${_PROFDATA_DIR}" PARENT_SCOPE)
    set(${OUT_DIR} "${_PROFDATA_DIR}" PARENT_SCOPE)
  else()
    set(${OUT_FLAGS} "" PARENT_SCOPE)
    set(${OUT_DIR} "" PARENT_SCOPE)
  endif()
endfunction()

# Get PGO use flags for the compiler
function(get_pgo_use_flags OUT_FLAGS PROFILE_DATA)
  if(PGO_TYPE STREQUAL "gcc")
    set(${OUT_FLAGS} "-fprofile-use;-fprofile-correction" PARENT_SCOPE)
  elseif(PGO_TYPE STREQUAL "clang")
    set(${OUT_FLAGS} "-fprofile-use=${PROFILE_DATA}" PARENT_SCOPE)
  else()
    set(${OUT_FLAGS} "" PARENT_SCOPE)
  endif()
endfunction()

# =============================================================================
# Compile Definitions
# =============================================================================

set(COMMON_DEFINITIONS "")

if(NOT HAS_STD_RANGES)
  list(APPEND COMMON_DEFINITIONS FLAG_STDRANGES)
endif()

# FLAG_BSD means we need libbsd (not a native BSD system)
if(NEED_LIBBSD)
  list(APPEND COMMON_DEFINITIONS FLAG_BSD)
endif()

if(JEMALLOC_TYPE STREQUAL "builtin")
  list(APPEND COMMON_DEFINITIONS FLAG_JEMALLOC_BUILTIN)
elseif(JEMALLOC_TYPE STREQUAL "external")
  list(APPEND COMMON_DEFINITIONS FLAG_JEMALLOC_EXTERNAL)
else()
  list(APPEND COMMON_DEFINITIONS FLAG_JEMALLOC_DISABLED)
endif()

# =============================================================================
# Source Files
# =============================================================================

set(CORE_SOURCES m42.cpp)
file(GLOB HEADER_FILES "*.hpp" "m42.h" "tbconfig.h")

# =============================================================================
# Helper Functions
# =============================================================================

# Add common dependencies to a target
function(add_common_dependencies target)
  target_link_libraries(${target} PRIVATE Threads::Threads)

  if(NEED_LIBBSD)
    target_link_libraries(${target} PRIVATE ${BSD_LIBRARY})
  endif()

  if(JEMALLOC_TYPE STREQUAL "external" AND JEMALLOC_FOUND)
    target_include_directories(${target} PRIVATE ${JEMALLOC_INCLUDE_DIRS})
    target_link_directories(${target} PRIVATE ${JEMALLOC_LIBRARY_DIRS})
    target_link_libraries(${target} PRIVATE ${JEMALLOC_LIBRARIES})
  endif()
endfunction()

# Create a standard executable target
function(add_chess_executable name source flags link_flags)
  add_executable(${name} ${source} ${CORE_SOURCES})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_compile_options(${name} PRIVATE ${COMMON_FLAGS} ${flags})
  target_compile_definitions(${name} PRIVATE ${COMMON_DEFINITIONS})
  if(link_flags)
    target_link_options(${name} PRIVATE ${link_flags})
  endif()
  add_common_dependencies(${name})
endfunction()

# =============================================================================
# Library Targets
# =============================================================================

if(BUILD_LIBRARIES)
  # Determine library flags based on CMAKE_BUILD_TYPE
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIB_COMPILE_FLAGS ${DEBUG_FLAGS})
    set(LIB_LINK_FLAGS ${DEBUG_LINK_FLAGS})
  else()
    # Release, RelWithDebInfo, MinSizeRel, or unset -> use optimized flags
    set(LIB_COMPILE_FLAGS ${OPT_LIB_FLAGS})
    set(LIB_LINK_FLAGS "")
  endif()

  # Shared library
  add_library(dummychess SHARED shared_object.cpp ${CORE_SOURCES})
  target_include_directories(dummychess PUBLIC ${CMAKE_SOURCE_DIR})
  target_compile_options(dummychess PRIVATE ${COMMON_FLAGS} ${LIB_COMPILE_FLAGS})
  target_compile_definitions(dummychess PRIVATE ${COMMON_DEFINITIONS} ${LIB_DEFINITIONS})
  set_target_properties(dummychess PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(LIB_LINK_FLAGS)
    target_link_options(dummychess PRIVATE ${LIB_LINK_FLAGS})
  endif()
  add_common_dependencies(dummychess)

  # Static library
  add_library(dummychess_static STATIC shared_object.cpp ${CORE_SOURCES})
  target_include_directories(dummychess_static PUBLIC ${CMAKE_SOURCE_DIR})
  target_compile_options(dummychess_static PRIVATE ${COMMON_FLAGS} ${LIB_COMPILE_FLAGS})
  target_compile_definitions(dummychess_static PRIVATE ${COMMON_DEFINITIONS} ${LIB_DEFINITIONS})
  set_target_properties(dummychess_static PROPERTIES OUTPUT_NAME dummychess)
endif()

# =============================================================================
# Executable Targets
# =============================================================================

if(BUILD_EXECUTABLES)
  # dummy_chess (debug build)
  add_chess_executable(dummy_chess simple.cpp "${DEBUG_FLAGS}" "${DEBUG_LINK_FLAGS}")

  # dummy_chess_uci_dbg (debug UCI engine)
  add_chess_executable(dummy_chess_uci_dbg uci.cpp "${DEBUG_FLAGS}" "${DEBUG_LINK_FLAGS}")

  # dummy_chess_curses (optimized, ncurses UI)
  if(BUILD_CURSES AND NCURSES_FOUND)
    add_executable(dummy_chess_curses ui.cpp ${CORE_SOURCES})
    target_include_directories(dummy_chess_curses PRIVATE ${CMAKE_SOURCE_DIR} ${NCURSES_INCLUDE_DIRS})
    target_compile_options(dummy_chess_curses PRIVATE ${COMMON_FLAGS} ${OPT_FLAGS})
    target_compile_definitions(dummy_chess_curses PRIVATE ${COMMON_DEFINITIONS})
    target_link_options(dummy_chess_curses PRIVATE ${OPT_FLAGS})
    target_link_libraries(dummy_chess_curses PRIVATE ${NCURSES_LIBRARIES})
    add_common_dependencies(dummy_chess_curses)
  endif()

  # dummy_chess_alphabeta (optimized, for alpha-beta analysis)
  add_chess_executable(dummy_chess_alphabeta alphabeta.cpp "${OPT_FLAGS}" "${OPT_FLAGS}")

  # =============================================================================
  # PGO Targets
  # =============================================================================

  get_pgo_generate_flags(PGO_GEN_FLAGS PGO_PROFDATA_DIR)

  if(PGO_TYPE STREQUAL "disabled")
    # No PGO - build directly with optimizations
    message(STATUS "Building without PGO")

    add_chess_executable(dummy_chess_uci uci.cpp "${OPT_FLAGS}" "${OPT_FLAGS}")
    target_compile_definitions(dummy_chess_uci PRIVATE MUTE_ERRORS)

    add_chess_executable(dummy_chess_bench bench.cpp "${OPT_FLAGS}" "${OPT_FLAGS}")

  else()
    # PGO enabled - create instrumented build, run benchmark, then optimized build
    message(STATUS "Building with PGO (${PGO_TYPE})")

    set(PGO_PROFDATA "${CMAKE_BINARY_DIR}/uci.profdata")

    # Step 1: Build instrumented UCI engine
    add_executable(dummy_chess_uci_instrumented uci.cpp ${CORE_SOURCES})
    target_include_directories(dummy_chess_uci_instrumented PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(dummy_chess_uci_instrumented PRIVATE ${COMMON_FLAGS} ${OPT_FLAGS} ${PGO_GEN_FLAGS})
    target_link_options(dummy_chess_uci_instrumented PRIVATE ${OPT_FLAGS} ${PGO_GEN_FLAGS})
    target_compile_definitions(dummy_chess_uci_instrumented PRIVATE ${COMMON_DEFINITIONS} MUTE_ERRORS)
    add_common_dependencies(dummy_chess_uci_instrumented)

    # Step 2: Run PGO benchmark to generate profile data
    if(PGO_TYPE STREQUAL "gcc")
      add_custom_command(
        OUTPUT ${PGO_PROFDATA}
        COMMAND ${CMAKE_COMMAND} -E echo "Running PGO benchmark (GCC)..."
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.gcda
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/pgo_bench.py $<TARGET_FILE:dummy_chess_uci_instrumented>
        COMMAND ${CMAKE_COMMAND} -E touch ${PGO_PROFDATA}
        DEPENDS dummy_chess_uci_instrumented
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating PGO profile data (GCC)"
        VERBATIM
      )
    elseif(PGO_TYPE STREQUAL "clang")
      add_custom_command(
        OUTPUT ${PGO_PROFDATA}
        COMMAND ${CMAKE_COMMAND} -E echo "Running PGO benchmark (Clang)..."
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${PGO_PROFDATA_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PGO_PROFDATA_DIR}
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/pgo_bench.py $<TARGET_FILE:dummy_chess_uci_instrumented>
        COMMAND ${LLVM_PROFDATA} merge -output=${PGO_PROFDATA} ${PGO_PROFDATA_DIR}
        DEPENDS dummy_chess_uci_instrumented
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating PGO profile data (Clang)"
        VERBATIM
      )
    endif()

    add_custom_target(pgo_profile DEPENDS ${PGO_PROFDATA})

    # Step 3: Get PGO use flags
    get_pgo_use_flags(PGO_USE_FLAGS ${PGO_PROFDATA})

    # Step 4: Build optimized UCI engine with profile data
    add_executable(dummy_chess_uci uci.cpp ${CORE_SOURCES})
    target_include_directories(dummy_chess_uci PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(dummy_chess_uci PRIVATE ${COMMON_FLAGS} ${OPT_FLAGS} ${PGO_USE_FLAGS})
    target_link_options(dummy_chess_uci PRIVATE ${OPT_FLAGS} ${PGO_USE_FLAGS})
    target_compile_definitions(dummy_chess_uci PRIVATE ${COMMON_DEFINITIONS} MUTE_ERRORS)
    add_common_dependencies(dummy_chess_uci)
    add_dependencies(dummy_chess_uci pgo_profile)

    # Suppress Clang warnings about profile data
    if(COMPILER_IS_CLANG)
      target_compile_options(dummy_chess_uci PRIVATE -Wno-backend-plugin)
    endif()

    # Step 5: Build benchmark with profile data
    add_executable(dummy_chess_bench bench.cpp ${CORE_SOURCES})
    target_include_directories(dummy_chess_bench PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(dummy_chess_bench PRIVATE ${COMMON_FLAGS} ${OPT_FLAGS} ${PGO_USE_FLAGS})
    target_link_options(dummy_chess_bench PRIVATE ${OPT_FLAGS} ${PGO_USE_FLAGS})
    target_compile_definitions(dummy_chess_bench PRIVATE ${COMMON_DEFINITIONS})
    add_common_dependencies(dummy_chess_bench)
    add_dependencies(dummy_chess_bench pgo_profile)

    if(COMPILER_IS_CLANG)
      target_compile_options(dummy_chess_bench PRIVATE -Wno-backend-plugin)
    endif()
  endif()
endif()

# =============================================================================
# Tests
# =============================================================================

enable_testing()

add_test(NAME perft_test
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/perft_test.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# =============================================================================
# Installation
# =============================================================================

if(BUILD_LIBRARIES)
  install(TARGETS dummychess dummychess_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

if(BUILD_EXECUTABLES)
  install(TARGETS dummy_chess dummy_chess_uci dummy_chess_uci_dbg dummy_chess_bench dummy_chess_alphabeta
    RUNTIME DESTINATION bin
  )

  if(BUILD_CURSES AND TARGET dummy_chess_curses)
    install(TARGETS dummy_chess_curses RUNTIME DESTINATION bin)
  endif()
endif()

if(BUILD_LIBRARIES)
  install(FILES ${HEADER_FILES} DESTINATION include/dummy_chess)
endif()

# =============================================================================
# Python Bindings
# =============================================================================

if(BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module)
  if(Python3_FOUND)
    # Use pybind11 from pip installation
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0)
      find_package(pybind11 CONFIG)
    endif()

    if(pybind11_FOUND)
      message(STATUS "Building Python bindings")
      message(STATUS "  Python: ${Python3_EXECUTABLE}")
      message(STATUS "  pybind11: ${pybind11_DIR}")

      # Python flags: like OPT_LIB_FLAGS but without -fno-exceptions (pybind11 needs exceptions)
      set(PYTHON_FLAGS -O3 -ffast-math -DNDEBUG -fno-trapping-math -fno-signed-zeros -march=native -DUSE_INTRIN)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        list(APPEND PYTHON_FLAGS -m64)
      endif()

      pybind11_add_module(_dummychess MODULE python/Bindings.cpp ${CORE_SOURCES})
      target_include_directories(_dummychess PRIVATE ${CMAKE_SOURCE_DIR})
      target_compile_options(_dummychess PRIVATE ${COMMON_FLAGS} ${PYTHON_FLAGS})
      target_compile_definitions(_dummychess PRIVATE ${COMMON_DEFINITIONS})
      add_common_dependencies(_dummychess)

      # Install into the package directory (for wheel building)
      install(TARGETS _dummychess DESTINATION .)
    else()
      message(WARNING "pybind11 not found, skipping Python bindings (install with: pip install pybind11)")
    endif()
  else()
    message(WARNING "Python3 not found, skipping Python bindings")
  endif()
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== dummy_chess Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Feature Detection:")
message(STATUS "  std::ranges: ${HAS_STD_RANGES}")
message(STATUS "  Native BSD: ${HAS_NATIVE_BSD}")
message(STATUS "  Need libbsd: ${NEED_LIBBSD}")
message(STATUS "  sanitize: ${SANITIZE_TYPE}")
message(STATUS "  jemalloc: ${JEMALLOC_TYPE}")
message(STATUS "  PGO: ${PGO_TYPE}")
message(STATUS "  ncurses: ${NCURSES_FOUND}")
if(PGO_TYPE STREQUAL "clang")
  message(STATUS "  llvm-profdata: ${LLVM_PROFDATA}")
endif()
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  OPTION_SUPPORT_SANITIZE: ${OPTION_SUPPORT_SANITIZE}")
message(STATUS "  OPTION_SUPPORT_JEMALLOC: ${OPTION_SUPPORT_JEMALLOC}")
message(STATUS "  OPTION_SUPPORT_PGO: ${OPTION_SUPPORT_PGO}")
message(STATUS "  BUILD_LIBRARIES: ${BUILD_LIBRARIES}")
message(STATUS "  BUILD_EXECUTABLES: ${BUILD_EXECUTABLES}")
message(STATUS "  BUILD_CURSES: ${BUILD_CURSES}")
message(STATUS "  BUILD_PYTHON: ${BUILD_PYTHON}")
message(STATUS "")
message(STATUS "Targets:")
if(BUILD_LIBRARIES)
  message(STATUS "  Libraries: dummychess (shared), dummychess_static")
endif()
if(BUILD_EXECUTABLES)
  message(STATUS "  Executables: dummy_chess, dummy_chess_uci, dummy_chess_uci_dbg, dummy_chess_bench, dummy_chess_alphabeta")
  if(BUILD_CURSES AND NCURSES_FOUND)
    message(STATUS "               dummy_chess_curses")
  endif()
  if(NOT PGO_TYPE STREQUAL "disabled")
    message(STATUS "  PGO intermediate: dummy_chess_uci_instrumented")
  endif()
endif()
if(BUILD_PYTHON AND TARGET _dummychess)
  message(STATUS "  Python: _dummychess")
endif()
message(STATUS "")
