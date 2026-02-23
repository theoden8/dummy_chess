cmake_minimum_required(VERSION 3.18)
project(dc0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- dummy_chess engine headers ---
set(ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(ENGINE_INCLUDE_DIRS
    "${ENGINE_DIR}"
    "${ENGINE_DIR}/external/fathom/src"
)

# --- dc0 headers (this directory) ---
set(DC0_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

# --- Engine source files needed for Board ---
set(ENGINE_SOURCES "${ENGINE_DIR}/m42.cpp")

# --- libtorch ---
option(DC0_USE_TORCH "Build targets that require libtorch" ON)
if(DC0_USE_TORCH)
    # Skip CUDA arch auto-detection (we know our GPU: RTX 4090 = sm_89)
    set(TORCH_CUDA_ARCH_LIST "8.9")
    set(CMAKE_CUDA_ARCHITECTURES "89")
    # Workaround: cmake <3.28 doesn't know CUDA standard 20
    set(CMAKE_CUDA_STANDARD 17)
    # Find CUDA compiler
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-12.4/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
    endif()

    # Find libtorch from the Python venv
    if(DEFINED ENV{Torch_DIR})
        set(Torch_DIR $ENV{Torch_DIR})
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
        execute_process(
            COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" -c
                "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_FIND_RESULT
        )
        if(TORCH_FIND_RESULT EQUAL 0)
            set(Torch_DIR "${TORCH_CMAKE_PREFIX}/Torch")
        endif()
    endif()

    # Workaround: PyTorch cmake references CUDA::nvToolsExt which may not exist
    # in newer CUDA toolkits (nvtx is header-only). Create a dummy target.
    if(NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()

    find_package(Torch REQUIRED)

    # --- Main binary: dc0 ---
    add_executable(dc0 dc0_main.cpp ${ENGINE_SOURCES})
    target_include_directories(dc0 PRIVATE
        ${DC0_INCLUDE_DIRS}
        ${ENGINE_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(dc0 "${TORCH_LIBRARIES}")
    target_compile_options(dc0 PRIVATE -O2)
    set_target_properties(dc0 PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
    )

    # --- UCI binary with dc0 neural network search: dc0_uci ---
    add_executable(dc0_uci "${ENGINE_DIR}/uci.cpp" ${ENGINE_SOURCES})
    target_include_directories(dc0_uci PRIVATE
        ${DC0_INCLUDE_DIRS}
        ${ENGINE_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_definitions(dc0_uci PRIVATE DC0_ENABLED)
    target_link_libraries(dc0_uci "${TORCH_LIBRARIES}")
    target_compile_options(dc0_uci PRIVATE -O2)
    # Engine headers have non-inline definitions that cause ODR violations
    target_link_options(dc0_uci PRIVATE "LINKER:--allow-multiple-definition")
    set_target_properties(dc0_uci PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# --- Testing ---
enable_testing()

# --- Google Test ---
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
# Prevent gtest from overriding compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# libtorch forces _GLIBCXX_USE_CXX11_ABI=0 (old ABI).  gtest must match,
# otherwise std::string symbols from test TUs won't resolve against libgtest.a.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
FetchContent_MakeAvailable(googletest)

# --- Unified test binary: dc0_tests ---
set(TEST_SOURCES
    tests/main.cpp
    tests/test_move_encoding.cpp
    tests/test_board_encoding.cpp
    tests/test_mcts.cpp
)

# Tests that require libtorch
if(DC0_USE_TORCH)
    list(APPEND TEST_SOURCES
        tests/test_network.cpp
        tests/test_selfplay.cpp
        tests/test_training.cpp
    )
endif()

add_executable(dc0_tests ${TEST_SOURCES} ${ENGINE_SOURCES})
target_include_directories(dc0_tests PRIVATE
    ${DC0_INCLUDE_DIRS}
    ${ENGINE_INCLUDE_DIRS}
)
target_link_libraries(dc0_tests GTest::gtest GTest::gtest_main)
target_compile_options(dc0_tests PRIVATE -O2)
# Engine headers (Zobrist.hpp, FEN.hpp, Piece.hpp) have non-inline function
# definitions that get compiled into every TU that includes them.  This is an
# upstream issue; --allow-multiple-definition is the pragmatic fix.
target_link_options(dc0_tests PRIVATE "LINKER:--allow-multiple-definition")

if(DC0_USE_TORCH)
    target_include_directories(dc0_tests PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(dc0_tests "${TORCH_LIBRARIES}")
    set_target_properties(dc0_tests PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# Register tests with ctest.  gtest_discover_tests auto-enumerates every
# TEST() / TEST_F() so `ctest` can run and filter them individually.
include(GoogleTest)
gtest_discover_tests(dc0_tests)
