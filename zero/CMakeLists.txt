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

# --- CUDA ---
# Workaround: cmake <3.28 doesn't know CUDA standard 20
set(CMAKE_CUDA_STANDARD 17)
# Default arch covers Ampere/Ada/Hopper.  Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# --- libtorch ---
# Discover torch's cmake prefix from the venv so that BOTH Torch and its
# transitive dependency Caffe2 are found from the venv — not from /usr/local.
if(DEFINED ENV{Torch_DIR})
    set(Torch_DIR $ENV{Torch_DIR})
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_result
    )
    if(_torch_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
        # Pin Caffe2_DIR so find_package(Caffe2) inside TorchConfig.cmake
        # picks the venv's Caffe2, not a stale system-installed one.
        set(Caffe2_DIR "${_torch_prefix}/Caffe2" CACHE PATH "" FORCE)
    endif()
endif()

# Workaround: PyTorch cmake references CUDA::nvToolsExt target which was
# removed in newer CUDA toolkits (nvtx is header-only since CUDA 12).
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)

# PyTorch cmake may also bake a full path to libnvToolsExt.so into
# TORCH_LIBRARIES.  Filter it out when the file doesn't exist.
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvToolsExt")

# Match libtorch's CXX11 ABI so all symbols agree at link time.
if(TORCH_CXX_FLAGS)
    string(FIND "${TORCH_CXX_FLAGS}" "-D_GLIBCXX_USE_CXX11_ABI=0" _abi0_pos)
    if(_abi0_pos GREATER_EQUAL 0)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

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

# --- Testing ---
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
    tests/main.cpp
    tests/test_move_encoding.cpp
    tests/test_board_encoding.cpp
    tests/test_mcts.cpp
    tests/test_network.cpp
    tests/test_selfplay.cpp
    tests/test_training.cpp
)

add_executable(dc0_tests ${TEST_SOURCES} ${ENGINE_SOURCES})
target_include_directories(dc0_tests PRIVATE
    ${DC0_INCLUDE_DIRS}
    ${ENGINE_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(dc0_tests GTest::gtest GTest::gtest_main "${TORCH_LIBRARIES}")
target_compile_options(dc0_tests PRIVATE -O2)
target_link_options(dc0_tests PRIVATE "LINKER:--allow-multiple-definition")
set_target_properties(dc0_tests PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
)

include(GoogleTest)
gtest_discover_tests(dc0_tests)
