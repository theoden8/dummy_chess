cmake_minimum_required(VERSION 3.14)
project(py_dummy_chess)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

include_directories(..)

find_package(Python3 REQUIRED COMPONENTS Development.Module)

# Use pybind11 from pip installation
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("pybind11 CMake dir: ${pybind11_DIR}")

find_package(pybind11 REQUIRED CONFIG)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

include_directories(${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})
message("Pybind11 includes: ${pybind11_INCLUDE_DIRS}")
message("Python includes: ${Python3_INCLUDE_DIRS}")
message("Python libraries: ${Python3_LIBRARIES}")
message("Pybind11 libraries: ${pybind11_LIBRARIES}")
message("Python executable: ${Python3_EXECUTABLE}")

# Set the library name
pybind11_add_module(_dummychess MODULE ../m42.cpp Bindings.cpp)

# Install into the dummy_chess package directory for wheel building
install(TARGETS _dummychess DESTINATION dummy_chess)
