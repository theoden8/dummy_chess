cmake_minimum_required(VERSION 3.14)
project(py_dummy_chess)

set(CMAKE_CXX_FLAGS "-std=c++20 -pedantic")
include_directories($(CMAKE_CURRENT_SOURCE_DIR)/..)

find_package(Boost REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(${Python3_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
message("Boost includes: ${Boost_INCLUDE_DIR}")
message("Python includes: ${Python3_INCLUDE_DIRS}")

set(dummy_chess_lib dummy_chess)
add_library(${dummy_chess_lib} SHARED ../m42.cpp Bindings.cpp)
set_target_properties(${dummy_chess_lib} PROPERTIES PREFIX "")
target_link_libraries(${dummy_chess_lib} ${Boost_LIBRARIES} ${Python3_LIBRARIES})

message("Boost libraries: ${Boost_LIBRARIES}")
message("Python libraries: ${Python3_LIBRARIES}")
