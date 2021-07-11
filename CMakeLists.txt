cmake_minimum_required (VERSION 3.0 FATAL_ERROR)
project (libtorch-lm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_executable(libtorch-lm main.cpp
                           src/util.cpp)
target_link_libraries(libtorch-lm "${TORCH_LIBRARIES}")
set_property(TARGET libtorch-lm PROPERTY CXX_STANDARD 14)
