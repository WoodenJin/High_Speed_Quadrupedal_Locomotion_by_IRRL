cmake_minimum_required(VERSION 3.10)
project(_flexible_robot)
set(CMAKE_CXX_STANDARD 14)

set(pybind11_DIR ~/raisim_build/share/cmake/pybind11)
set(raisimOgre_DIR ~/raisim_build/share/raisimOgre/cmake)
set(OGRE_DIR ~/raisim_build/lib/OGRE/cmake)

set(RSG_ENVIRONMENT_INCLUDE_PATH ./flex_gym/env/env/BlackPanther_V55)

find_package(pybind11 2.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(raisimOgre 0.6.0 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

message("[FLEXIBLE_ROBOT_RAISIM_GYM] including directory ${RSG_ENVIRONMENT_INCLUDE_PATH}")

pybind11_add_module(_flexible_robot flex_gym/env/raisim_gym.cpp)

target_link_libraries(_flexible_robot PRIVATE raisim::raisim raisim::raisimOgre yaml-cpp)

target_include_directories(_flexible_robot PUBLIC flex_gym/env ${RSG_ENVIRONMENT_INCLUDE_PATH} ${EIGEN3_INCLUDE_DIRS})

target_compile_options(_flexible_robot PRIVATE -mtune=native -fPIC -O3)
