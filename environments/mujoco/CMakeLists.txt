cmake_minimum_required(VERSION 3.16)

project(mujoco_environments)

# MuJoCo environments
add_library(
  libmjenv STATIC
  environment.cpp
  environment.hpp
  humanoid.hpp
  utilities.hpp
)

target_link_libraries(
  libmjenv
  absl::random_random
  glfw
  mujoco::mujoco
  torch
)

target_include_directories(libmjenv
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
