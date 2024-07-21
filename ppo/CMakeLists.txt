cmake_minimum_required(VERSION 3.16)

project(proximal_policy_optimization)

# threadpool
add_library(threadpool STATIC)

target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)

target_link_libraries(
  threadpool
  absl::base
)

target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# PPO
add_library(
  libppo STATIC
  agent.cpp
  agent.hpp
  distribution.cpp
  distribution.hpp
  batch_environment.hpp
  environment.hpp
  parser.cpp
  parser.hpp
  ppo.hpp
  running_statistics.hpp
  settings.cpp
  settings.hpp
  utilities.hpp
)

target_link_libraries(
  libppo
  threadpool
  torch
)

target_include_directories(libppo
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
