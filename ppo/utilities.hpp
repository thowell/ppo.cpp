// Copyright 2024 Taylor Howell

#ifndef PPO_UTILITIES_HPP_
#define PPO_UTILITIES_HPP_

#include <chrono>

// get duration in seconds since time point
// github.com/google-deepmind/mujoco_mpc/mjpc/utilities.h/c
template <typename T>
T GetDuration(std::chrono::steady_clock::time_point time) {
  return 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - time)
                      .count();
}

#endif  // PPO_UTILITIES_HPP_
