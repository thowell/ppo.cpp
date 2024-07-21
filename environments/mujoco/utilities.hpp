// Copyright 2024 Taylor Howell

#ifndef ENVIRONMENTS_MUJOCO_UTILITIES_HPP_
#define ENVIRONMENTS_MUJOCO_UTILITIES_HPP_

#include <cmath>

// Euler angle from quaternion
// https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/utils/torch_jit_utils.py#L176
template <typename T>
void EulerFromQuat(T* euler, const T* quat) {
  // qw, qx, qy, qz
  T qw = quat[0];
  T qx = quat[1];
  T qy = quat[2];
  T qz = quat[3];

  // roll
  T sinr_cosp = 2.0 * (qw * qx + qy * qz);
  T cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz;
  T roll = std::atan2(sinr_cosp, cosr_cosp);

  // pitch
  T sinp = 2.0 * (qw * qy - qz * qx);
  T pitch = std::asin(sinp);
  if (std::abs(sinp) >= 1) {
    pitch = std::copysign(M_PI / 2.0, sinp);
  }

  // yaw
  T siny_cosp = 2.0 * (qw * qz + qx * qy);
  T cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz;
  T yaw = std::atan2(siny_cosp, cosy_cosp);

  // angles
  euler[0] = std::fmod(roll, 2 * M_PI);
  euler[1] = std::fmod(pitch, 2 * M_PI);
  euler[2] = std::fmod(yaw, 2 * M_PI);
}

#endif  // ENVIRONMENTS_MUJOCO_UTILITIES_HPP_
