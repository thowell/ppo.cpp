// Copyright 2024 Taylor Howell

#ifndef ENVIRONMENTS_MUJOCO_HUMANOID_HPP_
#define ENVIRONMENTS_MUJOCO_HUMANOID_HPP_

#include <GLFW/glfw3.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "environment.hpp"
#include "utilities.hpp"

// https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py
template <typename T>
class NVHumanoidEnvironment : public MuJoCoEnvironment<T> {
 public:
  // default constructor
  NVHumanoidEnvironment()
      : MuJoCoEnvironment<T>(LoadTestModel(
            "../environments/mujoco/models/humanoid/nv_humanoid.xml")) {
    // initialize
    this->naction_ = 21;
    this->nobservation_ = 108 - 12;
    this->ndecimation_ = 4;
  }

  // step environment
  void Step(T* observation, T* reward, int* done, const T* action) override {
    // model
    mjModel* m = this->model_;

    // data
    mjData* d = this->data_;

    // set ctrl
    mju_copy(this->data_->ctrl, action, this->model_->nu);
    for (int i = 0; i < this->model_->nu; i++) {
      this->data_->ctrl[i] = mju_clip(
          this->data_->ctrl[i], this->model_->actuator_ctrlrange[2 * i],
          this->model_->actuator_ctrlrange[2 * i + 1]);
    }

    // step physics
    for (int t = 0; t < this->ndecimation_; t++) {
      mj_step(this->model_, this->data_);
    }

    // compute final forces and accelerations
    mj_rnePostConstraint(this->model_, this->data_);

    // observation
    Observation(observation);

    // status
    done[0] = 0;
    if (observation[0] < 0.8) {
      done[0] = 1;
    }

    // -- reward -- //
    reward[0] = 0.0;

    // direction headed reward
    if (observation[11] > 0.8) {
      reward[0] += 0.5;
    } else {
      reward[0] += 0.5 * observation[11] / 0.8;
    }

    // upright reward
    if (observation[10] > 0.93) {
      reward[0] += 0.1;
    }

    // action cost
    reward[0] -= 0.01 * mju_dot(action, action, this->model_->nu);

    // energy cost
    T motor_effort_ratio = 1.0;  // TODO(taylor): implement
    for (int i = 12; i < 33; i++) {
      reward[0] -= (std::abs(observation[i]) > 0.98) * 0.25 *
                   (std::abs(observation[i]) - 0.98) / 0.02 *
                   motor_effort_ratio;
    }

    for (int i = 0; i < 21; i++) {
      // electricity cost
      reward[0] -=
          0.05 * std::abs(action[i] * observation[33 + i]) * motor_effort_ratio;
    }

    // alive
    reward[0] += 2.0;

    // progress
    T to_target[3];
    mju_sub3(to_target, target_, d->qpos);
    to_target[2] = 0.0;
    T new_potential = -mju_normalize3(to_target) / env_timestep_;
    reward[0] += new_potential - potential_;
    potential_ = new_potential;

    // adjust for fallen agent
    if (observation[0] < 0.8) {
      reward[0] = -1.0;
    }
  }

  // reset environment
  void Reset(T* observation) override {
    // model
    mjModel* m = this->model_;

    // data
    mjData* d = this->data_;

    // default reset
    mj_resetData(m, d);

    // sampling token
    absl::BitGen key;

    // position reset
    for (int i = 7; i < m->nq; i++) {
      // sample noisy joint position
      T qi = m->qpos0[i] + 0.4 * absl::Uniform<T>(key, 0.0, 1.0) - 0.2;

      // set value
      d->qpos[i] = qi;
    }

    for (int i = 0; i < m->njnt; i++) {
      if (m->jnt_limited[i]) {
        int idx = m->jnt_qposadr[i];
        T lower = m->jnt_range[2 * i];
        T upper = m->jnt_range[2 * i + 1];

        // clip
        d->qpos[idx] = mju_clip(d->qpos[idx], lower, upper);
      }
    }

    // velocity reset
    for (int i = 6; i < m->nv; i++) {
      d->qvel[i] = 0.2 * absl::Uniform<T>(key, 0.0, 1.0) - 0.1;
    }

    // potential
    potential_ = -1000.0 / env_timestep_;

    // observation
    Observation(observation);
  }

  // clone
  std::unique_ptr<Environment<T>> Clone() const override {
    return std::make_unique<NVHumanoidEnvironment<T>>(*this);
  }

  void LookAt(mjvCamera* cam, const mjData* data) const override {
    cam->lookat[0] = data->qpos[0];
    cam->lookat[1] = data->qpos[1];
    cam->lookat[2] = data->qpos[2];
    cam->distance = 4.0;
  };

 private:
  void Observation(T* observation) {
    // model
    mjModel* m = this->model_;

    // data
    mjData* d = this->data_;

    // shift
    int shift = 0;

    // torso vertical position (1)
    T torso_z = d->qpos[2];
    observation[shift] = torso_z;
    shift += 1;

    // linear velocity (3)
    T* lin_vel = d->qvel;
    mju_copy(observation + shift, lin_vel, 3);
    shift += 3;

    // angular velocity (3)
    T* ang_vel = d->qvel + 3;
    mju_scl(observation + shift, ang_vel, scale_ang_vel_, 3);
    shift += 3;

    // yaw, roll, angle (1, 1, 1)
    T quat[4] = {d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]};
    T euler[3] = {0.0, 0.0, 0.0};
    EulerFromQuat(euler, quat);

    T angle_to_target =
        std::atan2(target_[1] - d->qpos[1], target_[0] - d->qpos[0]) - euler[2];

    observation[shift] = euler[2];
    shift += 1;

    observation[shift] = euler[0];
    shift += 1;

    observation[shift] = angle_to_target;
    shift += 1;

    // up and heading vector projection (2)
    T up[3] = {0.0, 0.0, 1.0};
    T up_world[3] = {0.0, 0.0, 0.0};
    mju_rotVecQuat(up_world, up, quat);

    T forward[3] = {1.0, 0.0, 0.0};
    T forward_world[3] = {0.0, 0.0, 0.0};
    mju_rotVecQuat(forward_world, forward, quat);

    T to_target[3];
    mju_sub3(to_target, target_, d->qpos);
    to_target[2] = 0.0;
    mju_normalize3(to_target);
    T forward_proj = mju_dot3(forward_world, to_target);

    observation[shift] = up_world[2];
    shift += 1;

    observation[shift] = forward_proj;
    shift += 1;

    // dof position (21)
    std::vector<T> qpos_limited(m->nq);
    mju_copy(qpos_limited.data(), d->qpos, m->nq);
    for (int i = 0; i < m->njnt; i++) {
      if (m->jnt_limited[i]) {
        int idx = m->jnt_qposadr[i];
        T qi = d->qpos[idx];
        T lower = m->jnt_range[2 * i];
        T upper = m->jnt_range[2 * i + 1];

        // unscale
        qpos_limited[idx] = (2 * qi - upper - lower) / (upper - lower);
      }
    }
    mju_copy(observation + shift, qpos_limited.data() + 7, 21);
    shift += 21;

    // dof velocity (21)
    T* dof_vel = d->qvel + 6;
    mju_scl(observation + shift, dof_vel, scale_dof_vel_, 21);
    shift += 21;

    // dof force (21)
    T* dof_force = d->qfrc_actuator + 6;
    mju_scl(observation + shift, dof_force, scale_dof_force_, 21);
    shift += 21;

    // sensor forces/torques (12)
    // TODO(taylor): implement

    // actions (21)
    T* actions = d->ctrl;
    mju_copy(observation + shift, actions, 21);
    shift += 21;
  }

  // internal data
  T target_[3] = {1000.0, 0.0, 0.0};

  // potentials
  T potential_;

  // scale
  T scale_ang_vel_ = 0.25;
  T scale_dof_vel_ = 0.1;
  T scale_dof_force_ = 0.01;

  // environment timestep
  T env_timestep_ = 0.0166;  // ndecimation * model.opt.timestep
};

#endif  // ENVIRONMENTS_MUJOCO_HUMANOID_HPP_
