// Copyright 2024 Taylor Howell

#ifndef PPO_BATCH_ENVIRONMENT_HPP_
#define PPO_BATCH_ENVIRONMENT_HPP_

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "agent.hpp"
#include "environment.hpp"
#include "running_statistics.hpp"
#include "threadpool.h"

template <typename T>
class BatchEnvironment : public Environment<T> {
 public:
  BatchEnvironment(const Environment<T>& env, int num_envs, int num_threads = 1,
                   bool autoreset = true, bool normalize_observation = true,
                   bool normalize_reward = true, T reward_gamma = 0.99,
                   std::tuple<T, T> observation_range = {-10.0, 10.0},
                   std::tuple<T, T> reward_range = {-10.0, 10.0},
                   int max_steps = 1000)
      : Environment<T>(env.NumAction(), env.NumObservation()),
        num_envs_(num_envs),
        pool_(num_threads),
        autoreset_(autoreset),
        normalize_observation_(normalize_observation),
        observation_statistics_(env.NumObservation()),
        normalize_reward_(normalize_reward),
        reward_gamma_(reward_gamma),
        observation_range_(observation_range),
        reward_range_(reward_range),
        max_steps_(max_steps) {
    // copy environment
    for (int i = 0; i < num_envs; i++) {
      environment_.emplace_back(env.Clone());
    }

    // normalize rewards
    for (int i = 0; i < num_envs; i++) {
      // per-environment reward statistics
      reward_statistics_.push_back(RunningStatistics<T>(1));
    }
    return_.resize(num_envs);
    std::fill(return_.begin(), return_.end(), 0.0);

    // steps
    steps_.resize(num_envs);
    std::fill(steps_.begin(), steps_.end(), 0);
  }

  // step environment
  void Step(T* observation, T* reward, int* done, const T* action) override {
    // step each environment
    int count_before = pool_.GetCount();
    for (int i = 0; i < num_envs_; i++) {
      pool_.Schedule([i, &envs = *this, &observation, &reward, &done,
                      &action]() {
        // environment
        Environment<T>* envi = envs.environment_[i].get();

        // observation
        int nobservation = envs.NumObservation();
        T* observationi = observation + i * nobservation;

        // reward
        T* rewardi = reward + i;

        // done
        int* donei = done + i;

        // action
        const T* actioni = action + i * envs.NumAction();

        // step
        envi->Step(observationi, rewardi, donei, actioni);
        envs.steps_[i] += 1;

        // max steps
        if (envs.steps_[i] >= envs.max_steps_) {
          donei[0] = 1;
        }

        // auto reset
        if (donei[0] && envs.autoreset_) {
          envi->Reset(observationi);
          envs.steps_[i] = 0;
        }

        // normalize reward
        if (envs.normalize_reward_) {
          // update return
          envs.return_[i] =
              envs.return_[i] * envs.reward_gamma_ * (1.0 - done[i]) +
              reward[i];

          // update statistics
          envs.reward_statistics_[i].Update(envs.return_.data() + i);

          // normalize
          reward[i] =
              reward[i] /
              std::sqrt(envs.reward_statistics_[i].Variance()[0] + 1.0e-5);

          // clip reward to range
          reward[i] = std::min(std::max(reward[i], get<0>(envs.reward_range_)),
                               get<1>(envs.reward_range_));
        }
      });
    }

    // wait
    pool_.WaitCount(count_before + num_envs_);
    pool_.ResetCount();

    // normalize observation
    if (normalize_observation_) {
      UpdateObservationStatistics(observation, num_envs_);
      NormalizeObservations(observation, num_envs_);
      ClipObservations(observation, num_envs_);
    }
  }

  // reset environment
  void Reset(T* observation) override {
    // reset each environment
    int count_before = pool_.GetCount();
    for (int i = 0; i < num_envs_; i++) {
      pool_.Schedule([i, &envs = *this, &observation]() {
        // observation
        int nobservation = envs.NumObservation();
        T* observationi = observation + i * nobservation;

        // reset
        envs.environment_[i].get()->Reset(observationi);
        envs.steps_[i] = 0;
      });
    }

    // wait
    pool_.WaitCount(count_before + num_envs_);
    pool_.ResetCount();

    // normalize observation
    if (normalize_observation_) {
      UpdateObservationStatistics(observation, num_envs_);
      NormalizeObservations(observation, num_envs_);
      ClipObservations(observation, num_envs_);
    }
  }

  // visualize agent
  void Visualize(Agent& agent, RunningStatistics<T>& obs_stats, int steps,
                 torch::Device device, torch::Device device_sim,
                 torch::Dtype device_type,
                 torch::Dtype device_sim_type) override {
    environment_[0].get()->Visualize(agent, obs_stats, steps, device,
                                     device_sim, device_type, device_sim_type);
  }

  // clone
  std::unique_ptr<Environment<T>> Clone() const override {
    std::cout << " BATCH ENVIRONMENT CLONE NOT PROPERLY IMPLEMENTED"
              << std::endl;
    return environment_[0].get()->Clone();
  }

  // number of environments
  int NumEnvironment() const { return num_envs_; }

  // copy observation statistics
  void ObservationStatistics(const BatchEnvironment<T>& env) {
    this->observation_statistics_ = env.observation_statistics_;
  }
  void ObservationStatistics(const RunningStatistics<T>& stats) {
    this->observation_statistics_ = stats;
  }

  // return observation statistics
  RunningStatistics<T> GetObservationStatistics() {
    return RunningStatistics<T>(observation_statistics_);
  }

 private:
  // update observation statistics with batch of observations
  void UpdateObservationStatistics(const T* batch, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
      const T* observation = batch + i * this->NumObservation();
      observation_statistics_.Update(observation);
    }
  }

  // normalize batch of observation
  void NormalizeObservations(T* batch, int batch_size) {
    int nobservation = this->NumObservation();
    std::vector<T> mean = observation_statistics_.Mean();
    std::vector<T> stddev = observation_statistics_.StandardDeviation();

    // TODO(taylor): parallel loop over batch
    for (int i = 0; i < batch_size; i++) {
      T* observation = batch + i * nobservation;
      for (int j = 0; j < nobservation; j++) {
        if (std::abs(stddev[j]) < 1.0e-7) {
          observation[j] = 0.0;
        } else {
          observation[j] = (observation[j] - mean[j]) / (stddev[j] + 1.0e-8);
        }
      }
    }
  }

  // clip batch of observations into range
  void ClipObservations(T* batch, int batch_size) {
    // environments
    // TODO(taylor): in parallel?
    for (int i = 0; i < batch_size; i++) {
      // observation
      int nobservation = this->NumObservation();
      T* observationi = batch + i * nobservation;

      // clip observation to range
      for (int j = 0; j < nobservation; j++) {
        observationi[j] =
            std::min(std::max(observationi[j], get<0>(observation_range_)),
                     get<1>(observation_range_));
      }
    }
  }

  int num_envs_;
  std::vector<std::unique_ptr<Environment<T>>> environment_;
  ThreadPool pool_;
  bool autoreset_;
  bool normalize_observation_;
  RunningStatistics<T> observation_statistics_;
  bool normalize_reward_;
  T reward_gamma_;
  std::vector<RunningStatistics<T>> reward_statistics_;
  std::vector<T> return_;
  std::tuple<T, T> observation_range_;
  std::tuple<T, T> reward_range_;
  std::vector<int> steps_;
  int max_steps_;
};

#endif  // PPO_BATCH_ENVIRONMENT_HPP_
