// Copyright 2024 Taylor Howell

#ifndef PPO_ENVIRONMENT_HPP_
#define PPO_ENVIRONMENT_HPP_

#include <torch/torch.h>

#include <memory>

#include "agent.hpp"
#include "running_statistics.hpp"

// environment
template <typename T>
class Environment {
 public:
  // default constructor
  Environment(int naction, int nobservation)
      : naction_(naction), nobservation_(nobservation) {}

  virtual ~Environment() = default;

  // step environment
  virtual void Step(T* observation, T* reward, int* done, const T* action) = 0;

  // reset environment
  virtual void Reset(T* observation) = 0;

  // visualize agent
  virtual void Visualize(Agent& agent, RunningStatistics<T>& obs_stats,
                         int steps, torch::Device device,
                         torch::Device device_sim, torch::Dtype device_type,
                         torch::Dtype device_sim_type) = 0;

  // clone
  virtual std::unique_ptr<Environment<T>> Clone() const = 0;

  // environment dimensions
  int NumAction() const { return naction_; }
  int NumObservation() const { return nobservation_; }

 protected:
  int naction_;
  int nobservation_;
};

#endif  // PPO_ENVIRONMENT_HPP_
