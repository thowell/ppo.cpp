// Copyright 2024 Taylor Howell

#ifndef PPO_AGENT_HPP_
#define PPO_AGENT_HPP_

#include <torch/torch.h>

#include <tuple>

#include "distribution.hpp"

// chatgpt generated
// Define a custom initialization function if required
torch::nn::Linear LayerInit(torch::nn::Linear layer,
                            float std = std::sqrt(2.0f),
                            float bias_const = 0.0f);

struct AgentImpl : torch::nn::Module {
  // networks
  torch::nn::Sequential features{nullptr};
  torch::nn::Sequential actor_mean{nullptr};
  torch::nn::Sequential critic{nullptr};
  torch::Tensor actor_logstd;

  // observation processing
  bool process_observation;
  torch::Tensor observation_shift;
  torch::Tensor observation_scale;

  AgentImpl(int observation_size, int action_size,
            bool process_observation = true) {
    // observation input
    this->process_observation = process_observation;
    observation_shift = torch::zeros({1, observation_size});
    observation_scale = torch::ones({1, observation_size});

    // Critic Network
    features = torch::nn::Sequential(
        LayerInit(torch::nn::Linear(observation_size, 400)), torch::nn::ELU(),
        LayerInit(torch::nn::Linear(400, 200)), torch::nn::ELU(),
        LayerInit(torch::nn::Linear(200, 100)), torch::nn::ELU());

    // Actor Network
    actor_mean = torch::nn::Sequential(
        LayerInit(torch::nn::Linear(100, action_size), 0.01));

    // Value Network
    critic = torch::nn::Sequential(LayerInit(torch::nn::Linear(100, 1), 1.0));

    actor_logstd = torch::zeros({1, action_size}, torch::requires_grad(true));

    register_module("features", features);
    register_module("critic", critic);
    register_module("actor_mean", actor_mean);
    register_parameter("actor_logstd", actor_logstd);
  }

  torch::Tensor GetValue(torch::Tensor x) {
    return critic->forward(features->forward(x));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  GetActionAndValue(torch::Tensor x, torch::Tensor action = {},
                    torch::Device device = torch::kCUDA) {
    auto f = features->forward(x);
    auto action_mean = actor_mean->forward(f);
    auto action_logstd_expanded = actor_logstd.expand_as(action_mean);
    auto action_std = torch::exp(action_logstd_expanded);
    auto probs = Normal(action_mean, action_std, device);

    if (!action.defined()) {
      action = probs.Sample();
    }

    auto log_prob = probs.LogProb(action).sum(1);
    auto entropy = probs.Entropy().sum(1);
    auto value = critic->forward(f);

    return std::make_tuple(action, log_prob, entropy, value);
  }

  torch::Tensor ProcessObservation(torch::Tensor x) {
    return (x - observation_shift) / observation_scale;
  }
};

TORCH_MODULE(Agent);

#endif  // PPO_AGENT_HPP_
