// Copyright 2024 Taylor Howell

#include "parser.hpp"

#include <iostream>
#include <string>

// parse command-line arguments (from Claude Sonnet 3.5)
ArgumentParser::ArgumentParser(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      std::string key = arg.substr(2);
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        args_[key] = argv[++i];
      } else {
        args_[key] = "true";
      }
    }
  }

  // available keys
  keys = {"num_thread",
          "device",
          "device_sim",
          "device_type",
          "device_sim_type",
          "num_envs",
          "num_steps",
          "minibatch_size",
          "learning_rate",
          "max_env_steps",
          "anneal_lr",
          "kl_threshold",
          "gamma",
          "gae_lambda",
          "update_epochs",
          "norm_adv",
          "clip_coef",
          "clip_vloss",
          "ent_coef",
          "vf_coef",
          "max_grad_norm",
          "optimizer_eps",
          "normalize_observation",
          "normalize_reward"
          "num_eval_envs",
          "max_eval_steps",
          "num_iter_per_eval"};
}

// get value based on key; if no key, return default value
std::string ArgumentParser::Get(const std::string& key,
                                const std::string& default_value) {
  return args_.count(key) ? args_[key] : default_value;
}

// check for key
bool ArgumentParser::Has(const std::string& key) {
  return args_.count(key) > 0;
}

// print information about parser
void ArgumentParser::Help() {
  std::cout << "Usage: [options]" << std::endl;
  std::cout << "Setup:" << std::endl;
  std::cout << "  --env: [humanoid, ]" << std::endl;
  std::cout << "  --train: run proximal policy optimization to optimize policy "
               "and value function"
            << std::endl;
  std::cout << "  --checkpoint: <filename in checkpoint directory>"
            << std::endl;
  std::cout << "  --load: <filename in checkpoint directory>" << std::endl;
  std::cout << "  --visualize: visualize policy" << std::endl;

  // hardware settings
  std::cout << "Hardware settings:" << std::endl;
  std::cout << "  --num_thread: number of threads/workers for collecting "
               "simulation experience [default: 20]"
            << std::endl;
  std::cout << "  --device: learning device [cpu, default: cuda, mps]"
            << std::endl;
  std::cout << "  --device_sim: simulation device [default: cpu, cuda, mps]"
            << std::endl;
  std::cout << "  --device_type: data type for device [default: float]"
            << std::endl;
  std::cout << "  --device_sim_type: data type for device_sim [default: double]"
            << std::endl;

  // PPO settings
  std::cout << "  --num_envs: number of learning environments for collection "
               "simulation experience"
            << std::endl;
  std::cout << "  --num_steps: number of environment steps" << std::endl;
  std::cout << "  --minibatch_size: size of minibatch" << std::endl;
  std::cout << "  --learning_rate: initial learning rate for policy and value "
               "function optimizer"
            << std::endl;
  std::cout << "  --max_env_steps: total number of environment steps to collect"
            << std::endl;
  std::cout << "  --anneal_lr: flag to anneal learning rate" << std::endl;
  std::cout
      << "  --kl_threshold: maximum KL divergence between old and new policies"
      << std::endl;
  std::cout << "  --gamma: discount factor for rewards" << std::endl;
  std::cout << "  --gae_lambda: factor for Generalized Advantage Estimation"
            << std::endl;
  std::cout
      << "  --update_epochs: number of batch updates per experience collection"
      << std::endl;
  std::cout << "  --norm_adv: flag for normalize advantages" << std::endl;
  std::cout << "  --clip_coef: value for PPO clip parameter" << std::endl;
  std::cout << "  --clip_vloss: flag for clipping value function loss"
            << std::endl;
  std::cout << "  --ent_coef: weight for entropy loss" << std::endl;
  std::cout << "  --vf_coef: weight for value function loss" << std::endl;
  std::cout << "  --max_grad_norm: maximum value for global L2-norm of "
               "parameter gradients"
            << std::endl;
  std::cout << "  --optimizer_eps: epsilson value for Adam optimizer"
            << std::endl;
  std::cout << "  --normalize_observation: normalize observations with running "
               "statistics"
            << std::endl;
  std::cout << "  --normalize_reward: normalize rewards with running statistics"
            << std::endl;

  // evaluation settings
  std::cout << "  --num_eval_envs: number of environments for evaluating "
               "policy performance"
            << std::endl;
  std::cout << "  --max_eval_steps: number of environments steps for policy "
               "performance"
            << std::endl;
  std::cout << "  --num_iter_per_eval: number of learning iterations per "
               "policy evaluation"
            << std::endl;
}
