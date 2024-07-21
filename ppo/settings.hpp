// Copyright 2024 Taylor Howell

#ifndef PPO_SETTINGS_HPP_
#define PPO_SETTINGS_HPP_

#include <torch/torch.h>

#include <string>

#include "parser.hpp"

struct Settings {
  // number of threads/workers
  int num_thread = 20;
  // number of parallel simulation environments
  int num_envs = 256;
  // torch
  torch::Device device = torch::kCPU;
  torch::Device device_sim = torch::kCPU;
  torch::Dtype device_type = torch::kFloat32;
  torch::Dtype device_sim_type = torch::kFloat64;
  // evaluation settings
  int num_eval_envs = 128;
  int max_eval_steps = 1000;
  int num_iter_per_eval = 10;
  bool visualize = false;
  // PPO settings
  int num_steps = 512;
  int minibatch_size = 32768;
  float learning_rate = 5.0e-4;
  int max_env_steps = 100000000;
  bool anneal_lr = true;
  float kl_threshold = 0.008;
  float gamma = 0.99f;
  float gae_lambda = 0.95f;
  int update_epochs = 5;
  bool norm_adv = true;
  float clip_coef = 0.2;
  bool clip_vloss = true;
  float ent_coef = 0.0f;
  float vf_coef = 4.0f;
  float max_grad_norm = 1.0f;
  float optimizer_eps = 1.0e-5;
  bool normalize_observation = true;
  bool normalize_reward = true;
  // checkpoint
  std::string checkpoint = "";
  // PPO settings computed based on above settings
  int batch_size;
  int num_minibatches;
  int num_iterations;
  // print settings
  void Print();
  // update setttings from parser
  void Parse(ArgumentParser* parser);
  // update dependent settings
  void Update();
};

#endif  // PPO_SETTINGS_HPP_
