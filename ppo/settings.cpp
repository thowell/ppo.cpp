// Copyright 2024 Taylor Howell

#include "settings.hpp"

#include <torch/torch.h>

#include <iostream>

#include "parser.hpp"

// print settings
void Settings::Print() {
  std::cout << "Settings: " << std::endl;
  std::cout << "  num_thread: " << num_thread << std::endl;
  std::cout << "  device: " << device << std::endl;
  std::cout << "  device_sim: " << device_sim << std::endl;
  std::cout << "  device_type: " << device_type << std::endl;
  std::cout << "  device_sim_type: " << device_sim_type << std::endl;
  std::cout << "  num_envs: " << num_envs << std::endl;
  std::cout << "  num_steps: " << num_steps << std::endl;
  std::cout << "  minibatch_size: " << minibatch_size << std::endl;
  std::cout << "  learning_rate: " << learning_rate << std::endl;
  std::cout << "  max_env_steps: " << max_env_steps << std::endl;
  std::cout << "  anneal_lr: " << (anneal_lr ? "true" : "false") << std::endl;
  std::cout << "  kl_threshold: " << kl_threshold << std::endl;
  std::cout << "  gamma: " << gamma << std::endl;
  std::cout << "  gae_lambda: " << gae_lambda << std::endl;
  std::cout << "  update_epochs: " << update_epochs << std::endl;
  std::cout << "  norm_adv: " << (norm_adv ? "true" : "false") << std::endl;
  std::cout << "  clip_coef: " << clip_coef << std::endl;
  std::cout << "  clip_vloss: " << (clip_vloss ? "true" : "false") << std::endl;
  std::cout << "  ent_coef: " << ent_coef << std::endl;
  std::cout << "  vf_coef: " << vf_coef << std::endl;
  std::cout << "  max_grad_norm: " << max_grad_norm << std::endl;
  std::cout << "  optimizer_eps: " << optimizer_eps << std::endl;
  std::cout << "  normalize_observation: "
            << (normalize_observation ? "true" : "false") << std::endl;
  std::cout << "  normalize_reward: " << (normalize_reward ? "true" : "false")
            << std::endl;
  std::cout << "  num_eval_envs: " << num_eval_envs << std::endl;
  std::cout << "  max_eval_steps: " << max_eval_steps << std::endl;
  std::cout << "  num_iter_per_eval: " << num_iter_per_eval << std::endl;
  std::cout << "  batch_size: " << batch_size << std::endl;
  std::cout << "  num_minibatches: " << num_minibatches << std::endl;
  std::cout << "  num_iterations: " << num_iterations << std::endl;
}

// update settings with arguments from parser
void Settings::Parse(ArgumentParser* parser) {
  // update settings from parser
  for (auto key : parser->keys) {
    if (parser->Has(key)) {
      if (key == "num_thread") {
        num_thread = std::stoi(parser->Get(key));
      } else if (key == "device") {
        if (parser->Get(key) == "cuda") {
          device = torch::kCUDA;
        } else if (parser->Get(key) == "cpu") {
          device = torch::kCPU;
        } else if (parser->Get(key) == "mps") {
          device = torch::kMPS;
        }
      } else if (key == "device_sim") {
        if (parser->Get(key) == "cuda") {
          device_sim = torch::kCUDA;
        } else if (parser->Get(key) == "cpu") {
          device_sim = torch::kCPU;
        } else if (parser->Get(key) == "mps") {
          device_sim = torch::kMPS;
        }
      } else if (key == "device_type") {
        if (parser->Get(key) == "float") {
          device_type = torch::kFloat32;
        } else if (parser->Get(key) == "double") {
          device_type = torch::kFloat64;
        }
      } else if (key == "device_sim_type") {
        if (parser->Get(key) == "float") {
          device_sim_type = torch::kFloat32;
        } else if (parser->Get(key) == "double") {
          device_sim_type = torch::kFloat64;
        }
      } else if (key == "num_envs") {
        num_envs = std::stoi(parser->Get(key));
      } else if (key == "num_steps") {
        num_steps = std::stoi(parser->Get(key));
      } else if (key == "minibatch_size") {
        minibatch_size = std::stoi(parser->Get(key));
      } else if (key == "learning_rate") {
        learning_rate = std::stof(parser->Get(key));
      } else if (key == "max_env_steps") {
        max_env_steps = std::stoi(parser->Get(key));
      } else if (key == "anneal_lr") {
        if (parser->Get(key) == "0" || parser->Get(key) == "false") {
          anneal_lr = false;
        } else if (parser->Get(key) == "1" || parser->Get(key) == "true") {
          anneal_lr = true;
        }
      } else if (key == "kl_threshold") {
        kl_threshold = std::stof(parser->Get(key));
      } else if (key == "gamma") {
        gamma = std::stof(parser->Get(key));
      } else if (key == "gae_lambda") {
        gae_lambda = std::stof(parser->Get(key));
      } else if (key == "update_epochs") {
        update_epochs = std::stoi(parser->Get(key));
      } else if (key == "norm_adv") {
        if (parser->Get(key) == "0" || parser->Get(key) == "false") {
          norm_adv = false;
        } else if (parser->Get(key) == "1" || parser->Get(key) == "true") {
          norm_adv = true;
        }
      } else if (key == "clip_coef") {
        clip_coef = std::stof(parser->Get(key));
      } else if (key == "clip_vloss") {
        if (parser->Get(key) == "0" || parser->Get(key) == "false") {
          clip_vloss = false;
        } else if (parser->Get(key) == "1" || parser->Get(key) == "true") {
          clip_vloss = true;
        }
      } else if (key == "ent_coef") {
        ent_coef = std::stof(parser->Get(key));
      } else if (key == "vf_coef") {
        vf_coef = std::stof(parser->Get(key));
      } else if (key == "max_grad_norm") {
        max_grad_norm = std::stof(parser->Get(key));
      } else if (key == "optimizer_eps") {
        optimizer_eps = std::stof(parser->Get(key));
      } else if (key == "normalize_observation") {
        if (parser->Get(key) == "0" || parser->Get(key) == "false") {
          normalize_observation = false;
        } else if (parser->Get(key) == "1" || parser->Get(key) == "true") {
          normalize_observation = true;
        }
      } else if (key == "normalize_reward") {
        if (parser->Get(key) == "0" || parser->Get(key) == "false") {
          normalize_reward = false;
        } else if (parser->Get(key) == "1" || parser->Get(key) == "true") {
          normalize_reward = true;
        }
      } else if (key == "num_eval_envs") {
        num_eval_envs = std::stoi(parser->Get(key));
      } else if (key == "max_eval_steps") {
        max_eval_steps = std::stoi(parser->Get(key));
      } else if (key == "num_iter_per_eval") {
        num_iter_per_eval = std::stoi(parser->Get(key));
      }
    }
  }
  if (parser->Has("visualize")) {
    visualize = true;
  }
  if (parser->Has("checkpoint")) {
    checkpoint = parser->Get("checkpoint");
  }
}

// update dependent settings
void Settings::Update() {
  batch_size = num_envs * num_steps;
  num_minibatches = static_cast<int>(std::floor(batch_size / minibatch_size));
  num_iterations = static_cast<int>(std::floor(max_env_steps / batch_size));
}
