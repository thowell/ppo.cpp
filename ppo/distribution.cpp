// Copyright 2024 Taylor Howell

#include "distribution.hpp"

#include <torch/torch.h>

#include <cmath>

// https://stackoverflow.com/a/71970807
Normal::Normal(const torch::Tensor& mean, const torch::Tensor& std,
               torch::Device device)
    : mean_(mean),
      stddev_(std),
      var_(std * std),
      log_std_(std.log()),
      device_(device) {}

torch::Tensor Normal::Sample() {
  torch::NoGradGuard no_grad;
  auto eps = torch::randn(this->mean_.sizes()).to(device_);
  return this->mean_ + eps * this->stddev_;
}

torch::Tensor Normal::LogProb(const torch::Tensor& value) {
  return -(value - this->mean_) * (value - this->mean_) / (2 * this->var_) -
         this->log_std_ - std::log(std::sqrt(2.0 * M_PI));
}

torch::Tensor Normal::Entropy() {
  return 0.5 + 0.5 * std::log(2.0 * M_PI) + torch::log(this->stddev_);
}
