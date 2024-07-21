// Copyright 2024 Taylor Howell

#ifndef PPO_DISTRIBUTION_HPP_
#define PPO_DISTRIBUTION_HPP_

#include <torch/torch.h>

// multivariate normal distribution
class Normal {
 public:
  Normal(const torch::Tensor& mean, const torch::Tensor& std,
         torch::Device device = torch::kCUDA);
  torch::Tensor Sample();
  torch::Tensor LogProb(const torch::Tensor& value);
  torch::Tensor Entropy();

 private:
  torch::Tensor mean_;
  torch::Tensor stddev_;
  torch::Tensor var_;
  torch::Tensor log_std_;
  torch::Device device_;
};

#endif  // PPO_DISTRIBUTION_HPP_
