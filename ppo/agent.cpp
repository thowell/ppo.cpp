// Copyright 2024 Taylor Howell

#include <torch/torch.h>

// chatgpt generated
// Define a custom initialization function if required
torch::nn::Linear LayerInit(torch::nn::Linear layer,
                            float std = std::sqrt(2.0f),
                            float bias_const = 0.0f) {
  torch::nn::init::orthogonal_(layer->weight, std);
  torch::nn::init::constant_(layer->bias, bias_const);
  return layer;
}
