#pragma once

#include "tensor.hpp"
#include <vector>

class SGD {
private:
  std::vector<Tensor *> parameters_;
  std::vector<Tensor> velocities_;
  float learning_rate_;
  float momentum_;

public:
  SGD(const std::vector<Tensor *> params, float lr, float momentum = 0.9f);

  void Step();
  void DecayLearningRate(float decay_factor);
  void ZeroGrad();
};
