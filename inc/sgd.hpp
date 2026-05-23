#pragma once

#include "tensor.hpp"
#include <vector>

class SGD {
private:
  std::vector<Tensor *> parameters;
  float learning_rate;

public:
  SGD(const std::vector<Tensor *> params, float lr);

  void Step();

  void ZeroGrad();
};
