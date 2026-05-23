#pragma once
#include "layer.hpp"
#include <vector>

class MaxPooling : public Layer {
private:
  int pool_size;
  int stride;
  Tensor mask_;

public:
  MaxPooling(int pool_size, int stride);

  Tensor Forward(const Tensor &input);
  Tensor Backward(const Tensor &grad_out);

  std::vector<Tensor *> GetParameters() { return {}; };
  std::string GetLayerName() const { return "MaxPooling"; };
};
