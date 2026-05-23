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

  Tensor Forward(const Tensor &input, bool is_training = false) override;
  Tensor Backward(const Tensor &grad_out) override;

  std::vector<Tensor *> GetParameters() override { return {}; };
  std::string GetLayerName() const override { return "MaxPooling"; };
};
