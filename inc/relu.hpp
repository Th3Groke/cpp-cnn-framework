#pragma once

#include "layer.hpp"
#include "tensor.hpp"
class ReLU : public Layer {

private:
  Tensor input_cache;

public:
  ReLU() = default;
  ~ReLU() override = default;
Tensor Forward(const Tensor &input, bool is_training=false) override;
  Tensor Backward(const Tensor &grad_out) override;

  std::string GetLayerName() const override { return "ReLU"; };
};
