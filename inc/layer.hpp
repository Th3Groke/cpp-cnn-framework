#pragma once
#include "tensor.hpp"
class Layer {
public:
  virtual ~Layer() = default;
  virtual Tensor Forward(const Tensor &input, bool is_training = false) = 0;
  virtual Tensor Backward(const Tensor &grad_out) = 0;
  virtual std::vector<Tensor *> GetParameters() { return {}; }
  virtual std::string GetLayerName() const = 0;
};
