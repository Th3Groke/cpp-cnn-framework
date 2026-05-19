#ifndef LAYER_H
#define LAYER_H

#include "tensor.hpp"
class Layer {
public:
  virtual ~Layer() = default;
  virtual Tensor Forward(const Tensor &input) = 0;
  virtual Tensor Backward(const Tensor &grad_out, float learning_rate) = 0;
  virtual std::string GetLayerName() const = 0;
};

#endif
