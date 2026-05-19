#ifndef RELU_HPP
#define RELU_HPP

#include "layer.hpp"
#include "tensor.hpp"
class ReLU : public Layer {

private:
  Tensor input_cache;

public:
  ReLU() = default;
  ~ReLU() override = default;
  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &grad_out, float learning_rate) override;

  std::string GetLayerName() const override { return "ReLU"; };
};

#endif
