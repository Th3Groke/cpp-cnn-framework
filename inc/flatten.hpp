#pragma once

#include "layer.hpp"
#include "tensor.hpp"

class Flatten : public Layer {
private:
  size_t batch_cache;
  size_t channels_cache;
  size_t rows_cache;
  size_t columns_cache;

public:
Tensor Forward(const Tensor &input, bool is_training=false) override;
  Tensor Backward(const Tensor &grad_out) override;

  // std::vector<Tensor *> GetParameters() override;
  std::string GetLayerName() const override { return "Flatten"; };
};
