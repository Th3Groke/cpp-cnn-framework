#pragma once

#include "layer.hpp"
#include <vector>

class Softmax : public Layer {
private:
  Tensor output_cache;

public:
Tensor Forward(const Tensor &input, bool is_training=false) override;
  Tensor Backward(const Tensor &grad_out) override;
  std::string GetLayerName() const override { return "Softmax"; };
  std::vector<Tensor *> GetParameters() override { return {}; };
};
