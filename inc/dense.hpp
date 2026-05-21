#pragma once
#include "layer.hpp"
#include "tensor.hpp"
class Dense : public Layer {
private:
  Tensor weights;
  Tensor biases;
  Tensor input_cache;

  size_t in_features;
  size_t out_features;

public:
  Dense(size_t input_size, size_t output_size);

  Tensor Forward(const Tensor &input) override;
  Tensor Backward(const Tensor &grad_out) override;

  std::vector<Tensor *> GetParameters() override;

  std::string GetLayerName() const override { return "Dense"; };
  // for unit tests
  const Tensor &GetWeights() const { return weights; };
  const Tensor &GetBiases() const { return biases; };
};
