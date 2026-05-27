#pragma once
#include "layer.hpp"
#include "tensor.hpp"

class Conv2d : public Layer {
private:
  int num_filters;
  int input_channels;
  int filter_size;

  Tensor filters;
  Tensor input_cache;
  Tensor biases;

public:
  Conv2d(int input_channels, int num_filters, int filter_size);
  Tensor Forward(const Tensor &input, bool is_training = false) override;
  Tensor Backward(const Tensor &grad_out) override;
  std::string GetLayerName() const override { return "Conv2D"; };
  Tensor &GetFilters() { return filters; };
  Tensor &GetBiases() { return biases; };
  std::vector<Tensor *> GetParameters() override {
    return {&filters, &biases};
  };
};
