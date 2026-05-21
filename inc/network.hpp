#pragma once

#include "layer.hpp"
#include "tensor.hpp"

class Network {
private:
  std::vector<Layer *> layers;

public:
  Network() {};
  ~Network() = default;
  void AddLayer(Layer *layer);

  Tensor Forward(Tensor input);

  Tensor Backward(Tensor grad_out);

  std::vector<Tensor *> GetParameters();
};
