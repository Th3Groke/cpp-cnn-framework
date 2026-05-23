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

  Tensor Forward(Tensor input, bool is_training = false);

  Tensor Backward(Tensor grad_out);

  std::vector<Tensor *> GetParameters();
  void SaveWeights(const std::string &filepath);
  void LoadWeights(const std::string &filepath);
};
