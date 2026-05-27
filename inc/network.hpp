#pragma once

#include "layer.hpp"
#include "tensor.hpp"
#include <memory>

class Network {
private:
  std::vector<std::unique_ptr<Layer>> layers;

public:
  Network() {};
  ~Network() = default;
  void AddLayer(std::unique_ptr<Layer> layer);

  Tensor Forward(const Tensor &input, bool is_training = false);

  Tensor Backward(Tensor grad_out);

  std::vector<Tensor *> GetParameters();
  void SaveWeights(const std::string &filename,
                   const std::string &directory = "trained");
  void LoadWeights(const std::string &filename,
                   const std::string &directory = "trained");
};
