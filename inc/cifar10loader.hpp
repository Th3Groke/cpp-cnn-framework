#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <sys/types.h>
#include <vector>

class Cifar10Loader {
private:
  std::string path;
  std::vector<int> labels;

  float NormalizePixelValue(uint8_t b) {
    return static_cast<float>(b) / 255.0f;
  }
  uint GetBufferIndex(int channel, int row, int column) {
    return (channel * 1024) + (row * 32) + column;
  }

public:
  Cifar10Loader() = default;

  int GetLabel(int index) { return labels[index]; }
  Tensor GetImageAsTensor(const Tensor &dataset, int index);

  void LoadCifar10(Tensor &tensor, const std::string &filepath);
  void LoadCifar10(Tensor &tensor, std::vector<std::string> &filepath);
};
