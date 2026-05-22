#ifndef CIFAR10LOADER_H
#define CIFAR10LOADER_H

#include "tensor.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
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
  Tensor GetImageAsTensor(const Tensor &dataset, int index) {
    Tensor single_image(1, 3, 32, 32);
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          single_image(0, c, h, w) = dataset(index, c, h, w);
        }
      }
    }
    return single_image;
  }

  void LoadCifar10(Tensor &tensor, const std::string &filepath) {
    std::ifstream is;
    is.open(filepath, std::ios::binary);
    if (!is) {
      std::cout << "The file could not be open!" << std::endl;
      return;
    }
    const int numChannels = 3;
    const int numRows = 32;
    const int numColumns = 32;
    std::vector<uint8_t> image_buffer = std::vector<uint8_t>(3072);
    for (size_t i = 0; i < tensor.getBatchSize(); i++) {
      uint8_t label;
      is.read(reinterpret_cast<char *>(&label), 1);
      labels.push_back(label);
      is.read(reinterpret_cast<char *>(image_buffer.data()), 3072);

      for (int c = 0; c < numChannels; c++) {
        for (int h = 0; h < numRows; h++) {
          for (int w = 0; w < numColumns; w++) {
            float neuronValue =
                NormalizePixelValue(image_buffer[GetBufferIndex(c, h, w)]);
            tensor(i, c, h, w) = neuronValue;
          }
        }
      }
    }
    is.close();
  }
};

#endif
