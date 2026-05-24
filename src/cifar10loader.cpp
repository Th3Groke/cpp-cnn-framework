#include "../inc/cifar10loader.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

const int NUM_CHANNELS = 3;
const int NUM_ROWS = 32;
const int NUM_COLUMNS = 32;
const int IMAGES_PER_FILE = 10000;
const int NUM_IMAGE_BYTES = NUM_COLUMNS * NUM_ROWS * NUM_CHANNELS;

Tensor Cifar10Loader::GetBatch(const Tensor &dataset, int index, int batch_size) {
  Tensor batch(batch_size, 3, 32, 32);
#pragma omp parallel for collapse(4)
  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          batch(i, c, h, w) = dataset(index + i, c, h, w);
        }
      }
    }
  }
  return batch;
}

std::vector<int> Cifar10Loader::GetLabels(int index, int batch_size) {
  std::vector<int> batch_labels;
  for (int i = 0; i < batch_size; i++) {
    batch_labels.push_back(labels[index + i]);
  }
  return batch_labels;
}

Tensor Cifar10Loader::GetImageAsTensor(const Tensor &dataset, int index) {
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

void Cifar10Loader::LoadCifar10(Tensor &tensor, const std::string &filepath) {
  std::ifstream is;
  is.open(filepath, std::ios::binary);
  if (!is) {
    throw std::runtime_error("The batch could not be loaded: " + filepath);
  }
  auto image_buffer = std::vector<uint8_t>(NUM_IMAGE_BYTES);
  for (size_t i = 0; i < tensor.getBatchSize(); i++) {
    uint8_t label;
    is.read(reinterpret_cast<char *>(&label), 1);
    labels.push_back(label);
    is.read(reinterpret_cast<char *>(image_buffer.data()), NUM_IMAGE_BYTES);

    for (int c = 0; c < NUM_CHANNELS; c++) {
      for (int h = 0; h < NUM_ROWS; h++) {
        for (int w = 0; w < NUM_COLUMNS; w++) {
          float neuronValue =
              NormalizePixelValue(image_buffer[GetBufferIndex(c, h, w)]);
          tensor(i, c, h, w) = neuronValue;
        }
      }
    }
  }
  is.close();
}
void Cifar10Loader::LoadCifar10(Tensor &tensor,
                                std::vector<std::string> &filepaths) {
  std::ifstream is;
  int global_i = 0;
  for (auto path : filepaths) {
    is.open(path, std::ios::binary);
    if (!is.is_open()) {
      throw std::runtime_error("Could not load batch: " + path);
    }
    auto image_buffer = std::vector<uint8_t>(NUM_IMAGE_BYTES);
    for (int i = 0; i < IMAGES_PER_FILE; i++) {
      uint8_t label;
      is.read(reinterpret_cast<char *>(&label), 1);
      labels.push_back(label);
      is.read(reinterpret_cast<char *>(image_buffer.data()), NUM_IMAGE_BYTES);
      for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int h = 0; h < NUM_ROWS; h++) {
          for (int w = 0; w < NUM_COLUMNS; w++) {
            float neuronValue =
                NormalizePixelValue(image_buffer[GetBufferIndex(c, h, w)]);
            tensor(global_i, c, h, w) = neuronValue;
          }
        }
      }
      global_i++;
    }
    is.close();
  }
}
