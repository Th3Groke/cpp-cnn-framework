#pragma once

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
class Tensor {
private:
  size_t totalBatchSize;
  size_t totalChannels;
  size_t totalRows;
  size_t totalColumns;

  size_t stride_N; // The batch stride
  size_t stride_C; // The channel stride
  size_t stride_H; // The row stride
  size_t stride_W; // The column stride

  std::vector<float> data;
  std::vector<float> gradient;

  void validateShapes(const Tensor &other) const {
    if (totalBatchSize != other.totalBatchSize ||
        totalChannels != other.totalChannels || totalRows != other.totalRows ||
        totalColumns != other.totalColumns) {
      throw std::runtime_error("Tensor shape mismatch!");
    }
  }

public:
  Tensor()
      : totalBatchSize(0), totalChannels(0), totalRows(0), totalColumns(0),
        stride_N(0), stride_C(0), stride_H(0), stride_W(0) {};
  Tensor(size_t totalBatchSize, size_t totalChannels, size_t totalRows,
         size_t totalColumns);

  ~Tensor() = default;

  inline size_t Index(size_t batchIndex, size_t channel, size_t row,
                      size_t column) const {
    return (batchIndex * stride_N) + (channel * stride_C) + (row * stride_H) +
           column;
  }

  // Accessors for data
  const std::vector<float> &GetData() const { return data; };
  void SetData(const std::vector<float> &vec) { data = vec; }
  float &operator()(size_t n, size_t c, size_t h, size_t w) {
    size_t idx = Index(n, c, h, w);
    assert(idx < data.size() && "Tensor index out of bounds!");
    return data[idx];
  }
  const float &operator()(size_t n, size_t c, size_t h, size_t w) const {
    size_t idx = Index(n, c, h, w);
    assert(idx < data.size() && "Tensor index out of bounds!");
    return data[idx];
  }

  // Accessors for gradients
  std::vector<float> &GetGradients() { return gradient; };

  float &grad(size_t n, size_t c, size_t h, size_t w) {
    size_t idx = Index(n, c, h, w);
    assert(idx < gradient.size() && "Gradient index out of bounds!");
    return gradient[idx];
  }
  const float &grad(size_t n, size_t c, size_t h, size_t w) const {
    size_t idx = Index(n, c, h, w);
    assert(idx < gradient.size() && "Gradient index out of bounds!");
    return gradient[idx];
  }

  // Shape getters
  size_t getBatchSize() const { return totalBatchSize; }
  size_t getChannels() const { return totalChannels; }
  size_t getRows() const { return totalRows; }
  size_t getColumns() const { return totalColumns; }

  // Math operations
  Tensor &operator+=(const Tensor &other);
  Tensor &operator-=(const Tensor &other);
  Tensor &operator*=(float scalar);

  // Element-wise multiplication (Hadamard product)
  Tensor &operator*=(const Tensor &other);

  static Tensor matmul(const Tensor &A, const Tensor &B);
  void Fill(float value);

  void FillRandomNormal(float mean, float std_dev);

  void Step(float learning_rate);
  void ZeroGrad();
  void Print(const std::string &name = "Tensor") const;
  void Save(std::ostream &out);
  void Load(std::ifstream &in);
};
