#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
class Tensor {
private:
  size_t totalBatchSize;
  size_t totalChannels;
  size_t totalRows;
  size_t totalColumns;

  size_t stride_W; // The column stride
  size_t stride_H; // The row stride
  size_t stride_C; // The channel stride
  size_t stride_N; // The batch stride

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
  Tensor(size_t totalBatchSize, size_t totalChannels, size_t totalRows,
         size_t totalColumns);

  ~Tensor() = default;

  inline size_t Index(size_t batchIndex, size_t channel, size_t row,
                      size_t column) const {
    return (batchIndex * stride_N) + (channel * stride_C) + (row * stride_H) +
           column;
  }

  // Accessors for data
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

  void FillRandomNormal(float mean, float std_dev);
  void ZeroGradients();
  void Print(const std::string &name = "Tensor") const;
};

#endif
