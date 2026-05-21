#include "../inc/tensor.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

Tensor::Tensor(size_t totalBatchSize, size_t totalChannels, size_t totalRows,
               size_t totalColumns) {
  this->totalBatchSize = totalBatchSize;
  this->totalChannels = totalChannels;
  this->totalRows = totalRows;
  this->totalColumns = totalColumns;

  this->stride_W = 1;
  this->stride_H = totalColumns * stride_W;
  this->stride_C = totalRows * stride_H;
  this->stride_N = totalChannels * stride_C;

  data = std::vector<float>(stride_N * totalBatchSize, 0.0f);
  gradient = std::vector<float>(stride_N * totalBatchSize, 0.0f);
}

void Tensor::FillRandomNormal(float mean, float std_dev) {
  std::random_device rd;
  std::mt19937 mt(rd());

  std::normal_distribution<float> nDist(mean, std_dev);

  for (float &val : data) {
    val = nDist(mt);
  }
}

void Tensor::Fill(float value) { std::fill(data.begin(), data.end(), value); }

void Tensor::Step(float learning_rate) {
  for (size_t i = 0; i < data.size(); i++) {
    data[i] -= learning_rate * gradient[i];
  }
}

void Tensor::ZeroGrad() { std::fill(gradient.begin(), gradient.end(), 0.0f); }

Tensor &Tensor::operator+=(const Tensor &other) {
  validateShapes(other);
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] += other.data[i];
  }
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
  validateShapes(other);
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] -= other.data[i];
  }
  return *this;
}

Tensor &Tensor::operator*=(float scalar) {
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] *= scalar;
  }
  return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
  validateShapes(other);
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] *= other.data[i];
  }
  return *this;
}

Tensor Tensor::matmul(const Tensor &A, const Tensor &B) {
  if (A.totalColumns != B.totalRows || A.totalBatchSize != B.totalBatchSize ||
      A.totalChannels != B.totalChannels) {
    throw std::runtime_error("Tensor MatMul shape mismatch!");
  }

  Tensor result(A.totalBatchSize, A.totalChannels, A.totalRows, B.totalColumns);

#pragma omp parallel for collapse(2)
  for (size_t n = 0; n < A.totalBatchSize; ++n) {
    for (size_t c = 0; c < A.totalChannels; ++c) {
      for (size_t i = 0; i < A.totalRows; ++i) {
        for (size_t k = 0; k < A.totalColumns; ++k) {
          float a_val = A(n, c, i, k);
          for (size_t j = 0; j < B.totalColumns; ++j) {
            result(n, c, i, j) += a_val * B(n, c, k, j);
          }
        }
      }
    }
  }
  return result;
}

void Tensor::Print(const std::string &name) const {
  std::cout << name << " (" << totalBatchSize << "x" << totalChannels << "x"
            << totalRows << "x" << totalColumns << "):" << std::endl;
  for (size_t n = 0; n < std::min(totalBatchSize, (size_t)1); ++n) {
    for (size_t c = 0; c < std::min(totalChannels, (size_t)1); ++c) {
      for (size_t i = 0; i < std::min(totalRows, (size_t)5); ++i) {
        for (size_t j = 0; j < std::min(totalColumns, (size_t)5); ++j) {
          std::cout << std::fixed << std::setprecision(4) << (*this)(n, c, i, j)
                    << " ";
        }
        if (totalColumns > 5)
          std::cout << "...";
        std::cout << std::endl;
      }
      if (totalRows > 5)
        std::cout << "..." << std::endl;
    }
  }
  std::cout << std::endl;
}
