#include "../inc/dropout.hpp"
#include <cstdlib>
#include <ctime>
#include <random>

Tensor Dropout::Forward(const Tensor &input, bool is_training) {
  if (mask_.getBatchSize() != input.getBatchSize() || 
      mask_.getChannels() != input.getChannels() ||
      mask_.getRows() != input.getRows() || 
      mask_.getColumns() != input.getColumns()) {
    mask_ = Tensor(input.getBatchSize(), input.getChannels(), input.getRows(),
                   input.getColumns());
  }
  if (!is_training) {
    return input;
  }
  Tensor output = input;
  float drop_rate = 0.3f;
  float keep_rate = 1.0f - drop_rate;
  float scale = 1.0f / keep_rate;

#pragma omp parallel
  {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 mt(rd());
    std::bernoulli_distribution coinflip(keep_rate);

#pragma omp for collapse(4)
    for (size_t n = 0; n < input.getBatchSize(); n++) {
      for (size_t c = 0; c < input.getChannels(); c++) {
        for (size_t h = 0; h < input.getRows(); h++) {
          for (size_t w = 0; w < input.getColumns(); w++) {
            if (coinflip(mt)) {
              mask_(n, c, h, w) = scale;
              output(n, c, h, w) *= scale;
            } else {
              mask_(n, c, h, w) = 0.0f;
              output(n, c, h, w) = 0.0f;
            }
          }
        }
      }
    }
  }
  return output;
}

Tensor Dropout::Backward(const Tensor &grad_out) {
  Tensor grad_in = grad_out;
#pragma omp parallel for collapse(4)
  for (size_t n = 0; n < grad_out.getBatchSize(); n++) {
    for (size_t c = 0; c < grad_out.getChannels(); c++) {
      for (size_t h = 0; h < grad_out.getRows(); h++) {
        for (size_t w = 0; w < grad_out.getColumns(); w++) {
          grad_in(n, c, h, w) *= mask_(n, c, h, w);
        }
      }
    }
  }
  return grad_in;
}
