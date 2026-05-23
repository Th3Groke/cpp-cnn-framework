#include "../inc/relu.hpp"

Tensor ReLU::Forward(const Tensor &input, bool is_training) {
  input_cache = input;
  Tensor output = input;
  for (size_t n = 0; n < output.getBatchSize(); n++) {
    for (size_t c = 0; c < output.getChannels(); c++) {
      for (size_t h = 0; h < output.getRows(); h++) {
        for (size_t w = 0; w < output.getColumns(); w++) {
          float &value = output(n, c, h, w);
          if (value < 0) {
            value = 0.0f;
          }
        }
      }
    }
  }
  return output;
}

Tensor ReLU::Backward(const Tensor &grad_out) {
  Tensor grad_input = Tensor(grad_out);
  for (size_t n = 0; n < input_cache.getBatchSize(); n++) {
    for (size_t c = 0; c < input_cache.getChannels(); c++) {
      for (size_t h = 0; h < input_cache.getRows(); h++) {
        for (size_t w = 0; w < input_cache.getColumns(); w++) {
          if (input_cache(n, c, h, w) <= 0) {
            grad_input(n, c, h, w) = 0.0f;
          }
        }
      }
    }
  }
  return grad_input;
}
