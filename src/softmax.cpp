#include "../inc/softmax.hpp"
#include <algorithm>
#include <cmath>

Tensor Softmax::Forward(const Tensor &input) {
  Tensor output(input.getBatchSize(), input.getChannels(), input.getRows(),
                input.getColumns());

  for (size_t n = 0; n < output.getBatchSize(); ++n) {
    float max_val = -1e9;
    for (size_t c = 0; c < output.getChannels(); ++c) {
      if (input(n, c, 0, 0) > max_val) {
        max_val = input(n, c, 0, 0);
      }
    }
    float sum_exp = 0.0f;
    for (size_t c = 0; c < output.getChannels(); ++c) {
      float e = std::exp(input(n, c, 0, 0) - max_val);
      output(n, c, 0, 0) = e;
      sum_exp += e;
    }
    for (size_t c = 0; c < input.getChannels(); ++c) {
      output(n, c, 0, 0) /= sum_exp;
    }
  }
  output_cache = output;
  return output;
}

Tensor Softmax::Backward(const Tensor &grad_out) {
  Tensor grad_input(grad_out.getBatchSize(), grad_out.getChannels(), 1, 1);

  for (size_t n = 0; n < grad_out.getBatchSize(); ++n) {
    float dot_product = 0.0f;
    for (size_t c = 0; c < grad_out.getChannels(); ++c) {
      dot_product += grad_out(n, c, 0, 0) * output_cache(n, c, 0, 0);
    }
    for (size_t c = 0; c < grad_out.getChannels(); ++c) {
      float y = output_cache(n, c, 0, 0);
      grad_input(n, c, 0, 0) = y * (grad_out(n, c, 0, 0) - dot_product);
    }
  }
  return grad_input;
}
