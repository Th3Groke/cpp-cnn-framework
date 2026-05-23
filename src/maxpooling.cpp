#include "../inc/maxpooling.hpp"

MaxPooling::MaxPooling(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {};

Tensor MaxPooling::Forward(const Tensor &input, bool is_training) {
  size_t new_r = input.getRows() / stride;
  size_t new_c = input.getColumns() / stride;
  Tensor grad_out(input.getBatchSize(), input.getChannels(), new_r, new_c);
  mask_ = Tensor(input.getBatchSize(), input.getChannels(), input.getRows(),
                 input.getColumns());
  mask_.Fill(0.0f);

  for (size_t n = 0; n < grad_out.getBatchSize(); n++) {
    for (size_t c = 0; c < grad_out.getChannels(); c++) {
      for (size_t h = 0; h < grad_out.getRows(); h++) {
        for (size_t w = 0; w < grad_out.getColumns(); w++) {
          // loop through the 2x2 block in input image find highest float and
          // save it to the grad_out
          float highest = -1e9f;
          int best_w = -1;
          int best_h = -1;
          for (int i = 0; i < stride; i++) {
            for (int j = 0; j < stride; j++) {
              int in_h = (h * stride) + i;
              int in_w = (w * stride) + j;
              if (input(n, c, in_h, in_w) >= highest) {
                highest = input(n, c, in_h, in_w);
                best_w = in_w;
                best_h = in_h;
              }
            }
          }
          mask_(n, c, best_h, best_w) = 1.0f;
          grad_out(n, c, h, w) = highest;
        }
      }
    }
  }
  return grad_out;
}

Tensor MaxPooling::Backward(const Tensor &grad_out) {
  Tensor grad_input(mask_.getBatchSize(), mask_.getChannels(), mask_.getRows(),
                    mask_.getColumns());
  grad_input.Fill(0.0f);
  for (size_t n = 0; n < grad_out.getBatchSize(); n++) {
    for (size_t c = 0; c < grad_out.getChannels(); c++) {
      for (size_t h = 0; h < grad_out.getRows(); h++) {
        for (size_t w = 0; w < grad_out.getColumns(); w++) {
          float grad_value = grad_out(n, c, h, w);
          for (int i = 0; i < stride; i++) {
            for (int j = 0; j < stride; j++) {
              int in_h = (h * stride) + i;
              int in_w = (w * stride) + j;
              grad_input(n, c, in_h, in_w) =
                  grad_out(n, c, h, w) * mask_(n, c, in_h, in_w);
            }
          }
        }
      }
    }
  }
  return grad_input;
}
