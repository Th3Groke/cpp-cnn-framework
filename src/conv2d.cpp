#include "../inc/conv2d.hpp"
#include <cmath>
#include <cstddef>

Conv2d::Conv2d(int input_channels, int num_filters, int filter_size)
    : num_filters(num_filters), input_channels(input_channels),
      filter_size(filter_size),
      filters(num_filters, input_channels, filter_size, filter_size),
      biases(num_filters, 1, 1, 1) {
  float fan_in = (float)(input_channels * filter_size * filter_size);
  float std_dev = std::sqrt(2.0f / fan_in);
  filters.FillRandomNormal(0.0f, std_dev);
  biases.Fill(0.0f);
};

Tensor Conv2d::Forward(const Tensor &input, bool is_training) {
  input_cache = input;
  Tensor output(input.getBatchSize(), num_filters,
                input.getRows() - filter_size + 1,
                input.getColumns() - filter_size + 1);

#pragma omp parallel for collapse(2)
  for (size_t n = 0; n < input.getBatchSize(); n++) {
    for (int f = 0; f < num_filters; f++) {
      for (size_t out_h = 0; out_h < output.getRows(); out_h++) {
        for (size_t out_w = 0; out_w < output.getColumns(); out_w++) {
          float sum = 0.0f;
          for (size_t c = 0; c < input.getChannels(); c++) {
            for (int f_h = 0; f_h < filter_size; f_h++) {
              for (int f_w = 0; f_w < filter_size; f_w++) {
                sum += input_cache(n, c, out_h + f_h, out_w + f_w) *
                       filters(f, c, f_h, f_w);
              }
            }
          }
          output(n, f, out_h, out_w) = sum + biases(f, 0, 0, 0);
        }
      }
    }
  }
  return output;
};
Tensor Conv2d::Backward(const Tensor &grad_out) {
  Tensor grad_input(input_cache.getBatchSize(), input_cache.getChannels(),
                    input_cache.getRows(), input_cache.getColumns());

  // Bias Gradients and update
  for (int f = 0; f < num_filters; f++) {
    float d_bias = 0.0f;
#pragma omp parallel for reduction(+ : d_bias) collapse(3)
    for (size_t n = 0; n < grad_out.getBatchSize(); n++) {
      for (size_t h = 0; h < grad_out.getRows(); h++) {
        for (size_t w = 0; w < grad_out.getColumns(); w++) {
          d_bias += grad_out(n, f, h, w);
        }
      }
    }
    biases.grad(f, 0, 0, 0) += d_bias;
  }

  // filter Gradients and update
#pragma omp parallel for collapse(2)
  for (int f = 0; f < num_filters; f++) {
    for (size_t c = 0; c < input_cache.getChannels(); c++) {
      for (int f_h = 0; f_h < filter_size; f_h++) {
        for (int f_w = 0; f_w < filter_size; f_w++) {
          float d_filter = 0.0f;

          for (size_t n = 0; n < grad_out.getBatchSize(); n++) {
            for (size_t out_h = 0; out_h < grad_out.getRows(); out_h++) {
              for (size_t out_w = 0; out_w < grad_out.getColumns(); out_w++) {
                d_filter += input_cache(n, c, out_h + f_h, out_w + f_w) *
                            grad_out(n, f, out_h, out_w);
              }
            }
          }
          filters.grad(f, c, f_h, f_w) += d_filter;
        }
      }
    }
  }

  // input gradients
#pragma omp parallel for collapse(2)
  for (size_t n = 0; n < input_cache.getBatchSize(); n++) {
    for (size_t c = 0; c < input_cache.getChannels(); c++) {
      for (size_t i_h = 0; i_h < input_cache.getRows(); i_h++) {
        for (size_t i_w = 0; i_w < input_cache.getColumns(); i_w++) {
          float d_input = 0.0f;
          for (int f = 0; f < num_filters; f++) {
            for (int f_h = 0; f_h < filter_size; f_h++) {
              for (int f_w = 0; f_w < filter_size; f_w++) {
                int out_h = (int)i_h - f_h;
                int out_w = (int)i_w - f_w;
                if (out_h >= 0 && out_h < (int)grad_out.getRows() &&
                    out_w >= 0 && out_w < (int)grad_out.getColumns()) {
                  d_input += grad_out(n, f, out_h, out_w) *
                             filters(f, c, f_h, f_w);
                }
              }
            }
          }
          grad_input(n, c, i_h, i_w) = d_input;
        }
      }
    }
  }
  return grad_input;
};
