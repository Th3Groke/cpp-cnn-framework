#include "../inc/dense.hpp"
#include <cmath>
#include <cstddef>

Dense::Dense(size_t input_size, size_t output_size) {
  in_features = input_size;
  out_features = output_size;
  weights = Tensor(1, 1, out_features, in_features);
  float fan_in = (float)input_size;
  float std_dev = std::sqrt(2.0f / fan_in);
  weights.FillRandomNormal(0.0f, std_dev);
  biases = Tensor(1, out_features, 1, 1);
  biases.Fill(0.0f);
}

std::vector<Tensor *> Dense::GetParameters() { return {&weights, &biases}; };
Tensor Dense::Forward(const Tensor &input, bool is_training) {
  input_cache = input;
  size_t batch_size = input.getBatchSize();
  Tensor output(batch_size, out_features, 1, 1);
#pragma omp parallel for collapse(2)
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t out = 0; out < out_features; out++) {
      float neuron_sum = biases(0, out, 0, 0);
      for (size_t in = 0; in < in_features; in++) {
        neuron_sum += input(n, in, 0, 0) * weights(0, 0, out, in);
      }
      output(n, out, 0, 0) = neuron_sum;
    }
  }
  return output;
}

Tensor Dense::Backward(const Tensor &grad_out) {
  size_t batch_size = input_cache.getBatchSize();
  Tensor grad_input(batch_size, in_features, 1, 1);

  // Bias gradients
  for (size_t out = 0; out < out_features; out++) {
    float d_bias = 0.0f;
#pragma omp parallel for reduction(+ : d_bias)
    for (size_t n = 0; n < batch_size; n++) {
      d_bias += grad_out(n, out, 0, 0);
    }
    biases.grad(0, out, 0, 0) += d_bias;
  }

  // Weight and Input gradients
#pragma omp parallel for collapse(2)
  for (size_t out = 0; out < out_features; out++) {
    for (size_t in = 0; in < in_features; in++) {
      float d_weight = 0.0f;
      for (size_t n = 0; n < batch_size; n++) {
        d_weight += input_cache(n, in, 0, 0) * grad_out(n, out, 0, 0);
      }
      weights.grad(0, 0, out, in) += d_weight;
    }
  }

#pragma omp parallel for collapse(2)
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t in = 0; in < in_features; in++) {
      float d_input = 0.0f;
      for (size_t out = 0; out < out_features; out++) {
        d_input += weights(0, 0, out, in) * grad_out(n, out, 0, 0);
      }
      grad_input(n, in, 0, 0) = d_input;
    }
  }

  return grad_input;
}
