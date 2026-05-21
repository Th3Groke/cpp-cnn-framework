#include "../inc/dense.hpp"
#include <cstddef>

Dense::Dense(size_t input_size, size_t output_size) {
  in_features = input_size;
  out_features = output_size;
  weights = Tensor(1, 1, out_features, in_features);
  weights.FillRandomNormal(0.0f, 0.1f);
  biases = Tensor(1, out_features, 1, 1);
  biases.Fill(0.0f);
}

std::vector<Tensor *> Dense::GetParameters() { return {&weights, &biases}; };
Tensor Dense::Forward(const Tensor &input) {
  input_cache = input;
  size_t batch_size = input.getBatchSize();
  Tensor output(batch_size, out_features, 1, 1);
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
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t out = 0; out < out_features; out++) {
      float d_out = grad_out(n, out, 0, 0);
      biases.grad(0, out, 0, 0) += d_out;
      for (size_t in = 0; in < in_features; in++) {
        weights.grad(0, 0, out, in) += input_cache(n, in, 0, 0) * d_out;
        grad_input(n, in, 0, 0) += weights(0, 0, out, in) * d_out;
      }
    }
  }
  return grad_input;
}
