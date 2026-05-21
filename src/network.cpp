#include "../inc/network.hpp"
#include <iostream>
#include <vector>

void Network::AddLayer(Layer *layer) { layers.push_back(layer); }

Tensor Network::Forward(Tensor input) {
  Tensor current = input;
  for (auto lr : layers) {
    current = lr->Forward(current);
  }
  return current;
}

Tensor Network::Backward(Tensor grad_out) {
  Tensor current_grad = grad_out;
  for (auto lr = layers.rbegin(); lr != layers.rend(); ++lr) {
    current_grad = (*lr)->Backward(current_grad);
  }
  return current_grad;
}

std::vector<Tensor *> Network::GetParameters() {
  std::vector<Tensor *> all_params;
  for (auto lr : layers) {
    for (auto param : lr->GetParameters()) {
      all_params.push_back(param);
    }
  }
  return all_params;
}
