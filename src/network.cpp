#include "../inc/network.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

void Network::AddLayer(std::unique_ptr<Layer> layer) {
  layers.push_back(std::move(layer));
}

Tensor Network::Forward(const Tensor &input, bool is_training) {
  Tensor current = input;
  for (const auto &lr : layers) {
    current = lr->Forward(current, is_training);
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
  for (const auto &lr : layers) {
    for (auto param : lr->GetParameters()) {
      all_params.push_back(param);
    }
  }
  return all_params;
}

void Network::SaveWeights(const std::string &filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open the file for saving weights: " +
                             filepath);
  }

  for (Tensor *param : GetParameters()) {
    param->Save(out);
  }
  out.close();
}

void Network::LoadWeights(const std::string &filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error(
        "Failed to open the file while loading the weights: " + filepath);
  }

  for (Tensor *param : GetParameters()) {
    param->Load(in);
  }
  in.close();
}
