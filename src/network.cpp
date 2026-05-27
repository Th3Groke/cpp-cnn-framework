#include "../inc/network.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
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

void Network::SaveWeights(const std::string &filename,
                          const std::string &directory) {
  std::filesystem::path dir_path = directory;

  if (!std::filesystem::exists(dir_path)) {
    std::cout << "Directory " << dir_path << " not found. creating now..."
              << std::endl;
    if (std::filesystem::create_directories(dir_path)) {
      std::cout << "Created directory!" << std::endl;
    } else {
      std::cerr << "Could not create the directory" << std::endl;
      return;
    }
  }
  std::filesystem::path full_path = dir_path / filename;
  std::ofstream out(full_path, std::ios::binary);

  if (!out.is_open()) {
    throw std::runtime_error("Failed to open the file for saving weights: " +
                             filename);
  }

  for (Tensor *param : GetParameters()) {
    param->Save(out);
  }
  out.close();
  std::cout << "weights saved to " << full_path << std::endl;
}

void Network::LoadWeights(const std::string &filename,
                          const std::string &directory) {
  std::filesystem::path dir_path = directory;
  if (!std::filesystem::exists(dir_path)) {
    throw std::runtime_error("Could not find the directory " +
                             dir_path.string());
  }

  std::filesystem::path full_path = dir_path / filename;
  std::ifstream in(full_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error(
        "Failed to open the file while loading the weights: " +
        full_path.string());
  }

  for (Tensor *param : GetParameters()) {
    param->Load(in);
  }
  in.close();
}
