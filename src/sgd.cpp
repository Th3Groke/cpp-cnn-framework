#include "../inc/sgd.hpp"
#include <vector>

SGD::SGD(std::vector<Tensor *> params, float lr) {
  parameters = params;
  learning_rate = lr;
};

void SGD::Step() {
  for (auto param : parameters) {
    param->Step(learning_rate);
  }
}

void SGD::ZeroGrad() {
  for (auto param : parameters) {
    param->ZeroGrad();
  }
}
