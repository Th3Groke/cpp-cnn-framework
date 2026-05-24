#include "../inc/sgd.hpp"
#include <cstddef>
#include <vector>

SGD::SGD(std::vector<Tensor *> params, float lr, float momentum)
    : parameters_(params), learning_rate_(lr), momentum_(momentum) {
  for (auto param : parameters_) {
    Tensor v(param->getBatchSize(), param->getChannels(), param->getRows(),
             param->getColumns());
    velocities_.push_back(v);
  }
};

void SGD::Step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    Tensor *param = parameters_[i];
    Tensor &velocity = velocities_[i];

#pragma omp parallel for collapse(4)
    for (size_t n = 0; n < param->getBatchSize(); n++) {
      for (size_t c = 0; c < param->getChannels(); c++) {
        for (size_t h = 0; h < param->getRows(); h++) {
          for (size_t w = 0; w < param->getColumns(); w++) {
            float grad = param->grad(n, c, h, w);
            velocity(n, c, h, w) = (momentum_ * velocity(n, c, h, w)) + grad;
            (*param)(n, c, h, w) =
                (*param)(n, c, h, w) - (learning_rate_ * velocity(n, c, h, w));
          }
        }
      }
    }
  }
}
void SGD::DecayLearningRate(float decay_factor) {
  learning_rate_ *= decay_factor;
}
void SGD::ZeroGrad() {
  for (auto param : parameters_) {
    param->ZeroGrad();
  }
}
