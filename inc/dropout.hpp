#include "layer.hpp"
#include "tensor.hpp"
#include <vector>

class Dropout : public Layer {
private:
  Tensor mask_;

public:
Tensor Forward(const Tensor &input, bool is_training=false) override;
  Tensor Backward(const Tensor &grad_out) override;
  std::string GetLayerName() const override { return "Dropout"; };
  std::vector<Tensor *> GetParameters() override { return {}; };
};
