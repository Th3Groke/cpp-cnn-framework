#include "../inc/conv2d.hpp"
#include "../inc/sgd.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("SGD Optimizer Pipeline Lifecycle", "[sgd][optimizer]") {
  // 1 Filter, 1 Channel, 2x2 Window
  Conv2d conv(1, 1, 2);

  // Force weights to 0.0f so the math is deterministic
  Tensor &f = const_cast<Tensor &>(conv.GetFilters());
  f(0, 0, 0, 0) = 0.0f;
  f(0, 0, 0, 1) = 0.0f;
  f(0, 0, 1, 0) = 0.0f;
  f(0, 0, 1, 1) = 0.0f;

  Tensor &b = const_cast<Tensor &>(conv.GetBiases());
  b(0, 0, 0, 0) = 0.0f;

  // Create inputs and gradients (all 1.0f for easy math) using the () operator
  Tensor input(1, 1, 3, 3);
  for (size_t h = 0; h < 3; ++h) {
    for (size_t w = 0; w < 3; ++w) {
      input(0, 0, h, w) = 1.0f;
    }
  }

  Tensor grad_out(1, 1, 2, 2);
  for (size_t h = 0; h < 2; ++h) {
    for (size_t w = 0; w < 2; ++w) {
      grad_out(0, 0, h, w) = 1.0f;
    }
  }

  // Execute the Pass
  conv.Forward(input);
  conv.Backward(grad_out);

  // 1. Initialize Optimizer with learning rate 0.1
  SGD sgd(conv.GetParameters(), 0.1f);

  // 2. Step the Weights
  sgd.Step();

  // Verify weights changed: 0.0 - (0.1 * 4.0) = -0.4
  REQUIRE_THAT(b(0, 0, 0, 0), Catch::Matchers::WithinRel(-0.4f, 0.0001f));
  REQUIRE_THAT(f(0, 0, 0, 0), Catch::Matchers::WithinRel(-0.4f, 0.0001f));

  // 3. Zero the Gradients
  sgd.ZeroGrad();

  // Verify gradients are wiped clean
  REQUIRE_THAT(b.grad(0, 0, 0, 0), Catch::Matchers::WithinRel(0.0f, 0.0001f));
  REQUIRE_THAT(f.grad(0, 0, 0, 0), Catch::Matchers::WithinRel(0.0f, 0.0001f));
}
