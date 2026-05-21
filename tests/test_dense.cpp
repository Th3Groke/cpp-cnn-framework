#include "../inc/dense.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("Dense Layer Forward and Backward Pass", "[dense]") {
  // 3 Input Features, 2 Output Features
  Dense dense(3, 2);
  // Override weights for deterministic testing
  Tensor &w = const_cast<Tensor &>(dense.GetWeights());
  // out 0 weights
  w(0, 0, 0, 0) = 0.1f;
  w(0, 0, 0, 1) = 0.2f;
  w(0, 0, 0, 2) = 0.3f;
  // out 1 weights
  w(0, 0, 1, 0) = 0.4f;
  w(0, 0, 1, 1) = 0.5f;
  w(0, 0, 1, 2) = 0.6f;

  Tensor &b = const_cast<Tensor &>(dense.GetBiases());
  b(0, 0, 0, 0) = 0.1f;
  b(0, 1, 0, 0) = -0.1f;

  // Create Input: (1 Batch, 3 Features, 1, 1)
  Tensor input(1, 3, 1, 1);
  input(0, 0, 0, 0) = 1.0f;
  input(0, 1, 0, 0) = 2.0f;
  input(0, 2, 0, 0) = 3.0f;

  // --- FORWARD PASS ---
  Tensor output = dense.Forward(input);

  REQUIRE(output.getBatchSize() == 1);
  REQUIRE(output.getChannels() == 2);

  // out_0 = (1.0*0.1 + 2.0*0.2 + 3.0*0.3) + 0.1 = 1.5
  REQUIRE_THAT(output(0, 0, 0, 0), Catch::Matchers::WithinRel(1.5f, 0.0001f));
  // out_1 = (1.0*0.4 + 2.0*0.5 + 3.0*0.6) - 0.1 = 3.1
  REQUIRE_THAT(output(0, 1, 0, 0), Catch::Matchers::WithinRel(3.1f, 0.0001f));

  // --- BACKWARD PASS ---
  Tensor grad_out(1, 2, 1, 1);
  grad_out(0, 0, 0, 0) = 1.0f; // Error of 1.0 for out_0
  grad_out(0, 1, 0, 0) = 2.0f; // Error of 2.0 for out_1

  Tensor grad_input = dense.Backward(grad_out);

  // Check Bias Gradients
  REQUIRE_THAT(b.grad(0, 0, 0, 0), Catch::Matchers::WithinRel(1.0f, 0.0001f));
  REQUIRE_THAT(b.grad(0, 1, 0, 0), Catch::Matchers::WithinRel(2.0f, 0.0001f));

  // Check Weight Gradients (grad_out * input)
  // w_grad for out_1, in_2 = 2.0 * 3.0 = 6.0
  REQUIRE_THAT(w.grad(0, 0, 1, 2), Catch::Matchers::WithinRel(6.0f, 0.0001f));

  // Check Input Gradients (grad_out * weights)
  // in_grad for feature 0 = (1.0 * w(out0,in0)) + (2.0 * w(out1,in0))
  //                       = (1.0 * 0.1) + (2.0 * 0.4) = 0.9
  REQUIRE_THAT(grad_input(0, 0, 0, 0),
               Catch::Matchers::WithinRel(0.9f, 0.0001f));

  // --- PARAMETER HOOK ---
  auto params = dense.GetParameters();
  REQUIRE(params.size() == 2);
}
