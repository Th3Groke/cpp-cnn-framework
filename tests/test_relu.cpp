#include "../inc/relu.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("ReLU Forward Pass applies max(0, x)", "[relu][forward]") {
  Tensor input(1, 1, 2, 2);
  input(0, 0, 0, 0) = 1.5f;   // Positive (stays)
  input(0, 0, 0, 1) = -0.5f;  // Negative (becomes 0)
  input(0, 0, 1, 0) = 0.0f;   // Zero (stays 0)
  input(0, 0, 1, 1) = -10.0f; // Negative (becomes 0)

  ReLU relu;
  Tensor output = relu.Forward(input);

  REQUIRE_THAT(output(0, 0, 0, 0), Catch::Matchers::WithinRel(1.5f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 0, 1), Catch::Matchers::WithinRel(0.0f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 1, 0), Catch::Matchers::WithinRel(0.0f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 1, 1), Catch::Matchers::WithinRel(0.0f, 0.0001f));
}

TEST_CASE("ReLU Backward Pass routes gradients correctly", "[relu][backward]") {
  Tensor input(1, 1, 2, 2);
  input(0, 0, 0, 0) = 2.0f;  // Gate open
  input(0, 0, 0, 1) = -1.0f; // Gate closed

  ReLU relu;
  // We must run Forward first to populate the input_cache
  relu.Forward(input);

  Tensor grad_out(1, 1, 2, 2);
  grad_out(0, 0, 0, 0) = 5.0f; // Should pass through
  grad_out(0, 0, 0, 1) = 3.0f; // Should be killed

  Tensor grad_in = relu.Backward(grad_out, 0.01f);

  REQUIRE_THAT(grad_in(0, 0, 0, 0), Catch::Matchers::WithinRel(5.0f, 0.0001f));
  REQUIRE_THAT(grad_in(0, 0, 0, 1), Catch::Matchers::WithinRel(0.0f, 0.0001f));
}
