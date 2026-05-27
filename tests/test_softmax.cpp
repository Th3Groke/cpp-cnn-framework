#include "../inc/softmax.hpp"
#include "vendor/catch_amalgamated.hpp"
#include <cmath>

TEST_CASE("The forward pass (Probability Check)", "[Softmax]") {
  Softmax softmax;
  Tensor tensor(1, 3, 1, 1);
  tensor(0, 0, 0, 0) = 1.0f;
  tensor(0, 1, 0, 0) = 2.0f;
  tensor(0, 2, 0, 0) = 3.0f;

  Tensor output = softmax.Forward(tensor);

  float sum = 0.0f;
  for (int c = 0; c < 3; c++) {
    REQUIRE(output(0, c, 0, 0) >= 0.0f);
    REQUIRE(output(0, c, 0, 0) <= 1.0f);
    sum += output(0, c, 0, 0);
  }
  REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0f, 0.0001f));

  REQUIRE(output(0, 2, 0, 0) >= output(0, 1, 0, 0));
  REQUIRE(output(0, 2, 0, 0) >= output(0, 0, 0, 0));
}

TEST_CASE("Numerical Stability") {
  Softmax softmax;
  Tensor tensor(1, 3, 1, 1);
  tensor(0, 0, 0, 0) = 1000.0f;
  tensor(0, 1, 0, 0) = 2000.0f;
  tensor(0, 2, 0, 0) = 3000.0f;

  Tensor output = softmax.Forward(tensor);

  REQUIRE(output(0, 2, 0, 0) == 1.0f);
}

TEST_CASE("The backwards pass (shape & flow)") {
  Softmax softmax;
  Tensor tensor(1, 3, 1, 1);
  tensor(0, 0, 0, 0) = 1.0f;
  tensor(0, 1, 0, 0) = 2.0f;
  tensor(0, 2, 0, 0) = 3.0f;

  // We still need to call this to populate output_cache
  softmax.Forward(tensor);

  Tensor grad_out(1, 3, 1, 1);
  grad_out(0, 0, 0, 0) = 1.0f;
  grad_out(0, 1, 0, 0) = -1.0f;
  grad_out(0, 2, 0, 0) = 0.5f;

  Tensor grad_input = softmax.Backward(grad_out);

  REQUIRE(tensor.getBatchSize() == grad_input.getBatchSize());
  REQUIRE(tensor.getChannels() == grad_input.getChannels());
  REQUIRE(tensor.getRows() == grad_input.getRows());
  REQUIRE(tensor.getColumns() == grad_input.getColumns());

  for (int c = 0; c < 3; c++) {
    REQUIRE_FALSE(std::isnan(grad_input(0, c, 0, 0)));
  }

  REQUIRE_THAT(grad_input.GetData(),
               Catch::Matchers::AnyMatch(Catch::Matchers::Predicate<float>(
                   [](float v) { return std::abs(v) > 1e-6f; },
                   "Contains at least one non-zero gradient")));
}
