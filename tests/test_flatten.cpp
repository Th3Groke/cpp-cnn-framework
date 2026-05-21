#include "../inc/flatten.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("Flatten Layer Reshapes Data Correctly", "[flatten]") {
  Flatten flatten;

  // Create a mock image batch: 2 images, 3 channels (RGB), 2x2 pixels
  Tensor input(2, 3, 2, 2);

  // Fill with sequential data so we can verify the memory order didn't get
  // scrambled
  float counter = 1.0f;
  for (size_t b = 0; b < 2; ++b) {
    for (size_t c = 0; c < 3; ++c) {
      for (size_t h = 0; h < 2; ++h) {
        for (size_t w = 0; w < 2; ++w) {
          input(b, c, h, w) = counter++;
        }
      }
    }
  }

  // --- FORWARD PASS ---
  Tensor output = flatten.Forward(input);

  // Verify Shape: Batch size should stay 2, channels should become 3 * 2 * 2 =
  // 12
  REQUIRE(output.getBatchSize() == 2);
  REQUIRE(output.getChannels() == 12);
  REQUIRE(output.getRows() == 1);
  REQUIRE(output.getColumns() == 1);

  // Verify Data Integrity: The underlying memory block should be completely
  // untouched
  const std::vector<float> &out_data = output.GetData();
  REQUIRE(out_data.size() == 24); // 2 * 12

  for (size_t i = 0; i < out_data.size(); ++i) {
    // Values should perfectly match our 1.0 to 24.0 counter
    REQUIRE_THAT(out_data[i], Catch::Matchers::WithinRel(
                                  static_cast<float>(i + 1), 0.0001f));
  }

  // --- BACKWARD PASS ---
  // Create a mock gradient returning from the Dense layer
  Tensor grad_out(2, 12, 1, 1);
  for (size_t b = 0; b < 2; ++b) {
    for (size_t f = 0; f < 12; ++f) {
      grad_out(b, f, 0, 0) = -1.0f; // Mock gradient error
    }
  }

  Tensor grad_input = flatten.Backward(grad_out);

  // Verify Shape: Should inflate exactly back to (2, 3, 2, 2)
  REQUIRE(grad_input.getBatchSize() == 2);
  REQUIRE(grad_input.getChannels() == 3);
  REQUIRE(grad_input.getRows() == 2);
  REQUIRE(grad_input.getColumns() == 2);

  // Verify Data Integrity: Everything should be -1.0f
  const std::vector<float> &grad_in_data = grad_input.GetData();
  for (size_t i = 0; i < grad_in_data.size(); ++i) {
    REQUIRE_THAT(grad_in_data[i], Catch::Matchers::WithinRel(-1.0f, 0.0001f));
  }

  // --- OPTIMIZER HOOK ---
  // Flatten has no weights to train
  auto params = flatten.GetParameters();
  REQUIRE(params.empty());
}
