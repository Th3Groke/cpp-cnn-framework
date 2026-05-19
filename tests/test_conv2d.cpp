#include "../inc/conv2d.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("Conv2D Memory Allocation and Initialization", "[conv2d][memory]") {
  // Setup: 16 filters, 3 color channels, 3x3 sliding window
  Conv2d conv(16, 3, 3);

  SECTION("Layer Identity Contract") {
    REQUIRE(conv.GetLayerName() == "Conv2D");
  }

  SECTION("Filters are allocated with correct 4D shapes") {
    const Tensor &f = conv.GetFilters();

    REQUIRE(f.getBatchSize() == 16); // 16 different filters
    REQUIRE(f.getChannels() == 3);   // looking at 3 input channels
    REQUIRE(f.getRows() == 3);       // 3 pixels tall
    REQUIRE(f.getColumns() == 3);    // 3 pixels wide

    // Verify that the Gaussian noise generator actually filled the tensor.
    // It is mathematically highly improbable that a random float is exactly
    // 0.0f.
    REQUIRE(f(0, 0, 0, 0) != 0.0f);
  }

  SECTION("Biases are allocated and default to zero") {
    const Tensor &b = conv.GetBiases();

    REQUIRE(b.getBatchSize() == 16); // 1 bias per filter
    REQUIRE(b.getChannels() == 1);
    REQUIRE(b.getRows() == 1);
    REQUIRE(b.getColumns() == 1);

    // Biases should start perfectly blank
    REQUIRE_THAT(b(0, 0, 0, 0), Catch::Matchers::WithinRel(0.0f, 0.0001f));
  }
}
TEST_CASE("Conv2D Forward Pass Math Verification", "[conv2d][forward]") {
  // 1 Batch, 1 Channel, 3x3 Image
  Tensor input(1, 1, 3, 3);
  input(0, 0, 0, 0) = 1.0f;
  input(0, 0, 0, 1) = 2.0f;
  input(0, 0, 0, 2) = 3.0f;
  input(0, 0, 1, 0) = 4.0f;
  input(0, 0, 1, 1) = 5.0f;
  input(0, 0, 1, 2) = 6.0f;
  input(0, 0, 2, 0) = 7.0f;
  input(0, 0, 2, 1) = 8.0f;
  input(0, 0, 2, 2) = 9.0f;

  // 1 Filter, 1 Channel, 2x2 Window
  Conv2d conv(1, 1, 2);

  // Testing Hack: Use const_cast to overwrite the random weights for
  // deterministic math
  Tensor &f = const_cast<Tensor &>(conv.GetFilters());
  f(0, 0, 0, 0) = 1.0f;
  f(0, 0, 0, 1) = 0.0f;
  f(0, 0, 1, 0) = 0.0f;
  f(0, 0, 1, 1) = 1.0f;

  Tensor &b = const_cast<Tensor &>(conv.GetBiases());
  b(0, 0, 0, 0) = 1.0f; // Add a bias of 1.0 to everything

  // Run the forward pass!
  Tensor output = conv.Forward(input);

  // Verify Output Dimensions
  REQUIRE(output.getBatchSize() == 1);
  REQUIRE(output.getChannels() == 1);
  REQUIRE(output.getRows() == 2);
  REQUIRE(output.getColumns() == 2);

  // Verify Calculus (Cross-Correlation + Bias)
  REQUIRE_THAT(output(0, 0, 0, 0), Catch::Matchers::WithinRel(7.0f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 0, 1), Catch::Matchers::WithinRel(9.0f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 1, 0), Catch::Matchers::WithinRel(13.0f, 0.0001f));
  REQUIRE_THAT(output(0, 0, 1, 1), Catch::Matchers::WithinRel(15.0f, 0.0001f));
}
TEST_CASE("Conv2D Backward Pass Bias Update", "[conv2d][backward][bias]") {
  // 2 Filters, 1 Input Channel, 3x3 Window
  Conv2d conv(2, 1, 3);

  Tensor dummy_input(1, 1, 4, 4);
  conv.Forward(dummy_input);

  Tensor grad_out(1, 2, 2, 2);

  // Filter 0's incoming gradients (Sum = 4.0)
  grad_out(0, 0, 0, 0) = 1.0f;
  grad_out(0, 0, 0, 1) = 1.0f;
  grad_out(0, 0, 1, 0) = 1.0f;
  grad_out(0, 0, 1, 1) = 1.0f;

  // Filter 1's incoming gradients (Sum = 8.0)
  grad_out(0, 1, 0, 0) = 2.0f;
  grad_out(0, 1, 0, 1) = 2.0f;
  grad_out(0, 1, 1, 0) = 2.0f;
  grad_out(0, 1, 1, 1) = 2.0f;

  // Run backward pass (No learning rate needed anymore!)
  conv.Backward(grad_out);

  const Tensor &b = conv.GetBiases();

  // Bias gradients should exactly match the sums
  REQUIRE_THAT(b.grad(0, 0, 0, 0), Catch::Matchers::WithinRel(4.0f, 0.0001f));
  REQUIRE_THAT(b.grad(1, 0, 0, 0), Catch::Matchers::WithinRel(8.0f, 0.0001f));
}
TEST_CASE("Conv2D Backward Pass Filter Update", "[conv2d][backward][filters]") {
  Conv2d conv(1, 1, 2);

  Tensor input(1, 1, 3, 3);
  input(0, 0, 0, 0) = 1.0f;
  input(0, 0, 0, 1) = 2.0f;
  input(0, 0, 0, 2) = 3.0f;
  input(0, 0, 1, 0) = 4.0f;
  input(0, 0, 1, 1) = 5.0f;
  input(0, 0, 1, 2) = 6.0f;
  input(0, 0, 2, 0) = 7.0f;
  input(0, 0, 2, 1) = 8.0f;
  input(0, 0, 2, 2) = 9.0f;

  conv.Forward(input);

  Tensor grad_out(1, 1, 2, 2);
  grad_out(0, 0, 0, 0) = 1.0f;
  grad_out(0, 0, 0, 1) = 1.0f;
  grad_out(0, 0, 1, 0) = 1.0f;
  grad_out(0, 0, 1, 1) = 1.0f;

  Tensor &f = const_cast<Tensor &>(conv.GetFilters());
  f(0, 0, 0, 0) = 0.0f;
  f(0, 0, 0, 1) = 0.0f;
  f(0, 0, 1, 0) = 0.0f;
  f(0, 0, 1, 1) = 0.0f;

  // Run backward pass!
  conv.Backward(grad_out);

  const Tensor &updated_f = conv.GetFilters();

  // Check the raw gradient accumulations
  // filter(0,0) sum: 1.0 + 2.0 + 4.0 + 5.0 = 12.0
  REQUIRE_THAT(updated_f.grad(0, 0, 0, 0),
               Catch::Matchers::WithinRel(12.0f, 0.0001f));

  // filter(0,1) sum: 2.0 + 3.0 + 5.0 + 6.0 = 16.0
  REQUIRE_THAT(updated_f.grad(0, 0, 0, 1),
               Catch::Matchers::WithinRel(16.0f, 0.0001f));
}
TEST_CASE("Conv2D Backward Pass Input Gradient Update",
          "[conv2d][backward][input]") {
  // 1 Filter, 1 Channel, 2x2 Window
  Conv2d conv(1, 1, 2);

  // Create a 3x3 input image
  Tensor input(1, 1, 3, 3);
  // Fill with dummy data just to shape the cache
  input.FillRandomNormal(0.0f, 1.0f);
  conv.Forward(input);

  // Create a fake grad_out tensor (1 Batch, 1 Channel, 2x2 Spatial)
  // We pretend the error is exactly 1.0 everywhere
  Tensor grad_out(1, 1, 2, 2);
  grad_out(0, 0, 0, 0) = 1.0f;
  grad_out(0, 0, 0, 1) = 1.0f;
  grad_out(0, 0, 1, 0) = 1.0f;
  grad_out(0, 0, 1, 1) = 1.0f;

  // Overwrite filters deterministically
  Tensor &f = const_cast<Tensor &>(conv.GetFilters());
  f(0, 0, 0, 0) = 1.0f;
  f(0, 0, 0, 1) = 2.0f;
  f(0, 0, 1, 0) = 3.0f;
  f(0, 0, 1, 1) = 4.0f;

  // Run backward pass!
  Tensor grad_input = conv.Backward(grad_out);

  // Verify Output Dimensions match the original 3x3 input
  REQUIRE(grad_input.getBatchSize() == 1);
  REQUIRE(grad_input.getChannels() == 1);
  REQUIRE(grad_input.getRows() == 3);
  REQUIRE(grad_input.getColumns() == 3);

  // Verify Calculus (Reverse Accumulation)
  // Top-left pixel only receives gradient from the first sliding window
  REQUIRE_THAT(grad_input(0, 0, 0, 0),
               Catch::Matchers::WithinRel(1.0f, 0.0001f));

  // Middle pixel (1,1) is touched by all 4 sliding windows!
  // It should accumulate: 1.0(bottom-right of filter) + 2.0(bottom-left)
  // + 3.0(top-right) + 4.0(top-left) = 10.0
  REQUIRE_THAT(grad_input(0, 0, 1, 1),
               Catch::Matchers::WithinRel(10.0f, 0.0001f));
}
