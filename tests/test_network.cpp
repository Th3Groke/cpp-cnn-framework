#include "../inc/conv2d.hpp"
#include "../inc/dense.hpp"
#include "../inc/flatten.hpp"
#include "../inc/network.hpp"
#include "../inc/sgd.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("Full Network Training Loop", "[network][integration]") {
  // 1. Build the Architecture
  Network net;

  // Input is 1 channel. We want 2 output channels, using a 2x2 filter.
  // A 3x3 input image will become a 2x2 spatial output.
  Conv2d conv(1, 2, 2);
  Flatten flatten;
  // Conv output is (Batch, 2 channels, 2 rows, 2 cols) = 8 total features
  Dense dense(8, 2);

  net.AddLayer(&conv);
  net.AddLayer(&flatten);
  net.AddLayer(&dense);

  // 2. Setup the Optimizer
  SGD optimizer(net.GetParameters(), 0.05f); // Learning rate 0.05

  // 3. Create Dummy Data
  Tensor input(1, 1, 3, 3);
  input.Fill(0.5f); // Gray image

  // Target: We want the network to output exactly [1.0, 0.0]
  float target_0 = 1.0f;
  float target_1 = 0.0f;

  // Record initial error so we can prove it goes down
  Tensor initial_output = net.Forward(input);
  float initial_error_0 = std::abs(initial_output(0, 0, 0, 0) - target_0);
  float initial_error_1 = std::abs(initial_output(0, 1, 0, 0) - target_1);
  float initial_total_error = initial_error_0 + initial_error_1;

  // 4. The Training Loop (50 Epochs)
  for (int epoch = 0; epoch < 50; epoch++) {
    // Forward Pass
    Tensor output = net.Forward(input);

    // Calculate Gradients (Derivative of Mean Squared Error: Output - Target)
    Tensor grad_out(1, 2, 1, 1);
    grad_out(0, 0, 0, 0) = output(0, 0, 0, 0) - target_0;
    grad_out(0, 1, 0, 0) = output(0, 1, 0, 0) - target_1;

    // Backward Pass
    net.Backward(grad_out);

    // Update Weights
    optimizer.Step();
    optimizer.ZeroGrad();
  }

  // 5. Verify the Network Learned
  Tensor final_output = net.Forward(input);
  float final_error_0 = std::abs(final_output(0, 0, 0, 0) - target_0);
  float final_error_1 = std::abs(final_output(0, 1, 0, 0) - target_1);
  float final_total_error = final_error_0 + final_error_1;

  // The total error must be significantly lower after training
  REQUIRE(final_total_error < initial_total_error);

  // The outputs should be converging toward [1.0, 0.0]
  REQUIRE_THAT(final_output(0, 0, 0, 0),
               Catch::Matchers::WithinAbs(1.0f, 0.1f));
  REQUIRE_THAT(final_output(0, 1, 0, 0),
               Catch::Matchers::WithinAbs(0.0f, 0.1f));
}
