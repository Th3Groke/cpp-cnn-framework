#include "../inc/cifar10loader.hpp"
#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"
#include <cstdio> // For std::remove (deleting the mock file)
#include <fstream>
#include <vector>

TEST_CASE("CIFAR-10 Loader correctly reads and normalizes binary bytes",
          "[loader][io]") {
  // 1. Setup: Define the mock file path and the predictable test data
  std::string mockFilePath = "test_mock_batch.bin";
  uint8_t expectedLabel = 5;
  uint8_t firstPixelValue = 127; // 127 / 255.0f should equal ~0.498f

  // 2. Setup: Generate the fake binary file
  std::ofstream out(mockFilePath, std::ios::binary);
  REQUIRE(out.is_open()); // Ensure the test file was actually created

  out.write(reinterpret_cast<char *>(&expectedLabel), 1);

  // Create a 3072-byte image buffer initialized to 0
  std::vector<uint8_t> fakeImage(3072, 0);
  fakeImage[0] = firstPixelValue; // Only set the very first Red pixel to 127
  out.write(reinterpret_cast<char *>(fakeImage.data()), 3072);
  out.close();

  SECTION("Loader populates the Tensor accurately") {
    // We only mock 1 image, so our Tensor only needs a batch size of 1
    Tensor t(1, 3, 32, 32);
    Cifar10Loader loader;

    // Execute the loader on our 1-image mock file
    loader.LoadCifar10(t, mockFilePath);

    // Verify: Did 127 turn into ~0.498f?
    float expectedFloat = 127.0f / 255.0f;
    REQUIRE_THAT(t(0, 0, 0, 0),
                 Catch::Matchers::WithinRel(expectedFloat, 0.0001f));

    // Verify: Did the rest of the empty space stay 0.0f?
    REQUIRE_THAT(t(0, 0, 0, 1), Catch::Matchers::WithinRel(0.0f, 0.0001f));
  }

  // 3. Teardown: Delete the fake file so we don't pollute the workspace
  std::remove(mockFilePath.c_str());
}
