#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"
#include <cstdio> // For std::remove
#include <fstream>

TEST_CASE("Tensor Binary Serialization", "[tensor][io]") {
  std::string test_filename = "test_tensor_weights.bin";

  SECTION("Save and Load perfectly restores 32-bit floats") {
    // 1. Setup a tensor and fill it with distinct verifiable data
    Tensor original(2, 3, 4, 4);

    float val = 1.0f;
    for (size_t n = 0; n < original.getBatchSize(); ++n) {
      for (size_t c = 0; c < original.getChannels(); ++c) {
        for (size_t r = 0; r < original.getRows(); ++r) {
          for (size_t col = 0; col < original.getColumns(); ++col) {
            original(n, c, r, col) = val;
            val += 0.5f; // Ensures every single index is unique
          }
        }
      }
    }

    // 2. Save it to disk
    std::ofstream out(test_filename, std::ios::binary);
    REQUIRE(out.is_open());
    original.Save(out);
    out.close();

    // 3. Load it into a brand new, empty tensor of the same shape
    Tensor loaded(2, 3, 4, 4);
    std::ifstream in(test_filename, std::ios::binary);
    REQUIRE(in.is_open());
    loaded.Load(in);
    in.close();

    // 4. Verify dimensions survived
    REQUIRE(loaded.getBatchSize() == original.getBatchSize());
    REQUIRE(loaded.getChannels() == original.getChannels());
    REQUIRE(loaded.getRows() == original.getRows());
    REQUIRE(loaded.getColumns() == original.getColumns());

    // 5. Verify every single float matches perfectly
    bool all_match = true;
    for (size_t n = 0; n < loaded.getBatchSize(); ++n) {
      for (size_t c = 0; c < loaded.getChannels(); ++c) {
        for (size_t r = 0; r < loaded.getRows(); ++r) {
          for (size_t col = 0; col < loaded.getColumns(); ++col) {
            if (loaded(n, c, r, col) != original(n, c, r, col)) {
              all_match = false;
            }
          }
        }
      }
    }
    REQUIRE(all_match == true);
  }

  SECTION(
      "Loading into a Tensor of the wrong shape throws std::runtime_error") {
    // 1. Setup and save a standard tensor
    Tensor original(2, 3, 4, 4);
    std::ofstream out(test_filename, std::ios::binary);
    original.Save(out);
    out.close();

    // 2. Create a tensor with the WRONG dimensions
    Tensor wrong_shape(1, 1, 32, 32);
    std::ifstream in(test_filename, std::ios::binary);
    REQUIRE(in.is_open());

    // 3. Catch2 expects our runtime_error to trigger and prevent a segfault
    REQUIRE_THROWS_AS(wrong_shape.Load(in), std::runtime_error);

    in.close();
  }

  // Teardown: Clean up the temporary binary file
  std::remove(test_filename.c_str());
}
