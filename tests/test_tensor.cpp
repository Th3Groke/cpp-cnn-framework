#include "../inc/tensor.hpp"
#include "vendor/catch_amalgamated.hpp"

TEST_CASE("Tensor Memory Allocation and Indexing", "[tensor][memory]") {
  // Initialize a small tensor: 2 batches, 1 channel, 3x3 image
  Tensor t(2, 1, 3, 3);

  SECTION("Out-of-bounds indexing triggers assertions") {
    // If you used asserts in your operator(), this helps document expected
    // boundaries. Catch2 doesn't easily catch C-style asserts without crashing,
    // but it's good to note here for your own testing methodology.
  }

  SECTION("Operator() correctly writes and reads memory") {
    // Write to the very first slot
    t(0, 0, 0, 0) = 42.5f;
    // Write to the dead center of the second batch
    t(1, 0, 1, 1) = -15.0f;
    // Write to the absolute last slot
    t(1, 0, 2, 2) = 3.14f;

    REQUIRE_THAT(t(0, 0, 0, 0), Catch::Matchers::WithinRel(42.5f, 0.0001f));
    REQUIRE_THAT(t(1, 0, 1, 1), Catch::Matchers::WithinRel(-15.0f, 0.0001f));
    REQUIRE_THAT(t(1, 0, 2, 2), Catch::Matchers::WithinRel(3.14f, 0.0001f));
  }
}

TEST_CASE("Tensor Scalar Arithmetic", "[tensor][math]") {
  Tensor a(1, 1, 2, 2);

  // Setup a predictable 2x2 matrix
  a(0, 0, 0, 0) = 1.0f;
  a(0, 0, 0, 1) = 2.0f;
  a(0, 0, 1, 0) = 3.0f;
  a(0, 0, 1, 1) = 4.0f;

  SECTION("Scalar Multiplication (*=)") {
    a *= 2.5f;

    REQUIRE_THAT(a(0, 0, 0, 0), Catch::Matchers::WithinRel(2.5f, 0.0001f));
    REQUIRE_THAT(a(0, 0, 0, 1), Catch::Matchers::WithinRel(5.0f, 0.0001f));
    REQUIRE_THAT(a(0, 0, 1, 0), Catch::Matchers::WithinRel(7.5f, 0.0001f));
    REQUIRE_THAT(a(0, 0, 1, 1), Catch::Matchers::WithinRel(10.0f, 0.0001f));
  }
}

TEST_CASE("Tensor Matrix Multiplication (Dot Product)",
          "[tensor][math][matmul]") {
  // Matrix A: 1x2
  Tensor A(1, 1, 1, 2);
  A(0, 0, 0, 0) = 1.0f;
  A(0, 0, 0, 1) = 2.0f;

  // Matrix B: 2x2
  Tensor B(1, 1, 2, 2);
  B(0, 0, 0, 0) = 3.0f;
  B(0, 0, 0, 1) = 4.0f;
  B(0, 0, 1, 0) = 5.0f;
  B(0, 0, 1, 1) = 6.0f;

  SECTION("1x2 multiplied by 2x2 results in 1x2") {
    // This assumes your matmul method returns a new Tensor
    Tensor C = Tensor::matmul(A, B);

    // Expected Math:
    // C(0,0) = (1*3) + (2*5) = 13
    // C(0,1) = (1*4) + (2*6) = 16

    // Uncomment these once your .matmul() is implemented!
    REQUIRE_THAT(C(0, 0, 0, 0), Catch::Matchers::WithinRel(13.0f, 0.0001f));
    REQUIRE_THAT(C(0, 0, 0, 1), Catch::Matchers::WithinRel(16.0f, 0.0001f));
  }
}
