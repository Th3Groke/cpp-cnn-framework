#include <iostream>

#include "inc/tensor.hpp"

int main() {
  try {
    Tensor A(1, 1, 2, 3);
    A.FillRandomNormal(0.0f, 1.0f);
    A.Print("Tensor A");

    Tensor B(1, 1, 3, 2);
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 2; ++j)
        B(0, 0, i, j) = 2.0f;
    B.Print("Tensor B");

    Tensor C = Tensor::matmul(A, B);
    C.Print("MatMul Result (A*B)");

    std::cout << "Testing shape validation (should throw):" << std::endl;
    A += C; // This should throw runtime_error
  } catch (const std::exception &e) {
    std::cout << "Caught expected error: " << e.what() << std::endl;
  }

  return 0;
}
