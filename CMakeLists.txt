# 1. Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.20)

# 2. Name your project and set the language
project(NeuralNetCPP VERSION 1.0 LANGUAGES CXX)

# 3. Require C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. Turn on aggressive compiler warnings (crucial for catching C++ bugs early)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# 5. Tell CMake where to find your header files
include_directories(${CMAKE_SOURCE_DIR}/inc)

# 6. Collect all your source files
set(SOURCES 
    src/cifar10loader.cpp
    src/conv2d.cpp
    src/dense.cpp
    src/dropout.cpp
    src/flatten.cpp
    src/maxpooling.cpp
    src/network.cpp
    src/relu.cpp
    src/sgd.cpp
    src/softmax.cpp
    src/tensor.cpp
    main.cpp
)

# 7. Create the executable
add_executable(NeuralNet ${SOURCES})

# 8. Testing Configuration
add_executable(RunTests 
    tests/test_tensor.cpp 
    tests/test_loader.cpp
    tests/test_relu.cpp
    tests/test_sgd.cpp
    tests/test_conv2d.cpp
    tests/test_dense.cpp
    tests/test_flatten.cpp
    tests/test_network.cpp
    tests/test_softmax.cpp
    tests/test_IO.cpp
    tests/vendor/catch_amalgamated.cpp 
    src/relu.cpp
    src/tensor.cpp
    src/softmax.cpp
    src/conv2d.cpp
    src/sgd.cpp
    src/dense.cpp
    src/flatten.cpp
    src/network.cpp
    src/cifar10loader.cpp
    src/dropout.cpp
    src/maxpooling.cpp
)

target_include_directories(RunTests PRIVATE 
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/tests/vendor
)

# 9. OpenMP Configuration
find_package(OpenMP REQUIRED)

# Apply to NeuralNet
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(NeuralNet PRIVATE -fopenmp)
    target_link_options(NeuralNet PRIVATE -fopenmp)
else()
    target_link_libraries(NeuralNet PUBLIC OpenMP::OpenMP_CXX)
endif()

# Apply to RunTests
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(RunTests PRIVATE -fopenmp)
    target_link_options(RunTests PRIVATE -fopenmp)
else()
    target_link_libraries(RunTests PUBLIC OpenMP::OpenMP_CXX)
endif()
