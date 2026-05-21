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
file(GLOB SOURCES "src/*.cpp" "main.cpp")

# 7. Create the executable
add_executable(NeuralNet ${SOURCES})

# 8. The OpenMP Windows Toolchain Fix
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Force LLVM/Clang to use its native OpenMP runtime.
    target_compile_options(NeuralNet PRIVATE -fopenmp)
    target_link_options(NeuralNet PRIVATE -fopenmp)
else()
    # Standard fallback for pure Linux/GCC environments
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(NeuralNet PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
# ==========================================
# TESTING CONFIGURATION (Manual Catch2)
# ==========================================

# Create the test executable. 
# We must compile the Catch2 .cpp file alongside your tests and your Tensor source.
add_executable(RunTests 
    tests/test_tensor.cpp 
    tests/test_loader.cpp
    tests/test_relu.cpp
    tests/test_sgd.cpp
    tests/test_conv2d.cpp
    tests/test_dense.cpp
    tests/test_flatten.cpp
    tests/vendor/catch_amalgamated.cpp 
    src/relu.cpp
    src/tensor.cpp
    src/conv2d.cpp
    src/sgd.cpp
    src/dense.cpp
    src/flatten.cpp
)

# Tell CMake where to find your headers AND the new Catch2 header
target_include_directories(RunTests PRIVATE 
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/tests/vendor
)

# OpenMP Windows Toolchain Fix (If you are using Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(RunTests PRIVATE -fopenmp)
    target_link_options(RunTests PRIVATE -fopenmp)
endif()
