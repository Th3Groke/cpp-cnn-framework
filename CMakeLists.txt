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

# 5. Find OpenMP for multi-threading (speeds up matrix multiplication)
find_package(OpenMP)

# 6. Tell CMake where to find your header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# 7. Collect all your source files
file(GLOB SOURCES "src/*.cpp")

# 8. Create the executable
add_executable(NeuralNet ${SOURCES})

# 9. Link OpenMP if it was found
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralNet PUBLIC OpenMP::OpenMP_CXX)
endif()
