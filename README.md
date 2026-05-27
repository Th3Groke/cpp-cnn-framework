# C++ Deep Learning Engine 🧠⚙️

![C++](https://img.shields.io/badge/C++-17%2F20-blue.svg?style=flat&logo=c%2B%2B)
![CMake](https://img.shields.io/badge/CMake-Build_System-brightgreen.svg?logo=cmake)
![OpenMP](https://img.shields.io/badge/OpenMP-Multi--Threading-orange.svg)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-success.svg)

A high-performance, from-scratch Convolutional Neural Network (CNN) framework written entirely in modern C++. 

This project was built as a final project for semester 4 of my studies. I've chosen it to explore the low-level mathematics and systems-engineering challenges of deep learning without relying on high-level tensor libraries like PyTorch or TensorFlow. It features custom forward and backward propagation, multithreaded tensor operations, and concurrent dataset I/O.

## 🚀 Key Engineering Features

* **Zero External ML Dependencies:** Every mathematical operation, from matrix convolution to the Cross-Entropy loss chain rule, is manually implemented.
* **CPU Multi-Threading (OpenMP):** Heavy tensor multiplications are accelerated across all available CPU cores using `#pragma omp parallel for collapse(N)` directives.
* **Asynchronous I/O (`std::async`):** Bypasses disk bottlenecks by loading 50,000 CIFAR-10 training images concurrently via modern C++ `<future>` workers.
* **Advanced Optimization:** Features a custom Stochastic Gradient Descent (SGD) optimizer augmented with **Momentum** to prevent gradient shock and accelerate convergence.
* **Inverted Dropout Regularization:** Implements thread-safe stochastic neuron deactivation to effectively combat network overfitting.
* **Memory Safety:** Strict architectural separation of concerns using `std::unique_ptr` graph management and C++17 `<filesystem>` for safe weight saving/loading.

## 🏗️ Architecture Pipeline

The current engine successfully trains on the **CIFAR-10** dataset (10 classes of complex RGB images), which I was using for the project. It achieves **~60.7% accuracy** on a purely CPU-driven architecture. 

The pipeline structure:
`Conv2D (3x3) -> ReLU -> MaxPooling (2x2) -> Conv2D (3x3) -> ReLU -> Flatten -> Dropout (0.3) -> Dense -> Softmax`


## 🛠️ Build and Run Instructions

This project uses CMake. It is cross-platform but requires a compiler that supports C++17/C++20 and OpenMP (GCC, Clang, or MSVC).

### 1. Clone the Repository
```bash
git clone [https://github.com/Th3Groke/cpp-cnn-framework.git](https://github.com/Th3Groke/cpp-cnn-framework.git)
cd cpp-cnn-framework
```

### 2. Prepare the dataset
Download the [CIFAR-10 binary dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and place the `data_batch_X.bin` files into a `data/` directory at the root of the project.

### 3. Configure the network
Setup the network in `main.cpp`

### 4. Build with CMake
```bash
cmake -B Build
cmake --build build
```

### 5. Run the Network
```bash
./build/NeuralNet
```
