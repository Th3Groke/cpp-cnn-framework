#include "inc/cifar10loader.hpp"
#include "inc/conv2d.hpp"
#include "inc/dense.hpp"
#include "inc/dropout.hpp"
#include "inc/flatten.hpp"
#include "inc/maxpooling.hpp"
#include "inc/network.hpp"
#include "inc/relu.hpp"
#include "inc/sgd.hpp"
#include "inc/softmax.hpp"
#include "inc/tensor.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

const int NUM_OF_BATCHES = 5;
const int epochs = 50;
const int num_of_samples = NUM_OF_BATCHES * 10000;
const int num_test_samples = 10000;
const int mini_batch_size = 32;
const bool TRAIN = false;
const std::string weights_path = "1.bin";
int main() {

  std::vector<std::string> batch_paths;
  for (int i = 1; i <= NUM_OF_BATCHES; i++) {
    std::string filepath = "data/data_batch_" + std::to_string(i) + ".bin";
    batch_paths.push_back(filepath);
  }
  std::cout << "Loading CIFAR-10..." << std::endl;
  Cifar10Loader loader;
  Tensor training_dataset(num_of_samples, 3, 32, 32);
  loader.LoadCifar10(training_dataset, batch_paths);
  std::cout << "Data loaded! Building network..." << std::endl;

  // building network
  Network net;
  net.AddLayer(std::make_unique<Conv2d>(3, 8, 3));
  net.AddLayer(std::make_unique<ReLU>());
  net.AddLayer(std::make_unique<MaxPooling>(2, 2));
  net.AddLayer(std::make_unique<Conv2d>(8, 16, 3));
  net.AddLayer(std::make_unique<ReLU>());
  net.AddLayer(std::make_unique<Flatten>());
  net.AddLayer(std::make_unique<Dropout>());
  net.AddLayer(std::make_unique<Dense>(2704, 10));

  Softmax softmax;

  SGD optimizer(net.GetParameters(), 0.0005f);
  // running training if true
  if (TRAIN) {
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      // Inner loop: Process mini-batches
      for (int i = 0; i < num_of_samples; i += mini_batch_size) {
        int current_batch_size = std::min(mini_batch_size, num_of_samples - i);
        Tensor input = loader.GetBatch(training_dataset, i, current_batch_size);
        std::vector<int> labels = loader.GetLabels(i, current_batch_size);

        // Forward pass
        Tensor logits = net.Forward(input, TRAIN);

        // passing through softmax
        Tensor probs = softmax.Forward(logits);

        // Cross-Entropy Loss Calculation and Gradient Output
        Tensor grad_out(current_batch_size, 10, 1, 1);
        float epsilon = 1e-7f;

        for (int b = 0; b < current_batch_size; b++) {
          int label = labels[b];
          float target_prob = probs(b, label, 0, 0);
          total_loss += -std::log(target_prob + epsilon);

          float max_val = -1e9f;
          int predicted_class = -1;
          for (int c = 0; c < 10; ++c) {
            float p = probs(b, c, 0, 0);
            if (p > max_val) {
              max_val = p;
              predicted_class = c;
            }
          }
          if (predicted_class == label)
            correct_predictions++;

          for (int c = 0; c < 10; ++c) {
            float target = (c == label) ? 1.0f : 0.0f;
            grad_out(b, c, 0, 0) =
                (probs(b, c, 0, 0) - target) / current_batch_size;
          }
        }

        // backwards pass
        net.Backward(grad_out);
        optimizer.Step();
        optimizer.ZeroGrad();
      }
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << " | Loss: " << (total_loss / num_of_samples)
                << " | Accuracy: "
                << (float)correct_predictions / num_of_samples * 100.0f << "%"
                << std::endl;
      optimizer.DecayLearningRate(0.95f);
    }
  } else {
    net.LoadWeights(weights_path);
  }

  // testing sequence
  std::cout << "\nStarting testing..." << std::endl;
  std::string test_path = "data/test_batch.bin";
  Tensor test_dataset(num_test_samples, 3, 32, 32);
  Cifar10Loader test_loader;
  test_loader.LoadCifar10(test_dataset, test_path);
  float test_loss = 0.0f;
  int test_correct = 0;

  for (size_t n = 0; n < num_test_samples; n += mini_batch_size) {
    int current_batch_size =
        std::min((size_t)mini_batch_size, num_test_samples - n);
    Tensor input = test_loader.GetBatch(test_dataset, n, current_batch_size);
    std::vector<int> labels = test_loader.GetLabels(n, current_batch_size);

    Tensor logits = net.Forward(input);
    Tensor probs = softmax.Forward(logits);

    float epsilon = 1e-7f;
    for (int b = 0; b < current_batch_size; b++) {
      int label = labels[b];
      float target_prob = probs(b, label, 0, 0);
      test_loss += -std::log(target_prob + epsilon);

      float max_prob = -1e9f;
      int predicted_class = -1;
      for (int c = 0; c < 10; c++) {
        float p = probs(b, c, 0, 0);
        if (p > max_prob) {
          max_prob = p;
          predicted_class = c;
        }
      }
      if (predicted_class == label) {
        test_correct++;
      }
    }
  }

  std::cout << "\n=================" << "\nTEST RESULTS"
            << "\n===============\n"
            << "num_of_samples: " << num_test_samples
            << " | number of epochs: " << epochs << '\n'
            << " Average Loss: " << (test_loss / num_test_samples) << " | "
            << "Accuracy: " << (float)test_correct / num_test_samples * 100.0f
            << "%" << std::endl;
  net.SaveWeights(weights_path);
  std::cout << "Weights saved to: " + weights_path << std::endl;
  return 0;
}
