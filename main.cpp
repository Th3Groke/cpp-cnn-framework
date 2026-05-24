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
#include <string>
#include <vector>

const int NUM_OF_BATCHES = 5;
const int epochs = 50;
const int num_of_samples = NUM_OF_BATCHES * 10000;
const int num_test_samples = 10000;
const int mini_batch_size = 32;
const bool TRAIN = true;
const std::string weights_path = "trained/1.bin";
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
  Conv2d conv1(3, 8, 3);
  ReLU relu1;
  MaxPooling pool1(2, 2);
  Conv2d conv2(8, 16, 3);
  ReLU relu2;
  Flatten flatten;
  Dense dense(2704, 10);
  Softmax softmax;
  Dropout dropout1;

  net.AddLayer(&conv1);
  net.AddLayer(&relu1);
  net.AddLayer(&pool1);
  net.AddLayer(&conv2);
  net.AddLayer(&relu2);
  net.AddLayer(&flatten);
  net.AddLayer(&dropout1);
  net.AddLayer(&dense);

  SGD optimizer(net.GetParameters(), 0.0005f);
  // running training if true
  if (TRAIN) {
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      // Inner loop: Process images
      for (int i = 0; i < num_of_samples; ++i) {
        Tensor input = loader.GetImageAsTensor(training_dataset, i);
        int label = loader.GetLabel(i);

        // Forward pass
        Tensor logits = net.Forward(input, TRAIN);

        // passing through softmax
        Tensor probs = softmax.Forward(logits);

        // Cross-Entropy Loss Calculation
        float epsilon = 1e-7f;
        float target_prob = probs(0, label, 0, 0);
        total_loss += -std::log(target_prob + epsilon);

        float max_val = -1e9f;
        int predicted_class = -1;
        for (int c = 0; c < 10; ++c) {
          float p = probs(0, c, 0, 0);
          if (p > max_val) {
            max_val = p;
            predicted_class = c;
          }
        }
        if (predicted_class == label)
          correct_predictions++;

        Tensor grad_out(1, 10, 1, 1);
        for (int c = 0; c < 10; ++c) {
          float target = (c == label) ? 1.0f : 0.0f;
          // The mathematical fusion shortcut: (Probability - Target)
          grad_out(0, c, 0, 0) = probs(0, c, 0, 0) - target;
        }

        // backwards pass
        net.Backward(grad_out);
        if ((i + 1) % mini_batch_size == 0 || (i + 1) % num_of_samples == 0) {
          optimizer.Step();
          optimizer.ZeroGrad();
        }
      }
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << " | Loss: " << (total_loss / num_of_samples)
                << " | Accuracy: "
                << (float)(correct_predictions /
                           (float)training_dataset.getBatchSize() * 100)
                << "%" << std::endl;
      optimizer.DecayLearningRate(0.95f);
    }
  } else {
    net.LoadWeights(weights_path);
  }

  // testing sequence
  std::string test_path = "data/test_batch.bin";
  Tensor test_dataset(num_test_samples, 3, 32, 32);
  Cifar10Loader test_loader;
  test_loader.LoadCifar10(test_dataset, test_path);
  float test_loss = 0.0f;
  int test_correct = 0;
  for (size_t n = 0; n < num_test_samples; n++) {
    Tensor input = test_loader.GetImageAsTensor(test_dataset, n);
    int label = test_loader.GetLabel(n);
    Tensor logits = net.Forward(input);
    Tensor probs = softmax.Forward(logits);

    float epsylon = 1e-7f;
    float target_prob = probs(0, label, 0, 0);
    test_loss += -std::log(target_prob + epsylon);
    float max_prob = -1e9f;
    int predicted_class = -1;
    for (int c = 0; c < 10; c++) {
      float p = probs(0, c, 0, 0);
      if (p > max_prob) {
        max_prob = p;
        predicted_class = c;
      }
    }
    if (predicted_class == label) {
      test_correct++;
    }
  }
  std::cout << "\n=================" << "\nTEST RESULTS"
            << "\n===============\n"
            << "num_of_samples: " << num_test_samples
            << " | number of epochs: " << epochs << '\n'
            << " Loss: " << test_loss << " | "
            << "Accuracy:" << (float)test_correct / num_test_samples * 100.0f
            << "%" << std::endl;
  net.SaveWeights(weights_path);
  std::cout << "weights saved to:" + weights_path << std::endl;
  return 0;
}
