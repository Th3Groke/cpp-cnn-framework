#include "inc/Cifar10Loader.hpp"
#include "inc/conv2d.hpp"
#include "inc/dense.hpp"
#include "inc/flatten.hpp"
#include "inc/network.hpp"
#include "inc/relu.hpp"
#include "inc/sgd.hpp"
#include "inc/softmax.hpp"
#include "inc/tensor.hpp"
#include <cmath>
#include <iostream>

const int epochs = 20;
const int num_of_samples = 10000;
const bool TRAIN = false;
const std::string weights_path = "trained/1.bin";
int main() {

  std::cout << "Loading CIFAR-10..." << std::endl;
  const std::string batch_path = "data/data_batch_1.bin";
  Cifar10Loader loader;
  Tensor training_dataset(num_of_samples, 3, 32, 32);
  loader.LoadCifar10(training_dataset, batch_path);
  std::cout << "Data loaded! Building network..." << std::endl;

  // building network
  Network net;
  Conv2d conv(3, 8, 3);
  ReLU relu;
  Flatten flatten;
  Dense dense(7200, 10);
  Softmax softmax;

  net.AddLayer(&conv);
  net.AddLayer(&relu);
  net.AddLayer(&flatten);
  net.AddLayer(&dense);

  SGD optimizer(net.GetParameters(), 0.0001f);

  if (TRAIN) {
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      // Inner loop: Process images
      for (int i = 0; i < num_of_samples; ++i) {
        Tensor input = loader.GetImageAsTensor(training_dataset, i);
        int label = loader.GetLabel(i);

        // 1. Forward Pass (Outputs RAW numbers from Dense layer)
        Tensor logits = net.Forward(input);

        // 2. Manually pass logits through your Softmax layer to get
        // probabilities
        Tensor probs = softmax.Forward(logits);

        // 3. Cross-Entropy Loss Calculation
        float epsilon = 1e-7f;
        float target_prob = probs(0, label, 0, 0);
        total_loss += -std::log(target_prob + epsilon);

        // 4. Track accuracy
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

        // 5. Build the STABLE Gradient Tensor for the Dense layer
        Tensor grad_out(1, 10, 1, 1);
        for (int c = 0; c < 10; ++c) {
          float target = (c == label) ? 1.0f : 0.0f;
          // The mathematical fusion shortcut: (Probability - Target)
          grad_out(0, c, 0, 0) = probs(0, c, 0, 0) - target;
        }

        // 6. Backward Pass (Flows directly into Dense::Backward)
        net.Backward(grad_out);

        optimizer.Step();
        optimizer.ZeroGrad();
      }
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << " | Loss: " << (total_loss / num_of_samples)
                << " | Accuracy: "
                << (float)(correct_predictions /
                           (float)training_dataset.getBatchSize() * 100)
                << "%" << std::endl;
    }
  } else {
    net.LoadWeights(weights_path);
  }

  // testing sequence
  std::string test_path = "data/test_batch.bin";
  Tensor test_dataset(num_of_samples, 3, 32, 32);
  Cifar10Loader test_loader;
  test_loader.LoadCifar10(test_dataset, test_path);
  float test_loss = 0.0f;
  int test_correct = 0;
  for (size_t n = 0; n < num_of_samples; n++) {
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
            << "num_of_samples: " << num_of_samples
            << " | number of epochs: " << epochs << " Loss: " << '\n'
            << test_loss << " | "
            << "Accuracy:" << (float)test_correct / num_of_samples * 100.0f
            << "%" << std::endl;
  net.SaveWeights(weights_path);
  std::cout << "weights saved to:" + weights_path << std::endl;
  return 0;
}
