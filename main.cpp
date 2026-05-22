#include "inc/Cifar10Loader.hpp"
#include "inc/conv2d.hpp"
#include "inc/dense.hpp"
#include "inc/flatten.hpp"
#include "inc/network.hpp"
#include "inc/relu.hpp"
#include "inc/sgd.hpp"
#include <iostream>

int main() {
  std::cout << "Loading CIFAR-10..." << std::endl;
  const std::string batch_path = "data/data_batch_1.bin";
  Cifar10Loader loader;
  Tensor dataset(10000, 3, 32, 32);
  loader.LoadCifar10(dataset, batch_path);
  std::cout << "Data loaded! Building network..." << std::endl;

  // building network
  Network net;
  Conv2d conv(3, 8, 3);
  ReLU relu;
  Flatten flatten;
  Dense dense(7200, 10);

  net.AddLayer(&conv);
  net.AddLayer(&relu);
  net.AddLayer(&flatten);
  net.AddLayer(&dense);

  SGD optimizer(net.GetParameters(), 0.0001f);

  std::cout << "Starting training..." << std::endl;
  int epochs = 5;
  for (int epoch = 0; epoch < epochs; epoch++) {
    float total_loss = 0.0f;
    int correct_predictions = 0;
    int num_of_samples = 100;

    for (int i = 0; i < num_of_samples; i++) {
      Tensor input = loader.GetImageAsTensor(dataset, i);
      int label = loader.GetLabel(i);

      Tensor output = net.Forward(input);

      Tensor grad_out(1, 10, 1, 1);
      float max_val = -9999.0f;
      int predicted_class = -1;
      for (int c = 0; c < 10; c++) {
        float target = (c == label) ? 1.0f : 0.0f;
        float pred = output(0, c, 0, 0);

        if (pred > max_val) {
          max_val = pred;
          predicted_class = c;
        }
        grad_out(0, c, 0, 0) = pred - target;

        total_loss += 0.5f * (pred - target) * (pred - target);
        if (predicted_class == label)
          correct_predictions++;

        net.Backward(grad_out);
        optimizer.Step();
        optimizer.ZeroGrad();
      }
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << " | Loss: " << (total_loss / num_of_samples)
                << " | Accuracy: " << correct_predictions << "%" << std::endl;
    }
  }
  return 0;
}
