#include "LoadCIFAR-10.hpp"
#include <iostream>
#include <torch/torch.h>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
public:
  CustomDataset(torch::Tensor data, torch::Tensor targets)
      : data_(data), targets_(targets) {}

  torch::data::Example<> get(size_t index) override {
    return {data_[index], targets_[index]};
  }

  torch::optional<size_t> size() const override { return data_.size(0); }

private:
  torch::Tensor data_;
  torch::Tensor targets_;
};

struct SimpleCNN : torch::nn::Module {
  SimpleCNN() {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
    conv2 = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    fc1 = register_module("fc1", torch::nn::Linear(64 * 8 * 8, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 10));
    pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = pool(torch::relu(conv1(x)));
    x = pool(torch::relu(conv2(x)));
    x = x.view({x.size(0), -1});
    x = torch::relu(fc1(x));
    x = fc2(x);
    return x;
  }

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::MaxPool2d pool{nullptr};
};

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU")
            << std::endl;

  // Replace this with your own CIFAR-10 loading code
  // Here we simulate random tensors
  auto [train_images, train_labels] =
      load_cifar10_bin("cifar-10-batches-bin/data_batch_1.bin");
  auto [test_images, test_labels] =
      load_cifar10_bin("cifar-10-batches-bin/test_batch.bin");

  train_images = train_images.to(device);
  train_labels = train_labels.to(device);
  test_images = test_images.to(device);
  test_labels = test_labels.to(device);

  auto train_dataset = CustomDataset(train_images, train_labels)
                           .map(torch::data::transforms::Stack<>());
  auto test_dataset = CustomDataset(test_images, test_labels)
                          .map(torch::data::transforms::Stack<>());

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(train_dataset),
          torch::data::DataLoaderOptions().batch_size(64));

  auto test_loader = torch::data::make_data_loader(
      std::move(test_dataset),
      torch::data::DataLoaderOptions().batch_size(100));

  auto model = std::make_shared<SimpleCNN>();
  model->to(device);

  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.001));
  torch::nn::CrossEntropyLoss criterion;

  for (int epoch = 0; epoch < 10; ++epoch) {
    model->train();
    size_t batch_idx = 0;
    double running_loss = 0.0;

    for (auto &batch : *train_loader) {
      auto inputs = batch.data.to(device);
      auto labels = batch.target.to(device);

      optimizer.zero_grad();
      auto outputs = model->forward(inputs);
      auto loss = criterion(outputs, labels);
      loss.backward();
      optimizer.step();

      running_loss += loss.item<double>();

      if (++batch_idx % 100 == 0) {
        std::cout << "[" << (epoch + 1) << ", " << batch_idx
                  << "] loss: " << (running_loss / 100) << std::endl;
        running_loss = 0.0;
      }
    }
  }

  model->eval();
  int64_t correct = 0, total = 0;

  for (auto &batch : *test_loader) {
    auto inputs = batch.data.to(device);
    auto labels = batch.target.to(device);

    auto outputs = model->forward(inputs);
    auto pred = outputs.argmax(1);
    correct += pred.eq(labels).sum().item<int64_t>();
    total += labels.size(0);
  }

  std::cout << "Test Accuracy: " << 100.0 * correct / total << "%" << std::endl;

  return 0;
}
