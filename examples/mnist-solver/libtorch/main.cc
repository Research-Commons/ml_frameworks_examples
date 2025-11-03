#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <vector>

struct NetImpl : torch::nn::Module {
  // First block
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv1x1_1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  torch::nn::MaxPool2d pool1{nullptr};

  // Second block
  torch::nn::Conv2d conv3{nullptr}, conv4{nullptr};
  torch::nn::BatchNorm2d bn3{nullptr}, bn4{nullptr};
  torch::nn::MaxPool2d pool2{nullptr};

  // Final block
  torch::nn::Conv2d conv5{nullptr};
  torch::nn::BatchNorm2d bn5{nullptr};
  torch::nn::AdaptiveAvgPool2d gap{nullptr};
  torch::nn::Conv2d fc{nullptr};

  torch::nn::Dropout dropout{nullptr};

  NetImpl() {
    // First block
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(10));
    conv2 = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(20));
    pool1 = register_module(
        "pool1",
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    conv1x1_1 = register_module(
        "conv1x1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 10, 1)));

    // Second block
    conv3 = register_module(
        "conv3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(20));
    conv4 = register_module(
        "conv4",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 20, 3).padding(1)));
    bn4 = register_module("bn4", torch::nn::BatchNorm2d(20));
    pool2 = register_module(
        "pool2",
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

    // Final block
    conv5 = register_module(
        "conv5",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 40, 3).padding(1)));
    bn5 = register_module("bn5", torch::nn::BatchNorm2d(40));
    gap = register_module("gap",
                          torch::nn::AdaptiveAvgPool2d(
                              torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module(
        "fc", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 10, 1)));

    dropout = register_module("dropout", torch::nn::Dropout(0.1));
  }

  torch::Tensor forward(torch::Tensor x) {
    // First block
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    x = pool1(x);
    x = dropout(x);
    x = torch::relu(conv1x1_1(x));

    // Second block
    x = torch::relu(bn3(conv3(x)));
    x = torch::relu(bn4(conv4(x)));
    x = pool2(x);
    x = dropout(x);

    // Final block
    x = torch::relu(bn5(conv5(x)));
    x = gap(x);
    x = fc(x);
    x = x.view({x.size(0), 10});
    return torch::log_softmax(x, /*dim=*/1);
  }
};
TORCH_MODULE(Net);

int main(int argc, char **argv) {
  const int64_t batch_size = (argc > 1) ? std::stoll(argv[1]) : 64;
  const int profile_batches = (argc > 2) ? std::stoll(argv[2]) : 1000;

  torch::Device device = torch::kCPU;
  std::cout << "Device: " << "CPU" << "\n";
  std::cout << "Batch size: " << batch_size << ", measure: " << profile_batches
            << " batches\n";

  Net net;
  net->to(device);
  net->eval();

  c10::InferenceMode guard(true); // best for pure inference

  // fixed random input shape: MNIST-like 1x28x28
  auto make_input = [&](int64_t bs) {
    return torch::randn(
        {bs, 1, 28, 28},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
  };

  // Measure
  std::vector<double> times_ms;
  times_ms.reserve(profile_batches);
  auto start_all = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < profile_batches; ++i) {
    auto x = make_input(batch_size);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto y = net->forward(x);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)y;
    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(dt);
  }
  auto end_all = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end_all - start_all).count();
  double avg_ms = total_ms / profile_batches;
  double throughput = 1000.0 / avg_ms; // batches/sec

  // simple stats
  double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
  double max_ms = *std::max_element(times_ms.begin(), times_ms.end());
  double sum = 0.0;
  for (double v : times_ms)
    sum += v;
  double mean = sum / times_ms.size();

  std::cout << "================ Profile =================\n";
  std::cout << "Batches measured: " << profile_batches << "\n";
  std::cout << "Avg latency: " << mean << " ms/batch\n";
  std::cout << "Min/Max: " << min_ms << " / " << max_ms << " ms\n";
  std::cout << "Throughput: " << throughput << " batches/sec\n";
  std::cout << "=========================================\n";
  return 0;
}
