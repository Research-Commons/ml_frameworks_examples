#include <chrono>
#include <iostream>
#include <torch/torch.h>

// ---------- BasicBlock ----------
struct BasicBlockImpl : torch::nn::Module {
  static constexpr int expansion = 1;

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::Sequential shortcut;

  BasicBlockImpl(int64_t in_ch, int64_t out_ch, int64_t stride = 1) {
    conv1 = register_module(
        "conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3)
                                       .stride(stride)
                                       .padding(1)
                                       .bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_ch));
    relu = register_module(
        "relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    conv2 = register_module(
        "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3)
                                       .stride(1)
                                       .padding(1)
                                       .bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_ch));

    shortcut = register_module("shortcut", torch::nn::Sequential());
    if (stride != 1 || in_ch != out_ch) {
      shortcut->push_back(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 1)
                                .stride(stride)
                                .bias(false)));
      shortcut->push_back(torch::nn::BatchNorm2d(out_ch));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x.clone();
    x = relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    if (!shortcut->is_empty())
      identity = shortcut->forward(identity);
    x = x + identity;
    x = relu(x);
    return x;
  }
};
TORCH_MODULE(BasicBlock);

// ---------- Bottleneck ----------
struct BottleneckImpl : torch::nn::Module {
  static constexpr int expansion = 4;

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::Sequential shortcut;

  BottleneckImpl(int64_t in_ch, int64_t out_ch, int64_t stride = 1) {
    conv1 = register_module(
        "conv1", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(in_ch, out_ch, 1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_ch));

    conv2 = register_module(
        "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3)
                                       .stride(stride)
                                       .padding(1)
                                       .bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_ch));

    conv3 = register_module(
        "conv3", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(out_ch, out_ch * expansion, 1)
                         .bias(false)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_ch * expansion));

    relu = register_module(
        "relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    shortcut = register_module("shortcut", torch::nn::Sequential());
    if (stride != 1 || in_ch != out_ch * expansion) {
      shortcut->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(in_ch, out_ch * expansion, 1)
              .stride(stride)
              .bias(false)));
      shortcut->push_back(torch::nn::BatchNorm2d(out_ch * expansion));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x.clone();
    x = relu(bn1(conv1(x)));
    x = relu(bn2(conv2(x)));
    x = bn3(conv3(x));
    if (!shortcut->is_empty())
      identity = shortcut->forward(identity);
    x = x + identity;
    x = relu(x);
    return x;
  }
};
TORCH_MODULE(Bottleneck);

// ---------- ResNet50 ----------
struct ResNet50Impl : torch::nn::Module {
  int64_t in_channels{64};

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d maxpool{nullptr};

  torch::nn::Sequential layer1, layer2, layer3, layer4;
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Linear fc{nullptr};

  ResNet50Impl(int64_t num_classes = 1000) {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(
                false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
    relu = register_module(
        "relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    maxpool = register_module(
        "maxpool", torch::nn::MaxPool2d(
                       torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

    // Use Impl in the builder
    layer1 =
        register_module("layer1", _make_layer_impl<BottleneckImpl>(64, 3, 1));
    layer2 =
        register_module("layer2", _make_layer_impl<BottleneckImpl>(128, 4, 2));
    layer3 =
        register_module("layer3", _make_layer_impl<BottleneckImpl>(256, 6, 2));
    layer4 =
        register_module("layer4", _make_layer_impl<BottleneckImpl>(512, 3, 2));

    avgpool = register_module("avgpool",
                              torch::nn::AdaptiveAvgPool2d(
                                  torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module(
        "fc", torch::nn::Linear(512 * BottleneckImpl::expansion, num_classes));

    _init_weights();
  }

  // Template on the Impl, wrap with ModuleHolder<Impl>, and use Impl::expansion
  template <typename Impl>
  torch::nn::Sequential _make_layer_impl(int64_t out_channels, int blocks,
                                         int64_t stride = 1) {
    torch::nn::Sequential layers;
    layers->push_back(
        torch::nn::ModuleHolder<Impl>(Impl(in_channels, out_channels, stride)));
    in_channels = out_channels * Impl::expansion;
    for (int i = 1; i < blocks; ++i) {
      layers->push_back(
          torch::nn::ModuleHolder<Impl>(Impl(in_channels, out_channels, 1)));
    }
    return layers;
  }

  void _init_weights() {
    for (auto &m : modules(/*include_self=*/false)) {
      if (auto *c = dynamic_cast<torch::nn::Conv2dImpl *>(m.get())) {
        torch::nn::init::kaiming_normal_(c->weight, /*a=*/0.0, torch::kFanOut,
                                         torch::kReLU);
        if (c->options.bias()) {
          torch::nn::init::constant_(c->bias, 0.0);
        }
      } else if (auto *b =
                     dynamic_cast<torch::nn::BatchNorm2dImpl *>(m.get())) {
        torch::nn::init::constant_(b->weight, 1.0);
        torch::nn::init::constant_(b->bias, 0.0);
      } else if (auto *l = dynamic_cast<torch::nn::LinearImpl *>(m.get())) {
        torch::nn::init::normal_(l->weight, 0.0, 0.01);
        torch::nn::init::constant_(l->bias, 0.0);
      }
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    x = relu(bn1(conv1(x)));
    x = maxpool(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avgpool(x);
    x = x.view({x.size(0), -1});
    x = fc(x);
    return x;
  }
};
TORCH_MODULE(ResNet50);

int main(int argc, char **argv) {
  torch::Device device = torch::kCPU;
  std::cout << "Device: CPU\n";

  const int64_t batch_size = (argc > 1) ? std::stoll(argv[1]) : 32;
  const int num_batches = (argc > 2) ? std::stoi(argv[2]) : 100;

  ResNet50 model(1000);
  model->to(device);
  model->eval();
  std::cout << model << "\n";

  auto make_input = [&](int64_t bs) {
    return torch::randn(
        {bs, 3, 224, 224},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
  };

  std::vector<double> times_ms;
  times_ms.reserve(num_batches);
  c10::InferenceMode guard(true);

  auto t_all0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_batches; ++i) {
    auto x = make_input(batch_size);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto y = model->forward(x);
    (void)y;
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(dt);
    std::cout << "Batch " << (i + 1) << "/" << num_batches << " latency: " << dt
              << " ms\n";
  }
  auto t_all1 = std::chrono::high_resolution_clock::now();

  // Stats
  double total_ms =
      std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
  double sum = 0.0, minv = 1e300, maxv = -1e300;
  for (double v : times_ms) {
    sum += v;
    if (v < minv)
      minv = v;
    if (v > maxv)
      maxv = v;
  }
  double mean = sum / times_ms.size();
  double throughput_bps = 1000.0 / mean;
  double throughput_ips = throughput_bps * batch_size;

  std::cout << "================= Profile =================\n";
  std::cout << "Batch size: " << batch_size << ", Measured: " << num_batches
            << " batches\n";
  std::cout << "Avg latency: " << mean << " ms/batch\n";
  std::cout << "Min/Max: " << minv << " / " << maxv << " ms\n";
  std::cout << "Throughput: " << throughput_bps << " batches/s"
            << "  (" << throughput_ips << " images/s)\n";
  std::cout << "Total wall time (measured loop): " << total_ms << " ms\n";
  std::cout << "===========================================\n";
  return 0;
}
