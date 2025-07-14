#include "CSVLoader.hpp"
#include "HighFive.hpp"
#include <cmath>

#define MIN_NUM_EPOCHS 50
#define MIN_LEARN_RATE 1e-3
#define MIN_NUM_THREADS 4

// ========================
// Multi-Layer Perceptron Model
// ========================
struct MLPImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

  MLPImpl(i64 input_dim) {
    // Fully-connected layers with ReLU activations
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 64));
    fc3 = register_module("fc3", torch::nn::Linear(64, 1));
  }

  // Forward pass through the MLP
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = fc3(x);
    return x;
  }
};

TORCH_MODULE(MLP);

// ========================
// Custom Dataset for (features, target)
// ========================
struct HousePriceDataset : torch::data::datasets::Dataset<HousePriceDataset> {
  torch::Tensor data_, targets_;

  HousePriceDataset(torch::Tensor data, torch::Tensor targets)
      : data_(std::move(data)), targets_(std::move(targets)) {}

  // Return one example at a time
  torch::data::Example<> get(size_t index) override {
    return {data_[index], targets_[index]};
  }

  // Return total number of samples
  torch::optional<size_t> size() const override { return data_.size(0); }
};

// ========================
// Compute Root Mean Squared Error (RMSE)
// ========================
double compute_rmse(torch::Tensor preds, torch::Tensor targets) {
  auto mse = torch::mse_loss(preds, targets);
  return std::sqrt(mse.item<double>());
}

void print_progress(int current, int total, double avg_loss) {
  constexpr const int bar_width = 50;
  const float progress = static_cast<float>(current) / total;

  // First line: Epoch info
  std::cout << "Epoch [" << current << "/" << total
            << "] Avg Loss: " << avg_loss << "            " << std::endl;

  // Second line: progress bar
  std::cout << "[";
  int pos = bar_width * progress;
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %   " << std::flush;

  if (current != total) {
    std::cout << "\033[F"; // ANSI escape code: move cursor up 1 line
  }
}

int main(int argc, char *argv[]) {
  // ========================
  // Configuration
  // ========================

  constexpr const i64 batch_size = 32;

  h5::Config cfg;
  cfg.register_default("epochs", MIN_NUM_EPOCHS);
  cfg.register_default("lr", MIN_LEARN_RATE);
  cfg.register_default("threads", MIN_NUM_THREADS);

  cfg.parse_cli(argc, argv);

  const i64 num_epochs = cfg.get<i64>("epochs");
  const double learning_rate = cfg.get<double>("lr");
  const int num_threads = cfg.get<int>("threads"); // note: variant holds i64

  torch::manual_seed(0);
  torch::set_num_threads(num_threads);

  std::cout << "=========== HOUSE PRICE PREDICTION ===========\n";

  std::cout << "=> Taking num_epochs: " << num_epochs << std::endl;
  std::cout << "=> Taking learning_rate: " << learning_rate << std::endl;
  std::cout << "==> Taking threads: " << num_threads << "\n\n";

  // ========================
  // Load Dataset
  // ========================
  auto [train_data, train_targets] = load_csv("generated/train_processed.csv");
  auto [test_data, test_targets] = load_csv("generated/test_processed.csv");

  const i64 input_dim = train_data.size(1);

  std::cout << ">> Input dimension detected: " << input_dim << "\n\n";

  // ========================
  // Prepare DataLoader
  // ========================
  std::cout << ">> Preparing DataLoader...\n";
  auto train_dataset = HousePriceDataset(train_data, train_targets)
                           .map(torch::data::transforms::Stack<>());
  auto train_loader = torch::data::make_data_loader(train_dataset, batch_size);

  // ========================
  // Model, Loss, Optimizer
  // ========================
  std::cout << ">> Initializing Model...\n";
  MLP model(input_dim);
  model->train();

  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(learning_rate));
  torch::nn::MSELoss loss_fn;

  std::cout << ">> Starting Training...\n\n";

  // ========================
  // Training Loop
  // ========================
  for (int epoch = 1; epoch <= num_epochs; ++epoch) {
    double epoch_loss = 0.0;
    int batch_idx = 0;

    for (auto &batch : *train_loader) {
      optimizer.zero_grad();

      auto inputs = batch.data;
      auto labels = batch.target;

      auto outputs = model->forward(inputs);
      auto loss = loss_fn(outputs, labels);

      loss.backward();
      optimizer.step();

      epoch_loss += loss.item<double>();
      ++batch_idx;
    }

    const double avg_loss = epoch_loss / batch_idx;

    print_progress(epoch, num_epochs, avg_loss);
  }

  std::cout << "\n>> Training Completed.\n";

  // ========================
  // Evaluation
  // ========================
  std::cout << ">> Evaluating on test set...\n";
  model->eval();

  auto test_preds = model->forward(test_data);
  double rmse = compute_rmse(test_preds, test_targets);

  std::cout << ">> Test RMSE: " << rmse << "\n";

  std::cout << "==============================================\n";

  return 0;
}
