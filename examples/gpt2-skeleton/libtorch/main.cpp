#include <torch/torch.h>
#include <iostream>
#include <string>
#include <chrono>
#include "gpt2.h"

using torch::indexing::Slice;

struct Args {
  int vocab = 32768;
  int seq = 128;
  int batch = 16;
  int layers = 4;
  int heads = 4;
  int embd = 256;
  int steps = 200;
  int warmup = 50;       // warmup steps (excluded from profiling)
  double lr = 3e-4;
  int threads = -1;      // -1 => default
  int log_interval = 50; // 0 => no logs
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&](int& dst) { if (i + 1 < argc) dst = std::stoi(argv[++i]); };
    auto nextd = [&](double& dst) { if (i + 1 < argc) dst = std::stod(argv[++i]); };
    if (k == "--vocab") next(a.vocab);
    else if (k == "--seq") next(a.seq);
    else if (k == "--batch") next(a.batch);
    else if (k == "--layers") next(a.layers);
    else if (k == "--heads") next(a.heads);
    else if (k == "--embd") next(a.embd);
    else if (k == "--steps") next(a.steps);
    else if (k == "--warmup") next(a.warmup);
    else if (k == "--lr") nextd(a.lr);
    else if (k == "--threads") next(a.threads);
    else if (k == "--log-interval") next(a.log_interval);
  }
  return a;
}

int main(int argc, char** argv) {
  torch::manual_seed(1337);
  auto args = parse_args(argc, argv);

  if (args.threads > 0) {
    at::set_num_threads(args.threads);
    at::set_num_interop_threads(std::max(1, args.threads / 2));
  }

  GPT2Config cfg;
  cfg.vocab_size = args.vocab;
  cfg.n_embd = args.embd;
  cfg.n_head = args.heads;
  cfg.n_layer = args.layers;
  cfg.max_seq_len = std::max(args.seq, cfg.max_seq_len);

  auto model = GPT2(cfg);
  model->train();
  model->to(torch::kCPU);

  torch::optim::AdamWOptions opt_opts(args.lr);
  torch::optim::AdamW optimizer(model->parameters(), opt_opts);

  auto start_time = std::chrono::steady_clock::now();
  auto step = [&](int i) {
    // Synthetic token batch
    auto tokens = torch::randint(cfg.vocab_size, {args.batch, args.seq}, torch::TensorOptions().dtype(torch::kLong));
    auto x = tokens.index({Slice(), Slice(0, args.seq - 1)}); // [B, T-1]
    auto y = tokens.index({Slice(), Slice(1, args.seq)});     // [B, T-1]

    optimizer.zero_grad();

    auto logits = model->forward(x); // [B, T-1, V]
    auto loss = torch::nn::functional::cross_entropy(
      logits.view({-1, cfg.vocab_size}),
      y.reshape({-1})
    );

    loss.backward();
    optimizer.step();

    return loss.item<double>();
  };

  // Warmup (excluded from timing/profile)
  for (int i = 0; i < args.warmup; ++i) {
    (void)step(i);
  }

  auto t0 = std::chrono::steady_clock::now();
  double last_print = 0.0;
  for (int i = 0; i < args.steps; ++i) {
    double loss = step(i);
    if (args.log_interval > 0 && (i % args.log_interval == 0 || i + 1 == args.steps)) {
      auto t1 = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      double tok = (double)args.batch * (args.seq - 1) * (i + 1);
      double tps = tok / dt;
      std::cout << "step " << i << " loss=" << loss << " tok/s=" << (long long)tps << "\n";
      last_print = loss;
    }
  }

  auto end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end - t0).count();
  std::cout << "Done in " << elapsed << "s\n";
  return 0;
}