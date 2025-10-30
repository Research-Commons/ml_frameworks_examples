#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>

struct GPT2Config {
  int vocab_size = 32768;
  int n_embd = 256;
  int n_head = 4;
  int n_layer = 4;
  int max_seq_len = 256;
};

struct CausalSelfAttentionImpl : torch::nn::Module {
  int n_head, n_embd, max_seq_len, head_dim;
  torch::nn::Linear c_attn{nullptr};
  torch::nn::Linear c_proj{nullptr};
  torch::Tensor causal_mask;

  CausalSelfAttentionImpl(int n_head, int n_embd, int max_seq_len);

  torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(CausalSelfAttention);

struct MLPImpl : torch::nn::Module {
  torch::nn::Linear fc{nullptr}, proj{nullptr};
  explicit MLPImpl(int n_embd);
  torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(MLP);

struct BlockImpl : torch::nn::Module {
  torch::nn::LayerNorm ln_1{nullptr}, ln_2{nullptr};
  CausalSelfAttention attn{nullptr};
  MLP mlp{nullptr};

  BlockImpl(const GPT2Config& cfg);
  torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(Block);

struct GPT2Impl : torch::nn::Module {
  GPT2Config cfg;
  torch::nn::Embedding wte{nullptr}; // token embeddings
  torch::nn::Embedding wpe{nullptr}; // position embeddings
  std::vector<Block> h;              // transformer blocks
  torch::nn::LayerNorm ln_f{nullptr};
  torch::nn::Linear lm_head{nullptr};

  explicit GPT2Impl(const GPT2Config& cfg);
  torch::Tensor forward(const torch::Tensor& idx); // returns logits [B,T,V]
};
TORCH_MODULE(GPT2);