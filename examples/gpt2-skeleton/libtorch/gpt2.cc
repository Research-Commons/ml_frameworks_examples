#include "gpt2.h"
#include <cmath>
#include <iostream>

using torch::indexing::Slice;

CausalSelfAttentionImpl::CausalSelfAttentionImpl(int n_head_, int n_embd_, int max_seq_len_)
  : n_head(n_head_), n_embd(n_embd_), max_seq_len(max_seq_len_) {
  TORCH_CHECK(n_embd % n_head == 0, "n_embd must be divisible by n_head");
  head_dim = n_embd / n_head;
  c_attn = register_module("c_attn", torch::nn::Linear(n_embd, 3 * n_embd));
  c_proj = register_module("c_proj", torch::nn::Linear(n_embd, n_embd));
  // Lower-triangular mask [maxT, maxT] (true allowed). We'll broadcast to [1,1,T,T].
  causal_mask = torch::tril(torch::ones({max_seq_len, max_seq_len}, torch::kBool));
}

torch::Tensor CausalSelfAttentionImpl::forward(const torch::Tensor& x) {
  // x: [B, T, C]
  auto B = x.size(0);
  auto T = x.size(1);
  auto C = x.size(2);

  auto qkv = c_attn->forward(x); // [B, T, 3C]
  auto q = qkv.slice(-1, 0, C);
  auto k = qkv.slice(-1, C, 2 * C);
  auto v = qkv.slice(-1, 2 * C, 3 * C);

  // [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
  q = q.view({B, T, n_head, head_dim}).permute({0, 2, 1, 3});
  k = k.view({B, T, n_head, head_dim}).permute({0, 2, 1, 3});
  v = v.view({B, T, n_head, head_dim}).permute({0, 2, 1, 3});

  // Scaled dot-product attention
  auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)head_dim); // [B, n_head, T, T]

  // Causal mask for current T
  auto cm = causal_mask.index({Slice(0, T), Slice(0, T)})
              .to(att.device())
              .view({1, 1, T, T});
  // Fill where mask is false (upper triangle)
  att = att.masked_fill(cm.logical_not(), -1e9);

  auto att_probs = torch::softmax(att, -1); // [B, n_head, T, T]
  auto y = torch::matmul(att_probs, v);     // [B, n_head, T, head_dim]

  y = y.permute({0, 2, 1, 3}).contiguous().view({B, T, C}); // [B, T, C]
  y = c_proj->forward(y);                                    // [B, T, C]
  return y;
}

MLPImpl::MLPImpl(int n_embd) {
  int hidden = 4 * n_embd;
  fc = register_module("fc", torch::nn::Linear(n_embd, hidden));
  proj = register_module("proj", torch::nn::Linear(hidden, n_embd));
}

torch::Tensor MLPImpl::forward(const torch::Tensor& x) {
  auto h = fc->forward(x);
  h = torch::gelu(h);
  h = proj->forward(h);
  return h;
}

BlockImpl::BlockImpl(const GPT2Config& cfg) {
  ln_1 = register_module("ln_1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
  ln_2 = register_module("ln_2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
  attn = register_module("attn", CausalSelfAttention(cfg.n_head, cfg.n_embd, cfg.max_seq_len));
  mlp  = register_module("mlp", MLP(cfg.n_embd));
}

torch::Tensor BlockImpl::forward(const torch::Tensor& x_in) {
  auto x = x_in + attn->forward(ln_1->forward(x_in));
  x = x + mlp->forward(ln_2->forward(x));
  return x;
}

GPT2Impl::GPT2Impl(const GPT2Config& cfg_) : cfg(cfg_) {
  wte = register_module("wte", torch::nn::Embedding(cfg.vocab_size, cfg.n_embd));
  wpe = register_module("wpe", torch::nn::Embedding(cfg.max_seq_len, cfg.n_embd));
  for (int i = 0; i < cfg.n_layer; ++i) {
    h.push_back(register_module("h" + std::to_string(i), Block(cfg)));
  }
  ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.n_embd})));
  lm_head = register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(cfg.n_embd, cfg.vocab_size).bias(false)));
}

torch::Tensor GPT2Impl::forward(const torch::Tensor& idx) {
  // idx: [B, T] int64
  auto B = idx.size(0);
  auto T = idx.size(1);

  auto pos = torch::arange(0, T, torch::TensorOptions().dtype(torch::kLong).device(idx.device()))
               .unsqueeze(0); // [1, T]

  auto tok = wte->forward(idx);         // [B, T, C]
  auto pos_emb = wpe->forward(pos);     // [1, T, C]
  auto x = tok + pos_emb;               // [B, T, C]
  for (auto& blk : h) {
    x = blk->forward(x);
  }
  x = ln_f->forward(x);
  auto logits = lm_head->forward(x);    // [B, T, V]
  return logits;
}