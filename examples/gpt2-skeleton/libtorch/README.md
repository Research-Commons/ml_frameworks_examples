# Minimal GPT2 for libtorch

## Goal

To "mimic" or replicate the architecture and underlying structure of GPT2 for the purpose of CPU profiling.

## What it can do

1. Decoder-only Transformer with:
2. Token and learned position embeddings (wte, wpe).
3. Pre-norm residual blocks: LayerNorm -> Attention -> residual, LayerNorm -> MLP -> residual.
4. Multi-head causal self-attention with a lower-triangular mask.
5. MLP with GELU and projection (hidden size 4x embedding).
6. Final LayerNorm and linear LM head to produce logits [B, T, V].
7. Variable B and T at runtime (up to cfg.max_seq_len).
8. Trainable end-to-end with standard cross-entropy for next-token prediction.
9. CPU-friendly, simple to profile with perf/flamegraphs.

## What it does not do / differences from “real” GPT‑2

1. No dropout at all (GPT‑2 uses attn/residual/MLP dropout).
2. No exact GPT‑2 weight initialization (defaults to LibTorch; GPT‑2 uses std=0.02 and some scaled projections).
3. No weight tying between wte and lm_head (often used).
4. No tokenizer/BPE merges, vocab, or pretrained weights; it only operates on integer token IDs you provide.
5. No generation utilities (no sampling, top‑k/top‑p, temperature, KV caching).
6. No KV cache for fast autoregressive inference.
7. No attention/MLP fused kernels, FlashAttention, quantization, or mixed precision (fp16/bf16).
8. No gradient checkpointing, activation rematerialization, or layer scaling tricks.
9. Fixed maximum context via cfg.max_seq_len; longer inputs will fail (wpe size and mask are bounded).
10. No padding/attention masks beyond causal; assumes dense sequences with left-to-right causality only.
11. No multi-device/distributed training; CPU-only unless you manually move to CUDA and link the CUDA build of LibTorch.
12. No full GPT‑2 config presets (e.g., small/medium/large/XL) or exact hyperparameters
