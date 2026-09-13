# Real-Time Optimized Virtual Character Engine

A from-scratch inference optimization pipeline for real-time interactive characters — LLM dialogue, TTS voice, and vision/perception, compressed and engineered for on-device, real-time deployment. Built to explore the actual technical surface of low-latency character inference: model compression, custom GPU kernels, and native game-engine integration.

## Why this project

Built to genuinely engage with what real-time character inference actually requires — not a benchmark exercise, but the full stack: compress a model, understand *why* an optimization does or doesn't work, write the custom kernels that off-the-shelf tools can't provide, and wire the result into a real game engine through a C++ native plugin.

## Architecture

- **Dialogue**: Qwen2.5-0.5B-Instruct, pruned and distilled
- **Voice**: Piper TTS (`en_US-lessac-medium`)
- **Perception**: Qwen2.5-VL-3B-Instruct
- **Engine integration**: C++ wrapper (ONNX Runtime C++ API) exposed to Unity via a native plugin, called asynchronously off the main thread

## Results summary

### LLM compression

| Technique | Result |
|---|---|
| GGUF quantization (Q4_K_M) | 942MB → 374MB, TPOT 114.4ms → 20.7ms (~5.5x faster) |
| Structured pruning (data-driven layer selection) | 4 layers removed, TPOT → 76.4ms (~33% faster), coherent |
| Structured pruning + distillation | 5 layers removed (broke the model on its own), **distillation recovered full coherence**, TPOT 74.7ms (~35% faster than baseline) |
| Unstructured pruning | Coherent to ~30-35% sparsity, collapses by 50% — no speed benefit (dense compute still processes zeros) |

### TTS compression

| Technique | Result |
|---|---|
| ONNX dynamic quantization | Made it **slower** (RTF 0.40 → 0.72-1.43) — small models can have less quantization overhead headroom than large ones |
| Unstructured pruning (ONNX weights) | Coherent to ~15% sparsity; audible degradation by 20-30% |

### Vision (Qwen2.5-VL-3B)

- Baseline: 2451.7ms/token (bf16 beat fp32 on CPU — counter to expectation, likely memory-bandwidth-bound, not compute-bound)
- **Quantization**: all three standard paths tested — GGUF (silently drops the vision encoder), ONNX/optimum (explicitly unsupported architecture), PyTorch dynamic quantization (works, but no speedup) — a genuine, current limitation of edge-deployment tooling for multimodal models, not an implementation gap

### KV-cache

- Measured real memory growth: 24.6KB/token → 234MB projected at a 10,000-token conversation
- Quantization (HQQ backend, 4-bit): broke coherence, no speed benefit — cache corruption compounds across generation, unlike weight quantization
- **Toy PagedAttention** (block-based memory manager): 96.1% memory savings vs. naive pre-allocation, correctly verified block pool reuse

### Speculative decoding

- Draft model: the pruned+distilled model from the compression phase (reused, not a separate download) — validated as a strong draft choice: 56.2% acceptance rate, 3.12 tokens per target forward call
- Real debugging story: an early "working" version silently diverged from the true baseline; root cause (found by comparing raw logits) was a missing repetition penalty the model's default generation config applies — fixed, then verified byte-for-byte against ground truth

### Custom Triton kernels (on an RTX 4090)

| Kernel | Result |
|---|---|
| Fused RMSNorm | 3.15x faster than unfused PyTorch |
| Fused INT8 dequantize + matmul | 1.79x faster than PyTorch's cuBLAS-backed matmul — exact numerical match |

### C++ wrapper + Unity integration

- Native C++ plugin (g++/MinGW, ONNX Runtime C++ API) loading Piper TTS, exposed to Unity as a `.dll`
- Async, non-blocking calls from Unity's main thread via a background task
- **Live demo: a character generates its own dialogue.** One button click runs a multi-turn conversation — the quantized LLM generates each line live (with real conversation history, via Qwen's chat format), a Python bridge converts it to phonemes, and the C++ TTS plugin speaks it — entirely inside the Unity Editor, not a pre-scripted or precomputed line.
- **Architecture note**: the LLM runs as a subprocess (`llm_cli.exe`, spawned from C#), not as an in-process native plugin. llama.cpp's runtime GPU/CPU backend discovery scans the directory of the *main executable* — inside the Unity Editor that's `Unity.exe`, nowhere near the plugin DLLs — so a `DllImport`-based in-process approach reliably failed to load the model. Running the LLM as a real subprocess sidesteps that entirely, at the cost of reloading the model from disk on every turn (~3-4s) rather than keeping it resident. Full debugging notes and measured end-to-end latency are in [`benchmark.md`](benchmark.md).
- **Note on which model runs live**: the Unity demo currently runs the quantized (GGUF, Q4_K_M) version of the *original, unpruned* model — the separately pruned+distilled checkpoint above hasn't yet been converted to GGUF. Quantization and pruning+distillation are both independently proven (see tables above); the live demo currently demonstrates the quantization + full engineering pipeline, not the pruned model specifically.

## Published artifact

The pruned + distilled LLM checkpoint is published publicly: [huggingface.co/Aks44/qwen2.5-0.5b-pruned-distilled-game](https://huggingface.co/Aks44/qwen2.5-0.5b-pruned-distilled-game)

## What I'd do next

- Extend distillation and speculative decoding to the vision model
- Real-time text-to-phoneme in C++ (currently precomputed/staged from Python, a deliberate scope boundary for this phase)
- A real attention kernel in Triton (RMSNorm and fused matmul are proven; full attention is the natural next step)
- Server-side (H100/A100) benchmarks alongside the consumer-edge numbers here

## A note on methodology

Every result in this repo was measured, not assumed — including the negative ones (TTS quantization made things slower; KV-cache quantization broke coherence). Where a "working" result turned out to be silently wrong (speculative decoding's early version), it's documented here, because catching that mattered more than the numbers themselves.