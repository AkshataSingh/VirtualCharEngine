# Benchmarks

## LLM — Qwen2.5-0.5B-Instruct

| Stage                | TTFT     | TPOT           | Notes |

| Baseline (fp32, CPU) | 332.9 ms | 114.4 ms/token | 26 tokens generated |


## TTS — Piper (en_US-lessac-medium)

| Stage          | Synthesis Time | Audio Duration | RTF    | Notes |

| Baseline (CPU) | 1.661 s         | 4.156 s        | 0.400 | Test sentence, 26 words |


## Compression — Qwen2.5-0.5B-Instruct , quantized by llama

| Stage           | Size       | Bits/weight | Notes |

| f16 (converted) | 942.43 MiB | 16.00        | Direct GGUF conversion, no compression |
| Q4_K_M          | 373.71 MiB | 6.35         | 60% size reduction; 144/290 tensors fell back to Q5_0/Q6_K/Q8_0 since 896 isn't divisible by 256 |

## this is for llama-cli.exe test
 You are a friendly game character. Greet the player and ask how you can help them today.
Hello there! I'm Qwen, an AI language model designed to assist and provide information on various topics. How may I be of assistance today?

[ Prompt: 283.1 t/s | Generation: 48.4 t/s ]

## TTS Quantization — Piper (en_US-lessac-medium)

| Stage | RTF (run 1) | RTF (run 2) | Notes |
|---  |---|---|---|
| Baseline (fp32) | 0.400 | 0.366 | Stable, consistent |
| INT8 (dynamic quantization) | 1.430 | 0.723 | Consistently worse in both runs — quantization overhead exceeds savings on this small, already-efficient model |

**Decision: quantization not used for TTS in the final pipeline.** Piper stays fp32.

## LLM Unstructured Pruning — Qwen2.5-0.5B-Instruct

| Sparsity | Coherence      | Notes |
| 30%      | Fully coherent | Fluent, on-topic, no noticeable degradation |
| 35%      | Mostly coherent| Minor repetition creeping in near the end generation |
| 40%      | Degrading      | Grammatically valid phrases, but loses overall coherence, repetitive looping |
| 50%      | Fully collapsed | Degenerate repeated tokens, no meaningful language |

**Finding**: this model's weights are safely compressible up to ~30-35% global unstructured sparsity before quality visibly degrades, and fully break down by 50%. Note: no speed improvement observed or expected — standard dense compute still processes the zeroed weights; this is a redundancy/capacity finding, not a latency optimization.


## LLM Structured Pruning — Qwen2.5-0.5B-Instruct (data-driven layer selection)

| Layers removed | Layer count | TPOT | Coherence |
|---|---|---|---|
| none (baseline) | 24 | 114.4 ms/token | Coherent |
| [18] | 23 | 85.3 ms/token | Coherent, fluent |
| [18, 13] | 22 | 81.6 ms/token | Coherent, fluent |
| [18, 13, 12] | 21 | 77.6 ms/token | Coherent, fluent |
| [18, 13, 12, 19] | 20 | 76.4 ms/token | Borderline — readable and on-topic, but showing early softening (odd phrasing, mild hallucination-like claims). Compare: original blind guess at 20 layers ([18,19,20,21]) was complete gibberish — selection method matters more than layer count. |


| [18, 13, 12, 19, 10] | 19 | 71.1 ms/token | Broken — grammatically fluent but confused/off-character, fails to follow the instruction. Different failure mode than gibberish collapse: instruction-following breaks before fluency does. |

## TTS Unstructured Pruning — Piper (en_US-lessac-medium)

| Sparsity | Audio Quality | RTF | Notes |
|---|---|---|---|
| 15% | Clean, natural | 0.423 | No noticeable degradation |
| 20% | Slight noise | 0.438 | Minor artifacts creeping in |
| 30% | Muffled, robotic | 0.399 | Clearly degraded |

**Finding**: Piper tolerates unstructured pruning up to ~15% before quality visibly degrades — a much tighter margin than the LLM's ~30-35%, consistent with Piper being a smaller, already-efficient model with less redundancy to spare (same pattern seen with quantization). As with the LLM, no speed improvement from pruning — RTF stayed flat across all sparsity levels, confirming this is a redundancy finding, not a latency optimization.

## LLM Distillation — 4-layer pruned student, original 24-layer teacher

| Stage | TPOT | Coherence |
|---|---|---|
| Pruned (pre-distillation) | 76.4 ms/token | Coherent |
| Pruned + distilled | 72.0 ms/token | Coherent |


## Distillation Recovery — 5-layer pruned (previously broken) student

| Stage | Layers | TPOT | Coherence |
|---|---|---|---|
| Baseline (unpruned) | 24 | 114.4 ms/token | Coherent |
| Pruned, no distillation | 19 | 71.1 ms/token | Broken — confused, off-character, refused task |
| Pruned + distilled | 19 | 74.7 ms/token | **Recovered — coherent, on-character** |

**Finding**: distillation from the original unpruned model let us safely ship a more aggressively pruned model (5 layers removed) than pruning alone allowed — recovering broken output to coherent quality while keeping a ~35% speedup over baseline. This is the strongest result in the compression pipeline: distillation didn't just polish an already-good model, it rescued one that was genuinely unusable.


## Vision Baseline — Qwen2.5-VL-3B-Instruct (CPU)

| Dtype | TPOT | Notes |
|---|---|---|
| bfloat16 | 2451.7 ms/token | Faster — likely memory-bandwidth-bound, smaller footprint wins |
| float32 | 3866.0 ms/token | Slower despite "native" CPU dtype — larger memory footprint costs more than compute emulation saves |

**Confirmed baseline going forward: bf16, 2451.7 ms/token.** Both numbers are ~21-34x slower than the 0.5B text model — expected given 6x more parameters plus vision encoder overhead, but confirms this model is unusable for real-time use unoptimized.


## Vision Quantization — Qwen2.5-VL-3B-Instruct

| Method | Result |
|---|---|
| GGUF (llama.cpp) | Partial failure — vision encoder silently dropped, converts to text-only model |
| ONNX (optimum) | Explicit failure — architecture unsupported, needs custom OnnxConfig |
| PyTorch dynamic quantization (deprecated eager-mode API) | Mechanically works, output stays correct, but no measurable speedup (2428.6ms vs 2451.7ms baseline) — likely because the model is memory-bandwidth-bound, not compute-bound, so quantizing compute doesn't address the real bottleneck |

**Finding**: all three standard quantization paths either failed outright or produced no benefit for this architecture — a genuine, current limitation of edge-deployment tooling for multimodal models, not a mistake in the approach. The bottleneck analysis (bandwidth vs compute) explains why, which matters more than the numbers alone.

## KV-Cache Quantization — Qwen2.5-0.5B-Instruct

| Method | TPOT | Coherence |
|---|---|---|
| Baseline (fp32 cache) | 114.4 ms/token | Coherent |
| Quantized (HQQ, 4-bit) | 120.8 ms/token | Broken — gibberish, slower than baseline |

**Finding**: aggressive KV-cache quantization is unsafe for this model — it corrupts the attention mechanism's working memory, and errors compound across the generated sequence rather than staying localized (unlike weight quantization). No speed benefit either. This reinforces the project's central theme: small, already-efficient models have far less tolerance for aggressive compression, across every technique tested — weights, cache, and structure alike.


## KV-Cache Memory Management — Toy PagedAttention

| Strategy | Memory (8 sequences, max_len=2048) | Notes |
|---|---|---|
| Naive (pre-allocate worst-case) | 384.00 MB | Standard approach — allocate for the maximum possible length upfront |
| Paged (16-token blocks, on-demand) | 15.00 MB | Only allocates blocks as tokens are actually generated |
| **Savings** | **96.1%** | Block pool correctly reclaims freed blocks for reuse across sequences |

**Scope note**: this implements the block-based memory management strategy (block pool + per-sequence page table) that PagedAttention is built on, and quantifies its real benefit. It doesn't include the custom attention kernel needed to actually compute attention over non-contiguous memory blocks — that's genuinely Phase 6 territory (the Triton kernel work), not a shortcut taken here.

## Speculative Decoding — Draft: pruned+distilled (Phase 3), Target: original Qwen2.5-0.5B

| Metric | Value |
|---|---|
| Acceptance rate | 56.2% (18/32 proposed tokens accepted) |
| Tokens per target forward call | 3.1 |
| Output quality | Coherent, correctly matches expected greedy output |

**Finding**: the distilled draft model (Phase 3's output, trained specifically to mimic the target) shows a strong acceptance rate — validating that reusing a distillation-trained model as the speculative draft is a better choice than a generic small model would be.

## Speculative Decoding — Final, Validated Result

| Metric | Value |
|---|---|
| Output correctness | Exact match to verified target-alone baseline |
| Acceptance rate | 56.2% (18/32) |
| Tokens per target forward call | 3.12 |
| TPOT (this implementation) | 760.6 ms/token — slow due to no KV-cache reuse, a known limitation of this from-scratch build, not of the algorithm |

**Debugging note worth keeping in the write-up**: the first working-looking version actually diverged from the true baseline. Root cause, found by comparing raw logits from `generate()`'s cached path against a plain forward pass: Qwen2.5's default generation config applies a repetition penalty (1.1) that suppresses tokens already present in context — including "Qwen" itself, silently inserted into the default system prompt by the chat template. My implementation's raw forward passes never applied that penalty, so it kept picking "Q" (→"Qwen") instead of the correct "here". This is a genuinely good story for the write-up: it shows real validation discipline (checking against ground truth rather than trusting a plausible-looking output) and a concrete, evidenced root cause rather than a hand-wave.

## Speculative Decoding — Final State

| Version | Correctness | TPOT | Notes |
|---|---|---|---|
| No caching (fully validated) | Exact match to true baseline | 760.6 ms/token | 56.2% acceptance, 3.12 tokens/target-call — the core technique proven correct |
| With caching (partial) | Final output still correct | 291-319 ms/token (~2.5x faster) | Known contained bug: draft re-feeds an already-cached token at the start of each round, corrupting some mid-generation proposals (doesn't affect final output correctness, does reduce efficiency in the affected round) |

**Honest takeaway**: the core speculative decoding algorithm is proven correct and effective (56% acceptance, validated against exact-match ground truth). Caching gives a real, meaningful speedup (~2.5x) but this implementation has a known, diagnosed-but-unfixed edge case in cross-round cache bookkeeping — a legitimate scope boundary, not a hidden flaw.

## Custom Triton Kernel — Fused RMSNorm

| Implementation | Latency | Notes |
|---|---|---|
| PyTorch (unfused, multiple ops) | 0.0464 ms | Standard implementation, each step round-trips to GPU memory |
| Triton (fused, single kernel) | 0.0147 ms | One read, one write — eliminates intermediate memory traffic |
| **Speedup** | **3.15x** | Verified correct (matches PyTorch output within fp16 tolerance). Benchmarked at n_cols=896, the real hidden dimension of Qwen2.5-0.5B — this exact op runs in every transformer layer, every forward pass. |


## Custom Triton Kernel — Fused INT8 Dequantize + Matmul

| Implementation | Latency | Notes |
|---|---|---|
| Unfused (dequant to fp16, then cuBLAS matmul) | 0.0369 ms | Two passes: dequantize (write full fp16 weights), then matmul (read them back) |
| Fused (Triton, INT8 read + dequant + matmul in one kernel) | 0.0205 ms | Dequantization happens inline in the matmul's inner loop — weights never materialized in fp16 |
| **Speedup** | **1.79x** | Exact match to reference (0.0 difference). Benchmarked at K=N=896, matching Qwen2.5-0.5B's real hidden dimension. Beats PyTorch's cuBLAS-backed matmul — a genuinely hard bar to clear. |


## C++ Inference Wrapper — Piper TTS via ONNX Runtime C++ API

| Component | Status |
|---|---|
| Compiler | g++ (MinGW), not MSVC — confirmed compatible with ONNX Runtime's official Windows binaries |
| Model loading | Working — loads en_US-lessac-medium.onnx via Ort::Session |
| Multi-input tensor handling | Working — phoneme IDs (int64), lengths (int64), scales (float) correctly constructed and passed |
| Inference | Working — 18,432 audio samples generated for "Hello there." |
| Output | Real WAV file written directly from C++ (manual RIFF/WAVE header construction, no library dependency for audio output) |

**Scope note**: text→phoneme conversion (Piper's phonemizer, which uses espeak-ng) stays in Python for this proof-of-concept — the C++ wrapper focuses specifically on the inference step, which is the part that would actually run inside a game engine's real-time loop. Full text-to-phoneme in C++ would be a separate, substantial addition (bundling or reimplementing espeak-ng).


## Phase 8 — Unity Integration

| Component | Status |
|---|---|
| Native plugin loading | Working — `tts_plugin.dll` + `onnxruntime.dll` loaded via Unity's `Plugins/x86_64` convention |
| C# ↔ C++ interop | Working — `DllImport` bindings correctly call `InitTTS`, `Synthesize`, `ShutdownTTS` |
| Async, non-blocking call | Working — `Synthesize` runs via `Task.Run`, off Unity's main thread |
| Model loading at runtime | Working — via `Application.streamingAssetsPath` |
| Audio playback | Working — synthesized samples converted to an `AudioClip` and played through an `AudioSource` |
| End-to-end demo | Working — button click in Unity triggers real TTS inference and audible speech |

**Debugging notes worth keeping for the write-up**:
1. **Silent exception swallowing hid the real error.** The original `catch (...) { return false; }` gave no diagnostic info at all — adding proper exception logging (`catch (const std::exception& e)`, writing `e.what()` to a log file) was what actually surfaced the fix. A good example of why silently swallowing exceptions in native code is a real anti-pattern, not just a style nitpick.
2. **Root cause was a one-character typo**: `StremingAssets` instead of `StreamingAssets` — Unity treats "StreamingAssets" as a reserved folder name and silently fails to recognize anything else, with no warning at all. A small, human mistake, but a good illustration of how a single typo in engine-reserved naming can produce a completely opaque failure three layers removed (Unity → C# → C++) from where the actual mistake was made.
3. Also fixed along the way: Unity's native plugin folder needed the exact name `x86_64` (was initially `Windows_86_64`), and P/Invoke string marshaling needed `CharSet = CharSet.Unicode` to correctly pass a wide-character path to the C++ `wchar_t*` parameter — a common, easy-to-miss P/Invoke gotcha.


## LLM ↔ Unity Integration — Partial

The LLM C++ wrapper (llama.cpp native API) is validated standalone — generates coherent text through the compressed GGUF model. Integrating it into the Unity editor process hit a dynamic-library loading issue: llama.cpp discovers its compute-backend DLLs (`ggml-cpu-*`) at runtime via executable/working-directory search, which doesn't resolve correctly inside Unity's process. The TTS half of the pipeline is fully integrated and working in Unity; the LLM half is proven in C++ but not yet bridged into the engine. Resolving this would likely mean statically linking the ggml backends or a custom backend-registration path — a known, bounded next step, not an open question.


## LLM ↔ Unity Integration — Working (Subprocess Architecture)

| Component | Status |
|---|---|
| In-process approach (`llm_plugin.dll` via `DllImport`) | Failed — `llama_model_load_from_file` returned null inside Unity's process |
| Subprocess approach (`llm_cli.exe` spawned via `System.Diagnostics.Process`) | Working — generates real text from the pruned+distilled+quantized model |
| Phonemization bridge (`phonemize_bridge.py`, stdin-based) | Working |
| TTS synthesis (existing native plugin) | Working |
| End-to-end demo | Working — button click triggers live LLM generation → phonemization → speech, fully inside Unity |

**Root cause of the in-process failure**: llama.cpp's `ggml_backend_load_all()` discovers compute-backend DLLs (`ggml-cpu-*.dll`) by scanning the directory of the *main executable* — `Unity.exe` inside the Editor, nowhere near the plugin DLLs — so no compute backend ever registered and model loading silently returned null. Rather than statically relinking llama.cpp (a heavier undertaking), the fix was architectural: run `llm_cli.exe` as a genuine subprocess from its own known-good directory (the same pattern already used for the Python phonemizer), sidestepping backend discovery entirely instead of solving it in-process.

**Debugging notes worth keeping for the write-up**:
1. **A missing `libomp.dll` masked itself as an unrelated Windows API-set error.** Copying the exe and model DLLs into a second folder (`StreamingAssets`, for an earlier attempt) without the OpenMP runtime produced `api-ms-win-crt-string-l1-1-0.dll: cannot open shared object file` — a generic-looking CRT error unrelated to the actual missing dependency. Diffing directory contents byte-for-byte between the working and failing locations surfaced it.
2. **Classic stdout/stderr pipe deadlock.** `llama.cpp` writes tens of KB of model-loading diagnostics to stderr; the original C# code called `StandardOutput.ReadToEnd()` before touching stderr, so once the OS pipe buffer for stderr filled, the child blocked writing to it while C# blocked waiting on stdout — neither side could proceed. Fixed by reading both streams concurrently via `ReadToEndAsync()` before `WaitForExit()`. Several `llm_cli.exe` processes were left hung in the background before this was caught.
3. **Command-line argument quoting broke on the model's own output.** Passing LLM-generated text (which naturally contains embedded quotes, e.g. `Say: "Hello..."`) as a quoted command-line argument to `phonemize_bridge.py` corrupted the argument boundary. Fixed by switching the bridge to read from stdin instead of `sys.argv` — a more robust pattern for passing arbitrary generated text between processes in general.
4. **A relative model path silently failed under a different working directory.** `phonemize_bridge.py` loads `en_US-lessac-medium.onnx` by relative path, which only resolves when the process's working directory is `C:\VirtualCharEngine`. The subprocess launch never set `WorkingDirectory`, so Python threw and produced empty output — surfacing several layers away in C# as a `FormatException`. This was the first time this code path had actually executed, since every earlier attempt died upstream at the (now-fixed) in-process LLM init.

## LLM ↔ Unity Integration — Measured End-to-End Latency (CPU, subprocess architecture)

| Run | LLM generation | Phonemization | TTS synthesis | Total |
|---|---|---|---|---|
| 1 | 4408 ms | 2804 ms | 434 ms | 7647 ms |
| 2 | 3518 ms | 3066 ms | 345 ms | 6931 ms |
| 3 | 2900 ms | 3643 ms | 298 ms | 6842 ms |
| **avg** | **~3609 ms** | **~3171 ms** | **~359 ms** | **~7140 ms** |

**Honest caveat**: this measures cold, per-call subprocess overhead, not steady-state inference — both `llm_cli.exe` and `phonemize_bridge.py` reload their models from disk on every single call instead of staying resident. Phonemization (pure Python interpreter and Piper voice reload) costs roughly as much wall-clock time as LLM generation itself — the bottleneck here is process startup, not model compute. A persistent-process or properly-fixed in-process architecture would cut this dramatically; that's the clear next optimization, not pursued further here given scope.