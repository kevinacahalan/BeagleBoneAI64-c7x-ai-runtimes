# Measured Hardware Results (BeagleBone AI-64 / TDA4VM)

All numbers below were measured directly on a live BeagleBone AI-64 (TI TDA4VM / J721E, C71x @ 1.0 GHz, MMA v1, L1D = 32 KiB cache, L2 = 64 KiB cache + 448 KiB SRAM, 7.625 MiB MSMC SRAM mapped, 1.5 GiB high-DDR LLM arena at `0x880000000` + 64 MiB DDR arena at `0xAC000000`) with **bit-exact or cosine `>= 0.998–1.000` verification against the Python/A72 reference**.

---

## Milestone 1: C7x MMA + Memory Hierarchy + W4A8 Microbenchmark Suite

- **Compiler:** TI C7000 CGT `1.2.0.STS` (`-mv7100 -O3`)
- **Linux <-> C7x round-trip (`ping`, 100 iters):** `2.29 µs` wall time (`~200` C7x cycles)

### 1. MMA C-Bank Row Depth Probe (`int8 × int8 → int32`)

| Rows tested | Host Emulator (`libc70-host-emulation.a`) | Live TDA4VM Silicon | Notes |
| --- | --- | --- | --- |
| 16 | PASS (0 mismatches) | **PASS** (0 mismatches) | 16 × 256 B = 4 KiB |
| 32 | WRAP at row 16 | **PASS** (0 mismatches) | Emulator bug confirmed |
| 48 | WRAP at row 16 | **PASS** (0 mismatches) | |
| 60 | WRAP at row 16 | **PASS** (0 mismatches) | `X_CRRSTPER = 240` |
| 63 | WRAP at row 16 | **PASS** (0 mismatches) | `X_CRRSTPER = 252` (max 8-bit `4*M`) |
| 64 | WRAP | WRAP (`X_CRRSTPER` wraps `(64*4)&0xff=0`) | Physical bank is 64 rows (16 KiB); `int32` readout maxes at 63 rows per bank |

### 2. C7x Streaming Engine (`SE0`) Memory Bandwidth

| Memory Tier | Buffer Size | SE0 512-bit Read | SE0 Copy Payload | SE0 Copy Total (R+W) |
| --- | --- | --- | --- | --- |
| **L2 SRAM** (`0x64801000`) | 128 KiB | **56.73 GB/s** | **56.78 GB/s** | **113.56 GB/s** |
| **MSMC SRAM** (`0x70200000`) | 1 MiB | **42.47 GB/s** | **22.05 GB/s** | **44.10 GB/s** |
| **LPDDR4 Arena** (`0xAC000000`) | 4 MiB | **10.23 GB/s** | **3.02 GB/s** | **6.04 GB/s** |

### 3. MMA `int8 × int8 → int32` GEMM (Prefill / Encoder Shapes)

Multi-tile accumulating MMA GEMM (`C_OPERATION0 = MUL`, `C_OPERATION1 = MULPLUS`, B-bank ping-pong via `__HWALDAB`, hardware transpose `B_ORDER = COL` for `BT` shapes, exact `int32` readout):

| Tier | Shape (`M × K × N`) | Mode | `block_m` | Latency | MAC/cycle | Throughput | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | `16 × 64 × 64` | `BN` | 16 | 1.0 µs | 68.2 | 136.4 GOPS | **PASS** |
| L2 | `60 × 256 × 256` | `BT` | 60 | 12.2 µs | 322.3 | **644.7 GOPS** | **PASS** |
| MSMC | `64 × 256 × 256` | `BT` | 16 | 31.8 µs | 131.8 | 263.7 GOPS | **PASS** |
| MSMC | `64 × 256 × 256` | `BT` | 32 | 21.6 µs | 194.2 | 388.3 GOPS | **PASS** |
| MSMC | `60 × 512 × 512` | `BT` | 60 | 45.2 µs | 347.8 | **695.5 GOPS** | **PASS** |
| MSMC | `120 × 512 × 512` | `BT` | 60 | 90.4 µs | 348.0 | **696.1 GOPS** | **PASS** |
| DDR | `60 × 512 × 512` | `BT` | 60 | 113.4 µs | 138.7 | 277.4 GOPS | **PASS** |
| DDR | `120 × 768 × 768` | `BT` | 60 | 447.5 µs | 158.2 | **316.3 GOPS** | **PASS** |

*Next GEMM optimization targets:* pre-packing weights into 64×64 tile-major layout (eliminating strided 2D SE1 DDR fetches) and B-stationary multi-M reuse + DRU double-buffering into L2 SRAM to push prefill GEMM past **2–4+ TOPS**.

### 4. Autoregressive Decode GEMV (`M = 1`): W8A8 MMA vs. Packed W4A8 Vector

| Tier | Shape (`1 × K × N`) | Kernel | Latency | Weight BW | Eq. INT8 BW | Speedup vs W8A8 | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | `1 × 256 × 256` | W8A8 MMA | 4.2 µs | 15.74 GB/s | 15.74 GB/s | 1.00× | **PASS** |
| MSMC | `1 × 256 × 768` | W8A8 MMA | 17.9 µs | 11.01 GB/s | 11.01 GB/s | 1.00× | **PASS** |
| MSMC | `1 × 512 × 512` | W8A8 MMA | 23.2 µs | 11.32 GB/s | 11.32 GB/s | 1.00× | **PASS** |
| MSMC | `1 × 512 × 2048` | W8A8 MMA | 91.9 µs | 11.40 GB/s | 11.40 GB/s | 1.00× | **PASS** |
| DDR | `1 × 512 × 512` | W8A8 MMA | 59.7 µs | 4.39 GB/s | 4.39 GB/s | 1.00× | **PASS** |
| DDR | `1 × 512 × 512` | **W4A8 Vec** | **31.3 µs** | 4.20 GB/s | **8.39 GB/s** | **1.91×** | **PASS** |
| DDR | `1 × 512 × 2048` | W8A8 MMA | 238.3 µs | 4.40 GB/s | 4.40 GB/s | 1.00× | **PASS** |
| DDR | `1 × 512 × 2048` | **W4A8 Vec** | **118.8 µs** | 4.41 GB/s | **8.83 GB/s** | **2.01×** | **PASS** |
| DDR | `1 × 768 × 768` | W8A8 MMA | 117.6 µs | 5.01 GB/s | 5.01 GB/s | 1.00× | **PASS** |
| DDR | `1 × 768 × 768` | **W4A8 Vec** | **52.8 µs** | 5.59 GB/s | **11.18 GB/s** | **2.23×** | **PASS** |

*Key insight:* On DDR decode (`M = 1`), the current single-row W4A8 vector kernel is **2.0–2.23× faster** than W8A8 MMA (`11.18 GB/s` equivalent INT8 bandwidth, beating raw 10.23 GB/s DDR read bandwidth by halving weight size). Unrolling W4A8 across 4 output rows and pre-packing weights into contiguous tiles will push W4A8 toward **~18–20 GB/s equivalent INT8** on DDR (and **40+ GB/s eq-INT8** in MSMC SRAM for models ≤ 7 MB).

### 5. Bare-Metal C7x `DRU0` Direct-TR DMA Bandwidth (`TR9` via `CHATOMIC[0]`)

Unlocked natively from bare-metal C7x (zero TI-SCI, zero RTOS) using `TR9` (`TYPE = 9`, matching `DRU_CAPABILITIES = 0x00000959000d0f20`) submitted via a single 512-bit atomic store to `CHATOMIC[0]` (`0x6D080000`) with an ordered 8-byte completion marker into non-cacheable L2 SRAM (`0x64800F00`):

| Transfer Path | Buffer Size | Measured Bandwidth | Latency / Copy | Verification |
| --- | --- | --- | --- | --- |
| **DDR → L2 SRAM** | 128 KiB | **10.92 GB/s** | 12.0 µs | **PASS** |
| **L2 SRAM → DDR** | 128 KiB | **10.79 GB/s** | 12.1 µs | **PASS** |
| **DDR → MSMC SRAM** | 1 MiB | **10.89 GB/s** | 96.3 µs | **PASS** |
| **MSMC SRAM → L2 SRAM** | 128 KiB | **55.93 GB/s** | 2.3 µs | **PASS** |
| **DDR → DDR** | 4 MiB | **5.07 GB/s** | 826.9 µs | **PASS** |
| **DDR → L2 → DDR round-trip** | 128 KiB | — | — | **PASS (byte-for-byte)** |

---

## Milestone 2: End-to-End W8A8 Transformer (`stories15M`) on C7x + MMA

- **Model:** `karpathy/llama2.c` `stories15M` (`dim=288→320`, `hidden=768`, `n_layers=6`, `n_heads=6`, `head_size=48`, `vocab=32000`, `seq_len=256`, tied INT8 embeddings/classifier)
- **Packed `.c7m` size:** `16.58 MiB` (`17,389,312` bytes), total DDR arena used with FP32 KV cache: `20.08 / 64.00 MiB`; leading `3/6` layers (`3.32 MiB`) staged in on-chip 4 MiB MSMC SRAM (`0x70200000`)
- **All linear projections (`wq, wk, wv, wo, w1, w2, w3, wcls`) execute on the C7x MMA** (`int8 × int8 → int32`); all non-matmul ops (RMSNorm, RoPE, Softmax, SwiGLU, MHA Attention, quant/dequant, residual adds, argmax) execute on the C7x 512-bit FP32 vector unit.

### 1. Live Silicon Correctness Verification (`c7x-ai llama verify`)

| Check | Result | Measured Error / Metric |
| --- | --- | --- |
| C7x silicon FP32 vector self-test (`C7X_OP_VEC_SELFTEST`) | **PASS** | `max_err = 4.77e-7` |
| Single decoder layer (`layer 0`, `pos = 0`) vs. Python reference | **PASS** | `max_abs_err = 8.34e-7`, `cosine = 1.0000000` (`0.080 ms`) |
| Final RMSNorm (`pos = 0`) vs. Python reference | **PASS** | `max_abs_err = 2.38e-5`, `cosine = 1.0000000` |
| All 32,000 output logits (`pos = 0`) vs. Python reference | **PASS** | `max_abs_err = 1.91e-6`, `cosine = 1.0000000` |
| Greedy (`temp = 0`) generation vs. FP32 `stories15M` reference | **PASS** | **40 / 40 exact token match** (`Once upon a time, there was a little girl named Lily...`) |

### 2. Optimization Progression (Measured Live on BeagleBone AI-64, 40 tokens)

| Build | C7x Compute | End-to-End (incl. A72) | Speedup vs v1 | Speedup vs 1-core A72 |
| --- | --- | --- | --- | --- |
| **1-core Cortex-A72 CPU (W8A8 baseline)** | — | `49.96 tok/s` | — | `1.0×` |
| **v1 — Initial C7x+MMA** (scalar `libc` `expf`, row-major 2D SE1, DDR-only, A72 `O_SYNC` logits read) | `8.440 ms` (`118.49 tok/s`) | `91.82 tok/s` | `1.00×` | `2.4×` DSP / `1.8×` e2e |
| **v2 — + 16-wide `vexpf_f16` SwiGLU/Softmax + fused C7x `argmax`** | `5.261 ms` (`190.06 tok/s`) | `186.02 tok/s` | `1.60×` / `2.03×` | `3.8×` DSP / `3.7×` e2e |
| **v3 — + 1D Tile-Major MMA (`64×64`) + 3-Layer MSMC Staging + Zero-Scalar Softmax** | `2.571 ms` (`388.98 tok/s`) | `373.19 tok/s` | `3.28×` / `4.06×` | `7.8×` DSP / `7.5×` e2e |
| **v4 — + 2D SE0/SE1 Prefetched Attention + SE0/SE1 Block-Max Dequant-Argmax** | **`1.912 ms` (`523.12 tok/s`)** | **`495.83 tok/s`** | **`4.41×` / `5.40×`** | **`10.5×` DSP / `9.9×` e2e** |

- **128-token prompt generation (`One day, a little robot`):** `128` tokens in `0.260 s` → **`491.69 tok/s` end-to-end** (`2.000 ms/tok = 500.09 tok/s` C7x DSP).

### 3. Per-Operator & Per-Layer Cycle Breakdown (v4, Live Silicon, 1.0 GHz C7x)

| Operator / Stage | 40-tok avg cycles | 40-tok avg (ms) | Share | 128-tok avg (ms) | Notes |
| --- | --- | --- | --- | --- | --- |
| **TOTAL (C7x DSP)** | **1,911,598** | **1.912 ms** | **100.0%** | **2.000 ms** | **523.1 tok/s** (40t) / **500.1 tok/s** (128t) |
| Classifier MMA (`1×320×32000`) | 1,078,483 | 1.078 ms | 56.4% | 1.068 ms | `10.24 MB` @ **9.50–9.59 GB/s** (93.7% of raw DDR ceiling) |
| FFN W1+W3 MMA (6 layers) | 208,418 | 0.208 ms | 10.9% | 0.206 ms | 3 layers MSMC, 3 layers DDR |
| QKV MMA (6 layers) | 135,726 | 0.136 ms | 7.1% | 0.134 ms | 3 layers MSMC, 3 layers DDR |
| Dequant `int32→fp32` + Argmax | 108,703 | 0.109 ms | 5.7% | 0.108 ms | SE0+SE1 single-pass block-max |
| FFN W2 MMA (6 layers) | 103,976 | 0.104 ms | 5.4% | 0.101 ms | 3 layers MSMC, 3 layers DDR |
| MHA Attention (6 layers) | 88,077 | 0.088 ms | 4.6% | 0.192 ms | 2D SE0/SE1 K/V prefetch + `vexpf_f16` softmax |
| Wo MMA (6 layers) | 44,997 | 0.045 ms | 2.4% | 0.045 ms | 3 layers MSMC, 3 layers DDR |
| Quant `fp32→int8` | 36,309 | 0.036 ms | 1.9% | 0.037 ms | 16-wide `float16` + `__float_to_int` |
| SwiGLU (6 layers) | 33,713 | 0.034 ms | 1.8% | 0.034 ms | Down from `3.024 ms` (**89× speedup**) |
| RoPE + KV write | 31,588 | 0.032 ms | 1.7% | 0.032 ms | Precomputed cos/sin tables |
| RMSNorm (13 calls) | 30,706 | 0.031 ms | 1.6% | 0.031 ms | 16-wide `float16` |
| Embedding lookup | 2,225 | 0.002 ms | 0.1% | 0.002 ms | Tied INT8 row dequant |

| Layer | Memory Tier | 40-tok avg cycles | 40-tok avg (ms) | 128-tok avg (ms) |
| --- | --- | --- | --- | --- |
| Layer 0 | **MSMC SRAM** | 81,168 | **0.081 ms** | 0.099 ms |
| Layer 1 | **MSMC SRAM** | 79,479 | **0.079 ms** | 0.098 ms |
| Layer 2 | **MSMC SRAM** | 79,091 | **0.079 ms** | 0.097 ms |
| Layer 3 | **LPDDR4** | 187,404 | **0.187 ms** | 0.203 ms |
| Layer 4 | **LPDDR4** | 185,404 | **0.185 ms** | 0.201 ms |
| Layer 5 | **LPDDR4** | 187,086 | **0.187 ms** | 0.201 ms |

---

## Milestone 3: Universal W4A8 MMA, Batched Prefill, Multi-Architecture LLMs, & DRU0 Async Double-Buffering

Extended the bare-metal runtime to be **model-agnostic** (`tools/convert_hf.py` + embedded `C7T1` BPE tokenizer + `512 MiB` DDR LLM arena at `0xC0000000`), added **MMA-accelerated tile-major group-64 W4A8** (`c7x_mma_gemm_w4a8_g64_f32`), **batched multi-token MMA prefill** (`C7X_OP_LLAMA_PREFILL`, `block_m <= 60`), **DRU0 async L2-SRAM ping-pong double-buffering** (zero-cache-flush via `SE0`/`SE1` L1D bypass), and verified **three distinct transformer architectures** on live BeagleBone AI-64 silicon:

### 1. Cross-Model Live Silicon Comparison (1.0 GHz C7x + MMA + DRU0)

| Model | Arch / Features | Quant | `.c7m` Size | MSMC Staged | Decode (C7x DSP) | Decode (E2E) | Batched Prefill (DSP) | Logits `cosine` / `max_err` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`stories15M`** (15M) | Llama2, 6L, 6/6 MHA, `hd=48`, 32k vocab | W8A8 | `16.58 MiB` | `3/6` layers | **523.1 tok/s** (`1.91 ms`) | **495.8 tok/s** | **1,950.2 tok/s** (32t) | `1.0000000` / `1.91e-6` (**40/40**) |
| **`stories15M`** (15M) | Llama2, 6L, 6/6 MHA, `hd=48`, 32k vocab | **W4A8-g64** | `9.25 MiB` | **`6/6` layers** | **844.8 tok/s** (`1.18 ms`) | **788.9 tok/s** | **2,360.2 tok/s** (32t) | `1.0000000` / `6.68e-6` (**40/40**) |
| **`SmolLM2-135M-Instruct`** (135M) | Llama, 30L, 9/3 GQA, `hd=64`, NeoX RoPE, `θ=100k`, 49k vocab | **W4A8-g64** | `73.65 MiB` | `2/30` layers | **102.2 tok/s** (`9.79 ms`) | **101.6 tok/s** | **241.4 tok/s** (19t) | `1.0000000` / `5.53e-5` |
| **`Qwen2.5-0.5B-Instruct`** (494M) | Qwen2, 24L, 14/2 GQA, **QKV biases**, NeoX RoPE, `ε=1e-6`, `θ=1M`, **152k vocab** | **W4A8-g64** | `269.20 MiB` | `0/24` layers | **33.6 tok/s** (`29.79 ms`) | **33.0 tok/s** | **100.0 tok/s** (23t) | `1.0000000` / `2.19e-5` (**6/6**) |

*(Note: `SmolLM2-135M` decode is `105.4 tok/s` over 42 short-context tokens and `102.2 tok/s` over 64 generated tokens after a 19-token chat prompt; `Qwen2.5-0.5B` decode is `33.57 tok/s` DSP / `32.78 tok/s` e2e over 20 tokens and `33.11 tok/s` DSP / `33.01 tok/s` e2e over 29 generated tokens after a 23-token chat prompt.)*

### 2. Impact of DRU0 Async L2-SRAM Ping-Pong on Large DDR Matrices (`>= 1 MiB`)

On DDR matrices `>= 1 MiB`, DRU0 prefetches the next tile chunk into L2 SRAM (`10.92 GB/s`) while MMA + `SE0`/`SE1` compute the active chunk from L2 SRAM (`59 GB/s`) with zero cache flushes:

| Operator (Live Silicon, W4A8-g64) | Direct `SE1` DDR | With `DRU0` L2 Ping-Pong | Speedup / Effective BW |
| --- | --- | --- | --- |
| `stories15M` Classifier (`32k × 320`) | `0.608 ms` | **`0.582 ms`** | `+4.5%` |
| `SmolLM2-135M` Classifier (`49k × 576`) | `1.657 ms` | **`1.494 ms`** | **`+10.9%`** (`10.6 GB/s` raw) |
| `Qwen2.5-0.5B` FFN W1+W3 (24 layers) | `12.324 ms` | **`11.544 ms`** | **`+6.8%`** |
| `Qwen2.5-0.5B` FFN W2 (24 layers) | `6.155 ms` | **`5.981 ms`** | `+2.9%` |
| `Qwen2.5-0.5B` Classifier (`152k × 896`, 76.6 MiB) | `7.983 ms` | **`7.033 ms`** | **`+13.5%`** (**`10.52 GB/s` raw = `21.0 GB/s` eq-INT8**) |
| **`Qwen2.5-0.5B` Total Decode** | `31.65 ms` (`31.6 tok/s`) | **`29.79 ms` (`33.6 tok/s`)** | **`+6.2%` overall** |

### 3. Per-Operator Breakdown: `SmolLM2-135M` & `Qwen2.5-0.5B` (Live Silicon, W4A8 + DRU0)

| Stage | `SmolLM2-135M` (64t avg) | Share | `Qwen2.5-0.5B` (29t avg) | Share |
| --- | --- | --- | --- | --- |
| **TOTAL (C7x DSP)** | **9.789 ms** (**102.2 tok/s**) | **100.0%** | **30.207 ms** (**33.1 tok/s**) | **100.0%** |
| FFN W1+W3 MMA | 3.107 ms | 31.7% | 11.544 ms | 38.2% |
| Classifier MMA | 1.494 ms | 15.3% | 7.033 ms | 23.3% |
| FFN W2 MMA | 1.550 ms | 15.8% | 5.982 ms | 19.8% |
| QKV MMA | 1.028 ms | 10.5% | 1.538 ms | 5.1% |
| Wo MMA | 0.624 ms | 6.4% | 1.184 ms | 3.9% |
| MHA Attention (2D SE0/SE1) | 0.979 ms | 10.0% | 0.918 ms | 3.0% |
| SwiGLU (`vexpf_f16`) | 0.331 ms | 3.4% | 0.866 ms | 2.9% |
| Quant `fp32→int8` | 0.266 ms | 2.7% | 0.430 ms | 1.4% |
| RMSNorm | 0.290 ms | 3.0% | 0.366 ms | 1.2% |
| RoPE + KV write + QKV bias | 0.042 ms | 0.4% | 0.051 ms | 0.2% |
| Embedding lookup | 0.016 ms | 0.2% | 0.025 ms | 0.1% |

---

## Milestone 4: 7.625 MiB MSMC, B-Stationary Prefill, INT8 KV (`2K–4K`), Qwen3 & 1.5B–1.7B Models, and Speculative Decoding

Expanded on-chip MSMC SRAM from `4 MiB → 7.625 MiB` (`0x70040000..0x707DFFFF` via L3 4 KiB edge page tables), expanded the high-DDR LLM arena to **`1.5 GiB`** (`0x880000000..0x8DFFFFFFF`, leaving ~2.1 GiB RAM for Linux), optimized B-stationary batched prefill (`C7X_OP_LLAMA_PREFILL`) with DRU0 L2 ping-pong + L2-fused RMSNorm/Attention/SwiGLU+Quant, added **per-head symmetric INT8 KV caching** (`3.76–3.88×` smaller KV), and added **Qwen3 QK-RMSNorm** (`q_norm`/`k_norm`, `q_dim != dim`, `head_size = 128`).

### 1. Live Silicon Multi-Architecture Matrix (6 Models, 15M → 1.7B, W4A8-g64)

| Model | Arch / Features | `.c7m` Size | MSMC Staged | KV Mode (`seq`) | Short-Ctx Decode (DSP / E2E) | Batched Prefill (`pp`) | Logits `cosine` | Greedy Parity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`stories15M`** (15M) | Llama2, 6L, 6/6 MHA, `hd=48`, 32k | `9.25 MiB` | `6/6` (`4.24 MiB`) | FP32 (`256`) | **871.7 / 808.6 tok/s** | **3,207.9 tok/s** (`pp32`) | `1.0000000` | **40/40** |
| **`SmolLM2-135M-Instruct`** (135M) | Llama, 30L, 9/3 GQA, `hd=64`, 49k | `73.65 MiB` | `4/30` (`7.61 MiB`) | FP32 (`512`) | **115.7 / 114.6 tok/s** | **395.2 tok/s** (`pp19`) / **323.9 tok/s** (`pp160`) | `1.0000000` | **exact** |
| **`Qwen2.5-0.5B-Instruct`** (494M) | Qwen2, 24L, 14/2 GQA, QKV bias, 152k | `269.20 MiB` | `0/24` | FP32 (`512`) | **35.05 / 34.9 tok/s** | **209.6 tok/s** (`pp60`) / **168.7 tok/s** (`pp146`) | `1.0000000` | **exact** |
| **`Qwen3-0.6B`** (596M) | Qwen3, 28L, 16/8 GQA, `q_dim=2048`, `hd=128`, **QK-norm** | `324.79 MiB` | `0/28` | FP32 / **INT8** (`2048`) | **26.93 / 26.61** (FP32)<br>**27.24 / 26.91 tok/s** (INT8) | **126.8 tok/s** (`pp24`) / **104.5 tok/s** (`pp146`) | `1.0000000` (FP32)<br>`0.99951` (INT8) | **10/10** (both) |
| **`Qwen2.5-1.5B-Instruct`** (1.54B) | Qwen2, 28L, 12/2 GQA, `hd=128`, QKV bias, 152k | `833.41 MiB` | `0/28` | FP32 / **INT8** (`2048`) | **11.60 / 11.53** (FP32)<br>**11.62 / 11.54 tok/s** (INT8) | **60.1 tok/s** (`pp24`) / **71.6 tok/s** (`pp146`) | `1.0000000` (FP32)<br>`0.99818` (INT8) | **10/10** (both) |
| **`Qwen3-1.7B`** (1.72B) | Qwen3, 28L, 16/8 GQA, `hd=128`, **QK-norm**, 152k | `928.22 MiB` | `0/28` | FP32 / **INT8** (`2048`) | **10.31 / 10.25** (FP32)<br>**10.34 / 10.27 tok/s** (INT8) | **57.5 tok/s** (`pp24`) | `1.0000000` (FP32)<br>`0.99931` (INT8) | **10/10** (both) |

### 2. Fast B-Stationary Batched Prefill Speedup (Milestone 3 → Milestone 4)

| Model | M3 Prefill (tok/s) | M4 B-Stationary + DRU0 + L2-Fused Prefill (tok/s) | Speedup |
| --- | --- | --- | --- |
| **`stories15M` (W4A8-g64, 32t)** | `2,360.2 tok/s` | **`3,207.9 tok/s`** | **`1.36×`** |
| **`SmolLM2-135M` (19t)** | `241.4 tok/s` | **`395.2 tok/s`** | **`1.64×`** |
| **`Qwen2.5-0.5B` (23t → 60t)** | `100.0 tok/s` | **`147.5 – 209.6 tok/s`** | **`1.48× – 2.10×`** |

### 3. Quantized INT8 KV Cache & Long Context (`2,048 – 4,096` Tokens)

| Model (`seq_len`) | FP32 KV Footprint | INT8 KV Footprint | Savings | Long-Context Test (Live Silicon) |
| --- | --- | --- | --- | --- |
| **`Qwen2.5-0.5B`** (`seq=4096`) | `96.00 MiB` | **`25.50 MiB`** | `3.76×` | **2,043 prompt tokens:** `48.30 s` (`42.3 tok/s`), decode @ pos ~2043 = **`14.4 tok/s`** (`2.60 ms/layer`)<br>**3,803 prompt tokens:** `158.17 s` (`24.0 tok/s`), decode @ pos ~3803 = **`9.5 tok/s`** (`4.07 ms/layer`) |
| **`Qwen3-0.6B`** (`seq=2048`) | `448.00 MiB` | **`115.50 MiB`** | `3.88×` | `logits cos = 0.99951`, **10/10 exact greedy**, decode **`27.24 tok/s`** (`+1.2%` vs FP32 KV) |
| **`Qwen2.5-1.5B`** (`seq=2048`) | `112.00 MiB` | **`28.88 MiB`** | `3.88×` | `logits cos = 0.99818`, **10/10 exact greedy**, decode **`11.62 tok/s`**, arena `895 / 1536 MiB` |
| **`Qwen3-1.7B`** (`seq=2048`) | `448.00 MiB` | **`115.50 MiB`** | `3.88×` | `logits cos = 0.99931`, **10/10 exact greedy**, decode **`10.34 tok/s`**, arena `1076 / 1536 MiB` |

### 4. Speculative Decoding Evaluation (`Qwen2.5-0.5B` Drafting for `Qwen2.5-1.5B`)

Both models load simultaneously into the 1.5 GiB arena (`1,159.9 / 1,536 MiB`) using `c7x-ai llama spec` with `C7X_LLAMA_REQ_ALL_ARGMAX` batched verification and exact KV rollback:

| Config | Parity vs Standard 1.5B | Accepted Draft Tokens | Tokens / Step | Draft Time | Target Verify Time (`M=K+1`) | Net Throughput | vs Standard 1.5B (`11.50 tok/s`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard `Qwen2.5-1.5B` (`M=1`) | Baseline | — | `1.00` | — | `86.8 ms` | **`11.50 tok/s`** | `1.00×` |
| Speculative `K = 3` (`M = 4`) | **100% exact** | `30 / 64` (`46.9%`) | `2.27` | `92.9 ms` | `254.9 ms` | `6.52 tok/s` | `0.57×` |
| Speculative `K = 4` (`M = 5`) | **100% exact** | `31 / 80` (`38.8%`) | `2.50` | `121.3 ms` | `285.6 ms` | `6.13 tok/s` | `0.53×` |
| Speculative `K = 5` (`M = 6`) | **100% exact** | `32 / 100` (`32.0%`) | `2.50` | `149.0 ms` | `306.5 ms` | `5.48 tok/s` | `0.48×` |

*Analysis:* See [`NOTES.md`](NOTES.md) — on TDA4VM C71x, group-64 W4A8 batched verification drains the MMA C-bank every `K=64` group, making `M=4` verification `2.94×` slower than `M=1` decode rather than `~1.0×` as on GPUs. The `c7x-ai llama spec` engine is shipped as a verified lossless diagnostic/research tool, while single-model decode remains the default fastest path on C71x.

---

## Milestone 6: Head-to-Head Benchmarks — C7x + MMA vs. `llama.cpp` on BeagleBone AI-64 Cortex-A72

Measured on the **exact same live BeagleBone AI-64 board** (dual Cortex-A72 @ 2.0 GHz vs. C71x + MMA @ 1.0 GHz). `llama.cpp` (`llama-bench`, built natively for `-mcpu=cortex-a72 -O3`, `GGML_OPENMP=OFF`) was benchmarked with `-t 1,2 -p 128 -n 64` across both `Q4_0` and `Q4_K_M` GGUF quants (mean of 3 reps for `135M–0.6B`, 2 reps for `1.5B`; std dev `<= 0.06 tok/s` across every entry).

### 1. Prompt Prefill Speed (`pp128` `llama.cpp` vs. `pp146–160` C7x + MMA)

| Model | `llama.cpp` `Q4_0` (1T) | `llama.cpp` `Q4_K_M` (1T) | `llama.cpp` `Q4_0` (2T) | `llama.cpp` `Q4_K_M` (2T) | **C7x + MMA (`W4A8-g64`)** | Speedup vs 1T (`Q4_0` / `Q4_K_M`) | Speedup vs 2T (`Q4_0` / `Q4_K_M`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`SmolLM2-135M`** | `28.72 tok/s` | `24.15 tok/s` | `55.58 tok/s` | `47.00 tok/s` | **`323.9 tok/s`** (`pp160`) | **`11.3×` / `13.4×`** | **`5.8×` / `6.9×`** |
| **`Qwen2.5-0.5B`** | `9.18 tok/s` | `7.59 tok/s` | `18.13 tok/s` | `15.01 tok/s` | **`168.7 tok/s`** (`pp146`) | **`18.4×` / `22.2×`** | **`9.3×` / `11.2×`** |
| **`Qwen3-0.6B`** | `7.19 tok/s` | `7.52 tok/s` | `14.21 tok/s` | `14.83 tok/s` | **`104.5 tok/s`** (`pp146`) | **`14.5×` / `13.9×`** | **`7.4×` / `7.0×`** |
| **`Qwen2.5-1.5B`** | `2.61 tok/s` | `2.65 tok/s` | `5.17 tok/s` | `5.25 tok/s` | **`71.6 tok/s`** (`pp146`) | **`27.4×` / `27.0×`** | **`13.8×` / `13.6×`** |

### 2. Autoregressive Token Decode Speed (`tg64`, Starting from Short Context)

| Model | `llama.cpp` `Q4_0` (1T) | `llama.cpp` `Q4_K_M` (1T) | `llama.cpp` `Q4_0` (2T) | `llama.cpp` `Q4_K_M` (2T) | **C7x + MMA (`W4A8-g64`)** *(post-146t)* | Speedup vs 1T (`Q4_0` / `Q4_K_M`) | Speedup vs 2T (`Q4_0` / `Q4_K_M`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`SmolLM2-135M`** | `20.59 tok/s` | `17.73 tok/s` | `35.17 tok/s` | `30.85 tok/s` | **`115.7 tok/s`** *(87.2)* | **`5.6×` / `6.5×`** | **`3.3×` / `3.8×`** |
| **`Qwen2.5-0.5B`** | `6.04 tok/s` | `5.20 tok/s` | `10.76 tok/s` | `9.45 tok/s` | **`35.05 tok/s`** *(30.2)* | **`5.8×` / `6.7×`** | **`3.3×` / `3.7×`** |
| **`Qwen3-0.6B`** | `4.61 tok/s` | `4.68 tok/s` | `8.91 tok/s` | `9.13 tok/s` | **`27.24 tok/s`** *(23.3)* | **`5.9×` / `5.8×`** | **`3.1×` / `3.0×`** |
| **`Qwen2.5-1.5B`** | `1.96 tok/s` | `1.94 tok/s` | `3.81 tok/s` | `3.75 tok/s` | **`11.62 tok/s`** *(11.3)* | **`5.9×` / `6.0×`** | **`3.1×` / `3.1×`** |

*Key takeaway:* Even against **both Cortex-A72 CPU cores saturated at 100%** running hand-tuned ARMv8 NEON `llama.cpp`, the bare-metal 1.0 GHz C7x + MMA runtime delivers **`5.8× – 13.8×` faster prompt prefill** (`11.3× – 27.4×` vs 1 core) and **`3.0× – 3.8×` faster token decode** (`5.6× – 6.7×` vs 1 core) — while leaving both A72 cores nearly idle for Linux, networking, and OpenAI SSE serving (`c7x-ai serve`).

---

## Milestone 7: Out-of-Tree `llama.cpp` Dynamic Backend (`libggml-c7x.so`) & `libc7x` Command-Buffer Runtime

Built an out-of-tree, zero-fork dynamic GGML backend plugin (`ggml-c7x/libggml-c7x.so`, loaded at runtime via `GGML_BACKEND_PATH` into unmodified `GGML_BACKEND_DL=ON` `llama.cpp`) over the new reusable `libc7x` runtime and `C7X_OP_EXEC_LIST` flat command-buffer executor. Runs real `Q4_0`, `Q4_1`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_K_M`, `F16`, and `F32` GGUFs directly on the C7x + MMA with a single `EXEC_LIST` RPC per forward pass (`graph splits = 2`, 100% transformer layers + LM head offloaded).

### 1. Live BeagleBone AI-64 Correctness Verification

| Suite | Result | Details |
| --- | --- | --- |
| `libc7x` hardware unit suite (`test_libc7x`) | **PASS** | `ping = 2.5 µs`, coalescing DDR/MSMC allocator, `Q4_0→W4A8(MSMC) cos=0.99746`, `Q8_0→W8A8(DDR) cos=0.999979` |
| `ggml-c7x` integration suite (`test_ggml_c7x_aarch64`) | **PASS** | Weight round-trip (`cos=0.999995`), partial `set_tensor`, tied `GET_ROWS+MUL_MAT`, OOB/dtype gating, 4,200-op auto-flush, 3D-`src1` MMA `Q4_0`/`Q4_K`/`Q8_0` (default & `GGML_C7X_W8A8=1`) |
| `llama.cpp` `test-backend-ops` (`SET_ROWS\|SOFT_MAX\|ROPE\|CPY\|CONT`) | **915 / 915 PASS** | Identical 915/915 on both live silicon and x86_64 hostemu |
| `llama.cpp` `test-backend-ops` (`MUL_MAT` `f16`/`f32`, batched/GQA/permuted) | **113 / 113 PASS** | Live silicon (`585/586` on hostemu; sole skip is 269 MiB > 256 MiB hostemu arena) |
| `c7x-ai bench gemv` stability (post-DDR drain barrier) | **10 / 10 PASS** | Zero intermittent stale-line mismatches across 10 consecutive runs |

### 2. Unmodified `llama.cpp` Head-to-Head (`llama-bench -fa 0 -t 2`, Live Silicon)

Comparing unmodified `llama-bench` on the exact same BeagleBone AI-64 with `-ngl 0 -t 2` (2-core Cortex-A72 NEON CPU) vs `-ngl 99` (`GGML_BACKEND_PATH=libggml-c7x.so`, default `W4A8-g64` and optional `GGML_C7X_W8A8=1`):

| Model (GGUF) | Size | 2T A72 `pp64` / `pp128` | **C7X0 `pp64` / `pp128`** *(W8A8)* | Prefill Speedup (`pp128`) | 2T A72 `tg32` | **C7X0 `tg32`** *(W8A8)* | Decode Speedup (`tg32`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`SmolLM2-135M` `Q4_0`** | `85.9 MiB` | `46.42` / `46.19 t/s` | **`165.08` / `175.07 t/s`** *(159.3 / 166.5)* | **`3.79×`** | `28.62 t/s` | **`47.71 t/s`** *(38.13)* | **`1.67×`** |
| **`Qwen2.5-0.5B` `Q4_K_M`** | `463.0 MiB` | `13.95` / `13.98 t/s` | **`110.21` / `118.33 t/s`** *(110.6 / **119.14**)* | **`8.46×`** *(8.52×)* | `8.60 t/s` | **`14.31 t/s`** *(13.87)* | **`1.66×`** |
| **`Qwen3-0.6B` `Q4_K_M`** (`hd=128`) | `372.7 MiB` | `12.62` / `12.65 t/s` | **`63.62` / `68.10 t/s`** *(61.8 / 65.3)* | **`5.38×`** | `7.75 t/s` | **`13.05 t/s`** *(10.68)* | **`1.68×`** |
| **`Qwen2.5-1.5B` `Q4_0`** (`hd=128`) | `1011.2 MiB` | `3.69` / `4.94 t/s` | **`55.75` / `63.02 t/s`** | **`12.76×`** | `3.56 t/s` | **`8.25 t/s`** | **`2.32×`** |

### 3. Accuracy Tradeoff: Default `W4A8-g64` vs. `GGML_C7X_W8A8=1` (`llama-perplexity`, `c=256`, 3 chunks)

| Model | Cortex-A72 CPU Reference | C7X0 Default (`W4A8-g64`) | C7X0 `GGML_C7X_W8A8=1` | Notes |
| --- | ---: | ---: | ---: | --- |
| **`SmolLM2-135M` `Q4_0`** | `100.98 ± 20.05` | `131.58 ± 25.39` (`+30.3%`) | **`104.20 ± 20.76` (`+3.2%`)** | `W4A8-g64` merges 2 Q4_0 group-32 blocks into symmetric g64; `W8A8` preserves full block scales |
| **`Qwen2.5-0.5B` `Q4_K_M`** | `66.41 ± 12.62` (`66.8 s`) | `75.98 ± 14.63` (`28.4 s`, `+14.4%`) | **`70.36 ± 13.42` (`+5.9%`)** | `W8A8=1` matches `W4A8` prefill speed (`119.14 t/s`) while slashing PPL delta by `2.4×` |

*Key optimizations behind the **58× speedup** from initial `ggml-c7x` bring-up (`2.99 → 175.07 t/s` on `SmolLM2-135M pp128`):* eliminating mid-graph CPU splits via `Q4_1`/`Q5_0`/`Q5_K`/`Q6_K` repacking, L2-SRAM-staged GQA attention (`stage_src0_to_l2` 2D `SE0` F16→F32 + unbranched `k_v=1,2,4,8` `SE0` streaming), 16-wide branchless Cody-Waite RoPE, contiguous-`dst` row-`memcpy` `CONT` (`15×` speedup), and 16-wide vectorized `SET_ROWS` for `llama.cpp`'s flattened transposed V-cache writes.
