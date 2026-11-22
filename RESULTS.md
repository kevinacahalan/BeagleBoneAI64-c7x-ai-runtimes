# Measured Hardware Results (BeagleBone AI-64 / TDA4VM)

All numbers below were measured directly on a live BeagleBone AI-64 (TI TDA4VM / J721E, C71x @ 1.0 GHz, MMA v1, L1D = 32 KiB cache, L2 = 64 KiB cache + 448 KiB SRAM, 4 MiB MSMC SRAM mapped, 64 MiB DDR tensor arena) with **bit-exact verification against the Cortex-A72 CPU reference** (`0` mismatches on every passing row).

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
