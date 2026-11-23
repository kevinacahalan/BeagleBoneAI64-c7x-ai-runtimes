# BeagleBone AI-64 C7x + MMA Transformer Runtime

A custom, bare-metal **TI C71x DSP + Matrix Multiply Accelerator (MMA)** local LLM inference engine and OpenAI-compatible appliance for the **BeagleBone AI-64 (TDA4VM / J721E)**, forked from [`BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo`](https://github.com/kevinacahalan/BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo).

By driving the C7x MMA, Streaming Engines (`SE0`/`SE1`), 512-bit vector unit, 448 KiB L2 SRAM, 7.625 MiB on-chip MSMC SRAM, and bare-metal Type-9 `DRU0` DMA directly (zero OpenVX / TIDL / RTOS overhead), this runtime runs modern 15M–1.7B HuggingFace transformers (`stories15M`, `SmolLM2-135M`, `Qwen2.5-0.5B`, `Qwen3-0.6B`, `Qwen2.5-1.5B-Instruct`, `Qwen3-1.7B`) up to **`13.8×` faster on prompt prefill** and **`3.8×` faster on token decode** than dual-core Cortex-A72 `llama.cpp` — while leaving both Arm Cortex-A72 cores free for Linux and serving.

---

## Highlights (Measured on Live TDA4VM Silicon)

### Head-to-Head: C7x + MMA (`W4A8-g64`) vs. `llama.cpp` on BeagleBone AI-64 Cortex-A72

| Model | C7x + MMA Prefill | `llama.cpp` 2T A72 Prefill (`Q4_0` / `Q4_K_M`) | **Prefill Speedup** (vs 2T / 1T) | C7x + MMA Decode | `llama.cpp` 2T A72 Decode (`Q4_0` / `Q4_K_M`) | **Decode Speedup** (vs 2T / 1T) |
| --- | --- | --- | --- | --- | --- | --- |
| **`SmolLM2-135M`** | **323.9 tok/s** *(pp160)* | `55.6` / `47.0 tok/s` | **`5.8×` – `6.9×`** *(11.3–13.4×)* | **115.7 tok/s** | `35.2` / `30.9 tok/s` | **`3.3×` – `3.8×`** *(5.6–6.5×)* |
| **`Qwen2.5-0.5B`** | **168.7 tok/s** *(pp146)* | `18.1` / `15.0 tok/s` | **`9.3×` – `11.2×`** *(18.4–22.2×)* | **35.1 tok/s** | `10.8` / `9.5 tok/s` | **`3.3×` – `3.7×`** *(5.8–6.7×)* |
| **`Qwen3-0.6B`** | **104.5 tok/s** *(pp146)* | `14.2` / `14.8 tok/s` | **`7.4×` – `7.0×`** *(14.5–13.9×)* | **27.2 tok/s** | `8.9` / `9.1 tok/s` | **`3.1×` – `3.0×`** *(5.9–5.8×)* |
| **`Qwen2.5-1.5B`** | **71.6 tok/s** *(pp146)* | `5.2` / `5.3 tok/s` | **`13.8×` – `13.6×`** *(27.4–27.0×)* | **11.6 tok/s** | `3.8` / `3.8 tok/s` | **`3.1×` – `3.1×`** *(5.9–6.0×)* |

Additional verified models on live silicon:
- **`stories15M` (15M, Llama2, W4A8-g64, 6/6 layers in MSMC SRAM):** **`871.7 tok/s`** C7x DSP (`808.6 tok/s` e2e), **`3,207.9 tok/s`** batched prefill, **40/40 exact greedy match** (`cosine = 1.0000000`).
- **`Qwen3-1.7B` (1.72B, QK-RMSNorm, `hd=128`, INT8 KV, `seq=2048`):** **`10.34 tok/s`** C7x DSP (`10.27 tok/s` e2e), **10/10 exact greedy match** (`cosine = 0.99931`).

### Core Architecture Features

- **7.625 MiB On-Chip MSMC SRAM (`0x70040000..0x707DFFFF`):** L3 4 KiB edge page tables map nearly the full 8 MiB MSMC while leaving hardware-unmapped guard bands around ATF and DMSC (pins 6/6 layers of `stories15M` or 4/30 layers of `SmolLM2-135M`).
- **1.5 GiB High-DDR LLM Arena (`0x880000000..0x8DFFFFFFF`):** Device-tree overlay reserves 1.5 GiB out of 4 GiB LPDDR4 (leaving ~2.1 GiB RAM including all low 32-bit DMA RAM for Linux), fitting 1.5B–1.7B models at `seq_len = 2048` (or two models simultaneously for speculative decoding).
- **Bare-Metal `DRU0` Direct-TR DMA + B-Stationary Batched Prefill:** Type-9 atomic 512-bit TRs (`10.92 GB/s` DDR→L2, `55.93 GB/s` MSMC→L2, zero TI-SCI/RTOS) power zero-cache-flush L2-SRAM ping-pong double-buffering (`10.52 GB/s` raw = `21.0 GB/s` eq-INT8) and B-stationary `block_m <= 60` batched MMA prefill.
- **Quantized INT8 KV Cache (`2048 – 4096` Context):** Per-head symmetric INT8 KV + 2D `SE0`/`SE1` sign-extending attention cuts KV memory & DDR traffic `3.76–3.88×` while preserving **10/10 exact greedy parity**; verified live up to `3,803` prompt tokens on `Qwen2.5-0.5B`.
- **Local LLM Appliance UX:** One-command HuggingFace Hub download & conversion (`tools/convert_hf.py --repo`), offline model inspector (`c7x-ai llama info`), interactive multi-turn terminal REPL with persistent KV reuse (`c7x-ai llama chat`), and a zero-dependency C HTTP/1.1 + SSE OpenAI-compatible API server (`c7x-ai serve`).
- **Bit-Exact x86_64 C7000 Host Emulation + CI:** Every MMA, vector, and transformer kernel compiles identically for TI's x86_64 host emulator (`g++ -D__C7X_HOSTEM__`) and TI C7000 CGT (`cl7x -mv7100 -O3`), verified automatically in GitHub Actions (`.github/workflows/ci.yml`).

See **[`RESULTS.md`](RESULTS.md)** for complete hardware tables, **[`NOTES.md`](NOTES.md)** for emulator-vs-silicon mismatches and compiler quirks, and **[`MEMORY.md`](MEMORY.md)** for the full address & cache map.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`dsp/`](dsp/) | Bare-metal C7x firmware (`c7x-ai.elf`): MMU/cache + 7.625 MiB MSMC L3 tables, bare-metal Type-9 `DRU0` DMA, MMA GEMM/W8A8/W4A8-g64 + B-stationary prefill + L2 ping-pong kernels, 512-bit FP32 vector & INT8 KV attention kernels, command-buffer executor (`C7X_OP_EXEC_LIST`), universal Llama/Qwen2/Qwen3 transformer engine, plus x86_64 host-emulation tests (`dsp/host_test/`) |
| [`libc7x/`](libc7x/) | Shared host C runtime library (`libc7x.a` for aarch64 hardware & `libc7x_hostemu.a` for x86_64 TI C7000 Host Emulation): coalescing DDR (1.5 GiB) + MSMC (7.625 MiB) arena allocator, `O_SYNC`-safe memory transfers, multi-process `flock` + ABI handshake, tile-major weight packers/unpackers, and single-RPC command-buffer builder |
| [`ggml-c7x/`](ggml-c7x/) | Out-of-tree dynamic `llama.cpp` / `ggml` backend plugin (`libggml-c7x.so`) built on `libc7x`: loadable into unmodified `GGML_BACKEND_DL=ON` builds of `llama.cpp` / `whisper.cpp` via `GGML_BACKEND_PATH`, with eager `WEIGHTS` tile-major repacking (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_K`/`Q6_K`/`Q8_0`/`Q4_K`), optional `GGML_C7X_W8A8=1` high-accuracy mode, MSMC hot-weight pinning, and Vision/Audio ops (`NORM`, `IM2COL`, `POOL_2D`, `CONV_2D`) |
| [`ort-c7x-ep/`](ort-c7x-ep/) | ONNX Runtime Execution Provider (`libort-c7x-ep.so`) over `libc7x` command buffers: zero-dependency standard `.onnx` binary protobuf loader & compiler supporting asymmetric INT8/UINT8 `QLinearConv` (standard + depthwise), `QLinearMatMul`, `Conv`, `MatMul`/`Gemm`, `LayerNormalization`, pooling, and activations in a single DSP RPC |
| [`linux-app/`](linux-app/) | Lean zero-dependency C host CLI (`c7x-ai`): remoteproc load/stop with `RP_MBOX_SHUTDOWN`, trace reader, hardware benchmark suite, `llama info`/`verify`/`run`/`chat`/`spec`, and `serve` (OpenAI HTTP/1.1 + SSE server with prefix KV caching) |
| [`tools/`](tools/) | Universal HuggingFace Hub downloader & converter (`convert_hf.py --repo <org/model>`, stdlib HTTP + NumPy only), embedded `C7T1` BPE tokenizer compiler, Python/NumPy reference (`llama_ref.py`), and golden trace exporter |
| [`beagle-linux/`](beagle-linux/) | Containerized build script (`scripts/build.sh`), board installer (`scripts/install-beagle.sh`), and DDR tensor arena overlay (`overlays/k3-j721e-beagleboneai64-c7x-ai.dtso`: 64 MiB @ `0xAC000000` + 1.5 GiB @ `0x880000000`) |
| [`containers/`](containers/) | Reproducible Podman/Docker images for TI C7000 CGT 1.2.0.STS + `g++` host emulation (`dsp-cgt/`) and Debian 13 aarch64 cross-compile (`linux/`) |
| [`.github/workflows/`](.github/workflows/) | GitHub Actions CI running containerized `./beagle-linux/scripts/build.sh all` and Python converter/reference self-tests |
| [`yocto/`](yocto/) | Minimal Poky Linux 6.18 `kas` + `meta-ai64` layer packaging `c7x-ai`, `c7x-ai.elf`, and the DDR arena overlay |

---

## Quick Start

### 1. Build & Run Host Emulation Tests (PC)

Requires `podman` or `docker` (no local toolchains needed):

```bash
./beagle-linux/scripts/build.sh all
```

Or run individual targets: `host-test`, `dsp`, `libc7x`, `ggml`, `ort`, `linux`, `overlay`, `bundle`.

### Using the `llama.cpp` Dynamic Backend (`libggml-c7x.so`)

Build `llama.cpp` with dynamic backends enabled (`-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON`), then point `GGML_BACKEND_PATH` at `libggml-c7x.so` (built for aarch64 BeagleBone AI-64 in `ggml-c7x/build-aarch64/` or x86_64 TI C7000 Host Emulation in `ggml-c7x/build-hostemu/`). Verified up to **`12.76×` faster prefill** (`63.02` vs `4.94 t/s` on `Qwen2.5-1.5B Q4_0`) and **`2.32×` faster decode** vs dual-core Cortex-A72:

```bash
# On BeagleBone AI-64 (after `c7x-ai load c7x-ai.elf`):
GGML_BACKEND_PATH=/path/to/libggml-c7x.so llama-bench --list-devices
GGML_BACKEND_PATH=/path/to/libggml-c7x.so llama-completion -m model-Q4_0.gguf -ngl 99 -fa off -p "Hello"
# Optional high-accuracy W8A8 packing mode (reduces Q4_0/Q4_K PPL delta by 2.4-10x):
GGML_C7X_W8A8=1 GGML_BACKEND_PATH=/path/to/libggml-c7x.so llama-completion -m model-Q4_K_M.gguf -ngl 99 -fa off -p "Hello"
```

### Using the ONNX Runtime C7x Execution Provider (`ort-c7x-ep`)

Build and run real non-LLM `.onnx` vision (`mobilenet_qlinear.onnx`) and audio (`audio_enc_block.onnx`) models on hostemu or live BeagleBone AI-64 silicon:

```bash
./beagle-linux/scripts/build.sh ort
# On BeagleBone AI-64:
/tmp/ort-c7x/test_ort_c7x_ep /tmp/ort-c7x/models 20
```

### 2. Deploy to the BeagleBone AI-64

```bash
./beagle-linux/scripts/build.sh bundle
scp -r out/c7x-ai-beagle USER@BOARD:
ssh -t USER@BOARD 'sudo ./c7x-ai-beagle/install-beagle.sh && sudo reboot'
# After reboot:
ssh USER@BOARD 'c7x-ai load c7x-ai.elf && c7x-ai status && c7x-ai bench all'
```

### 3. Convert Any HuggingFace Model in One Command

Zero mandatory Python dependencies beyond `numpy` (uses stdlib HTTP to pull `config.json`, `tokenizer.json`, and `safetensors` shards directly from HuggingFace Hub):

```bash
python3 tools/convert_hf.py --repo Qwen/Qwen2.5-1.5B-Instruct \
    --out models/qwen2.5-1.5b-w4a8.c7m --quant w4a8 --seq-len 2048
scp models/qwen2.5-1.5b-w4a8.c7m USER@BOARD:/tmp/
```

Supports Llama, SmolLM2, Qwen2/2.5, Qwen3 (with QK-RMSNorm), and Mistral checkpoints. *(For legacy `llama2.c` `.bin` models like `stories15M`, use `tools/llama_ref.py --quant w4a8 --pack ...`.)*

### 4. Inspect, Chat, or Serve an OpenAI-Compatible API on the Board

```bash
# Inspect model metadata, KV sizes, and MSMC SRAM fit (offline, instant)
c7x-ai llama info /tmp/qwen2.5-1.5b-w4a8.c7m

# Interactive multi-turn terminal chat (auto-detects ChatML / Qwen3 / Llama-3, reuses KV cache)
c7x-ai llama chat /tmp/qwen2.5-1.5b-w4a8.c7m -t 0.7 -P 0.9

# Start zero-dependency OpenAI-compatible HTTP/1.1 + SSE server on port 8080
c7x-ai serve /tmp/qwen2.5-1.5b-w4a8.c7m --host 0.0.0.0 --port 8080
```

Query the board from any client on your LAN with `curl` (supports `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `stream: false` and `stream: true`, plus automatic prefix KV caching across turns):

```bash
curl http://BOARD_IP:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b-w4a8.c7m",
    "messages": [{"role": "user", "content": "Explain the C7x MMA in two sentences."}],
    "temperature": 0,
    "max_tokens": 96,
    "stream": true
  }'
```

All CLI subcommands:
- `c7x-ai status | load [fw] | stop | trace [-f]` — DSP lifecycle & diagnostics
- `c7x-ai bench [all|ping|cbank|mem|dru|gemm|gemv]` — hardware microbenchmarks with bit-exact A72 verification
- `c7x-ai llama info <model.c7m>` — offline header, quant, tokenizer, KV footprint & MSMC staging summary
- `c7x-ai llama verify <model.c7m> [tok.bin] <golden.bin> [steps]` — live silicon kernel, layer, logit, prefill & greedy parity checks
- `c7x-ai llama run <model.c7m> [tok.bin] [-p prompt] [-n steps] [-t temp] [--kv-int8|--kv-fp32]` — one-shot generation with cycle profiling
- `c7x-ai llama chat <model.c7m> [-t temp] [-P topp] [--rep-penalty f] [--system msg] [--think|--no-think]` — persistent-KV multi-turn REPL (`/reset`, `/stats`, `/quit`)
- `c7x-ai llama spec <target.c7m> <draft.c7m> [-k draft_len]` — lossless speculative decoding with KV rollback
- `c7x-ai serve <model.c7m> [--host 0.0.0.0] [--port 8080]` — OpenAI-compatible HTTP/1.1 + SSE API server

---

## Roadmap

1. **Done — Lean C7x fork & microbenchmark harness:** Host-emulated & silicon-verified MMA `M×K×N` GEMM (`696 GOPS` MSMC), C-bank 63-row probe, L2/MSMC/DDR SE0 bandwidth, and W4A8 decode GEMV.
2. **Done — End-to-end `stories15M` W8A8 transformer (`523 tok/s` DSP / `496 tok/s` e2e):** 1D tile-major MMA (`9.59 GB/s`), MSMC SRAM staging, `vexpf_f16` SwiGLU/Softmax, 2D SE0/SE1 attention, 40/40 greedy match.
3. **Done — Model-agnostic W4A8 MMA engine, HuggingFace converter, & bare-metal `DRU0` DMA:** Embedded `C7T1` BPE tokenizer, `DRU0` Type-9 direct-TR async L2 ping-pong (`10.92 GB/s` DDR→L2, `55.93 GB/s` MSMC→L2), live `SmolLM2-135M` & `Qwen2.5-0.5B`.
4. **Done — 7.625 MiB MSMC, B-stationary prefill, INT8 KV (`2K–4K`), Qwen3 + 1.5B–1.7B models, & speculative decoding:** `1.5 GiB` high-DDR arena, `3,208 tok/s` `stories15M` / `395 tok/s` `SmolLM2-135M` / `210 tok/s` `Qwen2.5-0.5B` prefill, 4K-context INT8 KV, live `Qwen3-0.6B` (`27.2 tok/s`), `Qwen2.5-1.5B-Instruct` (`11.6 tok/s`), `Qwen3-1.7B` (`10.3 tok/s`), and documented speculative decoding analysis.
5. **Done — Local LLM Appliance UX:** `convert_hf.py --repo` stdlib Hub downloader, `c7x-ai llama info`, persistent-KV multi-turn `c7x-ai llama chat`, and live-verified OpenAI HTTP/1.1 + SSE `c7x-ai serve`.
6. **Done — Head-to-head `llama.cpp` benchmarks & GitHub Actions CI:** Verified **`5.8× – 13.8×` prefill** and **`3.0× – 3.8×` decode** speedup over dual-core Cortex-A72 `llama.cpp` (`11.3× – 27.4×` prefill / `5.6× – 6.7×` decode vs 1-core), plus automated containerized CI (`.github/workflows/ci.yml`).
7. **Done — Reusable `libc7x` runtime + out-of-tree `llama.cpp` dynamic backend (`libggml-c7x.so`):** Zero-fork dynamic plugin over `C7X_OP_EXEC_LIST` delivering up to **`12.76×` faster prefill** and **`2.32×` faster decode** vs dual-core Cortex-A72 on unmodified `llama-bench` (`135M–1.5B` GGUFs), `915/915` `test-backend-ops` on live silicon, plus optional `GGML_C7X_W8A8=1` high-accuracy mode.
8. **Done — Non-LLM expansion (Vision/Audio `ggml-c7x` ops & `ort-c7x-ep` ONNX Runtime Execution Provider):** Added `NORM`, `IM2COL`, `POOL_2D`, `CONV_2D` (`710/710` + `1,574/1,574` live silicon `test-backend-ops`, `2,030/2,030` hostemu), plus asymmetric INT8/UINT8 `QLinearConv`, `QLinearMatMul` (`19.6 GOPS`, `4.75×` vs A72), `DequantizeLinear`, and standalone `.onnx` execution (`mobilenet_qlinear.onnx` at `82 FPS`, `audio_enc_block.onnx` at `883 FPS`).

---

## License

MIT — see [`LICENSE`](LICENSE).
