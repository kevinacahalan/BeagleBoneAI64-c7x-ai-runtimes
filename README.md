# BeagleBone AI-64 C7x + MMA Transformer Runtime

A custom, bare-metal **TI C71x DSP + Matrix Multiply Accelerator (MMA)** inference runtime for the **BeagleBone AI-64 (TDA4VM / J721E)**, forked from [`BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo`](https://github.com/kevinacahalan/BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo) and stripped down to a lean Linux A72 + C7x DSP stack.

The goal is a hyper-optimized transformer / tiny-LLM runtime (e.g., `stories15M` → `SmolLM2-135M` / `Qwen2.5-0.5B`) that bypasses TI's OpenVX/TIDL framework overhead by driving the C7x MMA, Streaming Engines (`SE0`/`SE1`), 512-bit vector unit, 448 KiB L2 SRAM, 8 MiB on-chip MSMC SRAM, and DRU DMA directly from bare-metal C7x code.

## Highlights (Measured on Live TDA4VM Silicon)

- **Model-agnostic LLM runtime verified across three architectures on live silicon** (zero TIDL/OpenVX overhead; every linear projection runs on the C7x MMA, with all non-matmul ops on the 512-bit FP32 vector unit):
  - **`stories15M` (15M, Llama2, W4A8-g64, 6/6 layers in MSMC SRAM):** **`844.8 tok/s` C7x DSP (`788.9 tok/s` end-to-end)**, **`2,360 tok/s` batched prefill**, **40/40 exact greedy match** (`cosine = 1.0000000`, `max_abs_err = 6.68e-6`). *(W8A8: **`523.1 tok/s` DSP / `495.8 tok/s` e2e**, **`10.5×` faster than Cortex-A72 CPU**, 40/40 match.)*
  - **`SmolLM2-135M-Instruct` (135M, 30L, 3:1 GQA, NeoX RoPE, 49k vocab, W4A8-g64):** **`102.2 tok/s` C7x DSP (`101.6 tok/s` end-to-end)**, **`241.4 tok/s` batched prefill**, `cosine = 1.0000000` (`max_abs_err = 5.53e-5`).
  - **`Qwen2.5-0.5B-Instruct` (494M, 24L, 7:1 GQA, QKV biases, `eps=1e-6`, `theta=1M`, 151,936 vocab, W4A8-g64):** **`33.6 tok/s` C7x DSP (`33.0 tok/s` end-to-end)**, **`100.0 tok/s` batched prefill**, `cosine = 1.0000000` (`max_abs_err = 2.19e-5`).
- **Universal HuggingFace (`safetensors` + `config.json` + `tokenizer.json`) converter:** `tools/convert_hf.py` packs any standard Llama/SmolLM/Qwen2/Mistral-family model directly into a single self-contained `.c7m` file with an embedded `C7T1` byte-level BPE tokenizer — no external tokenizer file or per-model C code required.
- **Bare-metal C7x `DRU0` direct-TR DMA unlocked (zero TI-SCI, zero RTOS):** Type-9 atomic 512-bit TR submission to `CHATOMIC[0]` (`0x6D080000`) achieves **`10.92 GB/s` DDR→L2**, **`10.89 GB/s` DDR→MSMC**, and **`55.93 GB/s` MSMC→L2** (byte-exact verified), and powers zero-cache-flush L2-SRAM ping-pong double-buffering (`10.52 GB/s` raw = **`21.0 GB/s` eq-INT8** on `Qwen2.5-0.5B`'s 152k-vocab classifier, `+13.5%` faster than direct `SE1`!).
- **Expanded 512 MiB DDR LLM arena (`0xC0000000`):** Device-tree overlay reserves both `0xAC000000` (64 MiB) and `0xC0000000` (512 MiB), auto-detected by `c7x-ai llama`, fitting models up to ~500M+ parameters in W4A8.
- **Bit-exact x86_64 C7000 Host Emulation + Live Hardware loop:** every MMA, vector, and transformer kernel compiles identically for TI's x86_64 host emulator (`g++ -D__C7X_HOSTEM__`) and TI C7000 CGT (`cl7x -mv7100 -O3`).
- **Settled the MMA C-bank depth mystery on silicon:** TI's host emulator wraps at 16 rows, whereas live TDA4VM silicon verified **63 int32 rows per C bank** (16 KiB per bank, 32 KiB total). Raising `block_m` from `16` to `60` lifted MSMC GEMM **2.6×** to **696 GOPS** (`316 GOPS` from DDR) and enabled 60-token batched MMA prefill.

See **[`RESULTS.md`](RESULTS.md)** for full hardware tables, **[`NOTES.md`](NOTES.md)** for emulator-vs-silicon mismatches and compiler quirks, and **[`MEMORY.md`](MEMORY.md)** for the full address & cache map.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`dsp/`](dsp/) | Bare-metal C7x firmware (`c7x-ai.elf`): MMU/cache setup, bare-metal Type-9 `DRU0` DMA, MMA GEMM/W8A8/W4A8-g64 + L2 ping-pong kernels, 512-bit FP32 vector kernels, universal transformer & batched prefill engine, plus x86_64 host-emulation tests (`dsp/host_test/`) |
| [`linux-app/`](linux-app/) | Lean C host CLI (`c7x-ai`): remoteproc load/stop with clean `RP_MBOX_SHUTDOWN`, trace reader, hardware benchmark suite, and `llama verify`/`run` transformer runner with embedded `C7T1` & legacy `llama2.c` BPE tokenizers |
| [`tools/`](tools/) | Universal HuggingFace (`safetensors` + `tokenizer.json`) converter (`convert_hf.py`), Python/NumPy FP32/W8A8/W4A8 `llama2.c` reference (`llama_ref.py`), `.c7m` 64×64 tile-major packer, and golden activation/logit trace exporter |
| [`beagle-linux/`](beagle-linux/) | Containerized build script (`scripts/build.sh`), board installer (`scripts/install-beagle.sh`), and DDR tensor arena overlay (`overlays/k3-j721e-beagleboneai64-c7x-ai.dtso`: 64 MiB @ `0xAC000000` + 512 MiB @ `0xC0000000`) |
| [`containers/`](containers/) | Reproducible Podman/Docker images for TI C7000 CGT 1.2.0.STS + `g++` host emulation (`dsp-cgt/`) and Debian 13 aarch64 cross-compile (`linux/`) |
| [`yocto/`](yocto/) | Minimal Poky Linux 6.18 `kas` + `meta-ai64` layer packaging `c7x-ai`, `c7x-ai.elf`, and the DDR arena overlay |

---

## Quick Start

### 1. Build & Run Host Emulation Tests (PC)

Requires `podman` or `docker` (no local toolchains needed):

```bash
./beagle-linux/scripts/build.sh all
```

Or run individual targets:
- `./beagle-linux/scripts/build.sh host-test` — compile & run x86_64 C7000 MMA/vector host emulation tests
- `./beagle-linux/scripts/build.sh dsp` — build `dsp/build/c7x-ai.elf`
- `./beagle-linux/scripts/build.sh linux` — cross-compile `linux-app/c7x-ai`
- `./beagle-linux/scripts/build.sh overlay` — compile `beagle-linux/overlays/k3-j721e-beagleboneai64-c7x-ai.dtbo`
- `./beagle-linux/scripts/build.sh bundle` — package everything into `out/c7x-ai-beagle/`

### 2. Deploy & Benchmark on the BeagleBone AI-64

Copy the binary and firmware to the board (ensure `0xAC000000` is reserved via the overlay and other coprocessors aren't writing to `0xAC000000` or C7x L2):

```bash
scp linux-app/c7x-ai USER@BOARD:/usr/bin/c7x-ai
scp dsp/build/c7x-ai.elf USER@BOARD:/lib/firmware/c7x-ai.elf
ssh USER@BOARD 'c7x-ai load c7x-ai.elf && c7x-ai trace && c7x-ai bench all'
```

CLI subcommands:
- `c7x-ai status` — remoteproc state, C7x MMU/cache config, DRU0 status
- `c7x-ai load [fw]` — cleanly stop C7x (`RP_MBOX_SHUTDOWN`), load firmware, start
- `c7x-ai stop` — clean C7x shutdown + remoteproc stop
- `c7x-ai trace [-f]` — dump or follow C7x remoteproc trace log
- `c7x-ai bench [all|ping|cbank|mem|dru|gemm|gemv]` — run hardware benchmarks with bit-exact A72 verification
- `c7x-ai llama verify <model.c7m> [tokenizer.bin] <golden.bin> [steps]` — verify vector kernels, single layer, logits, batched prefill, and greedy tokens on silicon + A72 CPU baseline
- `c7x-ai llama run <model.c7m> [tokenizer.bin] [-p prompt] [-n steps] [-t temp] [-P topp] [-s seed]` — run live text generation on C7x + MMA with per-layer cycle profiling

### 3. Convert & Run Any HuggingFace LLM (`SmolLM2-135M`, `Qwen2.5-0.5B`, etc.)

```bash
# Convert any Llama/SmolLM/Qwen2/Mistral safetensors model into a self-contained .c7m
python3 tools/convert_hf.py --hf-dir models/qwen2.5-0.5b \
    --out models/qwen2.5-0.5b-w4a8.c7m --quant w4a8 --seq-len 512

scp models/qwen2.5-0.5b-w4a8.c7m USER@BOARD:/tmp/
ssh USER@BOARD 'c7x-ai llama run /tmp/qwen2.5-0.5b-w4a8.c7m -p "<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
" -n 64'
```

*(For legacy `llama2.c` `.bin` models like `stories15M`, use `tools/llama_ref.py --quant w4a8 --pack ...`.)*

---

## Roadmap

1. **Done — Lean C7x fork & microbenchmark harness:** host-emulated & silicon-verified MMA `M×K×N` GEMM (`ROW`/`COL`), C-bank 63-row probe (`696 GOPS` MSMC), L2/MSMC/DDR SE0 bandwidth, and W4A8 decode GEMV.
2. **Done — End-to-end `stories15M` W8A8 transformer on C7x + MMA (`523 tok/s` DSP / `496 tok/s` e2e):** 1D tile-major MMA weights (`9.59 GB/s`), 3-layer MSMC SRAM staging, `vexpf_f16` SwiGLU/Softmax, 2D SE0/SE1-prefetched attention, and 40/40 greedy token match (`10.5×` faster than Cortex-A72).
3. **Done — Model-agnostic W4A8 MMA engine, batched prefill, HuggingFace converter, & bare-metal DRU0 async double-buffering:** MMA-accelerated W4A8 (`845 tok/s` `stories15M`), 512 MiB DDR arena, embedded `C7T1` BPE tokenizer, live silicon verification of `SmolLM2-135M-Instruct` (`102 tok/s`, `241 tok/s` prefill) and `Qwen2.5-0.5B-Instruct` (`33.6 tok/s`, `100 tok/s` prefill), and bare-metal Type-9 `DRU0` direct-TR DMA (`10.92 GB/s` DDR→L2, `55.93 GB/s` MSMC→L2) with zero-cache-flush L2 ping-pong.

---

## License

MIT — see [`LICENSE`](LICENSE).
