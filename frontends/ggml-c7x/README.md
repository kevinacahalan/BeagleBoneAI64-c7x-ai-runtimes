# Path 2: `ggml-c7x` — Out-of-Tree Dynamic `llama.cpp` Backend (`frontends/ggml-c7x/`)

Dynamic `ggml` backend plugin (`libggml-c7x.so`) built on [`runtime/libc7x`](../../runtime/libc7x/). Loads at runtime via `GGML_BACKEND_PATH` into **unmodified** `GGML_BACKEND_DL=ON` builds of `llama.cpp` (and `whisper.cpp`), offloading full transformer + vision/audio graphs onto the BeagleBone AI-64 C7x + MMA in a single `C7X_OP_EXEC_LIST` RPC per pass.

## Live Silicon Highlights
- **Prefill (`pp128`):** Up to **`12.76×` faster** than 2-core Cortex-A72 (`SmolLM2-135M Q4_0 -fa 1`: **`185.40 t/s`** = `4.01×` A72; `Qwen2.5-0.5B Q4_K_M`: `118.33 t/s` = `8.46×` A72; `Qwen2.5-1.5B Q4_0`: `63.02 t/s` = `12.76×` A72).
- **Decode (`tg32`):** **`1.66× – 2.32×` faster** than 2-core Cortex-A72 (`SmolLM2-135M -fa 1`: **`55.34 t/s`**, `+16.0%` over `-fa 0` via native FlashAttention-2 + peephole fusion + lazy L1D cache invalidation; `-fa 1 -ctk q8_0 -ctv q8_0`: **`49.00 t/s`** at 2× smaller KV RAM; `Qwen2.5-1.5B`: `8.25 t/s`).
- **`test-backend-ops` on live silicon:** **`3,648/3,648` (`FLASH_ATTN_EXT`)**, `915/915` (`SET_ROWS|SOFT_MAX|ROPE|CPY|CONT`), `113/113` (`MUL_MAT`), `710/710` (`NORM|IM2COL|POOL_2D`), `1,574/1,574` (`CONV_2D cwhn=0`; `2,030/2,030` on hostemu).

## Build

Requires a clean checkout of upstream `llama.cpp` (defaults to `models/llama.cpp`, or pass `LLAMA_DIR=/path/to/llama.cpp`):

```bash
./platform/beagle-linux/scripts/build.sh ggml
# Installed onto the board by install-beagle.sh at:
#   /usr/lib/c7x-ai/libggml-c7x.so
```

## Run with Unmodified `llama.cpp`

Build `llama.cpp` with `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON` (and `-DGGML_OPENMP=OFF` on Cortex-A72), ensure `c7x-ai load c7x-ai.elf` has run, then:

```bash
GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-bench --list-devices
GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-bench -m model-Q4_0.gguf -ngl 99 -fa 1 -p 128 -n 32
GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-completion -m model-Q4_0.gguf -ngl 99 -p "Hello"
# Optional 2x smaller Q8_0 KV cache (supported natively by C7x FLASH_ATTN_EXT):
GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-completion -m model-Q4_0.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p "Hello"
```

## Environment Variables

| Variable | Effect |
| --- | --- |
| `GGML_C7X_W8A8=1` | Unpack 4-bit GGUF weights (`Q4_0`/`Q4_K`/etc.) to tile-major **W8A8** instead of `W4A8-g64`. Cuts perplexity delta by `2.4×–10×` (`+3%–6%` vs CPU) at nearly identical prefill speed, using ~2× weight RAM and ~15–20% slower decode. |
| `GGML_C7X_PROFILE=1` | Print per-op DSP cycle breakdown on stderr (`=2` for verbose per-node breakdown). |

## Supported Quant Formats & Operators
- **Repacked MMA weights:** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_K` (+ `F16`/`F32`/`Q8_0` general matmul & KV cache).
- **LLM ops:** `MUL_MAT`, `FLASH_ATTN_EXT` (`F16`/`F32`/`Q8_0` K&V, GQA, active-mask skipping, ALiBi, softcap, sinks), `RMS_NORM` (with peephole `RMS_NORM+MUL` fusion), `ROPE` (cached `freq_tab`/sin-cos, normal/NeoX/YaRN), `GLU` (with peephole `SILU+MUL` → SwiGLU fusion), `SOFT_MAX`, `ADD`, `MUL`, `SCALE`, `SILU`, `GELU`, `RELU`, `SIGMOID`, `TANH`, `GET_ROWS`, `SET_ROWS`, `CPY`, `CONT`, `VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`.
- **Vision & Audio ops (`whisper.cpp` / CNNs):** `NORM` (LayerNorm), `IM2COL` (1D/2D F32/F16), `POOL_2D` (Max/Avg), `CONV_2D` (F32/F16).
