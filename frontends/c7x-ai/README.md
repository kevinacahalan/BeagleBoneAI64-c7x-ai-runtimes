# Path 1: `c7x-ai` Native CLI & Fastest `.c7m` LLM Engine (`frontends/c7x-ai/`)

Zero-dependency C userspace CLI for the BeagleBone AI-64 serving two roles:

1. **Shared Board & DSP Lifecycle Control (used by all 3 paths):**
   - `c7x-ai load [c7x-ai.elf]` / `stop` / `status` / `trace [-f]` — clean remoteproc boot/shutdown (`RP_MBOX_SHUTDOWN` handshake) and live trace buffer inspection.
   - `c7x-ai bench [all|ping|cbank|mem|dru|gemm|gemv]` — hardware microbenchmarks with bit-exact Cortex-A72 verification.
2. **Fastest Monolithic `.c7m` LLM Runtime:**
   - Drives `runtime/dsp/c7x/transformer.c` directly for peak decode speed (`115.7 tok/s` `SmolLM2-135M`, `35.1 tok/s` `Qwen2.5-0.5B`, `27.2 tok/s` `Qwen3-0.6B`, `11.6 tok/s` `Qwen2.5-1.5B`, `10.3 tok/s` `Qwen3-1.7B`), 2K–4K **INT8 KV cache**, and MSMC hot-layer pinning (`871.7 tok/s` on `stories15M`).

## Build

```bash
./platform/beagle-linux/scripts/build.sh linux
# Output: frontends/c7x-ai/c7x-ai (aarch64 binary)
```

## Usage on BeagleBone AI-64

```bash
# Convert a HuggingFace model to native W4A8-g64 .c7m on your PC
python3 tools/convert_hf.py --repo Qwen/Qwen2.5-1.5B-Instruct \
    --out models/qwen2.5-1.5b-w4a8.c7m --quant w4a8 --seq-len 2048

# Inspect metadata, KV footprint & MSMC fit (offline)
c7x-ai llama info /tmp/qwen2.5-1.5b-w4a8.c7m

# Interactive multi-turn terminal chat (auto-detects ChatML/Qwen3/Llama-3, reuses KV cache)
c7x-ai llama chat /tmp/qwen2.5-1.5b-w4a8.c7m -t 0.7 -P 0.9

# Zero-dependency OpenAI-compatible HTTP/1.1 + SSE API server (with prefix KV caching)
c7x-ai serve /tmp/qwen2.5-1.5b-w4a8.c7m --host 0.0.0.0 --port 8080
```

Query the server from any machine on your LAN:

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

### All Subcommands
- `c7x-ai status | load [fw] | stop | trace [-f]`
- `c7x-ai bench [all|ping|cbank|mem|dru|gemm|gemv]`
- `c7x-ai llama info <model.c7m>`
- `c7x-ai llama verify <model.c7m> [tok.bin] <golden.bin> [steps]`
- `c7x-ai llama run <model.c7m> [tok.bin] [-p prompt] [-n steps] [-t temp] [--kv-int8|--kv-fp32]`
- `c7x-ai llama chat <model.c7m> [-t temp] [-P topp] [--rep-penalty f] [--system msg] [--think|--no-think]`
- `c7x-ai llama spec <target.c7m> <draft.c7m> [-k draft_len]`
- `c7x-ai serve <model.c7m> [--host 0.0.0.0] [--port 8080]`
