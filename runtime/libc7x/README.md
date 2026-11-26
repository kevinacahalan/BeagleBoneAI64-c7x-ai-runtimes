# `libc7x` — Reusable Host Runtime Library (`runtime/libc7x/`)

Shared C11 host runtime powering both [`ggml-c7x`](../../frontends/ggml-c7x/) and [`ort-c7x-ep`](../../frontends/ort-c7x-ep/). Abstracts hardware `/dev/mem` + L2 control-block RPCs and TI's x86_64 C7000 Host Emulation behind one API (`include/c7x_rt.h`).

## What It Provides

- **Coalescing Memory Allocators:** Manages both the 1.5 GiB high-DDR tensor arena (`C7X_MEM_DDR`, `0x880000000`) and 7.625 MiB on-chip MSMC SRAM (`C7X_MEM_MSMC`, `0x70040000`) with 64-byte alignment and automatic free-block coalescing.
- **Safe Host↔DSP Transfers:** `c7x_write()` / `c7x_read()` / `c7x_memset()` using aligned 64-bit chunks safe for uncached `O_SYNC` `/dev/mem` mappings (avoiding Arm64 alignment faults on unaligned `memcpy`).
- **Single-RPC Command Buffer Builder:** `c7x_cmdbuf_create()`, `c7x_cmdbuf_push_desc()`, `c7x_cmdbuf_push()`, and `c7x_cmdbuf_submit()` serialize hundreds or thousands of ops into one `C7X_OP_EXEC_LIST` DSP invocation with per-submission cycle timing.
- **Tile-Major Weight Packers:** Converts `Q4_0`, `Q4_1`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_K`, and `F32` weights into C7x MMA tile-major `W4A8-g64` / `W8A8` layouts plus asymmetric `QLinear` row-sum metadata (`c7x_pack_qlinear_weights()`).
- **Dual Transports:**
  - `build-aarch64/libc7x.a` — BeagleBone AI-64 hardware (`/dev/mem`, multi-process `flock`, ABI handshake).
  - `build-hostemu/libc7x_hostemu.a` — x86_64 TI C7000 Host Emulation executing the exact same `cmd_exec.c` / `mma_kernels.c` / `vec_kernels.c`.

## Building & Testing

```bash
./platform/beagle-linux/scripts/build.sh libc7x
```

Runs `build-hostemu/test_libc7x` automatically and cross-compiles `build-aarch64/libc7x.a` + `build-aarch64/test_libc7x`.
