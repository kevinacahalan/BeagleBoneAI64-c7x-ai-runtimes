# C7x Bare-Metal Firmware & Host Emulation (`runtime/dsp/`)

Bare-metal **TI C71x DSP + Matrix Multiply Accelerator (MMA)** firmware (`runtime/dsp/build/c7x-ai.elf`) and bit-exact x86_64 C7000 Host Emulation tests (`runtime/dsp/host_test/`) for the BeagleBone AI-64 (TDA4VM / J721E).

One firmware image serves **all three inference frontends**:
- **Composable command-buffer executor (`C7X_OP_EXEC_LIST`)** in `c7x/cmd_exec.c` (21 tensor ops for `ggml-c7x` and `ort-c7x-ep`)
- **Hand-fused monolithic transformer engine** in `c7x/transformer.c` (`C7X_OP_LLAMA_FORWARD` / `C7X_OP_LLAMA_PREFILL` for `c7x-ai` `.c7m` models)

## Source Layout

| Path | Role |
| --- | --- |
| `c7x/main.c` | Firmware entry point: brings up MMU/caches, initializes `DRU0`, registers remoteproc trace, polls mailbox shutdown (`RP_MBOX_SHUTDOWN`) and L2 SRAM control block (`0x64800000`) |
| `c7x/mmu.c`, `c7x/regs.asm` | Identity-mapped 48-bit C7x MMU page tables (L2 SRAM, 7.625 MiB MSMC SRAM via L3 4 KiB edge tables, 64 MiB + 1.5 GiB DDR arenas, peripherals/DRU0), cache sizing (32 KiB L1D + 64 KiB L2 cache / 448 KiB L2 SRAM), and J721E cache maintenance helpers |
| `c7x/dru.c`, `c7x/dru.h` | Bare-metal Compute Cluster 0 `DRU0` (`0x6D000000`) direct-TR driver using Type-9 TRs (`CHATOMIC[0]`) and NC-L2 completion markers (zero TI-SCI, zero RTOS) |
| `c7x/mma_kernels.c`, `c7x/mma_kernels.h` | C7x MMA kernels: multi-tile `int8×int8→int32` GEMM (`block_m <= 63`), 1D tile-major W8A8 GEMV, B-stationary group-64 W4A8 (`c7x_mma_gemm_w4a8_g64_f32`) with `DRU0` L2-SRAM ping-pong double-buffering, and C-bank depth probe |
| `c7x/vec_kernels.c`, `c7x/vec_kernels.h` | 512-bit FP32 vector unit kernels (`float16`): RMSNorm, Cody-Waite `vexpf_f16`, `vrecip_f16`, Softmax, SwiGLU, RoPE (interleaved & NeoX), dynamic `fp32→int8` quant, dequant/argmax, and 2D `SE0`/`SE1` GQA + INT8 KV attention |
| `c7x/cmd_exec.c`, `c7x/cmd_exec.h` | Flat command-buffer executor (`C7X_OP_EXEC_LIST`) implementing 21 ops: MMA `W4A8`/`W8A8` + F32/F16 matmul, norms, RoPE, softmax, elementwise, `GET_ROWS`/`SET_ROWS`/`CPY`/`CONT`, `IM2COL`, `POOL_2D`, `CONV_2D`, asymmetric INT8/UINT8 `QLINEAR_MATMUL`, `QLINEAR_CONV2D`, and `DEQUANTIZE_LIN` |
| `c7x/transformer.c`, `c7x/transformer.h` | Monolithic Llama/Qwen2/Qwen3 transformer engine with MSMC layer staging, INT8 KV cache, and B-stationary prefill |
| `c7x/bench.c` | On-silicon microbenchmarks (`ping`, `cbank`, `mem`, `dru`, `gemm`, `gemv`) |
| `c7x/c7x.cmd` | Linker script placing `_c_int00` at 2 MiB-aligned `0xA8200000` and resource table / trace buffer in non-cacheable `0xA8100000` |
| `include/` | Shared C7x/Linux ABI headers (`c7x_ai_proto.h`, `c7x_cmd_proto.h`, `c7x_llama_fmt.h`, `ai64_dsp_mbox.h`, `ai64_dsp_trace.h`) |
| `host_test/` | x86_64 C7000 Host Emulation tests (`test_mma.cpp`, `test_cmd.cpp`, `test_llama.cpp`) |

## Building & Testing

From the repository root:

```bash
./platform/beagle-linux/scripts/build.sh host-test   # compile & run x86_64 host-emulation tests
./platform/beagle-linux/scripts/build.sh dsp         # compile runtime/dsp/build/c7x-ai.elf
```

## Clean C7x Remoteproc Shutdown

A C71x has no local reset, so Linux powers the module off via the device manager, which refuses unless the core is idle (`can't stop rproc: -19`). Both TI's 6.18 kernel and `c7x-ai stop`/`load` send `RP_MBOX_SHUTDOWN` (`0xFFFFFF14`) on Mailbox cluster 4 FIFO 1; `include/ai64_dsp_mbox.h` replies with `RP_MBOX_SHUTDOWN_ACK` (`0xFFFFFF15`) on FIFO 0, disables interrupts, and parks the C7x in `IDLE`.

See [`../../docs/MEMORY.md`](../../docs/MEMORY.md), [`../../docs/NOTES.md`](../../docs/NOTES.md), and [`../../docs/RESULTS.md`](../../docs/RESULTS.md).
