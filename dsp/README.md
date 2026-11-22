# C7x Bare-Metal Firmware & Host Emulation (`dsp/`)

Bare-metal **TI C71x DSP + Matrix Multiply Accelerator (MMA)** firmware (`dsp/build/c7x-ai.elf`) and bit-exact x86_64 C7000 Host Emulation tests (`dsp/host_test/`) for the BeagleBone AI-64 (TDA4VM / J721E).

## Source Layout

| Path | Role |
| --- | --- |
| `c7x/main.c` | Firmware entry point: brings up MMU/caches, initializes `DRU0`, registers remoteproc trace, polls mailbox shutdown (`RP_MBOX_SHUTDOWN`) and L2 SRAM control block (`0x64800000`) |
| `c7x/mmu.c`, `c7x/regs.asm` | Identity-mapped 48-bit C7x MMU page tables (L2 SRAM, 4 MiB MSMC SRAM, 64 MiB + 512 MiB DDR arenas, peripherals/DRU0), cache sizing (32 KiB L1D + 64 KiB L2 cache / 448 KiB L2 SRAM), and J721E cache flush + double-`_mfence()` helpers |
| `c7x/dru.c`, `c7x/dru.h` | Bare-metal Compute Cluster 0 `DRU0` (`0x6D000000`) direct-TR driver using Type-9 TRs (`CHATOMIC[0]`) and NC-L2 completion markers (zero TI-SCI, zero RTOS) |
| `c7x/mma_kernels.c`, `c7x/mma_kernels.h` | C7x MMA kernels: multi-tile `int8×int8→int32` GEMM (`block_m <= 63`), 1D tile-major W8A8 GEMV, MMA-accelerated group-64 W4A8 (`c7x_mma_gemm_w4a8_g64_f32`) with `DRU0` L2-SRAM ping-pong double-buffering, and C-bank depth probe |
| `c7x/vec_kernels.c`, `c7x/vec_kernels.h` | 512-bit FP32 vector unit kernels (`float16`): RMSNorm, Cody-Waite `vexpf_f16`, `vrecip_f16`, Softmax, SwiGLU, RoPE (interleaved & NeoX half-split), dynamic `fp32→int8` quant, dequant/argmax, and 2D `SE0`/`SE1` GQA attention |
| `c7x/transformer.c`, `c7x/transformer.h` | Model-agnostic transformer inference engine: MSMC hot-layer staging, single-token decode (`C7X_OP_LLAMA_FORWARD`), and batched multi-token MMA prefill (`C7X_OP_LLAMA_PREFILL`) |
| `c7x/bench.c` | On-silicon microbenchmarks (`ping`, `cbank`, `mem`, `dru`, `gemm`, `gemv`) |
| `c7x/c7x.cmd` | Linker script placing `_c_int00` at 2 MiB-aligned `0xA8200000` and resource table / trace buffer in non-cacheable `0xA8100000` |
| `include/` | Shared C7x/Linux ABI headers: `c7x_ai_proto.h`, `c7x_llama_fmt.h`, `ai64_dsp_mbox.h`, `ai64_dsp_trace.h` |
| `host_test/` | x86_64 C7000 Host Emulation tests (`test_mma.cpp`, `test_llama.cpp`) compiled with `g++ -D__C7X_HOSTEM__` against `libc70-host-emulation.a` |

## Building & Testing

Use the containerized build script from the repository root (uses TI C7000 CGT `1.2.0.STS` in `containers/dsp-cgt/`):

```bash
./beagle-linux/scripts/build.sh host-test   # compile & run x86_64 host-emulation tests
./beagle-linux/scripts/build.sh dsp         # compile dsp/build/c7x-ai.elf
```

## Clean C7x Remoteproc Shutdown

A C71x has no local reset, so Linux powers the module off via the device manager, which refuses unless the core is idle (`can't stop rproc: -19`). Both TI's 6.18 kernel and `c7x-ai stop`/`load` send `RP_MBOX_SHUTDOWN` (`0xFFFFFF14`) on Mailbox cluster 4 FIFO 1; `dsp/include/ai64_dsp_mbox.h` replies with `RP_MBOX_SHUTDOWN_ACK` (`0xFFFFFF15`) on FIFO 0, disables interrupts, and parks the C7x in `IDLE`.

See [`../MEMORY.md`](../MEMORY.md) for the full address & cache map, [`../NOTES.md`](../NOTES.md) for emulator-vs-silicon quirks, and [`../RESULTS.md`](../RESULTS.md) for measured silicon numbers.
