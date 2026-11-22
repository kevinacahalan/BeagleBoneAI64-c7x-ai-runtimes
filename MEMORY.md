# Memory Map

All memory regions shared between Linux (2× Cortex-A72) and the bare-metal C7x DSP + MMA on the BeagleBone AI-64 (TI TDA4VM / J721E). The header [`dsp/include/c7x_ai_proto.h`](dsp/include/c7x_ai_proto.h) and MMU setup [`dsp/c7x/mmu.c`](dsp/c7x/mmu.c) are the source of truth.

## On-Chip Memory

| Address (SoC / C7x VA) | Size | C7x MMU Attr | Purpose |
| --- | --- | --- | --- |
| `0x64800000` | 4 KiB | Normal NC | Control / command / response block (`struct c7x_ctrl`) + top-of-page DRU0 NC-L2 completion marker slots (`0x64800E00`/`0x64800F00`) |
| `0x64801000` – `0x6486FFFF` | 444 KiB | Normal WB | C7x L2 SRAM scratchpad (transformer activations + unused tail automatically carved for DRU0 async ping-pong weight buffering; 64 KiB of 512 KiB L2 is L2 cache) |
| `0x70000000` – `0x7001FFFF` | 128 KiB | Unmapped | Reserved for ARM Trusted Firmware (ATF) — **do not touch** |
| `0x70200000` – `0x705FFFFF` | 4 MiB | Normal WB | On-chip MSMC SRAM arena (2 × 2 MiB MMU blocks; pinned leading transformer layers; cleanly clear of ATF and DMSC) |
| `0x707F0000` – `0x707FFFFF` | 64 KiB | Unmapped | Reserved for DMSC IPC — **do not touch** |

*(TDA4VM has ~7.75 MiB of usable MSMC SRAM between `0x70020000` and `0x707E7FFF`; the current MMU maps the aligned 4 MiB core at `0x70200000`.)*

## DDR Memory

| Address (SoC / C7x VA) | Size | C7x MMU Attr | Purpose |
| --- | --- | --- | --- |
| `0xA8000000` – `0xA81FFFFF` | 2 MiB | Normal NC | C7x DMA pool (`0xA8000000`), resource table & 4 KiB remoteproc trace buffer (`0xA8100000`) |
| `0xA8200000` – `0xA8FFFFFF` | 14 MiB | Normal WB | C7x firmware code (`_c_int00` at 2 MiB-aligned `0xA8200000`), data, stack, MMU tables |
| `0xAC000000` – `0xAFFFFFFF` | 64 MiB | Normal WB | Primary reserved DDR tensor arena (`c7x-ai-arena@ac000000`, `no-map` in `k3-j721e-beagleboneai64-c7x-ai.dtbo`) |
| `0xC0000000` – `0xDFFFFFFF` | 512 MiB | Normal WB | Large LLM DDR arena (`c7x-llm-arena@c0000000`, `no-map` in `k3-j721e-beagleboneai64-c7x-ai.dtbo`; auto-detected by `c7x-ai llama`, falls back to `0xAC000000`) |

## Peripherals Used by C7x

| Address | Purpose |
| --- | --- |
| `0x31F84000` | Mailbox cluster 4 (`RP_MBOX_SHUTDOWN` on FIFO 1, `RP_MBOX_SHUTDOWN_ACK` on FIFO 0) |
| `0x6D000000` | Compute Cluster 0 DRU0 (Data Routing Unit; bare-metal atomic Type-9 direct-TR via `CHATOMIC[0]` at `0x6D080000`) |

## Cache Coherency Notes

- C7x L1D (32 KiB) and L2 (64 KiB) are enabled write-back for L2 scratchpad, MSMC, and both DDR tensor arenas (`0xAC000000` and `0xC0000000`).
- Around every host-triggered job, C7x flushes/invalidates L1D then L2 (polling `L1DWBINV`/`L2WBINV` bit 0 until clear) with a **double `_mfence()`** (`__memory_fence(0x1)` twice) per TI J721E C71x silicon errata.
- Streaming Engines (`SE0`/`SE1`) bypass L1D and read directly from L2 SRAM (`0x64801000..0x6486FFFF`), so DRU0 DDR → L2 SRAM async ping-pong weight buffers require **zero cache invalidation** when consumed via `SE0`/`SE1`.
- Linux accesses `0x64800000`, `0xAC000000`, and `0xC0000000` via `/dev/mem` (`O_SYNC`).
