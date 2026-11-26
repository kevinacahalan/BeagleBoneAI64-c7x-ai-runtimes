/* SPDX-License-Identifier: MIT */
/*
 * Shared Linux A72 <-> C7x DSP control protocol and memory map.
 *
 * Included by both the C7x firmware (cl7x) and the Linux host tool (aarch64
 * gcc), so all fields are fixed-width little-endian types at explicit offsets.
 */
#ifndef C7X_AI_PROTO_H_
#define C7X_AI_PROTO_H_

#include <stdint.h>

/* C7x L2 SRAM: 512 KiB total, 64 KiB configured as L2 cache -> 448 KiB SRAM. */
#define C7X_L2_BASE          0x64800000ull
#define C7X_CTRL_ADDR        0x64800000ull /* 4 KiB non-cacheable control block */
#define C7X_CTRL_SIZE        0x00001000u
#define C7X_L2_SCRATCH_ADDR  0x64801000ull /* 444 KiB write-back scratchpad */
#define C7X_L2_SCRATCH_SIZE  0x0006F000u

/*
 * On-chip MSMC SRAM (8 MiB total on TDA4VM):
 *   0x70000000..0x7001FFFF (128 KiB): ARM Trusted Firmware (ATF) — untouchable
 *   0x70020000..0x7003FFFF (128 KiB): unmapped safety guard band
 *   0x70040000..0x707DFFFF (7.625 MiB): C7x MSMC SRAM arena (4 KiB L3 pages at edges)
 *   0x707E0000..0x707EFFFF (64 KiB):  unmapped safety guard band
 *   0x707F0000..0x707FFFFF (64 KiB):  DMSC IPC — untouchable
 */
#define C7X_MSMC_BASE          0x70040000ull
#define C7X_MSMC_SIZE          0x007A0000u /* 7,808 KiB = 7.625 MiB */

/* Shared reserved DDR tensor arenas (no-map in device tree). */
#define C7X_DDR_ARENA_BASE     0xAC000000ull /* 64 MiB primary/benchmark arena */
#define C7X_DDR_ARENA_SIZE     0x04000000u
#define C7X_LLM_ARENA_BASE     0xC0000000ull /* Legacy 512 MiB LLM arena (optional DT node) */
#define C7X_LLM_ARENA_SIZE     0x20000000u
#define C7X_LLM_HI_ARENA_BASE  0x880000000ull /* 1.5 GiB high-DDR LLM arena (optional DT node) */
#define C7X_LLM_HI_ARENA_SIZE  0x60000000u

/* Compute cluster 0 DRU (Data Routing Unit) MMIO base. */
#define C7X_DRU_BASE         0x6D000000ull

/* Mailbox cluster 4: Linux sends on FIFO 1, C7x sends on FIFO 0. */
#define C7X_MBOX_BASE        0x31F84000ull
#define C7X_MBOX_LINUX_TX    1u
#define C7X_MBOX_C7X_TX      0u

/* C71x core clock on BeagleBone AI-64 (1.0 GHz). */
#define C7X_CLOCK_HZ         1000000000ull

#define C7X_CTRL_MAGIC       0x49413743u /* "C7AI" */
#define C7X_CTRL_VERSION     1u

/* Command opcodes */
enum c7x_op {
	C7X_OP_PING          = 1u, /* Empty round-trip */
	C7X_OP_PROBE_CBANK   = 2u, /* Probe MMA C-bank row depth (16/32/48/64) */
	C7X_OP_MEM_BW_READ   = 3u, /* SE0 512-bit read bandwidth at src_addr */
	C7X_OP_MEM_BW_COPY   = 4u, /* SE0/vector copy bandwidth src_addr -> dst_addr */
	C7X_OP_DRU_COPY      = 5u, /* Bare-metal DRU direct-TR copy src_addr -> dst_addr */
	C7X_OP_GEMM_MMA_I8   = 6u, /* Arbitrary M x K x N int8*int8 -> int32 MMA GEMM */
	C7X_OP_GEMM_C_I8     = 7u, /* Reference plain-C M x K x N int8*int8 -> int32 on C7x */
	C7X_OP_GEMV_W4A8     = 8u, /* Packed int4 weights x int8 activations -> int32 on C7x */
	C7X_OP_CACHE_FLUSH   = 9u, /* Writeback + invalidate L1D/L2 */
	C7X_OP_VEC_SELFTEST  = 10u, /* Run built-in FP32 vector-unit kernel self-test */
	C7X_OP_LLAMA_FORWARD = 11u, /* Run W8A8/W4A8 transformer single-token decode */
	C7X_OP_LLAMA_PREFILL = 12u, /* Run batched multi-token MMA prefill */
	C7X_OP_EXEC_LIST     = 13u, /* Execute flat command buffer (struct c7x_cmd[m] at a_addr) */
};

/* Status codes */
enum c7x_status {
	C7X_STATUS_OK        = 0u,
	C7X_STATUS_BAD_OP    = 1u,
	C7X_STATUS_BAD_ARGS  = 2u,
	C7X_STATUS_NO_DRU    = 3u,
	C7X_STATUS_DRU_ERR   = 4u,
	C7X_STATUS_MISMATCH  = 5u,
	C7X_STATUS_CMD_ERR   = 6u,
};

/* Flags for GEMM/GEMV ops */
#define C7X_GEMM_B_TRANSPOSE (1u << 0) /* B is [N][K] row-major (load with B_ORDER=COL) */
#define C7X_GEMM_USE_MSMC    (1u << 1) /* Stage/run from MSMC when applicable */
#define C7X_GEMM_USE_L2      (1u << 2) /* Stage B tiles into L2 scratchpad */
#define C7X_GEMM_USE_DRU     (1u << 3) /* Prefetch B tiles via DRU into L2 ping/pong */
#define C7X_GEMM_BLOCK_M_SHIFT 8u      /* Optional C-bank row block override (0 = auto) */
#define C7X_GEMM_BLOCK_M_MASK  (0xFFu << C7X_GEMM_BLOCK_M_SHIFT)

/*
 * Control block at C7X_CTRL_ADDR (0x64800000, non-cacheable on C7x and
 * mapped O_SYNC by Linux). Keep every field 32-bit or 64-bit naturally aligned.
 */
struct c7x_ctrl {
	uint32_t magic;        /* C7x sets C7X_CTRL_MAGIC after init */
	uint32_t version;      /* C7X_CTRL_VERSION */
	uint32_t req_seq;      /* Linux increments after writing request fields */
	uint32_t done_seq;     /* C7x sets == req_seq after writing response */

	uint32_t op;           /* enum c7x_op */
	uint32_t status;       /* enum c7x_status */
	uint32_t flags;        /* op-specific flags */
	uint32_t iters;        /* repeat count for timing (0 treated as 1) */

	uint32_t m;            /* GEMM M, or byte count for BW/DRU ops */
	uint32_t k;            /* GEMM K, or C-bank probe depth */
	uint32_t n;            /* GEMM N */
	uint32_t reserved0;

	uint64_t a_addr;       /* Matrix A or src physical address */
	uint64_t b_addr;       /* Matrix B or dst physical address */
	uint64_t c_addr;       /* Matrix C (int32 output) physical address */

	uint64_t cycles;       /* Measured C7x __TSC cycles (kernel only) */
	uint64_t total_cycles; /* Measured C7x __TSC cycles (incl. cache maint) */
	uint64_t aux0;         /* Extra return value (checksum / probe result) */
	uint64_t aux1;

	/* Boot-time hardware info written once by C7x */
	uint32_t dnum;
	uint32_t scr_lo;
	uint32_t l1dcfg;
	uint32_t l2cfg;
	uint64_t dru_pid;
	uint64_t dru_caps;
	uint32_t dru_ok;
	uint32_t cbank_max_ok; /* Populated if self-probe ran, else 0 */
};

#endif /* C7X_AI_PROTO_H_ */
