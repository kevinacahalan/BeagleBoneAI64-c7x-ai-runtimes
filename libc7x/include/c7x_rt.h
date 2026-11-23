/* SPDX-License-Identifier: MIT */
/*
 * libc7x — Shared Linux Host Runtime Library for TI C7x + MMA Acceleration.
 *
 * Provides:
 *   - Hardware (/dev/mem + remoteproc + c7x_ctrl) and x86_64 C7000 Host
 *     Emulation transports behind one uniform C API.
 *   - Coalescing 64-byte-aligned sub-allocator over DDR (up to 1.5 GiB) and
 *     on-chip MSMC SRAM (7.625 MiB) arenas.
 *   - Safe O_SYNC /dev/mem read/write/zero helpers (avoids ARM64 DC ZVA SIGBUS).
 *   - Command-buffer builder + single-RPC submission (C7X_OP_EXEC_LIST).
 *   - Tile-major weight packers (FP32 / FP16 / GGML Q4_0 / GGML Q8_0 ->
 *     C7x W4A8-g64 & W8A8 MMA layouts).
 */
#ifndef LIBC7X_C7X_RT_H_
#define LIBC7X_C7X_RT_H_

#include <stddef.h>
#include <stdint.h>

#include "c7x_ai_proto.h"
#include "c7x_cmd_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c7x_ctx c7x_ctx_t;
typedef struct c7x_buf c7x_buf_t;
typedef struct c7x_cmdbuf c7x_cmdbuf_t;

enum c7x_mem_tier {
	C7X_MEM_DDR  = 0,
	C7X_MEM_MSMC = 1,
};

enum c7x_transport_kind {
	C7X_TRANSPORT_AUTO    = 0,
	C7X_TRANSPORT_HW      = 1,
	C7X_TRANSPORT_HOSTEMU = 2,
};

struct c7x_open_opts {
	enum c7x_transport_kind transport;
	size_t emu_ddr_bytes;  /* Default 256 MiB if 0 (hostemu only) */
	size_t emu_msmc_bytes; /* Default 7.625 MiB if 0 (hostemu only) */
};

struct c7x_rt_info {
	uint64_t ddr_base;
	size_t   ddr_total;
	size_t   ddr_free;
	uint64_t msmc_base;
	size_t   msmc_total;
	size_t   msmc_free;
	uint32_t dru_ok;
	uint32_t is_hostemu;
	uint32_t max_block_m;
};

struct c7x_exec_stats {
	uint64_t cycles;
	uint64_t total_cycles;
	double   wall_us;
	uint32_t cmds_done;
	uint32_t fail_idx;
};

/* Lifecycle */
c7x_ctx_t *c7x_open(const struct c7x_open_opts *opts);
void       c7x_close(c7x_ctx_t *ctx);
int        c7x_get_info(const c7x_ctx_t *ctx, struct c7x_rt_info *info);
int        c7x_ping(c7x_ctx_t *ctx, double *wall_us_out);

/* Arena memory allocator (64-byte aligned, coalescing free-list) */
c7x_buf_t        *c7x_alloc(c7x_ctx_t *ctx, size_t bytes, enum c7x_mem_tier tier);
void              c7x_free(c7x_ctx_t *ctx, c7x_buf_t *buf);
uint64_t          c7x_buf_phys(const c7x_buf_t *buf);
void             *c7x_buf_host_ptr(const c7x_buf_t *buf);
size_t            c7x_buf_size(const c7x_buf_t *buf);
enum c7x_mem_tier c7x_buf_tier(const c7x_buf_t *buf);

/* Safe host <-> device memory transfers (avoids ARM64 DC ZVA SIGBUS on O_SYNC) */
void c7x_write(c7x_ctx_t *ctx, c7x_buf_t *buf, size_t off, const void *src, size_t n);
void c7x_read(c7x_ctx_t *ctx, void *dst, const c7x_buf_t *buf, size_t off, size_t n);
void c7x_zero(c7x_ctx_t *ctx, c7x_buf_t *buf, size_t off, size_t n);
void c7x_write_ptr(c7x_ctx_t *ctx, void *dst_arena_ptr, const void *src, size_t n);
void c7x_read_ptr(c7x_ctx_t *ctx, void *dst, const void *src_arena_ptr, size_t n);
void c7x_zero_ptr(c7x_ctx_t *ctx, void *dst_arena_ptr, size_t n);

/* Translate host pointer inside an arena buffer to its 64-bit DSP address (0 if outside). */
uint64_t c7x_host_to_phys(const c7x_ctx_t *ctx, const void *host_ptr);

/* Command-buffer builder & single-RPC execution */
c7x_cmdbuf_t *c7x_cmdbuf_create(c7x_ctx_t *ctx, uint32_t max_cmds, size_t desc_bytes);
void          c7x_cmdbuf_destroy(c7x_cmdbuf_t *cb);
void          c7x_cmdbuf_reset(c7x_cmdbuf_t *cb);
int           c7x_cmdbuf_push(c7x_cmdbuf_t *cb, const struct c7x_cmd *cmd);
uint64_t      c7x_cmdbuf_push_desc(c7x_cmdbuf_t *cb, const void *desc, size_t bytes);
uint32_t      c7x_cmdbuf_count(const c7x_cmdbuf_t *cb);
int           c7x_cmdbuf_submit(c7x_ctx_t *ctx, c7x_cmdbuf_t *cb, struct c7x_exec_stats *stats);

/* Weight layout sizing & packers */
size_t c7x_w4a8_g64_packed_bytes(uint32_t n, uint32_t k, size_t *scales_off_out);
size_t c7x_w8a8_packed_bytes(uint32_t n, uint32_t k, size_t *scales_off_out);

void c7x_pack_f32_to_w4a8_g64(const float *w_f32, uint32_t n, uint32_t k,
			      uint8_t *w_packed, float *w_scales);
void c7x_pack_f32_ntile_to_w4a8_g64(const float *rows_f32, uint32_t valid_rows,
				    uint32_t k, uint32_t nt,
				    uint8_t *w_packed, float *w_scales);
void c7x_pack_f32_to_w8a8(const float *w_f32, uint32_t n, uint32_t k,
			  int8_t *w_tiled, float *w_scales);
void c7x_pack_q4_0_to_w4a8_g64(const void *q4_0_data, uint32_t n, uint32_t k,
			       uint8_t *w_packed, float *w_scales);
void c7x_pack_q8_0_to_w8a8(const void *q8_0_data, uint32_t n, uint32_t k,
			   int8_t *w_tiled, float *w_scales);
void c7x_pack_f16_to_w8a8(const uint16_t *f16_data, uint32_t n, uint32_t k,
			  int8_t *w_tiled, float *w_scales);

/* Unpack tile-major weights back to FP32 or GGML raw formats (for lossless get_tensor). */
void c7x_unpack_w4a8_g64_row_f32(const uint8_t *w_packed, const float *w_scales,
				 uint32_t row, uint32_t k, float *out_f32);
void c7x_unpack_w8a8_row_f32(const int8_t *w_tiled, const float *w_scales,
			     uint32_t row, uint32_t k, float *out_f32);
void c7x_unpack_w4a8_g64_to_q4_0(const uint8_t *w_packed, const float *w_scales,
				 uint32_t n, uint32_t k, void *q4_0_out);
void c7x_unpack_w8a8_to_q8_0(const int8_t *w_tiled, const float *w_scales,
			     uint32_t n, uint32_t k, void *q8_0_out);

#ifdef __cplusplus
}
#endif

#endif /* LIBC7X_C7X_RT_H_ */
