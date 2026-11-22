/* SPDX-License-Identifier: MIT */
/*
 * C7x MMA and vector-unit compute kernels.
 *
 * Compiles identically for both TI C7000 CGT (cl7x) and x86_64 C7000 Host
 * Emulation (g++ -D__C7X_HOSTEM__).
 */
#ifndef DSP_C7X_MMA_KERNELS_H_
#define DSP_C7X_MMA_KERNELS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Probe C-bank row depth (`depth` in 1..64) on a single 64x64 int8 tile:
 *   y[depth][64] = a[depth][64] * b[64][64]  (exact int32 output)
 * Uses a single C bank with C_CWRSTPER = X_CRRSTPER = depth.
 */
void c7x_mma_probe_cbank(const int8_t *a, const int8_t *b, int32_t *y, uint32_t depth);

/*
 * Arbitrary M x K x N int8 x int8 -> int32 GEMM on the C7x MMA.
 *
 *   K must be a positive multiple of 64.
 *   N must be a positive multiple of 64.
 *   M >= 1 (any positive integer; internally tiled by block_m).
 *
 *   b_transpose == 0: B is [K][N] row-major, C = A * B   (B_ORDER = ROW)
 *   b_transpose != 0: B is [N][K] row-major, C = A * B^T (B_ORDER = COL)
 *
 *   block_m: rows per MMA C-bank tile (e.g. 16, 32, or 64; 0 = default 16).
 */
void c7x_mma_gemm_i8_i32(const int8_t *a, const int8_t *b, int32_t *c,
			 uint32_t m, uint32_t k, uint32_t n,
			 uint32_t b_transpose, uint32_t block_m);

/*
 * 1D-streamed tile-major M=1 MMA GEMV:
 *   b_tiled: [N/64][K/64][64][64] int8 (contiguous 4 KiB tiles, row-major
 *            within each 64x64 tile, transposed on load via B_ORDER=COL).
 *   Opens SE1 once in pure 1D linear burst mode for full DDR/MSMC bandwidth.
 */
void c7x_mma_gemv_i8_i32_tiled(const int8_t *a, const int8_t *b_tiled,
			       int32_t *y, uint32_t k, uint32_t n);

/* Reference plain-C M x K x N int8 x int8 -> int32 GEMM. */
void c7x_ref_gemm_i8_i32(const int8_t *a, const int8_t *b, int32_t *c,
			 uint32_t m, uint32_t k, uint32_t n,
			 uint32_t b_transpose);

/*
 * Packed 4-bit signed weight GEMV (W4A8):
 *   a:        [K] int8 activation vector (K multiple of 128)
 *   w_packed: [N][K/2] uint8; byte [j*(K/2) + p] holds:
 *             low nibble  = signed int4 weight for k = 2*p
 *             high nibble = signed int4 weight for k = 2*p + 1
 *   y:        [N] int32 dot products (N multiple of 64)
 * Unpacks 128 int4 weights per 64-byte load in C7x vector registers and feeds
 * the MMA (or C7x vector dot product) so memory traffic is halved vs int8.
 */
void c7x_mma_gemv_w4a8(const int8_t *a, const uint8_t *w_packed, int32_t *y,
		       uint32_t k, uint32_t n);

void c7x_ref_gemv_w4a8(const int8_t *a, const uint8_t *w_packed, int32_t *y,
		       uint32_t k, uint32_t n);

/*
 * Tile-major group-64 W4A8 GEMM/GEMV on the C7x MMA (supports m == 1 decode
 * and m > 1 batched prefill with the exact same packed weights):
 *   a:        [m][k_pad] int8 activations (k_pad multiple of 64)
 *   a_scales: [m] float32 per-token activation scales
 *   w_packed: [n_pad/64][k_pad/64][32][64] uint8; vector v (0..31) byte c (0..63)
 *             holds low nibble = row 2v col c, high nibble = row 2v+1 col c.
 *   w_scales: [n_pad/64][k_pad/64][64] float32 group-64 scales per output row.
 *   out:      optional float32[m][out_dim] (may be NULL when m == 1 && argmax_out != NULL)
 */
void c7x_mma_gemm_w4a8_g64_f32(float *out, const int8_t *a, const float *a_scales,
			       const uint8_t *w_packed, const float *w_scales,
			       uint32_t m, uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
			       uint32_t block_m, int add, uint32_t *argmax_out);

void c7x_ref_gemm_w4a8_g64_f32(float *out, const int8_t *a, const float *a_scales,
			       const uint8_t *w_packed, const float *w_scales,
			       uint32_t m, uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
			       int add, uint32_t *argmax_out);

/*
 * 1D-streamed tile-major W8A8 MMA GEMM for arbitrary m >= 1 (batched prefill):
 *   b_tiled: [n/64][k/64][64][64] int8 contiguous MMA tiles.
 */
void c7x_mma_gemm_i8_i32_tiled(const int8_t *a, const int8_t *b_tiled, int32_t *c,
			       uint32_t m, uint32_t k, uint32_t n, uint32_t block_m);

/*
 * Provide a scratch buffer in L2 SRAM for asynchronous DRU0 weight ping-pong
 * double-buffering inside c7x_mma_gemm_w4a8_g64_f32 (no-op in host emulation).
 */
void c7x_mma_set_dru_l2_buf(void *buf, uint32_t bytes);

/* 512-bit Streaming Engine memory read & copy bandwidth kernels (bytes % 64 == 0). */
uint64_t c7x_mem_bw_read(const void *src, uint32_t bytes);
void c7x_mem_bw_copy(void *dst, const void *src, uint32_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* DSP_C7X_MMA_KERNELS_H_ */
