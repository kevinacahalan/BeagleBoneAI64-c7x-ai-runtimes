/* SPDX-License-Identifier: MIT */
/*
 * C7x MMA and vector-unit compute kernels.
 *
 * Shared between TI C7000 CGT (cl7x) and x86_64 C7000 Host Emulation
 * (g++ -D__C7X_HOSTEM__) so every kernel is verified bit-exact on PC before
 * running on live TDA4VM silicon.
 */

#ifdef __C7X_HOSTEM__
#include "c7x_host_emulation.h"
#else
#include <c7x.h>
#endif

#include <stdint.h>
#include <string.h>

#include "mma_kernels.h"
#if !defined(__C7X_HOSTEM__) && !defined(_HOST_BUILD)
#include "dru.h"
#endif

static uint8_t *g_dru_l2_buf = NULL;
static uint32_t g_dru_l2_bytes = 0u;

void c7x_mma_set_dru_l2_buf(void *buf, uint32_t bytes)
{
	g_dru_l2_buf   = (uint8_t *)buf;
	g_dru_l2_bytes = bytes;
}

#define MMA_TILE 64u

static inline __STRM_TEMPLATE strm_zero(void)
{
#ifdef __C7X_HOSTEM__
	return __uint16(0);
#else
	return (__uint16)(0);
#endif
}

/* Open SE on a 2D byte tile: icnt0 bytes wide (multiple of 64), icnt1 rows, row stride dim1. */
static inline void se_open_2d(int se, const void *p, uint32_t icnt0, uint32_t icnt1, int32_t dim1)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_2D;
	flags.ELETYPE = __SE_ELETYPE_8BIT;
	flags.VECLEN  = __SE_VECLEN_64ELEMS;

	params = __set_ICNT0(params, icnt0);
	params = __set_ICNT1_DIM1(params, icnt1, dim1);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

/* Open SE on a 1D byte stream (bytes multiple of 64). */
static inline void se_open_1d(int se, const void *p, uint32_t bytes)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_1D;
	flags.ELETYPE = __SE_ELETYPE_8BIT;
	flags.VECLEN  = __SE_VECLEN_64ELEMS;

	params = __set_ICNT0(params, bytes);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

/*
 * Transfer and store one 64-element int32 row (256 bytes = 4 x 64-byte vectors)
 * from the MMA C bank via paired __HWAXFER(0) / __HWARCV(0) instructions.
 * cl7x requires each __HWAXFER to be lexically paired with its __HWARCV in the
 * same function.
 */
#define MMA_XFER_STORE_ROW_I32(dst_ptr)                                           \
	do {                                                                      \
		uint8_t *_d = (uint8_t *)(dst_ptr);                               \
		uint32_t _q;                                                      \
		for (_q = 0u; _q < 4u; _q++) {                                    \
			__HWAXFER(0);                                             \
			__HWAADV();                                               \
			*(__uchar64_ptr)(_d + _q * 64u) = __HWARCV(0);            \
			__HWAADV();                                               \
		}                                                                 \
	} while (0)

void c7x_mma_probe_cbank(const int8_t *a, const int8_t *b, int32_t *y, uint32_t depth)
{
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t r;

	if (depth == 0u || depth > 64u) {
		return;
	}

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE      = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE      = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER      = __MMA_B_CONFIG_ROW;
	cfg.B_BSWPER     = MMA_TILE;
	cfg.B_BRSTPER    = MMA_TILE;

	cfg.C_ATYPE      = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE      = __MMA_C_CONFIG_BTYPE_INT8;
	cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
	cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
	cfg.C_OP0PER     = 0xffffffffu;
	cfg.C_OP1PER     = 0xffffffffu;
	cfg.C_BSWPER     = 0xffffffffu;
	cfg.C_HWLDDST    = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE   = __MMA_C_CONFIG_HWLDTYPE_INT32;
	cfg.C_CRSWPER    = 0xffffffffu;
	cfg.C_CRRSTPER   = depth;
	cfg.C_CWSWPER    = depth;
	cfg.C_CWRSTPER   = depth;

	cfg.X_CTYPE      = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE      = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT      = 0u;
	cfg.X_RE         = 0u;
	cfg.X_SAT        = 0u;
	cfg.X_ReLU       = 0u;
	cfg.X_CSWPER     = depth * 4u;
	cfg.X_CRRSTPER   = (depth * 4u) & 0xffu; /* 0 encodes 256 when depth == 64 */
	cfg.PARITYCTRL   = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	se_open_1d(1, b, MMA_TILE * MMA_TILE);
	se_open_1d(0, a, depth * MMA_TILE);

	for (r = 0; r < MMA_TILE; r++) {
		__HWALDB(__SE1ADV(uchar64));
		__HWAADV();
	}
	__SE1_CLOSE();

	for (r = 0; r < depth; r++) {
		__HWALDA(__SE0ADV(uchar64));
		__HWAOP(0);
		__HWAADV();
	}
	__SE0_CLOSE();

	__HWAFENCE_OPXFER();
	/* Drain host-emulation pipeline before XFER (empty macro on C7x silicon). */
	for (r = 0; r < 64u; r++) {
		__HWAADV();
	}

	for (r = 0; r < depth; r++) {
		MMA_XFER_STORE_ROW_I32(&y[r * MMA_TILE]);
	}
	__HWACLOSE(0);
}

/*
 * Compute a single M-block of size m_cur (1..64) across all N-tiles (N/64) and
 * all K-tiles (K/64) with K-accumulation inside the MMA C-banks, B-bank
 * ping-pong, and C-bank ping-pong across N-tiles.
 */
static void mma_gemm_mblock(const int8_t *a_blk, const int8_t *b, int32_t *c_blk,
			    uint32_t m_cur, uint32_t k, uint32_t n, uint32_t b_transpose)
{
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t k_tiles = k / MMA_TILE;
	uint32_t n_tiles = n / MMA_TILE;
	uint32_t nt, kt, r;
	int32_t b_stride = b_transpose ? (int32_t)k : (int32_t)n;

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE   = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE   = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER   = b_transpose ? __MMA_B_CONFIG_COL : __MMA_B_CONFIG_ROW;
	cfg.B_BSWPER  = MMA_TILE;
	cfg.B_BRSTPER = MMA_TILE;

	cfg.C_ATYPE   = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE   = __MMA_C_CONFIG_BTYPE_INT8;
	if (k_tiles == 1u) {
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
		cfg.C_OP0PER     = 0xffffffffu;
		cfg.C_OP1PER     = 0xffffffffu;
	} else {
		cfg.C_OPSTART    = __MMA_C_CONFIG_OPSTART_OPERATION0;
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MULPLUS;
		cfg.C_OP0PER     = m_cur;
		cfg.C_OP1PER     = (k_tiles - 1u) * m_cur;
	}
	/* Switch B read bank after every m_cur ops (one K-tile). */
	cfg.C_BSWPER   = m_cur;
	cfg.C_HWLDDST  = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE = __MMA_C_CONFIG_HWLDTYPE_INT32;
	/* Wrap C row index every m_cur ops; switch C bank after all k_tiles finish. */
	cfg.C_CRRSTPER = m_cur;
	cfg.C_CWRSTPER = m_cur;
	cfg.C_CRSWPER  = k_tiles * m_cur;
	cfg.C_CWSWPER  = k_tiles * m_cur;

	/* Exact int32 readout through X-stage transfer buffer (4 x 64-byte HWAXFER/HWARCV per row). */
	cfg.X_CTYPE    = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE    = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT    = 0u;
	cfg.X_RE       = 0u;
	cfg.X_SAT      = 0u;
	cfg.X_ReLU     = 0u;
	cfg.X_CRRSTPER = (m_cur * 4u) & 0xffu;
	cfg.X_CSWPER   = m_cur * 4u;
	cfg.PARITYCTRL = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	/* Preload the very first B tile (nt=0, kt=0) into B bank 0. */
	{
		const int8_t *b0 = b;

		se_open_2d(1, b0, MMA_TILE, MMA_TILE, b_stride);
		for (r = 0; r < MMA_TILE; r++) {
			__HWALDB(__SE1ADV(uchar64));
			__HWAADV();
		}
		__SE1_CLOSE();
	}

	for (nt = 0; nt < n_tiles; nt++) {
		for (kt = 0; kt < k_tiles; kt++) {
			const int8_t *a_tile = a_blk + kt * MMA_TILE;
			uint32_t next_kt = kt + 1u;
			uint32_t next_nt = nt;
			int has_next_b = 1;

			if (next_kt == k_tiles) {
				next_kt = 0u;
				next_nt = nt + 1u;
				if (next_nt == n_tiles) {
					has_next_b = 0;
				}
			}

			/* Open SE0 for A tile (m_cur x 64, row stride k). */
			se_open_2d(0, a_tile, MMA_TILE, m_cur, (int32_t)k);

			if (has_next_b) {
				const int8_t *b_next = b_transpose
					? (b + (next_nt * MMA_TILE) * k + next_kt * MMA_TILE)
					: (b + (next_kt * MMA_TILE) * n + next_nt * MMA_TILE);

				se_open_2d(1, b_next, MMA_TILE, MMA_TILE, b_stride);

				/*
				 * Overlap A-row compute with next B-tile load into the
				 * alternate B bank using __HWALDAB!
				 */
				for (r = 0; r < m_cur; r++) {
					__HWALDAB(__SE0ADV(uchar64), __SE1ADV(uchar64));
					__HWAOP(0);
					__HWAADV();
				}
				/* Finish loading remaining (64 - m_cur) rows of next B tile. */
				for (r = m_cur; r < MMA_TILE; r++) {
					__HWALDB(__SE1ADV(uchar64));
					__HWAADV();
				}
				__SE1_CLOSE();
			} else {
				for (r = 0; r < m_cur; r++) {
					__HWALDA(__SE0ADV(uchar64));
					__HWAOP(0);
					__HWAADV();
				}
			}
			__SE0_CLOSE();

			/*
			 * Drain previous N-tile's completed C bank during kt == 0 of the
			 * current N-tile (wait for previous ops via fence once before first xfer).
			 */
			if (nt > 0u && kt == 0u) {
				int32_t *c_prev = c_blk + (nt - 1u) * MMA_TILE;

				__HWAFENCE_OPXFER();
				for (r = 0; r < 64u; r++) {
					__HWAADV();
				}
				for (r = 0; r < m_cur; r++) {
					MMA_XFER_STORE_ROW_I32(c_prev + r * n);
				}
			}
		}
	}

	/* Read out the last N-tile from its C bank. */
	{
		int32_t *c_last = c_blk + (n_tiles - 1u) * MMA_TILE;

		__HWAFENCE_OPXFER();
		for (r = 0; r < 64u; r++) {
			__HWAADV();
		}
		for (r = 0; r < m_cur; r++) {
			MMA_XFER_STORE_ROW_I32(c_last + r * n);
		}
	}

	__HWACLOSE(0);
}

void c7x_mma_gemm_i8_i32(const int8_t *a, const int8_t *b, int32_t *c,
			 uint32_t m, uint32_t k, uint32_t n,
			 uint32_t b_transpose, uint32_t block_m)
{
	uint32_t m0;

	if (m == 0u || k == 0u || n == 0u || (k % MMA_TILE) != 0u || (n % MMA_TILE) != 0u) {
		return;
	}
	if (block_m == 0u || block_m > 63u) {
		block_m = 16u;
	}

	for (m0 = 0; m0 < m; m0 += block_m) {
		uint32_t m_cur = (m - m0 < block_m) ? (m - m0) : block_m;

		mma_gemm_mblock(a + m0 * k, b, c + m0 * n, m_cur, k, n, b_transpose);
	}
}

void c7x_mma_gemv_i8_i32_tiled(const int8_t *a, const int8_t *b_tiled,
			       int32_t *y, uint32_t k, uint32_t n)
{
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t k_tiles = k / MMA_TILE;
	uint32_t n_tiles = n / MMA_TILE;
	uint32_t nt, kt, r;

	if (k == 0u || n == 0u || (k % MMA_TILE) != 0u || (n % MMA_TILE) != 0u) {
		return;
	}

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE   = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE   = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER   = __MMA_B_CONFIG_COL;
	cfg.B_BSWPER  = MMA_TILE;
	cfg.B_BRSTPER = MMA_TILE;

	cfg.C_ATYPE   = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE   = __MMA_C_CONFIG_BTYPE_INT8;
	if (k_tiles == 1u) {
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
		cfg.C_OP0PER     = 0xffffffffu;
		cfg.C_OP1PER     = 0xffffffffu;
	} else {
		cfg.C_OPSTART    = __MMA_C_CONFIG_OPSTART_OPERATION0;
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MULPLUS;
		cfg.C_OP0PER     = 1u;
		cfg.C_OP1PER     = k_tiles - 1u;
	}
	cfg.C_BSWPER   = 1u;
	cfg.C_HWLDDST  = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE = __MMA_C_CONFIG_HWLDTYPE_INT32;
	cfg.C_CRRSTPER = 1u;
	cfg.C_CWRSTPER = 1u;
	cfg.C_CRSWPER  = k_tiles;
	cfg.C_CWSWPER  = k_tiles;

	cfg.X_CTYPE    = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE    = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT    = 0u;
	cfg.X_RE       = 0u;
	cfg.X_SAT      = 0u;
	cfg.X_ReLU     = 0u;
	cfg.X_CRRSTPER = 4u;
	cfg.X_CSWPER   = 4u;
	cfg.PARITYCTRL = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	/* Single contiguous 1D SE1 stream across all n_tiles * k_tiles * 4096 bytes. */
	se_open_1d(1, b_tiled, n * k);

	/* Preload tile (nt=0, kt=0) into B bank 0. */
	for (r = 0u; r < MMA_TILE; r++) {
		__HWALDB(__SE1ADV(uchar64));
		__HWAADV();
	}

	for (nt = 0u; nt < n_tiles; nt++) {
		for (kt = 0u; kt < k_tiles; kt++) {
			uchar64 a_vec = *(__uchar64_ptr)(uint8_t *)(a + kt * MMA_TILE);
			int is_last = (nt == n_tiles - 1u) && (kt == k_tiles - 1u);

			if (!is_last) {
				__HWALDAB(a_vec, __SE1ADV(uchar64));
				__HWAOP(0);
				__HWAADV();
				for (r = 1u; r < MMA_TILE; r++) {
					__HWALDB(__SE1ADV(uchar64));
					__HWAADV();
				}
			} else {
				__HWALDA(a_vec);
				__HWAOP(0);
				__HWAADV();
			}

			if (nt > 0u && kt == 0u) {
				__HWAFENCE_OPXFER();
				for (r = 0u; r < 64u; r++) {
					__HWAADV();
				}
				MMA_XFER_STORE_ROW_I32(y + (nt - 1u) * MMA_TILE);
			}
		}
	}

	__HWAFENCE_OPXFER();
	for (r = 0u; r < 64u; r++) {
		__HWAADV();
	}
	MMA_XFER_STORE_ROW_I32(y + (n_tiles - 1u) * MMA_TILE);

	__SE1_CLOSE();
	__HWACLOSE(0);
}

void c7x_ref_gemm_i8_i32(const int8_t *a, const int8_t *b, int32_t *c,
			 uint32_t m, uint32_t k, uint32_t n,
			 uint32_t b_transpose)
{
	uint32_t i, j, p;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			int32_t acc = 0;

			if (b_transpose) {
				for (p = 0; p < k; p++) {
					acc += (int32_t)a[i * k + p] * (int32_t)b[j * k + p];
				}
			} else {
				for (p = 0; p < k; p++) {
					acc += (int32_t)a[i * k + p] * (int32_t)b[p * n + j];
				}
			}
			c[i * n + j] = acc;
		}
	}
}

/*
 * Packed W4A8 GEMV using C7x 512-bit vector unpack + __dotp4_ext(char64, uchar64):
 *
 * Each byte of w_packed[j * (K/2) + p] holds two signed 4-bit weights:
 *   low  nibble -> signed int4 for k = 2*p
 *   high nibble -> signed int4 for k = 2*p + 1
 *
 * Algebraic trick for zero-overhead unsigned dotp4:
 *   For any 4-bit two's-complement nibble n in 0..15:
 *     signed(n) = (n ^ 8) - 8,  where u = (n ^ 8) is unsigned in 0..15!
 *   So sum_k a[k] * w_s4[k] = sum_k a[k] * u[k] - 8 * sum_k a[k].
 *
 * We split a[0..K-1] into a_even[K/2] and a_odd[K/2] once (O(K), negligible),
 * compute bias = 8 * sum(a) once, and stream w_packed via SE0 with 128 MACs per
 * 64-byte load (two __dotp4_ext(char64, uchar64) ops) and zero byte shuffles.
 */
#define W4A8_MAX_HALF_K 8192u

void c7x_mma_gemv_w4a8(const int8_t *a, const uint8_t *w_packed, int32_t *y,
		       uint32_t k, uint32_t n)
{
	static int8_t a_even[W4A8_MAX_HALF_K] __attribute__((aligned(64)));
	static int8_t a_odd[W4A8_MAX_HALF_K] __attribute__((aligned(64)));
	uint32_t half_k = k >> 1;
	uint32_t j, p;
	int32_t a_sum = 0;
	int32_t bias;
	const ulong8 mask0f = (ulong8)0x0F0F0F0F0F0F0F0Full;
	const ulong8 xor08  = (ulong8)0x0808080808080808ull;
	const ulong8 sh4    = (ulong8)4ull;

	if ((k % 128u) != 0u || half_k > W4A8_MAX_HALF_K) {
		c7x_ref_gemv_w4a8(a, w_packed, y, k, n);
		return;
	}

	for (p = 0; p < half_k; p++) {
		int8_t e = a[2u * p];
		int8_t o = a[2u * p + 1u];

		a_even[p] = e;
		a_odd[p]  = o;
		a_sum += (int32_t)e + (int32_t)o;
	}
	bias = 8 * a_sum;

	se_open_1d(0, w_packed, n * half_k);
	for (j = 0; j < n; j++) {
		int16 acc_v = (int16)0;
		int32_t tmp[16] __attribute__((aligned(64)));
		int32_t sum = -bias;
		uint32_t lane;

		for (p = 0; p < half_k; p += 64u) {
			ulong8 pk = __as_ulong8(__SE0ADV(uchar64));
			/*
			 * Unpack 128 signed 4-bit nibbles via native 64-bit vector ops
			 * (VANDD/VXORD/VSHRUD) — avoids cl7x 1.2.0.STS scalarizing 8-bit ops.
			 */
			uchar64 u_lo = __as_uchar64((pk & mask0f) ^ xor08);
			uchar64 u_hi = __as_uchar64(((pk >> sh4) & mask0f) ^ xor08);
			char64 ae = *(__char64_ptr)(int8_t *)&a_even[p];
			char64 ao = *(__char64_ptr)(int8_t *)&a_odd[p];

			acc_v += __dotp4_ext(ae, u_lo);
			acc_v += __dotp4_ext(ao, u_hi);
		}

		*(__int16_ptr)tmp = acc_v;
		for (lane = 0; lane < 16u; lane++) {
			sum += tmp[lane];
		}
		y[j] = sum;
	}
	__SE0_CLOSE();
}

void c7x_ref_gemv_w4a8(const int8_t *a, const uint8_t *w_packed, int32_t *y,
		       uint32_t k, uint32_t n)
{
	uint32_t j, p;
	uint32_t half_k = k >> 1;

	for (j = 0; j < n; j++) {
		const uint8_t *wj = w_packed + j * half_k;
		int32_t acc = 0;

		for (p = 0; p < half_k; p++) {
			uint8_t byte = wj[p];
			int8_t w0 = (int8_t)(byte << 4) >> 4;
			int8_t w1 = (int8_t)byte >> 4;

			acc += (int32_t)a[2u * p] * (int32_t)w0;
			acc += (int32_t)a[2u * p + 1u] * (int32_t)w1;
		}
		y[j] = acc;
	}
}

/*
 * Zero-borrow SWAR sign-extension of 64 packed 4-bit two's-complement nibbles
 * into two 64-byte signed int8 vectors using pure 64-bit ulong8 instructions
 * (VANDD/VXORD/VORD/VSUBD/VXORD — immune to cl7x 1.2.0.STS 8-bit vector bug).
 */
#define UNPACK_W4_PAIR_S8(pk_u8v, out_lo, out_hi)                                 \
	do {                                                                      \
		const ulong8 _m0f = (ulong8)0x0F0F0F0F0F0F0F0Full;                \
		const ulong8 _x08 = (ulong8)0x0808080808080808ull;                \
		const ulong8 _x80 = (ulong8)0x8080808080808080ull;                \
		const ulong8 _sh4 = (ulong8)4ull;                                 \
		ulong8 _pk = __as_ulong8(pk_u8v);                                 \
		ulong8 _lo = ((((_pk & _m0f) ^ _x08) | _x80) - _x08) ^ _x80;      \
		ulong8 _hi = (((((_pk >> _sh4) & _m0f) ^ _x08) | _x80) - _x08) ^ _x80; \
		(out_lo) = __as_uchar64(_lo);                                     \
		(out_hi) = __as_uchar64(_hi);                                     \
	} while (0)

static inline void se_open_1d_32b_mma(int se, const void *p, uint32_t n_elems)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_1D;
	flags.ELETYPE = __SE_ELETYPE_32BIT;
	flags.VECLEN  = __SE_VECLEN_16ELEMS;

	params = __set_ICNT0(params, n_elems);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

#define MMA_XFER_ACC_ROW_F32_SE0(acc0, acc1, acc2, acc3)                          \
	do {                                                                      \
		int16 _y0, _y1, _y2, _y3;                                         \
		__HWAXFER(0); __HWAADV(); _y0 = __as_int16(__HWARCV(0)); __HWAADV(); \
		__HWAXFER(0); __HWAADV(); _y1 = __as_int16(__HWARCV(0)); __HWAADV(); \
		__HWAXFER(0); __HWAADV(); _y2 = __as_int16(__HWARCV(0)); __HWAADV(); \
		__HWAXFER(0); __HWAADV(); _y3 = __as_int16(__HWARCV(0)); __HWAADV(); \
		(acc0) += __int_to_float(_y0) * __SE0ADV(float16);                \
		(acc1) += __int_to_float(_y1) * __SE0ADV(float16);                \
		(acc2) += __int_to_float(_y2) * __SE0ADV(float16);                \
		(acc3) += __int_to_float(_y3) * __SE0ADV(float16);                \
	} while (0)

static inline float hmax_f16_local(float16 v)
{
	float tmp[16] __attribute__((aligned(64)));
	float m;
	uint32_t i;

	*(__float16_ptr)tmp = v;
	m = tmp[0];
	for (i = 1u; i < 16u; i++) {
		if (tmp[i] > m) {
			m = tmp[i];
		}
	}
	return m;
}

static inline void finalize_w4_tile64(float *out, float16 a0, float16 a1,
				      float16 a2, float16 a3, float a_scale,
				      uint32_t col0, uint32_t out_dim, int add,
				      uint32_t *best_idx, float *best_val, int do_argmax)
{
	float16 as_v = (float16)a_scale;
	float16 v0 = a0 * as_v;
	float16 v1 = a1 * as_v;
	float16 v2 = a2 * as_v;
	float16 v3 = a3 * as_v;

	if (out) {
		if (col0 + 64u <= out_dim) {
			if (add) {
				*(__float16_ptr)(out + col0 + 0u)  += v0;
				*(__float16_ptr)(out + col0 + 16u) += v1;
				*(__float16_ptr)(out + col0 + 32u) += v2;
				*(__float16_ptr)(out + col0 + 48u) += v3;
			} else {
				*(__float16_ptr)(out + col0 + 0u)  = v0;
				*(__float16_ptr)(out + col0 + 16u) = v1;
				*(__float16_ptr)(out + col0 + 32u) = v2;
				*(__float16_ptr)(out + col0 + 48u) = v3;
			}
		} else {
			float tmp[64] __attribute__((aligned(64)));
			uint32_t rem = (out_dim > col0) ? (out_dim - col0) : 0u;
			uint32_t i;

			*(__float16_ptr)(tmp + 0u)  = v0;
			*(__float16_ptr)(tmp + 16u) = v1;
			*(__float16_ptr)(tmp + 32u) = v2;
			*(__float16_ptr)(tmp + 48u) = v3;
			for (i = 0u; i < rem; i++) {
				out[col0 + i] = add ? (out[col0 + i] + tmp[i]) : tmp[i];
			}
		}
	}

	if (do_argmax) {
		float16 vmax = __max(__max(v0, v1), __max(v2, v3));
		float tm = hmax_f16_local(vmax);

		if (tm > *best_val) {
			float tmp[64] __attribute__((aligned(64)));
			uint32_t rem = (col0 + 64u <= out_dim) ? 64u : ((out_dim > col0) ? (out_dim - col0) : 0u);
			uint32_t i;

			*(__float16_ptr)(tmp + 0u)  = v0;
			*(__float16_ptr)(tmp + 16u) = v1;
			*(__float16_ptr)(tmp + 32u) = v2;
			*(__float16_ptr)(tmp + 48u) = v3;
			for (i = 0u; i < rem; i++) {
				if (tmp[i] > *best_val) {
					*best_val = tmp[i];
					*best_idx = col0 + i;
				}
			}
		}
	}
}

/* Fast M == 1 register-resident MMA W4A8 group-64 GEMV (zero y_i32 buffer!). */
static void mma_gemv_w4a8_g64_m1_chunk(float *out, const int8_t *a, float a_scale,
				       const uint8_t *w_packed, const float *w_scales,
				       uint32_t k_pad, uint32_t n_tiles, uint32_t col_base,
				       uint32_t out_dim, int add,
				       uint32_t *best_idx_io, float *best_val_io, int do_argmax)
{
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t k_tiles = k_pad >> 6;
	uint32_t nt, kt, v, r;
	uint32_t best_idx = best_idx_io ? *best_idx_io : 0u;
	float best_val = best_val_io ? *best_val_io : -3.4e38f;
	float16 acc0 = (float16)0.0f, acc1 = (float16)0.0f;
	float16 acc2 = (float16)0.0f, acc3 = (float16)0.0f;

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE      = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE      = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER      = __MMA_B_CONFIG_COL;
	cfg.B_BSWPER     = MMA_TILE;
	cfg.B_BRSTPER    = MMA_TILE;

	cfg.C_ATYPE      = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE      = __MMA_C_CONFIG_BTYPE_INT8;
	cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
	cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
	cfg.C_OP0PER     = 0xffffffffu;
	cfg.C_OP1PER     = 0xffffffffu;
	cfg.C_BSWPER     = 1u;
	cfg.C_HWLDDST    = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE   = __MMA_C_CONFIG_HWLDTYPE_INT32;
	cfg.C_CRRSTPER   = 1u;
	cfg.C_CWRSTPER   = 1u;
	cfg.C_CRSWPER    = 1u;
	cfg.C_CWSWPER    = 1u;

	cfg.X_CTYPE      = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE      = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT      = 0u;
	cfg.X_RE         = 0u;
	cfg.X_SAT        = 0u;
	cfg.X_ReLU       = 0u;
	cfg.X_CRRSTPER   = 4u;
	cfg.X_CSWPER     = 4u;
	cfg.PARITYCTRL   = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	/* Dual contiguous 1D streams: SE1 = packed W4 weights, SE0 = group-64 scales! */
	se_open_1d(1, w_packed, n_tiles * k_tiles * 2048u);
	se_open_1d_32b_mma(0, w_scales, n_tiles * k_tiles * 64u);

	/* Preload tile (nt=0, kt=0): 32 packed 64B loads -> 64 MMA B-bank rows. */
	for (v = 0u; v < 32u; v++) {
		uchar64 r0, r1;

		UNPACK_W4_PAIR_S8(__SE1ADV(uchar64), r0, r1);
		__HWALDB(r0);
		__HWAADV();
		__HWALDB(r1);
		__HWAADV();
	}

	for (nt = 0u; nt < n_tiles; nt++) {
		for (kt = 0u; kt < k_tiles; kt++) {
			uchar64 a_vec = *(__uchar64_ptr)(uint8_t *)(a + (kt << 6));
			int is_last = (nt == n_tiles - 1u) && (kt == k_tiles - 1u);

			if (!is_last) {
				uchar64 r0, r1;

				UNPACK_W4_PAIR_S8(__SE1ADV(uchar64), r0, r1);
				__HWALDAB(a_vec, r0);
				__HWAOP(0);
				__HWAADV();
				__HWALDB(r1);
				__HWAADV();
				for (v = 1u; v < 32u; v++) {
					UNPACK_W4_PAIR_S8(__SE1ADV(uchar64), r0, r1);
					__HWALDB(r0);
					__HWAADV();
					__HWALDB(r1);
					__HWAADV();
				}
			} else {
				__HWALDA(a_vec);
				__HWAOP(0);
				__HWAADV();
			}

			/* Drain previous tile from alternate C bank and accumulate into float16 regs. */
			if (nt > 0u || kt > 0u) {
				uint32_t prev_nt = (kt == 0u) ? (nt - 1u) : nt;

				__HWAFENCE_OPXFER();
				for (r = 0u; r < 64u; r++) {
					__HWAADV();
				}
				MMA_XFER_ACC_ROW_F32_SE0(acc0, acc1, acc2, acc3);

				if (kt == 0u) {
					finalize_w4_tile64(out, acc0, acc1, acc2, acc3, a_scale,
							   col_base + (prev_nt << 6), out_dim, add,
							   &best_idx, &best_val, do_argmax);
					acc0 = (float16)0.0f;
					acc1 = (float16)0.0f;
					acc2 = (float16)0.0f;
					acc3 = (float16)0.0f;
				}
			}
		}
	}

	/* Drain final tile (n_tiles - 1, k_tiles - 1). */
	{
		__HWAFENCE_OPXFER();
		for (r = 0u; r < 64u; r++) {
			__HWAADV();
		}
		MMA_XFER_ACC_ROW_F32_SE0(acc0, acc1, acc2, acc3);
		finalize_w4_tile64(out, acc0, acc1, acc2, acc3, a_scale,
				   col_base + ((n_tiles - 1u) << 6), out_dim, add,
				   &best_idx, &best_val, do_argmax);
	}

	__SE0_CLOSE();
	__SE1_CLOSE();
	__HWACLOSE(0);

	if (best_idx_io) {
		*best_idx_io = best_idx;
	}
	if (best_val_io) {
		*best_val_io = best_val;
	}
}

/* Batched M > 1 MMA W4A8 group-64 GEMM (block_m up to 60 rows per C-bank!). */
static void mma_gemm_w4a8_g64_mblock_chunk(float *out_blk, const int8_t *a_blk, const float *a_scales_blk,
					   const uint8_t *w_packed, const float *w_scales,
					   uint32_t m_cur, uint32_t k_pad, uint32_t n_tiles,
					   uint32_t col_base, uint32_t out_dim, int add)
{
	static float tile_acc[64][64] __attribute__((aligned(64)));
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t k_tiles = k_pad >> 6;
	uint32_t nt, kt, v, r;

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE      = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE      = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER      = __MMA_B_CONFIG_COL;
	cfg.B_BSWPER     = MMA_TILE;
	cfg.B_BRSTPER    = MMA_TILE;

	cfg.C_ATYPE      = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE      = __MMA_C_CONFIG_BTYPE_INT8;
	cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
	cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
	cfg.C_OP0PER     = 0xffffffffu;
	cfg.C_OP1PER     = 0xffffffffu;
	cfg.C_BSWPER     = m_cur;
	cfg.C_HWLDDST    = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE   = __MMA_C_CONFIG_HWLDTYPE_INT32;
	cfg.C_CRRSTPER   = m_cur;
	cfg.C_CWRSTPER   = m_cur;
	cfg.C_CRSWPER    = m_cur;
	cfg.C_CWSWPER    = m_cur;

	cfg.X_CTYPE      = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE      = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT      = 0u;
	cfg.X_RE         = 0u;
	cfg.X_SAT        = 0u;
	cfg.X_ReLU       = 0u;
	cfg.X_CRRSTPER   = (m_cur * 4u) & 0xffu;
	cfg.X_CSWPER     = m_cur * 4u;
	cfg.PARITYCTRL   = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	se_open_1d(1, w_packed, n_tiles * k_tiles * 2048u);

	/* Preload tile (0, 0) */
	for (v = 0u; v < 32u; v++) {
		uchar64 r0, r1;

		UNPACK_W4_PAIR_S8(__SE1ADV(uchar64), r0, r1);
		__HWALDB(r0);
		__HWAADV();
		__HWALDB(r1);
		__HWAADV();
	}

	memset(tile_acc, 0, m_cur * 64u * sizeof(float));

	for (nt = 0u; nt < n_tiles; nt++) {
		for (kt = 0u; kt < k_tiles; kt++) {
			int is_last = (nt == n_tiles - 1u) && (kt == k_tiles - 1u);

			se_open_2d(0, a_blk + (kt << 6), MMA_TILE, m_cur, (int32_t)k_pad);
			if (!is_last) {
				uint32_t b_loaded = 0u;

				for (v = 0u; v < 32u; v++) {
					uchar64 b0, b1;

					UNPACK_W4_PAIR_S8(__SE1ADV(uchar64), b0, b1);
					if (b_loaded < m_cur) {
						__HWALDAB(__SE0ADV(uchar64), b0);
						__HWAOP(0);
					} else {
						__HWALDB(b0);
					}
					__HWAADV();
					b_loaded++;

					if (b_loaded < m_cur) {
						__HWALDAB(__SE0ADV(uchar64), b1);
						__HWAOP(0);
					} else {
						__HWALDB(b1);
					}
					__HWAADV();
					b_loaded++;
				}
			} else {
				for (r = 0u; r < m_cur; r++) {
					__HWALDA(__SE0ADV(uchar64));
					__HWAOP(0);
					__HWAADV();
				}
			}
			__SE0_CLOSE();

			if (nt > 0u || kt > 0u) {
				uint32_t prev_nt = (kt == 0u) ? (nt - 1u) : nt;
				uint32_t prev_kt = (kt == 0u) ? (k_tiles - 1u) : (kt - 1u);
				const float *sc = w_scales + ((size_t)prev_nt * k_tiles + prev_kt) * 64u;
				float16 s0 = *(__float16_ptr)(float *)(sc + 0u);
				float16 s1 = *(__float16_ptr)(float *)(sc + 16u);
				float16 s2 = *(__float16_ptr)(float *)(sc + 32u);
				float16 s3 = *(__float16_ptr)(float *)(sc + 48u);

				__HWAFENCE_OPXFER();
				for (r = 0u; r < 64u; r++) {
					__HWAADV();
				}
				for (r = 0u; r < m_cur; r++) {
					float *dst = tile_acc[r];
					int16 y0, y1, y2, y3;

					__HWAXFER(0); __HWAADV(); y0 = __as_int16(__HWARCV(0)); __HWAADV();
					__HWAXFER(0); __HWAADV(); y1 = __as_int16(__HWARCV(0)); __HWAADV();
					__HWAXFER(0); __HWAADV(); y2 = __as_int16(__HWARCV(0)); __HWAADV();
					__HWAXFER(0); __HWAADV(); y3 = __as_int16(__HWARCV(0)); __HWAADV();
					*(__float16_ptr)(dst + 0u)  += __int_to_float(y0) * s0;
					*(__float16_ptr)(dst + 16u) += __int_to_float(y1) * s1;
					*(__float16_ptr)(dst + 32u) += __int_to_float(y2) * s2;
					*(__float16_ptr)(dst + 48u) += __int_to_float(y3) * s3;
				}
				if (kt == 0u) {
					for (r = 0u; r < m_cur; r++) {
						float *src = tile_acc[r];
						finalize_w4_tile64(out_blk + r * out_dim,
								   *(__float16_ptr)(src + 0u),
								   *(__float16_ptr)(src + 16u),
								   *(__float16_ptr)(src + 32u),
								   *(__float16_ptr)(src + 48u),
								   a_scales_blk[r], col_base + (prev_nt << 6), out_dim, add,
								   NULL, NULL, 0);
					}
					memset(tile_acc, 0, m_cur * 64u * sizeof(float));
				}
			}
		}
	}

	/* Drain final tile (n_tiles - 1, k_tiles - 1). */
	{
		const float *sc = w_scales + ((size_t)(n_tiles - 1u) * k_tiles + (k_tiles - 1u)) * 64u;
		float16 s0 = *(__float16_ptr)(float *)(sc + 0u);
		float16 s1 = *(__float16_ptr)(float *)(sc + 16u);
		float16 s2 = *(__float16_ptr)(float *)(sc + 32u);
		float16 s3 = *(__float16_ptr)(float *)(sc + 48u);

		__HWAFENCE_OPXFER();
		for (r = 0u; r < 64u; r++) {
			__HWAADV();
		}
		for (r = 0u; r < m_cur; r++) {
			float *dst = tile_acc[r];
			int16 y0, y1, y2, y3;

			__HWAXFER(0); __HWAADV(); y0 = __as_int16(__HWARCV(0)); __HWAADV();
			__HWAXFER(0); __HWAADV(); y1 = __as_int16(__HWARCV(0)); __HWAADV();
			__HWAXFER(0); __HWAADV(); y2 = __as_int16(__HWARCV(0)); __HWAADV();
			__HWAXFER(0); __HWAADV(); y3 = __as_int16(__HWARCV(0)); __HWAADV();
			*(__float16_ptr)(dst + 0u)  += __int_to_float(y0) * s0;
			*(__float16_ptr)(dst + 16u) += __int_to_float(y1) * s1;
			*(__float16_ptr)(dst + 32u) += __int_to_float(y2) * s2;
			*(__float16_ptr)(dst + 48u) += __int_to_float(y3) * s3;
			finalize_w4_tile64(out_blk + r * out_dim,
					   *(__float16_ptr)(dst + 0u),
					   *(__float16_ptr)(dst + 16u),
					   *(__float16_ptr)(dst + 32u),
					   *(__float16_ptr)(dst + 48u),
					   a_scales_blk[r], col_base + ((n_tiles - 1u) << 6), out_dim, add,
					   NULL, NULL, 0);
		}
	}

	__SE1_CLOSE();
	__HWACLOSE(0);
}

void c7x_mma_gemm_w4a8_g64_f32(float *out, const int8_t *a, const float *a_scales,
			       const uint8_t *w_packed, const float *w_scales,
			       uint32_t m, uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
			       uint32_t block_m, int add, uint32_t *argmax_out)
{
	uint32_t n_tiles = n_pad >> 6;
	uint32_t k_tiles = k_pad >> 6;
	uint32_t m0;

	if (m == 0u || k_pad == 0u || n_pad == 0u || (k_pad % 64u) != 0u || (n_pad % 64u) != 0u) {
		return;
	}
	if (m == 1u) {
		uint32_t best_idx = 0u;
		float best_val = -3.4e38f;
		int do_argmax = (argmax_out != NULL);

#if !defined(__C7X_HOSTEM__) && !defined(_HOST_BUILD)
		if (g_dru_l2_buf != NULL && c7x_dru_is_ok() &&
		    (uintptr_t)w_packed >= 0xA8000000ull && n_tiles >= 4u) {
			uint32_t tile_w = k_tiles * 2048u;
			uint32_t tile_s = k_tiles * 256u;
			uint64_t total_b = (uint64_t)n_tiles * (uint64_t)(tile_w + tile_s);
			uint32_t half_b = (g_dru_l2_bytes >> 1) & ~63u;
			uint32_t chunk_nt = half_b / (tile_w + tile_s);

			/* Only use DRU ping-pong on large DDR matrices (>= 1 MiB) where DMA saturates LPDDR4. */
			if (total_b >= 1048576ull && chunk_nt >= 1u) {
				uint8_t *slot_buf[2];
				uint32_t nt0 = 0u, slot = 0u, cur_nt;

				if (chunk_nt > 8u) {
					chunk_nt = 8u;
				}
				slot_buf[0] = g_dru_l2_buf;
				slot_buf[1] = g_dru_l2_buf + half_b;

				/* Fast 1-tile prologue so MMA starts within ~2 us while chunk 1 prefetches. */
				cur_nt = 1u;
				c7x_dru_submit_pair_bytes(
					0u,
					(uint64_t)(uintptr_t)slot_buf[0],
					(uint64_t)(uintptr_t)w_packed,
					cur_nt * tile_w,
					(uint64_t)(uintptr_t)(slot_buf[0] + cur_nt * tile_w),
					(uint64_t)(uintptr_t)w_scales,
					cur_nt * tile_s);

				while (nt0 < n_tiles) {
					uint32_t next_nt0 = nt0 + cur_nt;
					uint32_t next_cnt = 0u;
					const uint8_t *lw = slot_buf[slot];
					const float   *ls = (const float *)(slot_buf[slot] + cur_nt * tile_w);

					c7x_dru_wait(0u);
					if (next_nt0 < n_tiles) {
						next_cnt = n_tiles - next_nt0;
						if (next_cnt > chunk_nt) {
							next_cnt = chunk_nt;
						}
						c7x_dru_submit_pair_bytes(
							0u,
							(uint64_t)(uintptr_t)slot_buf[slot ^ 1u],
							(uint64_t)(uintptr_t)(w_packed + (size_t)next_nt0 * tile_w),
							next_cnt * tile_w,
							(uint64_t)(uintptr_t)(slot_buf[slot ^ 1u] + next_cnt * tile_w),
							(uint64_t)(uintptr_t)(w_scales + (size_t)next_nt0 * k_tiles * 64u),
							next_cnt * tile_s);
					}
					mma_gemv_w4a8_g64_m1_chunk(out, a, a_scales[0], lw, ls,
								   k_pad, cur_nt, nt0 << 6, out_dim, add,
								   &best_idx, &best_val, do_argmax);
					nt0 = next_nt0;
					cur_nt = next_cnt;
					slot ^= 1u;
				}
				if (argmax_out) {
					*argmax_out = best_idx;
				}
				return;
			}
		}
#endif
		mma_gemv_w4a8_g64_m1_chunk(out, a, a_scales[0], w_packed, w_scales,
					   k_pad, n_tiles, 0u, out_dim, add,
					   &best_idx, &best_val, do_argmax);
		if (argmax_out) {
			*argmax_out = best_idx;
		}
		return;
	}
	if (block_m == 0u || block_m > 63u) {
#ifdef __C7X_HOSTEM__
		block_m = 16u;
#else
		block_m = 60u;
#endif
	}
	for (m0 = 0u; m0 < m; m0 += block_m) {
		uint32_t m_cur = (m - m0 < block_m) ? (m - m0) : block_m;

#if !defined(__C7X_HOSTEM__) && !defined(_HOST_BUILD)
		if (g_dru_l2_buf != NULL && c7x_dru_is_ok() &&
		    (uintptr_t)w_packed >= 0xA8000000ull && n_tiles >= 4u) {
			uint32_t tile_w = k_tiles * 2048u;
			uint64_t total_b = (uint64_t)n_tiles * (uint64_t)tile_w;
			uint32_t half_b = (g_dru_l2_bytes >> 1) & ~63u;
			uint32_t chunk_nt = half_b / tile_w;

			if (total_b >= 1048576ull && chunk_nt >= 1u) {
				uint8_t *slot_buf[2];
				uint32_t nt0 = 0u, slot = 0u, cur_nt;

				if (chunk_nt > 8u) {
					chunk_nt = 8u;
				}
				slot_buf[0] = g_dru_l2_buf;
				slot_buf[1] = g_dru_l2_buf + half_b;

				cur_nt = 1u;
				c7x_dru_submit_bytes(0u, (uint64_t)(uintptr_t)slot_buf[0],
						     (uint64_t)(uintptr_t)w_packed,
						     cur_nt * tile_w);

				while (nt0 < n_tiles) {
					uint32_t next_nt0 = nt0 + cur_nt;
					uint32_t next_cnt = 0u;
					const uint8_t *lw = slot_buf[slot];
					const float   *ws_chunk = w_scales + (size_t)nt0 * k_tiles * 64u;

					c7x_dru_wait(0u);
					if (next_nt0 < n_tiles) {
						next_cnt = n_tiles - next_nt0;
						if (next_cnt > chunk_nt) {
							next_cnt = chunk_nt;
						}
						c7x_dru_submit_bytes(
							0u,
							(uint64_t)(uintptr_t)slot_buf[slot ^ 1u],
							(uint64_t)(uintptr_t)(w_packed + (size_t)next_nt0 * tile_w),
							next_cnt * tile_w);
					}
					mma_gemm_w4a8_g64_mblock_chunk(
						out + m0 * out_dim, a + m0 * k_pad, a_scales + m0,
						lw, ws_chunk, m_cur, k_pad, cur_nt,
						nt0 << 6, out_dim, add);
					nt0 = next_nt0;
					cur_nt = next_cnt;
					slot ^= 1u;
				}
				continue;
			}
		}
#endif
		mma_gemm_w4a8_g64_mblock_chunk(out + m0 * out_dim, a + m0 * k_pad, a_scales + m0,
					       w_packed, w_scales, m_cur, k_pad, n_tiles,
					       0u, out_dim, add);
	}
}

void c7x_ref_gemm_w4a8_g64_f32(float *out, const int8_t *a, const float *a_scales,
			       const uint8_t *w_packed, const float *w_scales,
			       uint32_t m, uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
			       int add, uint32_t *argmax_out)
{
	uint32_t k_tiles = k_pad >> 6;
	uint32_t n_tiles = n_pad >> 6;
	uint32_t i, nt, kt, r, c;
	uint32_t best_idx = 0u;
	float best_val = -3.4e38f;

	for (i = 0u; i < m; i++) {
		const int8_t *ai = a + i * k_pad;
		float as = a_scales[i];

		for (nt = 0u; nt < n_tiles; nt++) {
			for (r = 0u; r < 64u; r++) {
				uint32_t row = (nt << 6) + r;
				float acc = 0.0f;

				if (row >= out_dim) {
					continue;
				}
				for (kt = 0u; kt < k_tiles; kt++) {
					const uint8_t *tile = w_packed + ((size_t)nt * k_tiles + kt) * 2048u;
					const uint8_t *vec = tile + (r >> 1) * 64u;
					float sc = w_scales[((size_t)nt * k_tiles + kt) * 64u + r];
					int32_t dot = 0;

					for (c = 0u; c < 64u; c++) {
						uint8_t byte = vec[c];
						int8_t w = (r & 1u)
							   ? ((int8_t)byte >> 4)
							   : ((int8_t)(byte << 4) >> 4);
						dot += (int32_t)ai[(kt << 6) + c] * (int32_t)w;
					}
					acc += (float)dot * sc;
				}
				acc *= as;
				if (out) {
					out[i * out_dim + row] = add ? (out[i * out_dim + row] + acc) : acc;
				}
				if (argmax_out && i == 0u && acc > best_val) {
					best_val = acc;
					best_idx = row;
				}
			}
		}
	}
	if (argmax_out) {
		*argmax_out = best_idx;
	}
}

/* 1D-streamed tile-major W8A8 MMA GEMM for arbitrary m >= 1 (batched prefill). */
static void mma_gemm_i8_i32_tiled_mblock(const int8_t *a_blk, const int8_t *b_tiled,
					 int32_t *c_blk, uint32_t m_cur,
					 uint32_t k, uint32_t n)
{
	__HWA_CONFIG_REG cfg;
	__HWA_OFFSET_REG off;
	uint32_t k_tiles = k / MMA_TILE;
	uint32_t n_tiles = n / MMA_TILE;
	uint32_t nt, kt, r;

	memset(&cfg, 0, sizeof(cfg));
	memset(&off, 0, sizeof(off));

	cfg.A_ATYPE   = __MMA_A_CONFIG_ATYPE_INT8;
	cfg.B_BTYPE   = __MMA_B_CONFIG_SIZE8;
	cfg.B_ORDER   = __MMA_B_CONFIG_COL;
	cfg.B_BSWPER  = MMA_TILE;
	cfg.B_BRSTPER = MMA_TILE;

	cfg.C_ATYPE   = __MMA_C_CONFIG_ATYPE_SA;
	cfg.C_BTYPE   = __MMA_C_CONFIG_BTYPE_INT8;
	if (k_tiles == 1u) {
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MUL;
		cfg.C_OP0PER     = 0xffffffffu;
		cfg.C_OP1PER     = 0xffffffffu;
	} else {
		cfg.C_OPSTART    = __MMA_C_CONFIG_OPSTART_OPERATION0;
		cfg.C_OPERATION0 = __MMA_C_CONFIG_MUL;
		cfg.C_OPERATION1 = __MMA_C_CONFIG_MULPLUS;
		cfg.C_OP0PER     = m_cur;
		cfg.C_OP1PER     = (k_tiles - 1u) * m_cur;
	}
	cfg.C_BSWPER   = m_cur;
	cfg.C_HWLDDST  = __MMA_C_CONFIG_HWLDDST_X1;
	cfg.C_HWLDTYPE = __MMA_C_CONFIG_HWLDTYPE_INT32;
	cfg.C_CRRSTPER = m_cur;
	cfg.C_CWRSTPER = m_cur;
	cfg.C_CRSWPER  = k_tiles * m_cur;
	cfg.C_CWSWPER  = k_tiles * m_cur;

	cfg.X_CTYPE    = __MMA_X_CONFIG_CTYPE_INT32;
	cfg.X_XTYPE    = __MMA_X_CONFIG_XTYPE_INT32;
	cfg.X_SHIFT    = 0u;
	cfg.X_RE       = 0u;
	cfg.X_SAT      = 0u;
	cfg.X_ReLU     = 0u;
	cfg.X_CRRSTPER = (m_cur * 4u) & 0xffu;
	cfg.X_CSWPER   = m_cur * 4u;
	cfg.PARITYCTRL = __MMA_NORMAL;

	__HWAOPEN(cfg, off, __MMA_OPEN_FSM_RESET);
	__HWAADV();

	se_open_1d(1, b_tiled, n * k);
	for (r = 0u; r < MMA_TILE; r++) {
		__HWALDB(__SE1ADV(uchar64));
		__HWAADV();
	}

	for (nt = 0u; nt < n_tiles; nt++) {
		for (kt = 0u; kt < k_tiles; kt++) {
			int is_last = (nt == n_tiles - 1u) && (kt == k_tiles - 1u);

			se_open_2d(0, a_blk + kt * MMA_TILE, MMA_TILE, m_cur, (int32_t)k);
			if (!is_last) {
				for (r = 0u; r < m_cur; r++) {
					__HWALDAB(__SE0ADV(uchar64), __SE1ADV(uchar64));
					__HWAOP(0);
					__HWAADV();
				}
				for (r = m_cur; r < MMA_TILE; r++) {
					__HWALDB(__SE1ADV(uchar64));
					__HWAADV();
				}
			} else {
				for (r = 0u; r < m_cur; r++) {
					__HWALDA(__SE0ADV(uchar64));
					__HWAOP(0);
					__HWAADV();
				}
			}
			__SE0_CLOSE();

			if (nt > 0u && kt == 0u) {
				int32_t *c_prev = c_blk + (nt - 1u) * MMA_TILE;

				__HWAFENCE_OPXFER();
				for (r = 0u; r < 64u; r++) {
					__HWAADV();
				}
				for (r = 0u; r < m_cur; r++) {
					MMA_XFER_STORE_ROW_I32(c_prev + r * n);
				}
			}
		}
	}

	{
		int32_t *c_last = c_blk + (n_tiles - 1u) * MMA_TILE;

		__HWAFENCE_OPXFER();
		for (r = 0u; r < 64u; r++) {
			__HWAADV();
		}
		for (r = 0u; r < m_cur; r++) {
			MMA_XFER_STORE_ROW_I32(c_last + r * n);
		}
	}

	__SE1_CLOSE();
	__HWACLOSE(0);
}

void c7x_mma_gemm_i8_i32_tiled(const int8_t *a, const int8_t *b_tiled, int32_t *c,
			       uint32_t m, uint32_t k, uint32_t n, uint32_t block_m)
{
	uint32_t m0;

	if (m == 0u || k == 0u || n == 0u || (k % MMA_TILE) != 0u || (n % MMA_TILE) != 0u) {
		return;
	}
	if (m == 1u) {
		c7x_mma_gemv_i8_i32_tiled(a, b_tiled, c, k, n);
		return;
	}
	if (block_m == 0u || block_m > 63u) {
#ifdef __C7X_HOSTEM__
		block_m = 16u;
#else
		block_m = 60u;
#endif
	}
	for (m0 = 0u; m0 < m; m0 += block_m) {
		uint32_t m_cur = (m - m0 < block_m) ? (m - m0) : block_m;

		mma_gemm_i8_i32_tiled_mblock(a + m0 * k, b_tiled, c + m0 * n, m_cur, k, n);
	}
}

uint64_t c7x_mem_bw_read(const void *src, uint32_t bytes)
{
	uint32_t nvec = bytes / 64u;
	uint32_t i;
	ulong8 acc0 = (ulong8)0;
	ulong8 acc1 = (ulong8)0;
	uint64_t tmp[8];
	uint64_t out = 0;

	if (nvec == 0u) {
		return 0;
	}
	se_open_1d(0, src, nvec * 64u);
	for (i = 0; i + 1u < nvec; i += 2u) {
		acc0 += __as_ulong8(__SE0ADV(uchar64));
		acc1 += __as_ulong8(__SE0ADV(uchar64));
	}
	if (i < nvec) {
		acc0 += __as_ulong8(__SE0ADV(uchar64));
	}
	__SE0_CLOSE();
	acc0 += acc1;
	*(__ulong8_ptr)tmp = acc0;
	for (i = 0; i < 8u; i++) {
		out ^= tmp[i];
	}
	return out;
}

void c7x_mem_bw_copy(void *dst, const void *src, uint32_t bytes)
{
	uint32_t nvec = bytes / 64u;
	uint32_t i;
	uint8_t *d = (uint8_t *)dst;

	if (nvec == 0u) {
		return;
	}
	se_open_1d(0, src, nvec * 64u);
	for (i = 0; i < nvec; i++) {
		*(__uchar64_ptr)(d + i * 64u) = __SE0ADV(uchar64);
	}
	__SE0_CLOSE();
}
