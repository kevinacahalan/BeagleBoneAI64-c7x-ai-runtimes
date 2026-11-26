/* SPDX-License-Identifier: MIT */
/*
 * C7x Command-Buffer Executor (C7X_OP_EXEC_LIST).
 *
 * Executes a flat array of `struct c7x_cmd` tensor operations in one DSP RPC
 * with L2 scratchpad reuse, DRU0 L2 ping-pong double buffering, and zero
 * intermediate cache flushes.
 *
 * Compiles identically for:
 *   - TI C7000 CGT (cl7x -mv7100 -O3)
 *   - x86_64 C7000 Host Emulation (g++ -D__C7X_HOSTEM__ -D_HOST_BUILD)
 */

#ifdef __C7X_HOSTEM__
#include "c7x_host_emulation.h"
#else
#include <c7x.h>
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cmd_exec.h"
#include "mma_kernels.h"
#include "vec_kernels.h"
#if !defined(__C7X_HOSTEM__) && !defined(_HOST_BUILD)
#include "mmu.h"
#endif

#define CMD_ALIGN64(x) (((x) + 63u) & ~63u)

static inline __STRM_TEMPLATE cmd_strm_zero(void)
{
#ifdef __C7X_HOSTEM__
	return __uint16(0);
#else
	return (__uint16)(0);
#endif
}

static inline void cmd_se_open_1d_f32(int se, const void *p, uint32_t n_elems)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = cmd_strm_zero();

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

static inline void cmd_se_open_2d_f32(int se, const void *p, uint32_t icnt0,
				      uint32_t icnt1, int32_t dim1)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = cmd_strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_2D;
	flags.ELETYPE = __SE_ELETYPE_32BIT;
	flags.VECLEN  = __SE_VECLEN_16ELEMS;

	params = __set_ICNT0(params, icnt0);
	params = __set_ICNT1_DIM1(params, icnt1, dim1);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

static inline void cmd_se_open_1d_u16_u32(int se, const void *p, uint32_t n_elems)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = cmd_strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_1D;
	flags.ELETYPE = __SE_ELETYPE_16BIT;
	flags.PROMOTE = __SE_PROMOTE_2X_ZEROEXT;
	flags.VECLEN  = __SE_VECLEN_16ELEMS;

	params = __set_ICNT0(params, n_elems);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

static inline void cmd_se_open_2d_u16_u32(int se, const void *p, uint32_t icnt0,
					  uint32_t icnt1, int32_t dim1)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = cmd_strm_zero();

	flags.DIMFMT  = __SE_DIMFMT_2D;
	flags.ELETYPE = __SE_ELETYPE_16BIT;
	flags.PROMOTE = __SE_PROMOTE_2X_ZEROEXT;
	flags.VECLEN  = __SE_VECLEN_16ELEMS;

	params = __set_ICNT0(params, icnt0);
	params = __set_ICNT1_DIM1(params, icnt1, dim1);
	params = __set_SE_FLAGS(params, &flags, NULL);

	if (se) {
		__SE1_OPEN(p, params);
	} else {
		__SE0_OPEN(p, params);
	}
}

/* Exact 16-wide branchless IEEE-754 binary16 (zero-extended uint16) -> binary32. */
static inline float16 cmd_vf16_to_f32(uint16 h)
{
	uint16 sign = (h & (uint16)0x8000u) << (uint16)16;
	uint16 exp  = (h >> (uint16)10) & (uint16)0x1Fu;
	uint16 mant = h & (uint16)0x03FFu;

	uint16 norm_bits = sign | ((exp + (uint16)112u) << (uint16)23) | (mant << (uint16)13);
	uint16 inf_bits  = sign | (uint16)0x7F800000u | (mant << (uint16)13);
	float16 sub_f    = __int_to_float(__as_int16(mant)) * (float16)5.9604644775390625e-8f;
	uint16 sub_bits  = __as_uint16(sub_f) | sign;

	__vpred is_zero_sub = __cmp_eq_pred(exp, (uint16)0u);
	__vpred is_inf_nan  = __cmp_eq_pred(exp, (uint16)31u);
	uint16 res = __select(is_zero_sub, sub_bits, norm_bits);
	res = __select(is_inf_nan, inf_bits, res);
	return __as_float16(res);
}

/*
 * 16-wide branchless sin/cos using 3-part Cody-Waite pi/2 reduction and
 * Cephes degree-7/8 minimax polynomials (~1e-6 max error for |x| <= 1e5,
 * where quadrant index q < 2^16 keeps q * pi2_hi exact in IEEE-754 float32).
 */
static inline void cmd_vsincos_store_f16(float16 x, float16 mscale,
					 float *s_dst, float *c_dst)
{
	const float16 inv_pi2 = (float16)0.63661977236758134308f;
	const float16 pi2_hi  = (float16)1.5703125f;
	const float16 pi2_mid = (float16)4.837512969970703125e-4f;
	const float16 pi2_lo  = (float16)7.549789948768648e-8f;
	int16 q = __float_to_int(x * inv_pi2);
	float16 qf = __int_to_float(q);
	float16 r = ((x - qf * pi2_hi) - qf * pi2_mid) - qf * pi2_lo;
	float16 r2 = r * r;
	float16 sr = r + r * r2 * ((float16)(-1.6666654611e-1f) +
		     r2 * ((float16)8.3321608736e-3f + r2 * (float16)(-1.9515295891e-4f)));
	float16 cr = (float16)1.0f + r2 * ((float16)(-0.5f) +
		     r2 * ((float16)4.166664568298827e-2f +
		     r2 * ((float16)(-1.388731625493765e-3f) +
		     r2 * (float16)2.443315711809948e-5f)));
	uint16 qu = __as_uint16(q);
	__vpred swap = __cmp_eq_pred(qu & (uint16)1u, (uint16)1u);
	float16 s0 = __select(swap, cr, sr);
	float16 c0 = __select(swap, sr, cr);
	uint16 s_sign = (qu & (uint16)2u) << (uint16)30;
	uint16 c_sign = ((qu + (uint16)1u) & (uint16)2u) << (uint16)30;

	*(__float16_ptr)s_dst = __as_float16(__as_uint16(s0) ^ s_sign) * mscale;
	*(__float16_ptr)c_dst = __as_float16(__as_uint16(c0) ^ c_sign) * mscale;
}

static inline float cmd_hmax_f16(float16 v)
{
	float tmp[16] __attribute__((aligned(64)));
	float m;
	uint32_t i;

	*(__float16_ptr)tmp = v;
	m = tmp[0];
	for (i = 1u; i < 16u; i++) {
		if (tmp[i] > m) m = tmp[i];
	}
	return m;
}

static inline float cmd_hsum_f16(float16 v)
{
	float tmp[16] __attribute__((aligned(64)));
	float s0, s1, s2, s3;

	*(__float16_ptr)tmp = v;
	s0 = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
	s1 = (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
	s2 = (tmp[8] + tmp[9]) + (tmp[10] + tmp[11]);
	s3 = (tmp[12] + tmp[13]) + (tmp[14] + tmp[15]);
	return (s0 + s1) + (s2 + s3);
}

/* 16-wide FP32 exp(x), clamped to +/-87.3365 to avoid +Inf -> NaN in recip. */
static inline float16 cmd_vexpf_f16(float16 x)
{
	const float16 c_lo    = (float16)(-87.33654475f);
	const float16 c_hi    = (float16)(87.33654475f);
	const float16 inv_ln2 = (float16)(1.4426950408889634f);
	const float16 ln2_hi  = (float16)(0.693359375f);
	const float16 ln2_lo  = (float16)(-2.12194440e-4f);
	const float16 c1      = (float16)(1.0f);
	const float16 c2      = (float16)(0.5f);
	const float16 c3      = (float16)(0.1666666716f);
	const float16 c4      = (float16)(0.0416666679f);
	const float16 c5      = (float16)(0.0083333338f);
	const float16 c6      = (float16)(0.0013888889f);
	float16 xc, kf, r, p, pow2k;
	int16 k, exp_bits;

	xc = __min(__max(x, c_lo), c_hi);
	k  = __float_to_int(xc * inv_ln2);
	kf = __int_to_float(k);
	r  = (xc - kf * ln2_hi) - kf * ln2_lo;

	p = c6 * r + c5;
	p = p * r + c4;
	p = p * r + c3;
	p = p * r + c2;
	p = p * r + c1;
	p = p * r + c1;

	exp_bits = (k + (int16)127) << (int16)23;
	pow2k = __as_float16(exp_bits);
	return __select(__cmp_lt_pred(x, (float16)(-88.0f)), (float16)0.0f, p * pow2k);
}

static inline float16 cmd_vrecip_f16(float16 d)
{
	const float16 two = (float16)2.0f;
	float16 r = __recip(d);

	r = r * (two - d * r);
	r = r * (two - d * r);
	return r;
}

static inline float cmd_expf_scalar(float x)
{
	float16 v = cmd_vexpf_f16((float16)x);
	float tmp[16] __attribute__((aligned(64)));

	*(__float16_ptr)tmp = v;
	return tmp[0];
}

/* Exact IEEE-754 binary16 <-> binary32 conversion (zero external LUTs). */
static inline float cmd_f16_to_f32(uint16_t h)
{
	uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
	uint32_t exp  = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x03FFu;
	uint32_t bits;
	float f;

	if (exp == 0u) {
		if (mant == 0u) {
			bits = sign;
		} else {
			/* Subnormal */
			uint32_t e = 127u - 14u;
			while ((mant & 0x0400u) == 0u) {
				mant <<= 1;
				e--;
			}
			mant &= 0x03FFu;
			bits = sign | (e << 23) | (mant << 13);
		}
	} else if (exp == 31u) {
		bits = sign | 0x7F800000u | (mant << 13);
	} else {
		bits = sign | ((exp + 112u) << 23) | (mant << 13);
	}
	memcpy(&f, &bits, sizeof(f));
	return f;
}

static inline uint16_t cmd_f32_to_f16(float f)
{
	uint32_t bits;
	uint32_t sign, exp, mant;
	int32_t new_exp;

	memcpy(&bits, &f, sizeof(bits));
	sign = (bits >> 16) & 0x8000u;
	exp  = (bits >> 23) & 0xFFu;
	mant = bits & 0x007FFFFFu;

	if (exp == 255u) {
		/* Inf or NaN */
		if (mant != 0u) {
			return (uint16_t)(sign | 0x7E00u | (mant >> 13));
		}
		return (uint16_t)(sign | 0x7C00u);
	}
	new_exp = (int32_t)exp - 112;
	if (new_exp >= 31) {
		return (uint16_t)(sign | 0x7C00u);
	}
	if (new_exp <= 0) {
		if (new_exp < -10) {
			return (uint16_t)sign;
		}
		mant = (mant | 0x00800000u) >> (uint32_t)(1 - new_exp);
		/* Round to nearest even */
		mant += 0x00000FFFu + ((mant >> 13) & 1u);
		return (uint16_t)(sign | (mant >> 13));
	}
	/* Normal with round to nearest even */
	mant += 0x00000FFFu + ((mant >> 13) & 1u);
	if (mant & 0x00800000u) {
		mant = 0u;
		new_exp++;
		if (new_exp >= 31) {
			return (uint16_t)(sign | 0x7C00u);
		}
	}
	return (uint16_t)(sign | ((uint32_t)new_exp << 10) | (mant >> 13));
}

/* Fast 16-wide IEEE-754 binary32 -> binary16 (round-to-nearest-even). */
static inline void cmd_vf32_to_f16_store(float16 fv, uint16_t *dst)
{
	uint16 bits = __as_uint16(fv);
	uint16 sign = (bits >> (uint16)16) & (uint16)0x8000u;
	uint16 exp  = (bits >> (uint16)23) & (uint16)0xFFu;
	uint16 mant = bits & (uint16)0x007FFFFFu;
	uint16 round = (uint16)0x00000FFFu + ((mant >> (uint16)13) & (uint16)1u);
	uint16 m_rnd = mant + round;
	uint16 carry = (m_rnd >> (uint16)23) & (uint16)1u;
	uint16 e_norm = (exp - (uint16)112u) + carry;
	uint16 h_norm = sign | (e_norm << (uint16)10) | ((m_rnd & (uint16)0x007FFFFFu) >> (uint16)13);
	uint32_t tmp[16] __attribute__((aligned(64)));
	float ftmp[16] __attribute__((aligned(64)));
	uint32_t etmp[16] __attribute__((aligned(64)));
	uint32_t i;

	*(__uint16_ptr)tmp   = h_norm;
	*(__float16_ptr)ftmp = fv;
	*(__uint16_ptr)etmp  = exp;
	for (i = 0u; i < 16u; i++) {
		uint32_t e = etmp[i];
		dst[i] = (e <= 112u || e >= 142u) ? cmd_f32_to_f16(ftmp[i]) : (uint16_t)tmp[i];
	}
}

/* GGML Q8_0 (34 bytes per 32 elements) and Q4_0 (18 bytes per 32 elements) */
static void dequant_row_q8_0(const uint8_t *src, float *dst, uint32_t n)
{
	uint32_t nb = n / 32u;
	uint32_t b, j;
	int32_t itmp[16] __attribute__((aligned(64)));
	float ftmp[16]   __attribute__((aligned(64)));

	for (b = 0u; b < nb; b++) {
		const uint8_t *blk = src + (uint64_t)b * 34ull;
		uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
		float d = cmd_f16_to_f32(dh);
		float16 dv = (float16)d;
		const int8_t *qs = (const int8_t *)(blk + 2u);
		float *out = dst + b * 32u;
		uint32_t half;

		for (half = 0u; half < 32u; half += 16u) {
			for (j = 0u; j < 16u; j++) {
				itmp[j] = (int32_t)qs[half + j];
			}
			{
				float16 fv = __int_to_float(*(__int16_ptr)itmp) * dv;
				float *dp = out + half;
				if ((((uintptr_t)dp) & 63u) == 0u) {
					*(__float16_ptr)dp = fv;
				} else {
					*(__float16_ptr)ftmp = fv;
					memcpy(dp, ftmp, 16u * sizeof(float));
				}
			}
		}
	}
}

static float dot_q8_0_f32(const float *a, const uint8_t *b_q8, uint32_t k)
{
	uint32_t nb = k / 32u;
	uint32_t b, j;
	float sum = 0.0f;
	int32_t itmp[16] __attribute__((aligned(64)));
	float ftmp[16]   __attribute__((aligned(64)));

	for (b = 0u; b < nb; b++) {
		const uint8_t *blk = b_q8 + (uint64_t)b * 34ull;
		uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
		float d = cmd_f16_to_f32(dh);
		const int8_t *qs = (const int8_t *)(blk + 2u);
		const float *ap = a + b * 32u;
		float16 acc = (float16)0.0f;
		uint32_t half;

		for (half = 0u; half < 32u; half += 16u) {
			for (j = 0u; j < 16u; j++) {
				itmp[j] = (int32_t)qs[half + j];
			}
			memcpy(ftmp, ap + half, 16u * sizeof(float));
			acc += *(__float16_ptr)ftmp * __int_to_float(*(__int16_ptr)itmp);
		}
		sum += cmd_hsum_f16(acc) * d;
	}
	return sum;
}

static void quant_row_q8_0(const float *src, uint8_t *dst, uint32_t n)
{
	uint32_t nb = n / 32u;
	uint32_t b, j;
	float ftmp0[16]  __attribute__((aligned(64)));
	float ftmp1[16]  __attribute__((aligned(64)));
	int32_t qtmp[16] __attribute__((aligned(64)));

	for (b = 0u; b < nb; b++) {
		const float *in = src + b * 32u;
		uint8_t *blk = dst + (uint64_t)b * 34ull;
		float amax, d, id;
		uint16_t dh;
		int8_t *qs = (int8_t *)(blk + 2u);
		float16 v0, v1, av;

		memcpy(ftmp0, in,       16u * sizeof(float));
		memcpy(ftmp1, in + 16u, 16u * sizeof(float));
		v0 = *(__float16_ptr)ftmp0;
		v1 = *(__float16_ptr)ftmp1;
		av = __max(__max(v0, -v0), __max(v1, -v1));
		amax = cmd_hmax_f16(av);

		d = amax / 127.0f;
		id = (d != 0.0f) ? (1.0f / d) : 0.0f;
		dh = cmd_f32_to_f16(d);
		blk[0] = (uint8_t)(dh & 0xFFu);
		blk[1] = (uint8_t)(dh >> 8);
		{
			float16 idv = (float16)id;
			int16 q0 = __min(__max(__float_to_int(v0 * idv), (int16)(-127)), (int16)127);
			int16 q1 = __min(__max(__float_to_int(v1 * idv), (int16)(-127)), (int16)127);
			*(__int16_ptr)qtmp = q0;
			for (j = 0u; j < 16u; j++) qs[j] = (int8_t)qtmp[j];
			*(__int16_ptr)qtmp = q1;
			for (j = 0u; j < 16u; j++) qs[16u + j] = (int8_t)qtmp[j];
		}
	}
}

static void dequant_row_q4_0(const uint8_t *src, float *dst, uint32_t n)
{
	uint32_t nb = n / 32u;
	uint32_t b, j;

	for (b = 0u; b < nb; b++) {
		const uint8_t *blk = src + (uint64_t)b * 18ull;
		uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
		float d = cmd_f16_to_f32(dh);
		const uint8_t *qs = blk + 2u;
		float *out = dst + b * 32u;

		for (j = 0u; j < 16u; j++) {
			int32_t q0 = (int32_t)(qs[j] & 0x0Fu) - 8;
			int32_t q1 = (int32_t)(qs[j] >> 4) - 8;

			out[j]       = (float)q0 * d;
			out[j + 16u] = (float)q1 * d;
		}
	}
}

/* -------------------------------------------------------------------------- */
/* Individual command implementations                                         */
/* -------------------------------------------------------------------------- */

static int exec_matmul_w4a8_g64(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const uint8_t *w_packed = (const uint8_t *)(uintptr_t)cmd->src0;
	const float   *a_f32    = (const float *)(uintptr_t)cmd->src1;
	const float   *w_scales = (const float *)(uintptr_t)cmd->src2;
	float         *out      = (float *)(uintptr_t)cmd->dst;
	uint32_t m = cmd->p0;
	uint32_t k = cmd->p1;
	uint32_t n = cmd->p2;
	uint32_t k_pad = CMD_ALIGN64(k);
	uint32_t n_pad = CMD_ALIGN64(n);
	uint32_t a_stride = cmd->p4 ? cmd->p4 : k;
	uint32_t o_stride = cmd->p5 ? cmd->p5 : n;
	int add = (cmd->flags & C7X_CMD_FLAG_ADD) ? 1 : 0;
	uint32_t max_bm, bm_req, row_off;
	uint32_t dru_reserve, avail;
	uint8_t *l2 = (uint8_t *)l2_scratch;

	if (!w_packed || !a_f32 || !w_scales || !out || m == 0u || k == 0u || n == 0u) {
		return -1;
	}

#ifdef __C7X_HOSTEM__
	max_bm = 16u;
#else
	max_bm = 60u;
#endif
	bm_req = cmd->p3;
	if (bm_req > 0u && bm_req < max_bm) {
		max_bm = bm_req;
	}

	/* Reserve L2 tail for DRU0 ping-pong if we have room (on silicon). */
	dru_reserve = (l2_bytes >= 262144u) ? 196608u : 0u;
	avail = (l2_bytes > dru_reserve) ? (l2_bytes - dru_reserve) : l2_bytes;
	/* Each chunk needs ALIGN64(bm * k_pad) + ALIGN64(bm * 4) bytes (64B margin). */
	if (avail < CMD_ALIGN64(k_pad) + 64u) {
		dru_reserve = 0u;
		avail = l2_bytes;
	}
	if (avail < CMD_ALIGN64(k_pad) + 64u) {
		return -2; /* L2 scratch too small for even 1 row */
	}
	{
		uint32_t per_row = k_pad + (uint32_t)sizeof(float);
		uint32_t fit = (avail - 64u) / per_row;
		if (fit == 0u) fit = 1u;
		if (fit < max_bm) max_bm = fit;
		while (max_bm > 1u &&
		       CMD_ALIGN64(max_bm * k_pad) + CMD_ALIGN64(max_bm * (uint32_t)sizeof(float)) > avail) {
			max_bm--;
		}
	}

	for (row_off = 0u; row_off < m; row_off += max_bm) {
		uint32_t bm = m - row_off;
		uint32_t a_q_bytes, a_s_bytes, used, r;
		int8_t *a_q;
		float  *a_s;

		if (bm > max_bm) bm = max_bm;
		a_q_bytes = CMD_ALIGN64(bm * k_pad);
		a_s_bytes = CMD_ALIGN64(bm * (uint32_t)sizeof(float));
		a_q = (int8_t *)l2;
		a_s = (float *)(l2 + a_q_bytes);
		used = a_q_bytes + a_s_bytes;

		for (r = 0u; r < bm; r++) {
			c7x_quantize_f32_i8(a_q + (uint64_t)r * k_pad,
					    a_f32 + (uint64_t)(row_off + r) * a_stride,
					    k, k_pad, &a_s[r]);
		}

		if ((cmd->flags & C7X_CMD_FLAG_USE_DRU) && l2_bytes > used + 4096u) {
			c7x_mma_set_dru_l2_buf(l2 + used, l2_bytes - used);
		} else {
			c7x_mma_set_dru_l2_buf(NULL, 0u);
		}

		if (o_stride == n || bm == 1u) {
			c7x_mma_gemm_w4a8_g64_f32(out + (uint64_t)row_off * o_stride,
						  a_q, a_s, w_packed, w_scales,
						  bm, k_pad, n_pad, n, bm, add, NULL);
		} else {
			/* Strided output rows: run row-by-row if o_stride != n. */
			for (r = 0u; r < bm; r++) {
				c7x_mma_gemm_w4a8_g64_f32(out + (uint64_t)(row_off + r) * o_stride,
							  a_q + (uint64_t)r * k_pad, &a_s[r],
							  w_packed, w_scales,
							  1u, k_pad, n_pad, n, 1u, add, NULL);
			}
		}
	}

	c7x_mma_set_dru_l2_buf(NULL, 0u);
	return 0;
}

static int exec_matmul_w8a8(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const int8_t *w_tiled  = (const int8_t *)(uintptr_t)cmd->src0;
	const float  *a_f32    = (const float *)(uintptr_t)cmd->src1;
	const float  *w_scales = (const float *)(uintptr_t)cmd->src2;
	float        *out      = (float *)(uintptr_t)cmd->dst;
	uint32_t m = cmd->p0;
	uint32_t k = cmd->p1;
	uint32_t n = cmd->p2;
	uint32_t k_pad = CMD_ALIGN64(k);
	uint32_t n_pad = CMD_ALIGN64(n);
	uint32_t a_stride = cmd->p4 ? cmd->p4 : k;
	uint32_t o_stride = cmd->p5 ? cmd->p5 : n;
	int add = (cmd->flags & C7X_CMD_FLAG_ADD) ? 1 : 0;
	uint32_t max_bm, bm_req, row_off;
	uint8_t *l2 = (uint8_t *)l2_scratch;

	if (!w_tiled || !a_f32 || !w_scales || !out || m == 0u || k == 0u || n == 0u) {
		return -1;
	}

#ifdef __C7X_HOSTEM__
	max_bm = 16u;
#else
	max_bm = 60u;
#endif
	bm_req = cmd->p3;
	if (bm_req > 0u && bm_req < max_bm) {
		max_bm = bm_req;
	}

	/* Choose bm and n_chunk so a_q[bm*k_pad] + y_i32[bm*n_chunk] fit in l2_bytes. */
	if (l2_bytes < CMD_ALIGN64(k_pad) + 64u * (uint32_t)sizeof(int32_t) + 128u) {
		return -2;
	}

	/* Cap bm so a_q + a_s + at least one 64-col y_i32 tile per row fit in l2_bytes. */
	{
		uint32_t per_row_min = k_pad + 64u * (uint32_t)sizeof(int32_t) + (uint32_t)sizeof(float);
		uint32_t fit_bm = (l2_bytes - 128u) / per_row_min;
		if (fit_bm == 0u) fit_bm = 1u;
		if (fit_bm < max_bm) max_bm = fit_bm;
		while (max_bm > 1u &&
		       CMD_ALIGN64(max_bm * k_pad) + CMD_ALIGN64(max_bm * (uint32_t)sizeof(float)) +
		       max_bm * 64u * (uint32_t)sizeof(int32_t) > l2_bytes) {
			max_bm--;
		}
	}

	for (row_off = 0u; row_off < m; row_off += max_bm) {
		uint32_t bm = m - row_off;
		uint32_t a_q_bytes, a_s_bytes, y_avail, n_chunk, n_off, r;
		int8_t  *a_q;
		float   *a_s;
		int32_t *y_i32;

		if (bm > max_bm) bm = max_bm;
		a_q_bytes = CMD_ALIGN64(bm * k_pad);
		a_s_bytes = CMD_ALIGN64(bm * (uint32_t)sizeof(float));
		a_q   = (int8_t *)l2;
		a_s   = (float *)(l2 + a_q_bytes);
		y_i32 = (int32_t *)(l2 + a_q_bytes + a_s_bytes);
		y_avail = l2_bytes - (a_q_bytes + a_s_bytes);

		n_chunk = (y_avail / (bm * (uint32_t)sizeof(int32_t))) & ~63u;
		if (n_chunk == 0u) {
			return -2;
		}
		if (n_chunk > n_pad) {
			n_chunk = n_pad;
		}

		for (r = 0u; r < bm; r++) {
			c7x_quantize_f32_i8(a_q + (uint64_t)r * k_pad,
					    a_f32 + (uint64_t)(row_off + r) * a_stride,
					    k, k_pad, &a_s[r]);
		}

		for (n_off = 0u; n_off < n_pad; n_off += n_chunk) {
			uint32_t nc = n_pad - n_off;
			uint32_t nc_valid;
			const int8_t *w_sub;

			if (nc > n_chunk) nc = n_chunk;
			nc_valid = (n > n_off) ? (n - n_off) : 0u;
			if (nc_valid > nc) nc_valid = nc;
			if (nc_valid == 0u) break;

			w_sub = w_tiled + (uint64_t)(n_off >> 6) * (uint64_t)(k_pad >> 6) * 4096ull;
			if (bm == 1u) {
				c7x_mma_gemv_i8_i32_tiled(a_q, w_sub, y_i32, k_pad, nc);
			} else {
				c7x_mma_gemm_i8_i32_tiled(a_q, w_sub, y_i32, bm, k_pad, nc, bm);
			}

			for (r = 0u; r < bm; r++) {
				float *dst_row = out + (uint64_t)(row_off + r) * o_stride + n_off;
				const int32_t *yr = y_i32 + (uint64_t)r * nc;
				if (add) {
					c7x_dequant_add_i32_f32(dst_row, yr, w_scales + n_off,
								a_s[r], nc_valid);
				} else {
					c7x_dequant_i32_f32(dst_row, yr, w_scales + n_off,
							    a_s[r], nc_valid);
				}
			}
		}
	}
	return 0;
}

static float dot_f32_f32(const float *a, const float *b, uint32_t k)
{
	uint32_t k_vec = k & ~15u;
	uint32_t i = 0u;
	float sum = 0.0f;

	if (k_vec >= 16u) {
		float16 acc = (float16)0.0f;
		cmd_se_open_1d_f32(0, a, k_vec);
		cmd_se_open_1d_f32(1, b, k_vec);
		for (; i < k_vec; i += 16u) {
			float16 va = __SE0ADV(float16);
			float16 vb = __SE1ADV(float16);
			acc += va * vb;
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
		sum = cmd_hsum_f16(acc);
	}
	for (; i < k; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

static float dot_f16_f32(const float *a, const uint16_t *b16, uint32_t k)
{
	uint32_t k_vec = k & ~15u;
	uint32_t i = 0u;
	float sum = 0.0f;

	if (k_vec >= 16u) {
		float16 acc = (float16)0.0f;
		cmd_se_open_1d_u16_u32(0, b16, k_vec);
		cmd_se_open_1d_f32(1, a, k_vec);
		for (; i < k_vec; i += 16u) {
			float16 vb = cmd_vf16_to_f32(__SE0ADV(uint16));
			float16 va = __SE1ADV(float16);
			acc += va * vb;
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
		sum = cmd_hsum_f16(acc);
	}
	for (; i < k; i++) {
		sum += a[i] * cmd_f16_to_f32(b16[i]);
	}
	return sum;
}

/*
 * Writeback + invalidate L1D after writing to L2 SRAM scratchpad via CPU stores.
 * Needed before any later command in the same EXEC_LIST uses DRU0 DMA into L2 SRAM
 * (DRU0 is non-coherent with L1D, so stale dirty L1D lines must not later evict
 * over DRU0-fetched weights).
 */
static inline void cmd_l1d_wbinv_after_l2_scratch(void)
{
#if !defined(__C7X_HOSTEM__) && !defined(_HOST_BUILD)
	__memory_fence(0x1);
	__memory_fence(0x1);
	ai64_l1d_wbinv_all();
	__memory_fence(0x1);
	__memory_fence(0x1);
#endif
}

/*
 * L2-staged F32 dot kernel over a 64B-aligned L2 matrix B_l2[n][k_v * 16] and
 * 64B-aligned L2 vector a_l2[k_v * 16]. Uses SE0 (56 GB/s L2 SRAM streaming,
 * zero L1D thrash) after a single L1D writeback per KV head.
 */
static void l2_matvec_f32(const float *a_l2, const float *b_l2,
			  uint32_t k_v, uint32_t n, float *o)
{
	uint32_t in, v;
	uint32_t stride = k_v << 4;

	if (k_v == 1u) {
		float16 a0 = *(__float16_ptr)(float *)a_l2;
		cmd_se_open_1d_f32(0, b_l2, n * 16u);
		for (in = 0u; in < n; in++) {
			o[in] = cmd_hsum_f16(a0 * __SE0ADV(float16));
		}
		__SE0_CLOSE();
		return;
	}
	if (k_v == 2u) {
		float16 a0 = *(__float16_ptr)(float *)a_l2;
		float16 a1 = *(__float16_ptr)(float *)(a_l2 + 16u);
		cmd_se_open_1d_f32(0, b_l2, n * 32u);
		for (in = 0u; in < n; in++) {
			float16 acc = a0 * __SE0ADV(float16);
			acc += a1 * __SE0ADV(float16);
			o[in] = cmd_hsum_f16(acc);
		}
		__SE0_CLOSE();
		return;
	}
	if (k_v == 4u) {
		float16 a0 = *(__float16_ptr)(float *)a_l2;
		float16 a1 = *(__float16_ptr)(float *)(a_l2 + 16u);
		float16 a2 = *(__float16_ptr)(float *)(a_l2 + 32u);
		float16 a3 = *(__float16_ptr)(float *)(a_l2 + 48u);
		cmd_se_open_1d_f32(0, b_l2, n * 64u);
		for (in = 0u; in < n; in++) {
			float16 acc = a0 * __SE0ADV(float16);
			acc += a1 * __SE0ADV(float16);
			acc += a2 * __SE0ADV(float16);
			acc += a3 * __SE0ADV(float16);
			o[in] = cmd_hsum_f16(acc);
		}
		__SE0_CLOSE();
		return;
	}
	if (k_v == 8u) {
		float16 a0 = *(__float16_ptr)(float *)a_l2;
		float16 a1 = *(__float16_ptr)(float *)(a_l2 + 16u);
		float16 a2 = *(__float16_ptr)(float *)(a_l2 + 32u);
		float16 a3 = *(__float16_ptr)(float *)(a_l2 + 48u);
		float16 a4 = *(__float16_ptr)(float *)(a_l2 + 64u);
		float16 a5 = *(__float16_ptr)(float *)(a_l2 + 80u);
		float16 a6 = *(__float16_ptr)(float *)(a_l2 + 96u);
		float16 a7 = *(__float16_ptr)(float *)(a_l2 + 112u);
		cmd_se_open_1d_f32(0, b_l2, n * 128u);
		for (in = 0u; in < n; in++) {
			float16 acc0 = a0 * __SE0ADV(float16);
			float16 acc1 = a1 * __SE0ADV(float16);
			acc0 += a2 * __SE0ADV(float16);
			acc1 += a3 * __SE0ADV(float16);
			acc0 += a4 * __SE0ADV(float16);
			acc1 += a5 * __SE0ADV(float16);
			acc0 += a6 * __SE0ADV(float16);
			acc1 += a7 * __SE0ADV(float16);
			o[in] = cmd_hsum_f16(acc0 + acc1);
		}
		__SE0_CLOSE();
		return;
	}
	cmd_se_open_1d_f32(0, b_l2, n * stride);
	for (in = 0u; in < n; in++) {
		float16 acc = (float16)0.0f;
		for (v = 0u; v < k_v; v++) {
			acc += *(__float16_ptr)(float *)(a_l2 + (v << 4)) * __SE0ADV(float16);
		}
		o[in] = cmd_hsum_f16(acc);
	}
	__SE0_CLOSE();
}

/* Stage n rows x k_copy elements of DDR/MSMC F16, Q8_0, or F32 into 64B-aligned L2 F32 [n][k_pad16]. */
static void stage_src0_to_l2(const uint8_t *p0_base, uint64_t nb01,
			     uint32_t n, uint32_t k_copy, uint32_t k_pad16,
			     uint32_t k_row_cap, int src0_fmt, float *b_l2)
{
	uint32_t k_vec = k_copy & ~15u;
	uint32_t in, ik;

	if (src0_fmt == 2) {
		/* GGML Q8_0 (34 bytes per 32 elements): dequantize only active 32-element blocks. */
		uint32_t k_q32 = (k_copy + 31u) & ~31u;
		if (k_q32 > k_row_cap) k_q32 = k_row_cap;
		for (in = 0u; in < n; in++) {
			const uint8_t *sq = p0_base + (uint64_t)in * nb01;
			float *dst = b_l2 + (uint64_t)in * k_pad16;
			if (k_q32 <= k_pad16) {
				dequant_row_q8_0(sq, dst, k_q32);
				for (ik = k_copy; ik < k_pad16; ik++) {
					dst[ik] = 0.0f;
				}
			} else {
				/* Last partial 32-elem block straddles k_pad16 (k_copy <= k_pad16 < k_q32). */
				float blk32[32] __attribute__((aligned(64)));
				uint32_t full = k_q32 - 32u;
				if (full > 0u) {
					dequant_row_q8_0(sq, dst, full);
				}
				dequant_row_q8_0(sq + (uint64_t)(full / 32u) * 34ull, blk32, 32u);
				for (ik = full; ik < k_copy; ik++) {
					dst[ik] = blk32[ik - full];
				}
				for (ik = k_copy; ik < k_pad16; ik++) {
					dst[ik] = 0.0f;
				}
			}
		}
		return;
	}

	if (src0_fmt == 1) {
		if (k_pad16 > 0u && k_pad16 <= k_row_cap && (nb01 % sizeof(uint16_t)) == 0u &&
		    nb01 <= 0x7FFFFFFFull) {
			/*
			 * Single 2D SE0 decode for the full k_pad16 width (valid since k_pad16 <= k_row_cap),
			 * then zero any [k_copy..k_pad16-1] pad lanes in L1D/L2 (NaN-safe for uninit KV).
			 */
			cmd_se_open_2d_u16_u32(0, p0_base, k_pad16, n, (int32_t)(nb01 / sizeof(uint16_t)));
			for (in = 0u; in < n; in++) {
				float *dst = b_l2 + (uint64_t)in * k_pad16;
				for (ik = 0u; ik < k_pad16; ik += 16u) {
					*(__float16_ptr)&dst[ik] = cmd_vf16_to_f32(__SE0ADV(uint16));
				}
				for (ik = k_copy; ik < k_pad16; ik++) {
					dst[ik] = 0.0f;
				}
			}
			__SE0_CLOSE();
			return;
		}
		for (in = 0u; in < n; in++) {
			const uint16_t *s16 = (const uint16_t *)(p0_base + (uint64_t)in * nb01);
			float *dst = b_l2 + (uint64_t)in * k_pad16;
			ik = 0u;
			if (k_vec >= 16u) {
				cmd_se_open_1d_u16_u32(0, s16, k_vec);
				for (; ik < k_vec; ik += 16u) {
					*(__float16_ptr)&dst[ik] = cmd_vf16_to_f32(__SE0ADV(uint16));
				}
				__SE0_CLOSE();
			}
			for (; ik < k_copy; ik++) {
				dst[ik] = cmd_f16_to_f32(s16[ik]);
			}
			for (; ik < k_pad16; ik++) {
				dst[ik] = 0.0f;
			}
		}
	} else {
		if (k_pad16 > 0u && k_pad16 <= k_row_cap && (nb01 % sizeof(float)) == 0u &&
		    nb01 <= 0x7FFFFFFFull) {
			cmd_se_open_2d_f32(0, p0_base, k_pad16, n, (int32_t)(nb01 / sizeof(float)));
			for (in = 0u; in < n; in++) {
				float *dst = b_l2 + (uint64_t)in * k_pad16;
				for (ik = 0u; ik < k_pad16; ik += 16u) {
					*(__float16_ptr)&dst[ik] = __SE0ADV(float16);
				}
				for (ik = k_copy; ik < k_pad16; ik++) {
					dst[ik] = 0.0f;
				}
			}
			__SE0_CLOSE();
			return;
		}
		for (in = 0u; in < n; in++) {
			const float *sf = (const float *)(p0_base + (uint64_t)in * nb01);
			float *dst = b_l2 + (uint64_t)in * k_pad16;
			memcpy(dst, sf, (size_t)k_copy * sizeof(float));
			for (ik = k_copy; ik < k_pad16; ik++) {
				dst[ik] = 0.0f;
			}
		}
	}
}

static int exec_matmul_f32(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *src1 = (const uint8_t *)(uintptr_t)cmd->src1;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_matmul_f32_desc *d =
		(const struct c7x_matmul_f32_desc *)(uintptr_t)cmd->src3;
	int src0_f16  = (cmd->flags & C7X_CMD_FLAG_SRC0_F16)  ? 1 : 0;
	int src0_q8_0 = (cmd->flags & C7X_CMD_FLAG_SRC0_Q8_0) ? 1 : 0;
	int src0_fmt  = src0_q8_0 ? 2 : (src0_f16 ? 1 : 0);
	uint32_t elem0 = src0_q8_0 ? 34u : (src0_f16 ? (uint32_t)sizeof(uint16_t) : (uint32_t)sizeof(float));
	uint32_t k, n, m, b2, b3, r2, r3;
	uint32_t i3, i02, r_i2, im, in;

	if (!src0 || !src1 || !dst || !d) {
		return -1;
	}
	k  = d->ne0[0];
	n  = d->ne0[1];
	m  = d->ne1[1];
	b2 = d->ne1[2];
	b3 = d->ne1[3];
	if (d->ne1[0] != k || d->ne0[2] == 0u || d->ne0[3] == 0u) {
		return -2;
	}
	if (src0_q8_0 && ((k & 31u) != 0u || d->nb0[0] != 34ull)) {
		return -3;
	}
	r2 = b2 / d->ne0[2];
	r3 = b3 / d->ne0[3];
	if (r2 == 0u) r2 = 1u;
	if (r3 == 0u) r3 = 1u;

	/*
	 * Fast L2-staged path for contiguous dim-0 src0/src1/dst (covers Q*K^T and P*V!):
	 * Group all r2 GQA query heads sharing KV head i02, stage active src0[n][k_max]
	 * into L2 SRAM ONCE per KV head, then compute all r2 * m queries from L1D/L2 SRAM!
	 */
	if (l2_scratch && l2_bytes >= 4096u &&
	    d->nb0[0] == elem0 && d->nb1[0] == sizeof(float) && d->nb_dst[0] == sizeof(float) &&
	    n > 0u && k > 0u) {
		for (i3 = 0u; i3 < b3; i3++) {
			uint32_t i03 = i3 / r3;
			for (i02 = 0u; i02 < d->ne0[2]; i02++) {
				const uint8_t *p0_base = src0 + (uint64_t)i03 * d->nb0[3] + (uint64_t)i02 * d->nb0[2];
				uint32_t k_max = 0u;
				uint32_t k_pad16;
				uint64_t b_bytes, a_bytes;

				/* Find max active (non-trailing-zero) K across all r2*m queries for this KV head. */
				for (r_i2 = 0u; r_i2 < r2; r_i2++) {
					uint32_t i2 = i02 * r2 + r_i2;
					const uint8_t *p1_base;
					if (i2 >= b2) break;
					p1_base = src1 + (uint64_t)i3 * d->nb1[3] + (uint64_t)i2 * d->nb1[2];
					for (im = 0u; im < m; im++) {
						const float *a = (const float *)(p1_base + (uint64_t)im * d->nb1[1]);
						uint32_t ke = k;
						while (ke > k_max && a[ke - 1u] == 0.0f) {
							ke--;
						}
						if (ke > k_max) {
							k_max = ke;
						}
					}
				}
				if (k_max == 0u) {
					for (r_i2 = 0u; r_i2 < r2; r_i2++) {
						uint32_t i2 = i02 * r2 + r_i2;
						uint8_t *pd_base;
						if (i2 >= b2) break;
						pd_base = dst + (uint64_t)i3 * d->nb_dst[3] + (uint64_t)i2 * d->nb_dst[2];
						for (im = 0u; im < m; im++) {
							memset(pd_base + (uint64_t)im * d->nb_dst[1], 0, (size_t)n * sizeof(float));
						}
					}
					continue;
				}

				k_pad16 = (k_max + 15u) & ~15u;
				a_bytes = (uint64_t)k_pad16 * sizeof(float);
				b_bytes = (uint64_t)n * (uint64_t)k_pad16 * sizeof(float);

				if (a_bytes + b_bytes <= (uint64_t)l2_bytes) {
					float *a_l2 = (float *)l2_scratch;
					float *b_l2 = (float *)((uint8_t *)l2_scratch + a_bytes);
					uint32_t k_v = k_pad16 >> 4;

					stage_src0_to_l2(p0_base, d->nb0[1], n, k_max, k_pad16, k, src0_fmt, b_l2);

					for (r_i2 = 0u; r_i2 < r2; r_i2++) {
						uint32_t i2 = i02 * r2 + r_i2;
						const uint8_t *p1_base;
						uint8_t *pd_base;
						if (i2 >= b2) break;
						p1_base = src1 + (uint64_t)i3 * d->nb1[3] + (uint64_t)i2 * d->nb1[2];
						pd_base = dst  + (uint64_t)i3 * d->nb_dst[3] + (uint64_t)i2 * d->nb_dst[2];
						for (im = 0u; im < m; im++) {
							const float *a = (const float *)(p1_base + (uint64_t)im * d->nb1[1]);
							float       *o = (float *)(pd_base + (uint64_t)im * d->nb_dst[1]);
							uint32_t ik;
							memcpy(a_l2, a, (size_t)k_max * sizeof(float));
							for (ik = k_max; ik < k_pad16; ik++) {
								a_l2[ik] = 0.0f;
							}
							l2_matvec_f32(a_l2, b_l2, k_v, n, o);
						}
					}
					continue;
				}

				/* Fallback when src0 slice exceeds L2 SRAM: tile n into L2-sized strips. */
				if (a_bytes + (uint64_t)k_pad16 * sizeof(float) <= (uint64_t)l2_bytes) {
					float *a_l2 = (float *)l2_scratch;
					float *b_l2 = (float *)((uint8_t *)l2_scratch + a_bytes);
					uint32_t k_v = k_pad16 >> 4;
					uint32_t n_chunk = (uint32_t)(((uint64_t)l2_bytes - a_bytes) /
								      ((uint64_t)k_pad16 * sizeof(float)));
					uint32_t n0;

					for (n0 = 0u; n0 < n; n0 += n_chunk) {
						uint32_t nc = (n - n0 < n_chunk) ? (n - n0) : n_chunk;
						stage_src0_to_l2(p0_base + (uint64_t)n0 * d->nb0[1],
								 d->nb0[1], nc, k_max, k_pad16, k, src0_fmt, b_l2);
						for (r_i2 = 0u; r_i2 < r2; r_i2++) {
							uint32_t i2 = i02 * r2 + r_i2;
							const uint8_t *p1_base;
							uint8_t *pd_base;
							if (i2 >= b2) break;
							p1_base = src1 + (uint64_t)i3 * d->nb1[3] + (uint64_t)i2 * d->nb1[2];
							pd_base = dst  + (uint64_t)i3 * d->nb_dst[3] + (uint64_t)i2 * d->nb_dst[2];
							for (im = 0u; im < m; im++) {
								const float *a = (const float *)(p1_base + (uint64_t)im * d->nb1[1]);
								float       *o = (float *)(pd_base + (uint64_t)im * d->nb_dst[1]) + n0;
								uint32_t ik;
								memcpy(a_l2, a, (size_t)k_max * sizeof(float));
								for (ik = k_max; ik < k_pad16; ik++) {
									a_l2[ik] = 0.0f;
								}
								l2_matvec_f32(a_l2, b_l2, k_v, nc, o);
							}
						}
					}
					continue;
				}

				/* Ultra-wide K (> 200 KiB per row): direct streaming fallback. */
				for (r_i2 = 0u; r_i2 < r2; r_i2++) {
					uint32_t i2 = i02 * r2 + r_i2;
					const uint8_t *p1_base;
					uint8_t *pd_base;
					if (i2 >= b2) break;
					p1_base = src1 + (uint64_t)i3 * d->nb1[3] + (uint64_t)i2 * d->nb1[2];
					pd_base = dst  + (uint64_t)i3 * d->nb_dst[3] + (uint64_t)i2 * d->nb_dst[2];
					for (im = 0u; im < m; im++) {
						const float *a = (const float *)(p1_base + (uint64_t)im * d->nb1[1]);
						float       *o = (float *)(pd_base + (uint64_t)im * d->nb_dst[1]);
						if (src0_q8_0) {
							for (in = 0u; in < n; in++) {
								o[in] = dot_q8_0_f32(a, p0_base + (uint64_t)in * d->nb0[1], k);
							}
						} else if (src0_f16) {
							for (in = 0u; in < n; in++) {
								o[in] = dot_f16_f32(a, (const uint16_t *)(p0_base + (uint64_t)in * d->nb0[1]), k_max);
							}
						} else {
							for (in = 0u; in < n; in++) {
								o[in] = dot_f32_f32(a, (const float *)(p0_base + (uint64_t)in * d->nb0[1]), k_max);
							}
						}
					}
				}
			}
		}
		return 0;
	}

	if (src0_q8_0) {
		return -4;
	}

	for (i3 = 0u; i3 < b3; i3++) {
		uint32_t i03 = i3 / r3;
		uint32_t i2;
		for (i2 = 0u; i2 < b2; i2++) {
			uint32_t i02_s = i2 / r2;
			const uint8_t *p0_base = src0 + (uint64_t)i03 * d->nb0[3] + (uint64_t)i02_s * d->nb0[2];
			const uint8_t *p1_base = src1 + (uint64_t)i3  * d->nb1[3] + (uint64_t)i2  * d->nb1[2];
			uint8_t       *pd_base = dst  + (uint64_t)i3  * d->nb_dst[3] + (uint64_t)i2 * d->nb_dst[2];

			for (im = 0u; im < m; im++) {
				/* General strided fallback */
				for (in = 0u; in < n; in++) {
					const uint8_t *brow = p0_base + (uint64_t)in * d->nb0[1];
					const uint8_t *arow = p1_base + (uint64_t)im * d->nb1[1];
					float acc = 0.0f;
					uint32_t ik;
					for (ik = 0u; ik < k; ik++) {
						float av = *(const float *)(arow + (uint64_t)ik * d->nb1[0]);
						float bv = src0_f16
							   ? cmd_f16_to_f32(*(const uint16_t *)(brow + (uint64_t)ik * d->nb0[0]))
							   : *(const float *)(brow + (uint64_t)ik * d->nb0[0]);
						acc += av * bv;
					}
					*(float *)(pd_base + (uint64_t)im * d->nb_dst[1] + (uint64_t)in * d->nb_dst[0]) = acc;
				}
			}
		}
	}
	return 0;
}

static void rmsnorm_unweighted_row_f32(float *out, const float *x, uint32_t n, float eps)
{
	float16 ss_v = (float16)0.0f;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;
	float ss, inv;
	float16 inv_v;

	if (n_vec > 0u) {
		cmd_se_open_1d_f32(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 xv = __SE0ADV(float16);
			ss_v += xv * xv;
		}
		__SE0_CLOSE();
	}
	ss = cmd_hsum_f16(ss_v);
	for (; i < n; i++) {
		ss += x[i] * x[i];
	}

	inv = 1.0f / sqrtf(ss / (float)n + eps);
	inv_v = (float16)inv;

	i = 0u;
	if (n_vec > 0u) {
		cmd_se_open_1d_f32(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 xv = __SE0ADV(float16);
			*(__float16_ptr)&out[i] = xv * inv_v;
		}
		__SE0_CLOSE();
	}
	for (; i < n; i++) {
		out[i] = x[i] * inv;
	}
}

static int exec_rms_norm_f32(const struct c7x_cmd *cmd)
{
	const float *src0   = (const float *)(uintptr_t)cmd->src0;
	const float *weight = cmd->src1 ? (const float *)(uintptr_t)cmd->src1 : NULL;
	float       *dst    = (float *)(uintptr_t)cmd->dst;
	uint32_t n      = cmd->p0;
	uint32_t n_rows = cmd->p1 ? cmd->p1 : 1u;
	uint32_t s_str  = cmd->p2 ? cmd->p2 : n;
	uint32_t d_str  = cmd->p3 ? cmd->p3 : n;
	float eps       = cmd->f0;
	uint32_t r;

	if (!src0 || !dst || n == 0u) {
		return -1;
	}
	for (r = 0u; r < n_rows; r++) {
		const float *xr = src0 + (uint64_t)r * s_str;
		float       *dr = dst  + (uint64_t)r * d_str;
		if (weight) {
			c7x_rmsnorm_f32(dr, xr, weight, n, eps);
		} else {
			rmsnorm_unweighted_row_f32(dr, xr, n, eps);
		}
	}
	return 0;
}

static void vec_binop_row_f32(float *dst, const float *a, const float *b,
			      uint32_t n, int is_mul)
{
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;

	if (n_vec > 0u) {
		cmd_se_open_1d_f32(0, a, n_vec);
		cmd_se_open_1d_f32(1, b, n_vec);
		if (is_mul) {
			for (; i < n_vec; i += 16u) {
				float16 va = __SE0ADV(float16);
				float16 vb = __SE1ADV(float16);
				*(__float16_ptr)&dst[i] = va * vb;
			}
		} else {
			for (; i < n_vec; i += 16u) {
				float16 va = __SE0ADV(float16);
				float16 vb = __SE1ADV(float16);
				*(__float16_ptr)&dst[i] = va + vb;
			}
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
	}
	if (is_mul) {
		for (; i < n; i++) dst[i] = a[i] * b[i];
	} else {
		for (; i < n; i++) dst[i] = a[i] + b[i];
	}
}

static int exec_binop_f32(const struct c7x_cmd *cmd, int is_mul)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *src1 = (const uint8_t *)(uintptr_t)cmd->src1;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;

	if (!src0 || !src1 || !dst) {
		return -1;
	}

	if (cmd->src3 == 0ull) {
		/* Fast contiguous / row-broadcast path */
		uint32_t n_cols = cmd->p0;
		uint32_t n_rows = cmd->p1 ? cmd->p1 : 1u;
		uint32_t s1_rows = cmd->p2 ? cmd->p2 : n_rows;
		uint32_t s1_cols = cmd->p3 ? cmd->p3 : n_cols;
		uint32_t r, c;

		for (r = 0u; r < n_rows; r++) {
			const float *a = (const float *)src0 + (uint64_t)r * n_cols;
			const float *b = (const float *)src1 + (uint64_t)(r % s1_rows) * s1_cols;
			float       *d = (float *)dst + (uint64_t)r * n_cols;

			if (s1_cols == n_cols) {
				vec_binop_row_f32(d, a, b, n_cols, is_mul);
			} else if (s1_cols == 1u) {
				float bv = b[0];
				if (is_mul) {
					for (c = 0u; c < n_cols; c++) d[c] = a[c] * bv;
				} else {
					for (c = 0u; c < n_cols; c++) d[c] = a[c] + bv;
				}
			} else {
				for (c = 0u; c < n_cols; c++) {
					float bv = b[c % s1_cols];
					d[c] = is_mul ? (a[c] * bv) : (a[c] + bv);
				}
			}
		}
		return 0;
	}

	/* General 4D broadcasting + strided path */
	{
		const struct c7x_4d_binop_desc *desc =
			(const struct c7x_4d_binop_desc *)(uintptr_t)cmd->src3;
		uint32_t i3, i2, i1, i0;
		int row_contig;

		if (desc->ne1[0] == 0u || desc->ne1[1] == 0u ||
		    desc->ne1[2] == 0u || desc->ne1[3] == 0u) {
			return -2;
		}
		row_contig = (desc->nb0[0] == sizeof(float) &&
			      desc->nb1[0] == sizeof(float) &&
			      desc->nb_dst[0] == sizeof(float) &&
			      desc->ne1[0] == desc->ne_dst[0]);

		for (i3 = 0u; i3 < desc->ne_dst[3]; i3++) {
			uint32_t s1_3 = i3 % desc->ne1[3];
			for (i2 = 0u; i2 < desc->ne_dst[2]; i2++) {
				uint32_t s1_2 = i2 % desc->ne1[2];
				for (i1 = 0u; i1 < desc->ne_dst[1]; i1++) {
					uint32_t s1_1 = i1 % desc->ne1[1];
					const uint8_t *p0 = src0 + (uint64_t)i3 * desc->nb0[3] +
							    (uint64_t)i2 * desc->nb0[2] +
							    (uint64_t)i1 * desc->nb0[1];
					const uint8_t *p1 = src1 + (uint64_t)s1_3 * desc->nb1[3] +
							    (uint64_t)s1_2 * desc->nb1[2] +
							    (uint64_t)s1_1 * desc->nb1[1];
					uint8_t *pd = dst + (uint64_t)i3 * desc->nb_dst[3] +
						      (uint64_t)i2 * desc->nb_dst[2] +
						      (uint64_t)i1 * desc->nb_dst[1];

					if (row_contig) {
						vec_binop_row_f32((float *)pd, (const float *)p0,
								  (const float *)p1,
								  desc->ne_dst[0], is_mul);
					} else {
						for (i0 = 0u; i0 < desc->ne_dst[0]; i0++) {
							uint32_t s1_0 = i0 % desc->ne1[0];
							float a = *(const float *)(p0 + (uint64_t)i0 * desc->nb0[0]);
							float b = *(const float *)(p1 + (uint64_t)s1_0 * desc->nb1[0]);
							*(float *)(pd + (uint64_t)i0 * desc->nb_dst[0]) =
								is_mul ? (a * b) : (a + b);
						}
					}
				}
			}
		}
	}
	return 0;
}

static int exec_scale_f32(const struct c7x_cmd *cmd)
{
	const float *src0 = (const float *)(uintptr_t)cmd->src0;
	float       *dst  = (float *)(uintptr_t)cmd->dst;
	uint32_t n_cols = cmd->p0;
	uint32_t n_rows = cmd->p1 ? cmd->p1 : 1u;
	uint32_t s_str  = cmd->p2 ? cmd->p2 : n_cols;
	uint32_t d_str  = cmd->p3 ? cmd->p3 : n_cols;
	float scale = cmd->f0;
	float bias  = cmd->f1;
	float16 sv  = (float16)scale;
	float16 bv  = (float16)bias;
	uint32_t r, i, n_vec = n_cols & ~15u;

	if (!src0 || !dst || n_cols == 0u) {
		return -1;
	}
	for (r = 0u; r < n_rows; r++) {
		const float *xr = src0 + (uint64_t)r * s_str;
		float       *dr = dst  + (uint64_t)r * d_str;
		i = 0u;
		if (n_vec > 0u) {
			cmd_se_open_1d_f32(0, xr, n_vec);
			for (; i < n_vec; i += 16u) {
				float16 xv = __SE0ADV(float16);
				*(__float16_ptr)&dr[i] = xv * sv + bv;
			}
			__SE0_CLOSE();
		}
		for (; i < n_cols; i++) {
			dr[i] = xr[i] * scale + bias;
		}
	}
	return 0;
}

static inline float16 cmd_vsilu_f16(float16 x)
{
	const float16 one = (float16)1.0f;
	float16 e = cmd_vexpf_f16(-x);
	return x * cmd_vrecip_f16(one + e);
}

static inline float16 cmd_vsigmoid_f16(float16 x)
{
	const float16 one = (float16)1.0f;
	return cmd_vrecip_f16(one + cmd_vexpf_f16(-x));
}

static inline float16 cmd_vtanh_f16(float16 x)
{
	/* tanh(x) = 1 - 2 / (exp(2x) + 1) */
	const float16 one = (float16)1.0f;
	const float16 two = (float16)2.0f;
	float16 e2x = cmd_vexpf_f16(two * x);
	return one - two * cmd_vrecip_f16(e2x + one);
}

static inline float16 cmd_vgelu_f16(float16 x)
{
	/* 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
	const float16 half = (float16)0.5f;
	const float16 one  = (float16)1.0f;
	const float16 c_k  = (float16)0.7978845608f;
	const float16 c_a  = (float16)0.044715f;
	float16 inner = c_k * (x + c_a * x * x * x);
	return half * x * (one + cmd_vtanh_f16(inner));
}

static inline float16 cmd_vgelu_quick_f16(float16 x)
{
	const float16 c = (float16)(-1.702f);
	const float16 one = (float16)1.0f;
	return x * cmd_vrecip_f16(one + cmd_vexpf_f16(c * x));
}

static float scalar_unary(float x, uint32_t subop)
{
	switch (subop) {
	case C7X_UNARY_SILU:
		return x / (1.0f + cmd_expf_scalar(-x));
	case C7X_UNARY_GELU: {
		float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
		float e2 = cmd_expf_scalar(2.0f * inner);
		float th = 1.0f - 2.0f / (e2 + 1.0f);
		return 0.5f * x * (1.0f + th);
	}
	case C7X_UNARY_GELU_QUICK:
		return x / (1.0f + cmd_expf_scalar(-1.702f * x));
	case C7X_UNARY_RELU:
		return x > 0.0f ? x : 0.0f;
	case C7X_UNARY_SIGMOID:
		return 1.0f / (1.0f + cmd_expf_scalar(-x));
	case C7X_UNARY_TANH: {
		float e2 = cmd_expf_scalar(2.0f * x);
		return 1.0f - 2.0f / (e2 + 1.0f);
	}
	case C7X_UNARY_NEG:
		return -x;
	case C7X_UNARY_EXP:
		if (x > 88.72283905f) return INFINITY;
		if (x < -103.972077f) return 0.0f;
		return expf(x);
	case C7X_UNARY_ABS:
		return x < 0.0f ? -x : x;
	case C7X_UNARY_SQR:
		return x * x;
	case C7X_UNARY_SQRT:
		return sqrtf(x);
	default:
		return x;
	}
}

static int exec_unary_f32(const struct c7x_cmd *cmd)
{
	const float *src0 = (const float *)(uintptr_t)cmd->src0;
	float       *dst  = (float *)(uintptr_t)cmd->dst;
	uint32_t n = cmd->p0;
	uint32_t subop = cmd->p1;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;

	if (!src0 || !dst || n == 0u) {
		return -1;
	}
	if (subop > C7X_UNARY_SQRT) {
		return -2;
	}

	if (n_vec > 0u && subop != C7X_UNARY_SQRT && subop != C7X_UNARY_EXP) {
		const float16 zero = (float16)0.0f;
		cmd_se_open_1d_f32(0, src0, n_vec);
		switch (subop) {
		case C7X_UNARY_SILU:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vsilu_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_GELU:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vgelu_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_GELU_QUICK:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vgelu_quick_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_RELU:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = __max(__SE0ADV(float16), zero);
			}
			break;
		case C7X_UNARY_SIGMOID:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vsigmoid_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_TANH:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vtanh_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_NEG:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = -__SE0ADV(float16);
			}
			break;
		case C7X_UNARY_EXP:
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&dst[i] = cmd_vexpf_f16(__SE0ADV(float16));
			}
			break;
		case C7X_UNARY_ABS:
			for (; i < n_vec; i += 16u) {
				float16 xv = __SE0ADV(float16);
				*(__float16_ptr)&dst[i] = __max(xv, -xv);
			}
			break;
		case C7X_UNARY_SQR:
			for (; i < n_vec; i += 16u) {
				float16 xv = __SE0ADV(float16);
				*(__float16_ptr)&dst[i] = xv * xv;
			}
			break;
		default:
			break;
		}
		__SE0_CLOSE();
	}
	for (; i < n; i++) {
		dst[i] = scalar_unary(src0[i], subop);
	}
	return 0;
}

static int exec_glu_f32(const struct c7x_cmd *cmd)
{
	const float *src0 = (const float *)(uintptr_t)cmd->src0;
	const float *src1 = cmd->src1 ? (const float *)(uintptr_t)cmd->src1 : NULL;
	float       *dst  = (float *)(uintptr_t)cmd->dst;
	uint32_t n_out   = cmd->p0;
	uint32_t n_rows  = cmd->p1 ? cmd->p1 : 1u;
	uint32_t subop   = cmd->p2;
	int swapped      = (cmd->flags & C7X_CMD_FLAG_GLU_SWAP) ? 1 : 0;
	uint32_t s0_str  = cmd->p4 ? cmd->p4 : (src1 ? n_out : 2u * n_out);
	uint32_t s1_str  = cmd->p5 ? cmd->p5 : n_out;
	uint32_t r, i, n_vec = n_out & ~15u;

	if (!src0 || !dst || n_out == 0u) {
		return -1;
	}
	if (subop > C7X_GLU_GEGLU_QUICK) {
		return -2;
	}

	for (r = 0u; r < n_rows; r++) {
		const float *gate;
		const float *up;
		float *dr = dst + (uint64_t)r * n_out;

		if (src1) {
			/* Split tensors: GGML_OP_GLU with src1 always applies activation to src0, multiplies by src1 */
			gate = src0 + (uint64_t)r * s0_str;
			up   = src1 + (uint64_t)r * s1_str;
		} else {
			/* Single concatenated tensor [2*n_out]: swapped=0 -> gate=first half, up=second half */
			const float *row = src0 + (uint64_t)r * s0_str;
			gate = swapped ? (row + n_out) : row;
			up   = swapped ? row : (row + n_out);
		}

		if (subop == C7X_GLU_SWIGLU) {
			c7x_swiglu_f32(dr, gate, up, n_out);
		} else {
			i = 0u;
			if (n_vec > 0u) {
				const float16 zero = (float16)0.0f;
				cmd_se_open_1d_f32(0, gate, n_vec);
				cmd_se_open_1d_f32(1, up, n_vec);
				if (subop == C7X_GLU_GEGLU) {
					for (; i < n_vec; i += 16u) {
						float16 g = cmd_vgelu_f16(__SE0ADV(float16));
						float16 u = __SE1ADV(float16);
						*(__float16_ptr)&dr[i] = g * u;
					}
				} else if (subop == C7X_GLU_GEGLU_QUICK) {
					for (; i < n_vec; i += 16u) {
						float16 g = cmd_vgelu_quick_f16(__SE0ADV(float16));
						float16 u = __SE1ADV(float16);
						*(__float16_ptr)&dr[i] = g * u;
					}
				} else {
					for (; i < n_vec; i += 16u) {
						float16 g = __max(__SE0ADV(float16), zero);
						float16 u = __SE1ADV(float16);
						*(__float16_ptr)&dr[i] = g * u;
					}
				}
				__SE0_CLOSE();
				__SE1_CLOSE();
			}
			for (; i < n_out; i++) {
				uint32_t uop = (subop == C7X_GLU_GEGLU) ? C7X_UNARY_GELU :
					       (subop == C7X_GLU_GEGLU_QUICK) ? C7X_UNARY_GELU_QUICK :
					       C7X_UNARY_RELU;
				dr[i] = scalar_unary(gate[i], uop) * up[i];
			}
		}
	}
	return 0;
}

static int exec_soft_max_f32(const struct c7x_cmd *cmd)
{
	const uint8_t *src0  = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *mask  = cmd->src1 ? (const uint8_t *)(uintptr_t)cmd->src1 : NULL;
	const float   *sinks = cmd->src2 ? (const float *)(uintptr_t)cmd->src2 : NULL;
	uint8_t       *dst   = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_softmax_desc *d =
		(const struct c7x_softmax_desc *)(uintptr_t)cmd->src3;
	int mask_f16 = (cmd->flags & C7X_CMD_FLAG_MASK_F16) ? 1 : 0;
	uint32_t i3, i2, i1, i0;

	if (!src0 || !dst || !d || d->ne0 == 0u) {
		return -1;
	}

	for (i3 = 0u; i3 < d->ne3; i3++) {
		for (i2 = 0u; i2 < d->ne2; i2++) {
			float slope = 1.0f;
			if (d->max_bias > 0.0f) {
				uint32_t h = i2;
				float base = (h < d->n_head_log2) ? d->m0 : d->m1;
				int32_t exp_i = (h < d->n_head_log2)
						? (int32_t)(h + 1u)
						: (int32_t)(2u * (h - d->n_head_log2) + 1u);
				slope = powf(base, (float)exp_i);
			}
			for (i1 = 0u; i1 < d->ne1; i1++) {
				const float *x = (const float *)(src0 + (uint64_t)i3 * d->nb03 +
								 (uint64_t)i2 * d->nb02 +
								 (uint64_t)i1 * d->nb01);
				float *y = (float *)(dst + (uint64_t)i3 * d->nb_dst3 +
						     (uint64_t)i2 * d->nb_dst2 +
						     (uint64_t)i1 * d->nb_dst1);
				const uint8_t *mrow = NULL;
				float maxv = -INFINITY;
				float sum = 0.0f;

				if (mask && d->mask_ne1 > 0u) {
					uint32_t m3 = d->mask_ne3 ? (i3 % d->mask_ne3) : 0u;
					uint32_t m2 = d->mask_ne2 ? (i2 % d->mask_ne2) : 0u;
					uint32_t m1 = i1 % d->mask_ne1;
					mrow = mask + (uint64_t)m3 * d->mask_nb3 +
					       (uint64_t)m2 * d->mask_nb2 +
					       (uint64_t)m1 * d->mask_nb1;
				}

				/* Pass 1: vectorized scale + mask into y and find max */
				{
					uint32_t n_vec = d->ne0 & ~15u;
					i0 = 0u;
					if (n_vec >= 16u) {
						float16 scale_v = (float16)d->scale;
						float16 slope_v = (float16)slope;
						float16 max_v   = (float16)(-INFINITY);

						cmd_se_open_1d_f32(0, x, n_vec);
						if (mrow) {
							if (mask_f16) {
								cmd_se_open_1d_u16_u32(1, mrow, n_vec);
								for (; i0 < n_vec; i0 += 16u) {
									float16 v = __SE0ADV(float16) * scale_v +
										    slope_v * cmd_vf16_to_f32(__SE1ADV(uint16));
									*(__float16_ptr)&y[i0] = v;
									max_v = __max(max_v, v);
								}
								__SE1_CLOSE();
							} else {
								cmd_se_open_1d_f32(1, mrow, n_vec);
								for (; i0 < n_vec; i0 += 16u) {
									float16 v = __SE0ADV(float16) * scale_v +
										    slope_v * __SE1ADV(float16);
									*(__float16_ptr)&y[i0] = v;
									max_v = __max(max_v, v);
								}
								__SE1_CLOSE();
							}
						} else {
							for (; i0 < n_vec; i0 += 16u) {
								float16 v = __SE0ADV(float16) * scale_v;
								*(__float16_ptr)&y[i0] = v;
								max_v = __max(max_v, v);
							}
						}
						__SE0_CLOSE();
						maxv = cmd_hmax_f16(max_v);
					}
					for (; i0 < d->ne0; i0++) {
						float v = x[i0] * d->scale;
						if (mrow) {
							float mv = mask_f16
								   ? cmd_f16_to_f32(((const uint16_t *)mrow)[i0])
								   : ((const float *)mrow)[i0];
							v += slope * mv;
						}
						y[i0] = v;
						if (v > maxv) {
							maxv = v;
						}
					}
				}
				if (sinks) {
					float sv = sinks[i2];
					if (sv > maxv) maxv = sv;
				}

				/* Pass 2: vectorized exp(y - maxv) and sum */
				{
					uint32_t n_vec = d->ne0 & ~15u;
					float16 max_v = (float16)maxv;
					float16 sum_v = (float16)0.0f;
					i0 = 0u;
					if (n_vec > 0u) {
						cmd_se_open_1d_f32(0, y, n_vec);
						for (; i0 < n_vec; i0 += 16u) {
							float16 ev = cmd_vexpf_f16(__SE0ADV(float16) - max_v);
							*(__float16_ptr)&y[i0] = ev;
							sum_v += ev;
						}
						__SE0_CLOSE();
					}
					sum = cmd_hsum_f16(sum_v);
					for (; i0 < d->ne0; i0++) {
						float ev = cmd_expf_scalar(y[i0] - maxv);
						y[i0] = ev;
						sum += ev;
					}
					if (sinks) {
						sum += cmd_expf_scalar(sinks[i2] - maxv);
					}
				}

				/* Pass 3: normalize */
				if (sum > 0.0f) {
					float inv = 1.0f / sum;
					float16 inv_v = (float16)inv;
					uint32_t n_vec = d->ne0 & ~15u;
					i0 = 0u;
					if (n_vec > 0u) {
						cmd_se_open_1d_f32(0, y, n_vec);
						for (; i0 < n_vec; i0 += 16u) {
							*(__float16_ptr)&y[i0] = __SE0ADV(float16) * inv_v;
						}
						__SE0_CLOSE();
					}
					for (; i0 < d->ne0; i0++) {
						y[i0] *= inv;
					}
				}
			}
		}
	}
	return 0;
}

/* GGML YaRN / standard RoPE frequency helpers */
static float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base)
{
	return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) /
	       (2.0f * logf(base));
}

static void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
				float beta_fast, float beta_slow, uint32_t truncate, float dims[2])
{
	float start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
	float end   = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);
	if (truncate) {
		start = floorf(start);
		end   = ceilf(end);
	}
	dims[0] = start < 0.0f ? 0.0f : start;
	dims[1] = end > (float)(n_dims - 1) ? (float)(n_dims - 1) : end;
}

static float rope_yarn_ramp(float low, float high, int i0)
{
	float y = ((float)i0 / 2.0f - low) / ((high - low) > 0.001f ? (high - low) : 0.001f);
	if (y < 0.0f) y = 0.0f;
	if (y > 1.0f) y = 1.0f;
	return 1.0f - y;
}

static int exec_rope_f32(const struct c7x_cmd *cmd)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	const int32_t *pos  = (const int32_t *)(uintptr_t)cmd->src1;
	const float   *ff   = cmd->src2 ? (const float *)(uintptr_t)cmd->src2 : NULL;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_rope_desc *d =
		(const struct c7x_rope_desc *)(uintptr_t)cmd->src3;
	uint32_t ne0, ne1, ne2, ne3, n_dims;
	int is_neox;
	float theta_scale = 1.0f, corr_dims[2];
	uint32_t i3, i2, i1, i0;

	if (!src0 || !pos || !dst || !d) {
		return -1;
	}
	ne0 = d->ne0[0];
	ne1 = d->ne0[1];
	ne2 = d->ne0[2];
	ne3 = d->ne0[3];
	n_dims = d->n_dims;
	is_neox = (d->mode & C7X_ROPE_MODE_NEOX) ? 1 : 0;
	if (n_dims > ne0 || (n_dims & 1u) != 0u) {
		return -2;
	}

	{
		struct rope_freq_key {
			uint32_t n_dims;
			uint32_t n_ctx_orig;
			int32_t  truncate;
			float    freq_base;
			float    freq_scale;
			float    ext_factor;
			float    beta_fast;
			float    beta_slow;
			float    attn_factor;
		};
		static struct rope_freq_key g_rope_fkey;
		static int g_rope_fvalid = 0;
		static float g_rope_freq_tab[128] __attribute__((aligned(64)));
		static float g_rope_mscale = 1.0f;
		static int g_rope_scvalid = 0;
		static int32_t g_rope_sc_pos = 0;
		static float g_rope_c_tab[128] __attribute__((aligned(64)));
		static float g_rope_s_tab[128] __attribute__((aligned(64)));

		float freq_tab_local[128] __attribute__((aligned(64)));
		float c_tab_local[128]    __attribute__((aligned(64)));
		float s_tab_local[128]    __attribute__((aligned(64)));
		float *freq_tab = freq_tab_local;
		float *c_tab    = c_tab_local;
		float *s_tab    = s_tab_local;
		uint32_t n_pairs = n_dims >> 1;
		int use_tab = (n_pairs <= 128u);
		int can_cache = (use_tab && ff == NULL);
		float mscale_base = d->attn_factor;

		if (can_cache) {
			struct rope_freq_key fk;
			memset(&fk, 0, sizeof(fk));
			fk.n_dims      = n_dims;
			fk.n_ctx_orig  = d->n_ctx_orig;
			fk.truncate    = d->truncate;
			fk.freq_base   = d->freq_base;
			fk.freq_scale  = d->freq_scale;
			fk.ext_factor  = d->ext_factor;
			fk.beta_fast   = d->beta_fast;
			fk.beta_slow   = d->beta_slow;
			fk.attn_factor = d->attn_factor;

			if (g_rope_fvalid && memcmp(&fk, &g_rope_fkey, sizeof(fk)) == 0) {
				freq_tab    = g_rope_freq_tab;
				mscale_base = g_rope_mscale;
			} else {
				float tb = 1.0f;
				uint32_t k_pad16 = (n_pairs + 15u) & ~15u;

				theta_scale = powf(d->freq_base, -2.0f / (float)n_dims);
				corr_dims[0] = 0.0f;
				corr_dims[1] = 0.0f;
				if (d->ext_factor != 0.0f) {
					rope_yarn_corr_dims((int)n_dims, (int)d->n_ctx_orig, d->freq_base,
							    d->beta_fast, d->beta_slow, d->truncate, corr_dims);
					mscale_base *= (1.0f + 0.1f * logf(1.0f / d->freq_scale));
				}
				for (i0 = 0u; i0 < n_pairs; i0++) {
					float f_extrap = tb;
					float f_interp = d->freq_scale * f_extrap;
					float f_eff = f_interp;
					if (d->ext_factor != 0.0f) {
						float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)(i0 * 2u)) * d->ext_factor;
						f_eff = f_interp * (1.0f - ramp_mix) + f_extrap * ramp_mix;
					}
					g_rope_freq_tab[i0] = f_eff;
					tb *= theta_scale;
				}
				for (; i0 < k_pad16 && i0 < 128u; i0++) {
					g_rope_freq_tab[i0] = 0.0f;
				}
				g_rope_fkey    = fk;
				g_rope_mscale  = mscale_base;
				g_rope_fvalid  = 1;
				g_rope_scvalid = 0;
				freq_tab       = g_rope_freq_tab;
			}
		} else {
			theta_scale = powf(d->freq_base, -2.0f / (float)n_dims);
			corr_dims[0] = 0.0f;
			corr_dims[1] = 0.0f;
			if (d->ext_factor != 0.0f) {
				rope_yarn_corr_dims((int)n_dims, (int)d->n_ctx_orig, d->freq_base,
						    d->beta_fast, d->beta_slow, d->truncate, corr_dims);
				mscale_base *= (1.0f + 0.1f * logf(1.0f / d->freq_scale));
			}
			if (use_tab) {
				float tb = 1.0f;
				uint32_t k_pad16 = (n_pairs + 15u) & ~15u;
				for (i0 = 0u; i0 < n_pairs; i0++) {
					float inv_freq = ff ? (1.0f / ff[i0]) : 1.0f;
					float f_extrap = tb * inv_freq;
					float f_interp = d->freq_scale * f_extrap;
					float f_eff = f_interp;
					if (d->ext_factor != 0.0f) {
						float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)(i0 * 2u)) * d->ext_factor;
						f_eff = f_interp * (1.0f - ramp_mix) + f_extrap * ramp_mix;
					}
					freq_tab[i0] = f_eff;
					tb *= theta_scale;
				}
				for (; i0 < k_pad16 && i0 < 128u; i0++) {
					freq_tab[i0] = 0.0f;
				}
			}
		}

		for (i3 = 0u; i3 < ne3; i3++) {
			for (i2 = 0u; i2 < ne2; i2++) {
				int32_t p_i32 = pos[i2];
				float p = (float)p_i32;

				/* Compute 16 (sin, cos) pairs at once via vectorized Cody-Waite polynomial (or reuse cache!). */
				if (use_tab) {
					if (can_cache && g_rope_scvalid && g_rope_sc_pos == p_i32) {
						c_tab = g_rope_c_tab;
						s_tab = g_rope_s_tab;
					} else {
						float *cd = can_cache ? g_rope_c_tab : c_tab_local;
						float *sd = can_cache ? g_rope_s_tab : s_tab_local;
						if (fabsf(p) <= 100000.0f) {
							float16 pv = (float16)p;
							float16 mv = (float16)mscale_base;
							uint32_t k_pad16 = (n_pairs + 15u) & ~15u;
							for (i0 = 0u; i0 < k_pad16; i0 += 16u) {
								float16 th = pv * *(__float16_ptr)&freq_tab[i0];
								cmd_vsincos_store_f16(th, mv, &sd[i0], &cd[i0]);
							}
						} else {
							for (i0 = 0u; i0 < n_pairs; i0++) {
								float th = p * freq_tab[i0];
								cd[i0] = cosf(th) * mscale_base;
								sd[i0] = sinf(th) * mscale_base;
							}
						}
						if (can_cache) {
							g_rope_sc_pos  = p_i32;
							g_rope_scvalid = 1;
						}
						c_tab = cd;
						s_tab = sd;
					}
				}

				for (i1 = 0u; i1 < ne1; i1++) {
					const float *x = (const float *)(src0 + (uint64_t)i3 * d->nb0[3] +
									 (uint64_t)i2 * d->nb0[2] +
									 (uint64_t)i1 * d->nb0[1]);
					float *y = (float *)(dst + (uint64_t)i3 * d->nb_dst[3] +
							     (uint64_t)i2 * d->nb_dst[2] +
							     (uint64_t)i1 * d->nb_dst[1]);

					if (use_tab) {
						if (!is_neox) {
							for (i0 = 0u; i0 < n_dims; i0 += 2u) {
								float c = c_tab[i0 >> 1];
								float s = s_tab[i0 >> 1];
								float x0 = x[i0];
								float x1 = x[i0 + 1u];
								y[i0]      = x0 * c - x1 * s;
								y[i0 + 1u] = x0 * s + x1 * c;
							}
						} else {
							uint32_t half = n_dims >> 1;
							uint32_t h_vec = half & ~15u;
							i0 = 0u;
							if (h_vec >= 16u && (((uintptr_t)y | (uintptr_t)(y + half)) & 63u) == 0u) {
								cmd_se_open_1d_f32(0, x, h_vec);
								cmd_se_open_1d_f32(1, x + half, h_vec);
								for (; i0 < h_vec; i0 += 16u) {
									float16 cv = *(__float16_ptr)&c_tab[i0];
									float16 sv = *(__float16_ptr)&s_tab[i0];
									float16 v0 = __SE0ADV(float16);
									float16 v1 = __SE1ADV(float16);
									*(__float16_ptr)&y[i0]        = v0 * cv - v1 * sv;
									*(__float16_ptr)&y[i0 + half] = v0 * sv + v1 * cv;
								}
								__SE0_CLOSE();
								__SE1_CLOSE();
							}
							for (; i0 < half; i0++) {
								float c = c_tab[i0];
								float s = s_tab[i0];
								float x0 = x[i0];
								float x1 = x[i0 + half];
								y[i0]        = x0 * c - x1 * s;
								y[i0 + half] = x0 * s + x1 * c;
							}
						}
					} else {
						float theta_base = p;
						for (i0 = 0u; i0 < n_dims; i0 += 2u) {
							float inv_freq = ff ? (1.0f / ff[i0 / 2u]) : 1.0f;
							float theta_extrap = theta_base * inv_freq;
							float theta_interp = d->freq_scale * theta_extrap;
							float theta = theta_interp;
							float c, s, x0, x1;

							if (d->ext_factor != 0.0f) {
								float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * d->ext_factor;
								theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
							}
							c = cosf(theta) * mscale_base;
							s = sinf(theta) * mscale_base;

							if (!is_neox) {
								x0 = x[i0];
								x1 = x[i0 + 1u];
								y[i0]      = x0 * c - x1 * s;
								y[i0 + 1u] = x0 * s + x1 * c;
							} else {
								uint32_t ic = i0 / 2u;
								x0 = x[ic];
								x1 = x[ic + n_dims / 2u];
								y[ic]              = x0 * c - x1 * s;
								y[ic + n_dims / 2u] = x0 * s + x1 * c;
							}
							theta_base *= theta_scale;
						}
					}
					/* Copy unrotated tail dimensions */
					for (i0 = n_dims; i0 < ne0; i0++) {
						y[i0] = x[i0];
					}
				}
			}
		}
	}
	return 0;
}

static void unpack_tiled_w4a8_row(const uint8_t *wq, const float *ws,
				  uint32_t row, uint32_t dim, uint32_t dim_pad,
				  float *out)
{
	uint32_t k_tiles = dim_pad >> 6;
	uint32_t nt = row >> 6;
	uint32_t r  = row & 63u;
	uint32_t kt, c;

	for (kt = 0u; kt < k_tiles; kt++) {
		const uint8_t *tile = wq + ((uint64_t)nt * k_tiles + kt) * 2048ull;
		const uint8_t *vec  = tile + (r >> 1) * 64u;
		float sc = ws[((uint64_t)nt * k_tiles + kt) * 64ull + r];
		uint32_t col0 = kt << 6;

		for (c = 0u; c < 64u; c++) {
			if (col0 + c < dim) {
				uint8_t byte = vec[c];
				int8_t w = (r & 1u)
					   ? ((int8_t)byte >> 4)
					   : ((int8_t)(byte << 4) >> 4);
				out[col0 + c] = (float)w * sc;
			}
		}
	}
}

static void unpack_tiled_w8a8_row(const int8_t *wq, const float *ws,
				  uint32_t row, uint32_t dim, uint32_t dim_pad,
				  float *out)
{
	uint32_t k_tiles = dim_pad >> 6;
	uint32_t nt = row >> 6;
	uint32_t r  = row & 63u;
	float sc = ws[row];
	uint32_t kt;

	for (kt = 0u; kt < k_tiles; kt++) {
		uint32_t col0 = kt << 6;
		uint32_t rem = (dim > col0) ? (dim - col0) : 0u;
		uint32_t chunk = rem > 64u ? 64u : rem;
		if (chunk > 0u) {
			const int8_t *src = wq + ((uint64_t)nt * k_tiles + kt) * 4096ull + r * 64u;
			c7x_dequant_i8_f32(out + col0, src, sc, chunk);
		}
	}
}

static int exec_get_rows(const struct c7x_cmd *cmd)
{
	const uint8_t *src0   = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *idx    = (const uint8_t *)(uintptr_t)cmd->src1;
	const float   *scales = cmd->src2 ? (const float *)(uintptr_t)cmd->src2 : NULL;
	uint8_t       *dst    = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_get_rows_desc *d =
		(const struct c7x_get_rows_desc *)(uintptr_t)cmd->src3;
	uint32_t dtype = cmd->p4;
	uint32_t i12, i11, i10, c;

	if (!src0 || !idx || !dst || !d) {
		return -1;
	}
	for (i12 = 0u; i12 < d->ne12; i12++) {
		for (i11 = 0u; i11 < d->ne11; i11++) {
			for (i10 = 0u; i10 < d->ne10; i10++) {
				int32_t row_idx = *(const int32_t *)(idx + (uint64_t)i10 * d->nb10 +
								     (uint64_t)i11 * d->nb11 +
								     (uint64_t)i12 * d->nb12);
				float *out = (float *)(dst + (uint64_t)i10 * d->nb_dst1 +
						       (uint64_t)i11 * d->nb_dst2 +
						       (uint64_t)i12 * d->nb_dst3);
				if (row_idx < 0 || (uint32_t)row_idx >= d->ne01) {
					return -2;
				}
				if (dtype == C7X_DTYPE_W4A8_G64) {
					if (!scales) return -3;
					unpack_tiled_w4a8_row(src0, scales, (uint32_t)row_idx,
							      d->ne00, d->k_pad, out);
				} else if (dtype == C7X_DTYPE_W8A8) {
					if (!scales) return -3;
					unpack_tiled_w8a8_row((const int8_t *)src0, scales, (uint32_t)row_idx,
							      d->ne00, d->k_pad, out);
				} else {
					const uint8_t *srow = src0 + (uint64_t)row_idx * d->nb01 +
							      (uint64_t)i11 * d->nb02 +
							      (uint64_t)i12 * d->nb03;
					if (dtype == C7X_DTYPE_F32) {
						memcpy(out, srow, (size_t)d->ne00 * sizeof(float));
					} else if (dtype == C7X_DTYPE_F16) {
						const uint16_t *h = (const uint16_t *)srow;
						for (c = 0u; c < d->ne00; c++) {
							out[c] = cmd_f16_to_f32(h[c]);
						}
					} else if (dtype == C7X_DTYPE_Q8_0) {
						dequant_row_q8_0(srow, out, d->ne00);
					} else if (dtype == C7X_DTYPE_Q4_0) {
						dequant_row_q4_0(srow, out, d->ne00);
					} else {
						return -4;
					}
				}
			}
		}
	}
	return 0;
}

static int exec_cpy(const struct c7x_cmd *cmd)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_cpy_desc *d =
		(const struct c7x_cpy_desc *)(uintptr_t)cmd->src3;
	uint32_t stype = cmd->p0;
	uint32_t dtype = cmd->p1;
	uint32_t i3, i2, i1, i0;

	if (!src0 || !dst || !d) {
		return -1;
	}

	/* Fast path when src and dst shapes match and row-quantized or row-contiguous */
	if (d->ne0[0] == d->ne_dst[0] && d->ne0[1] == d->ne_dst[1] &&
	    d->ne0[2] == d->ne_dst[2] && d->ne0[3] == d->ne_dst[3]) {
		if ((stype == C7X_DTYPE_F32 || stype == C7X_DTYPE_F16) && dtype == C7X_DTYPE_Q8_0) {
			uint32_t nb = d->ne0[0] / 32u;
			float tmp32[32];
			uint32_t b, j;
			for (i3 = 0u; i3 < d->ne0[3]; i3++) {
				for (i2 = 0u; i2 < d->ne0[2]; i2++) {
					for (i1 = 0u; i1 < d->ne0[1]; i1++) {
						const uint8_t *srow = src0 + (uint64_t)i3 * d->nb0[3] +
								      (uint64_t)i2 * d->nb0[2] +
								      (uint64_t)i1 * d->nb0[1];
						uint8_t *o = dst + (uint64_t)i3 * d->nb_dst[3] +
							     (uint64_t)i2 * d->nb_dst[2] +
							     (uint64_t)i1 * d->nb_dst[1];
						if (stype == C7X_DTYPE_F32 && d->nb0[0] == sizeof(float)) {
							quant_row_q8_0((const float *)srow, o, d->ne0[0]);
						} else {
							for (b = 0u; b < nb; b++) {
								for (j = 0u; j < 32u; j++) {
									const uint8_t *ep = srow + (uint64_t)(b * 32u + j) * d->nb0[0];
									tmp32[j] = (stype == C7X_DTYPE_F16)
										   ? cmd_f16_to_f32(*(const uint16_t *)ep)
										   : *(const float *)ep;
								}
								quant_row_q8_0(tmp32, o + (uint64_t)b * 34ull, 32u);
							}
						}
					}
				}
			}
			return 0;
		}
		if (stype == C7X_DTYPE_Q8_0 && (dtype == C7X_DTYPE_F32 || dtype == C7X_DTYPE_F16)) {
			uint32_t nb = d->ne0[0] / 32u;
			float tmp32[32];
			uint32_t b, j;
			for (i3 = 0u; i3 < d->ne0[3]; i3++) {
				for (i2 = 0u; i2 < d->ne0[2]; i2++) {
					for (i1 = 0u; i1 < d->ne0[1]; i1++) {
						const uint8_t *s = src0 + (uint64_t)i3 * d->nb0[3] +
								   (uint64_t)i2 * d->nb0[2] +
								   (uint64_t)i1 * d->nb0[1];
						uint8_t *orow = dst + (uint64_t)i3 * d->nb_dst[3] +
								(uint64_t)i2 * d->nb_dst[2] +
								(uint64_t)i1 * d->nb_dst[1];
						if (dtype == C7X_DTYPE_F32 && d->nb_dst[0] == sizeof(float)) {
							dequant_row_q8_0(s, (float *)orow, d->ne0[0]);
						} else {
							for (b = 0u; b < nb; b++) {
								dequant_row_q8_0(s + (uint64_t)b * 34ull, tmp32, 32u);
								for (j = 0u; j < 32u; j++) {
									uint8_t *dp = orow + (uint64_t)(b * 32u + j) * d->nb_dst[0];
									if (dtype == C7X_DTYPE_F16) {
										*(uint16_t *)dp = cmd_f32_to_f16(tmp32[j]);
									} else {
										*(float *)dp = tmp32[j];
									}
								}
							}
						}
					}
				}
			}
			return 0;
		}
	}

	/* Fast row-memcpy path when same dtype, contiguous dim-0 in src0, and either
	 * matching shapes OR fully contiguous dst (covers CONT of permuted views like kqv_out!). */
	if (stype == dtype &&
	    (stype == C7X_DTYPE_F32 || stype == C7X_DTYPE_F16 || stype == C7X_DTYPE_I32)) {
		uint32_t esz = (stype == C7X_DTYPE_F16) ? 2u : 4u;
		if (d->nb0[0] == esz && d->nb_dst[0] == esz) {
			int dst_contig = (d->nb_dst[1] == (uint64_t)d->ne_dst[0] * esz) &&
					 (d->nb_dst[2] == (uint64_t)d->ne_dst[1] * d->nb_dst[1]) &&
					 (d->nb_dst[3] == (uint64_t)d->ne_dst[2] * d->nb_dst[2]);
			size_t row_bytes = (size_t)d->ne0[0] * esz;

			if (dst_contig) {
				uint8_t *dp = dst;
				for (i3 = 0u; i3 < d->ne0[3]; i3++) {
					for (i2 = 0u; i2 < d->ne0[2]; i2++) {
						for (i1 = 0u; i1 < d->ne0[1]; i1++) {
							const uint8_t *sp = src0 + (uint64_t)i3 * d->nb0[3] +
									    (uint64_t)i2 * d->nb0[2] +
									    (uint64_t)i1 * d->nb0[1];
							memcpy(dp, sp, row_bytes);
							dp += row_bytes;
						}
					}
				}
				return 0;
			}
			if (d->ne0[0] == d->ne_dst[0] && d->ne0[1] == d->ne_dst[1] &&
			    d->ne0[2] == d->ne_dst[2] && d->ne0[3] == d->ne_dst[3]) {
				for (i3 = 0u; i3 < d->ne0[3]; i3++) {
					for (i2 = 0u; i2 < d->ne0[2]; i2++) {
						for (i1 = 0u; i1 < d->ne0[1]; i1++) {
							const uint8_t *sp = src0 + (uint64_t)i3 * d->nb0[3] +
									    (uint64_t)i2 * d->nb0[2] +
									    (uint64_t)i1 * d->nb0[1];
							uint8_t *dp = dst + (uint64_t)i3 * d->nb_dst[3] +
								      (uint64_t)i2 * d->nb_dst[2] +
								      (uint64_t)i1 * d->nb_dst[1];
							memcpy(dp, sp, row_bytes);
						}
					}
				}
				return 0;
			}
		}
	}

	/* General elementwise linear-index copy (handles arbitrary reshape/permute/view!) */
	{
		uint64_t total = (uint64_t)d->ne0[0] * d->ne0[1] * d->ne0[2] * d->ne0[3];
		uint64_t idx = 0ull;
		uint32_t d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u;

		if (stype == C7X_DTYPE_Q8_0 || dtype == C7X_DTYPE_Q8_0) {
			return -2;
		}
		for (i3 = 0u; i3 < d->ne0[3]; i3++) {
			for (i2 = 0u; i2 < d->ne0[2]; i2++) {
				for (i1 = 0u; i1 < d->ne0[1]; i1++) {
					for (i0 = 0u; i0 < d->ne0[0]; i0++) {
						const uint8_t *sp = src0 + (uint64_t)i3 * d->nb0[3] +
								    (uint64_t)i2 * d->nb0[2] +
								    (uint64_t)i1 * d->nb0[1] +
								    (uint64_t)i0 * d->nb0[0];
						uint8_t *dp = dst + (uint64_t)d3 * d->nb_dst[3] +
							      (uint64_t)d2 * d->nb_dst[2] +
							      (uint64_t)d1 * d->nb_dst[1] +
							      (uint64_t)d0 * d->nb_dst[0];
						if (stype == C7X_DTYPE_I32 && dtype == C7X_DTYPE_I32) {
							*(int32_t *)dp = *(const int32_t *)sp;
						} else {
							float val = (stype == C7X_DTYPE_F16)
								    ? cmd_f16_to_f32(*(const uint16_t *)sp)
								    : *(const float *)sp;
							if (dtype == C7X_DTYPE_F16) {
								*(uint16_t *)dp = cmd_f32_to_f16(val);
							} else {
								*(float *)dp = val;
							}
						}
						idx++;
						if (++d0 == d->ne_dst[0]) {
							d0 = 0u;
							if (++d1 == d->ne_dst[1]) {
								d1 = 0u;
								if (++d2 == d->ne_dst[2]) {
									d2 = 0u;
									d3++;
								}
							}
						}
					}
				}
			}
		}
		(void)total;
		(void)idx;
	}
	return 0;
}

static int exec_set_rows(const struct c7x_cmd *cmd)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *src1 = (const uint8_t *)(uintptr_t)cmd->src1;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_set_rows_desc *d =
		(const struct c7x_set_rows_desc *)(uintptr_t)cmd->src3;
	uint32_t dst_dtype = cmd->p0;
	int idx_i64 = (cmd->flags & C7X_CMD_FLAG_IDX_I64) ? 1 : 0;
	uint32_t i3, i2, i1, i0;

	if (!src0 || !src1 || !dst || !d) {
		return -1;
	}

	/*
	 * Fast path for flattened transposed V-cache writes (llama.cpp cpy_v):
	 * ne0 = [1, N, 1, 1], nb0[1] == 4 (contiguous F32 along i1), dst_dtype == F16.
	 * Pre-validate indices, then stream src0 16-wide via SE0 + branchless vf32->f16.
	 */
	if (d->ne0[0] == 1u && d->ne0[2] == 1u && d->ne0[3] == 1u &&
	    d->nb0[1] == sizeof(float) && dst_dtype == C7X_DTYPE_F16) {
		uint32_t n1 = d->ne0[1];
		uint32_t n1_vec = n1 & ~15u;
		const float *sp = (const float *)src0;
		uint16_t tmp16[16];
		uint32_t j;

		for (i1 = 0u; i1 < n1; i1++) {
			const uint8_t *ip = src1 + (uint64_t)i1 * d->nb1[0];
			int64_t row_idx = idx_i64
					  ? *(const int64_t *)ip
					  : (int64_t)(*(const int32_t *)ip);
			if (row_idx < 0 || (uint64_t)row_idx >= (uint64_t)d->ne_dst[1]) {
				return -2;
			}
		}
		i1 = 0u;
		if (n1_vec >= 16u) {
			cmd_se_open_1d_f32(0, sp, n1_vec);
			for (; i1 < n1_vec; i1 += 16u) {
				cmd_vf32_to_f16_store(__SE0ADV(float16), tmp16);
				for (j = 0u; j < 16u; j++) {
					const uint8_t *ip = src1 + (uint64_t)(i1 + j) * d->nb1[0];
					int64_t row_idx = idx_i64
							  ? *(const int64_t *)ip
							  : (int64_t)(*(const int32_t *)ip);
					*(uint16_t *)(dst + (uint64_t)row_idx * d->nb_dst[1]) = tmp16[j];
				}
			}
			__SE0_CLOSE();
		}
		for (; i1 < n1; i1++) {
			const uint8_t *ip = src1 + (uint64_t)i1 * d->nb1[0];
			int64_t row_idx = idx_i64
					  ? *(const int64_t *)ip
					  : (int64_t)(*(const int32_t *)ip);
			*(uint16_t *)(dst + (uint64_t)row_idx * d->nb_dst[1]) = cmd_f32_to_f16(sp[i1]);
		}
		return 0;
	}

	/*
	 * Fast path for 3D transposed V-cache writes: ne0[0] == 1, row_idx depends only on i1
	 * (ne1[1] <= 1 && ne1[2] <= 1), and src0 is contiguous along i2 (nb0[2] == 4).
	 */
	if (d->ne0[0] == 1u && d->ne0[2] > 0u && d->ne1[1] <= 1u && d->ne1[2] <= 1u &&
	    d->nb0[2] == sizeof(float) &&
	    (dst_dtype == C7X_DTYPE_F16 || dst_dtype == C7X_DTYPE_F32)) {
		uint32_t n2 = d->ne0[2];
		uint32_t n2_vec = n2 & ~15u;

		for (i3 = 0u; i3 < d->ne0[3]; i3++) {
			for (i1 = 0u; i1 < d->ne0[1]; i1++) {
				const uint8_t *ip = src1 + (uint64_t)i1 * d->nb1[0];
				int64_t row_idx = idx_i64
						  ? *(const int64_t *)ip
						  : (int64_t)(*(const int32_t *)ip);
				const float *sp = (const float *)(src0 + (uint64_t)i3 * d->nb0[3] +
								  (uint64_t)i1 * d->nb0[1]);
				uint8_t *dp_base;

				if (row_idx < 0 || (uint64_t)row_idx >= (uint64_t)d->ne_dst[1]) {
					return -2;
				}
				dp_base = dst + (uint64_t)i3 * d->nb_dst[3] +
					  (uint64_t)row_idx * d->nb_dst[1];
				if (dst_dtype == C7X_DTYPE_F16) {
					uint16_t tmp16[16];
					uint32_t j;
					i2 = 0u;
					if (n2_vec >= 16u) {
						cmd_se_open_1d_f32(0, sp, n2_vec);
						for (; i2 < n2_vec; i2 += 16u) {
							cmd_vf32_to_f16_store(__SE0ADV(float16), tmp16);
							for (j = 0u; j < 16u; j++) {
								*(uint16_t *)(dp_base + (uint64_t)(i2 + j) * d->nb_dst[2]) = tmp16[j];
							}
						}
						__SE0_CLOSE();
					}
					for (; i2 < n2; i2++) {
						*(uint16_t *)(dp_base + (uint64_t)i2 * d->nb_dst[2]) = cmd_f32_to_f16(sp[i2]);
					}
				} else {
					for (i2 = 0u; i2 < n2; i2++) {
						*(float *)(dp_base + (uint64_t)i2 * d->nb_dst[2]) = sp[i2];
					}
				}
			}
		}
		return 0;
	}

	for (i3 = 0u; i3 < d->ne0[3]; i3++) {
		uint32_t idx_i2 = d->ne1[2] ? (i3 % d->ne1[2]) : 0u;
		for (i2 = 0u; i2 < d->ne0[2]; i2++) {
			uint32_t idx_i1 = d->ne1[1] ? (i2 % d->ne1[1]) : 0u;
			for (i1 = 0u; i1 < d->ne0[1]; i1++) {
				const uint8_t *ip = src1 + (uint64_t)i1 * d->nb1[0] +
						    (uint64_t)idx_i1 * d->nb1[1] +
						    (uint64_t)idx_i2 * d->nb1[2];
				int64_t row_idx = idx_i64
						  ? *(const int64_t *)ip
						  : (int64_t)(*(const int32_t *)ip);
				const float *sp = (const float *)(src0 + (uint64_t)i3 * d->nb0[3] +
								  (uint64_t)i2 * d->nb0[2] +
								  (uint64_t)i1 * d->nb0[1]);
				uint8_t *dp;

				if (row_idx < 0 || (uint64_t)row_idx >= (uint64_t)d->ne_dst[1]) {
					return -2;
				}
				dp = dst + (uint64_t)i3 * d->nb_dst[3] +
				     (uint64_t)i2 * d->nb_dst[2] +
				     (uint64_t)row_idx * d->nb_dst[1];
				if (dst_dtype == C7X_DTYPE_F32) {
					if (d->ne0[0] == 1u) {
						*(float *)dp = *sp;
					} else {
						memcpy(dp, sp, (size_t)d->ne0[0] * sizeof(float));
					}
				} else if (dst_dtype == C7X_DTYPE_F16) {
					uint16_t *h = (uint16_t *)dp;
					if (d->ne0[0] == 1u) {
						*h = cmd_f32_to_f16(*sp);
					} else {
						uint32_t n_vec = d->ne0[0] & ~15u;
						i0 = 0u;
						if (n_vec >= 16u) {
							cmd_se_open_1d_f32(0, sp, n_vec);
							for (; i0 < n_vec; i0 += 16u) {
								cmd_vf32_to_f16_store(__SE0ADV(float16), &h[i0]);
							}
							__SE0_CLOSE();
						}
						for (; i0 < d->ne0[0]; i0++) {
							h[i0] = cmd_f32_to_f16(sp[i0]);
						}
					}
				} else if (dst_dtype == C7X_DTYPE_Q8_0) {
					quant_row_q8_0(sp, dp, d->ne0[0]);
				} else {
					return -3;
				}
			}
		}
	}
	return 0;
}

/* -------------------------------------------------------------------------- */
/* Phase 3: Non-LLM Tensor Ops (LayerNorm, Im2Col, Pool2D, Conv2D, QLinear)   */
/* -------------------------------------------------------------------------- */

static void norm_row_f32(float *y, const float *x, uint32_t n, float eps,
			 const float *gamma, const float *beta)
{
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;
	float sum = 0.0f, mean, var = 0.0f, scale;

	if (n_vec >= 16u) {
		float16 acc = (float16)0.0f;
		cmd_se_open_1d_f32(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			acc += __SE0ADV(float16);
		}
		__SE0_CLOSE();
		sum = cmd_hsum_f16(acc);
	}
	for (; i < n; i++) {
		sum += x[i];
	}
	mean = sum / (float)n;

	i = 0u;
	if (n_vec >= 16u) {
		float16 mv = (float16)mean;
		float16 ss = (float16)0.0f;
		cmd_se_open_1d_f32(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 d = __SE0ADV(float16) - mv;
			*(__float16_ptr)&y[i] = d;
			ss += d * d;
		}
		__SE0_CLOSE();
		var = cmd_hsum_f16(ss);
	}
	for (; i < n; i++) {
		float d = x[i] - mean;
		y[i] = d;
		var += d * d;
	}
	var /= (float)n;
	scale = 1.0f / sqrtf(var + eps);

	i = 0u;
	if (n_vec >= 16u) {
		float16 sv = (float16)scale;
		cmd_se_open_1d_f32(0, y, n_vec);
		if (gamma && beta) {
			cmd_se_open_1d_f32(1, gamma, n_vec);
			for (; i < n_vec; i += 16u) {
				float16 v = __SE0ADV(float16) * sv * __SE1ADV(float16) +
					    *(__float16_ptr)(float *)&beta[i];
				*(__float16_ptr)&y[i] = v;
			}
			__SE1_CLOSE();
		} else if (gamma) {
			cmd_se_open_1d_f32(1, gamma, n_vec);
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&y[i] = __SE0ADV(float16) * sv * __SE1ADV(float16);
			}
			__SE1_CLOSE();
		} else if (beta) {
			cmd_se_open_1d_f32(1, beta, n_vec);
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&y[i] = __SE0ADV(float16) * sv + __SE1ADV(float16);
			}
			__SE1_CLOSE();
		} else {
			for (; i < n_vec; i += 16u) {
				*(__float16_ptr)&y[i] = __SE0ADV(float16) * sv;
			}
		}
		__SE0_CLOSE();
	}
	for (; i < n; i++) {
		float v = y[i] * scale;
		if (gamma) v *= gamma[i];
		if (beta)  v += beta[i];
		y[i] = v;
	}
}

static int exec_norm_f32(const struct c7x_cmd *cmd)
{
	const uint8_t *src0  = (const uint8_t *)(uintptr_t)cmd->src0;
	uint8_t       *dst   = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_norm_desc *d = (const struct c7x_norm_desc *)(uintptr_t)cmd->src3;
	const float   *gamma = NULL;
	const float   *beta  = NULL;
	uint32_t i3, i2, i1, i0;

	if (!src0 || !dst || !d || d->ne0[0] == 0u) {
		return -1;
	}
	if (d->has_gamma) {
		gamma = (const float *)(uintptr_t)cmd->src1;
		if (!gamma) return -2;
	}
	if (d->has_beta) {
		beta = (const float *)(uintptr_t)cmd->src2;
		if (!beta) return -2;
	}

	if (d->nb0[0] == sizeof(float) && d->nb_dst[0] == sizeof(float)) {
		for (i3 = 0u; i3 < d->ne0[3]; i3++) {
			for (i2 = 0u; i2 < d->ne0[2]; i2++) {
				for (i1 = 0u; i1 < d->ne0[1]; i1++) {
					const float *x = (const float *)(src0 + (uint64_t)i3 * d->nb0[3] +
									 (uint64_t)i2 * d->nb0[2] +
									 (uint64_t)i1 * d->nb0[1]);
					float *y = (float *)(dst + (uint64_t)i3 * d->nb_dst[3] +
							     (uint64_t)i2 * d->nb_dst[2] +
							     (uint64_t)i1 * d->nb_dst[1]);
					norm_row_f32(y, x, d->ne0[0], d->eps, gamma, beta);
				}
			}
		}
		return 0;
	}

	/* General strided dim-0 fallback */
	for (i3 = 0u; i3 < d->ne0[3]; i3++) {
		for (i2 = 0u; i2 < d->ne0[2]; i2++) {
			for (i1 = 0u; i1 < d->ne0[1]; i1++) {
				const uint8_t *x = src0 + (uint64_t)i3 * d->nb0[3] +
						   (uint64_t)i2 * d->nb0[2] +
						   (uint64_t)i1 * d->nb0[1];
				uint8_t *y = dst + (uint64_t)i3 * d->nb_dst[3] +
					     (uint64_t)i2 * d->nb_dst[2] +
					     (uint64_t)i1 * d->nb_dst[1];
				float sum = 0.0f, mean, var = 0.0f, scale;
				for (i0 = 0u; i0 < d->ne0[0]; i0++) {
					sum += *(const float *)(x + (uint64_t)i0 * d->nb0[0]);
				}
				mean = sum / (float)d->ne0[0];
				for (i0 = 0u; i0 < d->ne0[0]; i0++) {
					float v = *(const float *)(x + (uint64_t)i0 * d->nb0[0]) - mean;
					*(float *)(y + (uint64_t)i0 * d->nb_dst[0]) = v;
					var += v * v;
				}
				scale = 1.0f / sqrtf(var / (float)d->ne0[0] + d->eps);
				for (i0 = 0u; i0 < d->ne0[0]; i0++) {
					float v = *(float *)(y + (uint64_t)i0 * d->nb_dst[0]) * scale;
					if (gamma) v *= gamma[i0];
					if (beta)  v += beta[i0];
					*(float *)(y + (uint64_t)i0 * d->nb_dst[0]) = v;
				}
			}
		}
	}
	return 0;
}

static int exec_im2col(const struct c7x_cmd *cmd)
{
	const uint8_t *src1 = (const uint8_t *)(uintptr_t)cmd->src1;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_im2col_desc *d = (const struct c7x_im2col_desc *)(uintptr_t)cmd->src3;
	uint32_t stype = cmd->p0;
	uint32_t dtype = cmd->p1;
	uint32_t in, ioh, iow, iic, ikh, ikw;
	uint64_t col_stride;

	if (!src1 || !dst || !d) {
		return -1;
	}
	if ((stype != C7X_DTYPE_F32 && stype != C7X_DTYPE_F16) ||
	    (dtype != C7X_DTYPE_F32 && dtype != C7X_DTYPE_F16)) {
		return -2;
	}
	col_stride = (uint64_t)d->IC * (uint64_t)d->KH * (uint64_t)d->KW;

	for (in = 0u; in < d->N; in++) {
		for (ioh = 0u; ioh < d->OH; ioh++) {
			for (iow = 0u; iow < d->OW; iow++) {
				uint64_t patch_idx = ((uint64_t)in * d->OH + ioh) * d->OW + iow;
				uint64_t dst_off = patch_idx * col_stride;

				for (iic = 0u; iic < d->IC; iic++) {
					const uint8_t *s_chan = src1 + (uint64_t)in * d->src_nb_n +
								(uint64_t)iic * d->src_nb_c;
					uint64_t c_base = dst_off + (uint64_t)iic * d->KH * d->KW;

					for (ikh = 0u; ikh < d->KH; ikh++) {
						int32_t iih = (int32_t)ioh * d->s1 + (int32_t)ikh * d->d1 - d->p1;
						int h_ok = (iih >= 0 && (uint32_t)iih < d->IH);
						const uint8_t *s_row = h_ok ? (s_chan + (uint64_t)iih * d->src_nb1) : NULL;

						for (ikw = 0u; ikw < d->KW; ikw++) {
							int32_t iiw = (int32_t)iow * d->s0 + (int32_t)ikw * d->d0 - d->p0;
							uint64_t di = c_base + (uint64_t)ikh * d->KW + ikw;

							if (!h_ok || iiw < 0 || (uint32_t)iiw >= d->IW) {
								if (dtype == C7X_DTYPE_F32) {
									((float *)dst)[di] = 0.0f;
								} else {
									((uint16_t *)dst)[di] = 0u;
								}
							} else if (stype == C7X_DTYPE_F32 && dtype == C7X_DTYPE_F32) {
								((float *)dst)[di] = ((const float *)s_row)[iiw];
							} else if (stype == C7X_DTYPE_F32 && dtype == C7X_DTYPE_F16) {
								((uint16_t *)dst)[di] = cmd_f32_to_f16(((const float *)s_row)[iiw]);
							} else if (stype == C7X_DTYPE_F16 && dtype == C7X_DTYPE_F16) {
								((uint16_t *)dst)[di] = ((const uint16_t *)s_row)[iiw];
							} else {
								((float *)dst)[di] = cmd_f16_to_f32(((const uint16_t *)s_row)[iiw]);
							}
						}
					}
				}
			}
		}
	}
	return 0;
}

static int exec_pool_2d_f32(const struct c7x_cmd *cmd)
{
	const uint8_t *src0 = (const uint8_t *)(uintptr_t)cmd->src0;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_pool2d_desc *d = (const struct c7x_pool2d_desc *)(uintptr_t)cmd->src3;
	int src0_f16 = (cmd->flags & C7X_CMD_FLAG_SRC0_F16) ? 1 : 0;
	uint32_t pl, oy, ox;
	int32_t ky, kx;
	float inv_ka;

	if (!src0 || !dst || !d || d->k0 <= 0 || d->k1 <= 0) {
		return -1;
	}
	if (d->op > C7X_POOL_AVG_EXCL_PAD) {
		return -2;
	}
	inv_ka = 1.0f / (float)(d->k0 * d->k1);

	for (pl = 0u; pl < d->n_planes; pl++) {
		const uint8_t *splane = src0 + (uint64_t)pl * d->src_nb2;
		uint8_t       *dplane = dst  + (uint64_t)pl * d->dst_nb2;

		for (oy = 0u; oy < d->OH; oy++) {
			float *drow = (float *)(dplane + (uint64_t)oy * d->dst_nb1);
			int32_t iy0 = (int32_t)oy * d->s1 - d->p1;

			for (ox = 0u; ox < d->OW; ox++) {
				int32_t ix0 = (int32_t)ox * d->s0 - d->p0;
				float res = (d->op == C7X_POOL_MAX) ? -3.402823466e+38f : 0.0f;
				uint32_t cnt = 0u;

				for (ky = 0; ky < d->k1; ky++) {
					int32_t iy = iy0 + ky;
					const uint8_t *srow;
					if (iy < 0 || (uint32_t)iy >= d->IH) continue;
					srow = splane + (uint64_t)iy * d->src_nb1;

					for (kx = 0; kx < d->k0; kx++) {
						int32_t ix = ix0 + kx;
						float v;
						if (ix < 0 || (uint32_t)ix >= d->IW) continue;
						v = src0_f16
						    ? cmd_f16_to_f32(((const uint16_t *)srow)[ix])
						    : ((const float *)srow)[ix];
						if (d->op == C7X_POOL_MAX) {
							if (v > res) res = v;
						} else {
							res += v;
						}
						cnt++;
					}
				}
				if (d->op == C7X_POOL_AVG) {
					res *= inv_ka;
				} else if (d->op == C7X_POOL_AVG_EXCL_PAD) {
					res = (cnt > 0u) ? (res / (float)cnt) : 0.0f;
				}
				drow[ox] = res;
			}
		}
	}
	return 0;
}

static void build_conv2d_patch_f32(const uint8_t *src_n, const struct c7x_conv2d_f32_desc *d,
				   uint32_t oy, uint32_t ox, uint32_t k_pad16, float *a_l2)
{
	uint32_t ic, ky, kx, idx = 0u;

	for (ic = 0u; ic < d->IC; ic++) {
		const uint8_t *sc = src_n + (uint64_t)ic * d->src_nb[2];
		for (ky = 0u; ky < d->KH; ky++) {
			int32_t sy = (int32_t)oy * d->s1 + (int32_t)ky * d->d1 - d->p1;
			int y_ok = (sy >= 0 && (uint32_t)sy < d->IH);
			const uint8_t *sy_p = y_ok ? (sc + (uint64_t)sy * d->src_nb[1]) : NULL;
			for (kx = 0u; kx < d->KW; kx++) {
				int32_t sx = (int32_t)ox * d->s0 + (int32_t)kx * d->d0 - d->p0;
				if (!y_ok || sx < 0 || (uint32_t)sx >= d->IW) {
					a_l2[idx++] = 0.0f;
				} else {
					a_l2[idx++] = *(const float *)(sy_p + (uint64_t)sx * d->src_nb[0]);
				}
			}
		}
	}
	while (idx < k_pad16) {
		a_l2[idx++] = 0.0f;
	}
}

static int exec_conv_2d_f32(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const uint8_t *kern = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *src1 = (const uint8_t *)(uintptr_t)cmd->src1;
	const float   *bias = NULL;
	uint8_t       *dst  = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_conv2d_f32_desc *d =
		(const struct c7x_conv2d_f32_desc *)(uintptr_t)cmd->src3;
	int kern_f16 = (cmd->flags & C7X_CMD_FLAG_SRC0_F16) ? 1 : 0;
	int do_relu  = (cmd->flags & C7X_CMD_FLAG_RELU) ? 1 : 0;
	uint32_t elem_k = kern_f16 ? (uint32_t)sizeof(uint16_t) : (uint32_t)sizeof(float);
	uint32_t k_col, k_pad16, oc_chunk, oc0, in, oy, ox, oc;
	uint64_t a_bytes, k_row_bytes;

	if (!kern || !src1 || !dst || !d || d->IC == 0u || d->OC == 0u ||
	    d->KH == 0u || d->KW == 0u) {
		return -1;
	}
	if (d->has_bias) {
		bias = (const float *)(uintptr_t)cmd->src2;
		if (!bias) return -2;
	}

	k_col = d->IC * d->KH * d->KW;
	k_pad16 = (k_col + 15u) & ~15u;
	a_bytes = (uint64_t)k_pad16 * sizeof(float);
	k_row_bytes = (uint64_t)k_col * elem_k;

	/* L2-staged kernel strips + 56 GB/s SE0 l2_matvec_f32 across all N*OH*OW patches. */
	if (l2_scratch && l2_bytes >= 4096u &&
	    a_bytes + (uint64_t)k_pad16 * sizeof(float) + 256u <= (uint64_t)l2_bytes) {
		/* Reserve a_l2[k_pad16], out_tmp[oc_chunk] (aligned 64B), and b_l2[oc_chunk][k_pad16]. */
		uint32_t avail = l2_bytes - (uint32_t)a_bytes;
		uint32_t per_oc = k_pad16 * (uint32_t)sizeof(float) + (uint32_t)sizeof(float);
		float *a_l2 = (float *)l2_scratch;
		uint32_t k_v = k_pad16 >> 4;

		oc_chunk = avail / per_oc;
		if (oc_chunk > d->OC) oc_chunk = d->OC;
		if (oc_chunk > 0u) {
			uint32_t out_bytes = ((oc_chunk * (uint32_t)sizeof(float)) + 63u) & ~63u;
			float *out_tmp = (float *)((uint8_t *)l2_scratch + a_bytes);
			float *b_l2    = (float *)((uint8_t *)out_tmp + out_bytes);

			/* Re-check oc_chunk after 64B alignment of out_tmp. */
			if (a_bytes + out_bytes + (uint64_t)oc_chunk * k_pad16 * sizeof(float) > (uint64_t)l2_bytes) {
				oc_chunk--;
			}
			if (oc_chunk > 0u) {
				for (oc0 = 0u; oc0 < d->OC; oc0 += oc_chunk) {
					uint32_t occ = (d->OC - oc0 < oc_chunk) ? (d->OC - oc0) : oc_chunk;
					stage_src0_to_l2(kern + (uint64_t)oc0 * k_row_bytes,
							 k_row_bytes, occ, k_col, k_pad16, k_col,
							 kern_f16, b_l2);
					for (in = 0u; in < d->N; in++) {
						const uint8_t *src_n = src1 + (uint64_t)in * d->src_nb[3];
						uint8_t *dst_n = dst + (uint64_t)in * d->dst_nb[3];
						for (oy = 0u; oy < d->OH; oy++) {
							uint8_t *dst_y = dst_n + (uint64_t)oy * d->dst_nb[1];
							for (ox = 0u; ox < d->OW; ox++) {
								uint8_t *dst_xy = dst_y + (uint64_t)ox * d->dst_nb[0];
								build_conv2d_patch_f32(src_n, d, oy, ox, k_pad16, a_l2);
								l2_matvec_f32(a_l2, b_l2, k_v, occ, out_tmp);
								for (oc = 0u; oc < occ; oc++) {
									float v = out_tmp[oc];
									if (bias) v += bias[oc0 + oc];
									if (do_relu && v < 0.0f) v = 0.0f;
									*(float *)(dst_xy + (uint64_t)(oc0 + oc) * d->dst_nb[2]) = v;
								}
							}
						}
					}
				}
				return 0;
			}
		}
	}

	/* Direct scalar fallback for huge kernels exceeding L2 SRAM */
	for (in = 0u; in < d->N; in++) {
		const uint8_t *src_n = src1 + (uint64_t)in * d->src_nb[3];
		uint8_t *dst_n = dst + (uint64_t)in * d->dst_nb[3];
		for (oy = 0u; oy < d->OH; oy++) {
			for (ox = 0u; ox < d->OW; ox++) {
				for (oc = 0u; oc < d->OC; oc++) {
					const uint8_t *kbase = kern + (uint64_t)oc * k_row_bytes;
					float acc = bias ? bias[oc] : 0.0f;
					uint32_t ic, ky, kx, ki = 0u;
					for (ic = 0u; ic < d->IC; ic++) {
						const uint8_t *sc = src_n + (uint64_t)ic * d->src_nb[2];
						for (ky = 0u; ky < d->KH; ky++) {
							int32_t sy = (int32_t)oy * d->s1 + (int32_t)ky * d->d1 - d->p1;
							if (sy < 0 || (uint32_t)sy >= d->IH) {
								ki += d->KW;
								continue;
							}
							const uint8_t *sy_p = sc + (uint64_t)sy * d->src_nb[1];
							for (kx = 0u; kx < d->KW; kx++) {
								int32_t sx = (int32_t)ox * d->s0 + (int32_t)kx * d->d0 - d->p0;
								if (sx >= 0 && (uint32_t)sx < d->IW) {
									float xv = *(const float *)(sy_p + (uint64_t)sx * d->src_nb[0]);
									float wv = kern_f16
										   ? cmd_f16_to_f32(((const uint16_t *)kbase)[ki])
										   : ((const float *)kbase)[ki];
									acc += xv * wv;
								}
								ki++;
							}
						}
					}
					if (do_relu && acc < 0.0f) acc = 0.0f;
					*(float *)(dst_n + (uint64_t)oc * d->dst_nb[2] +
						   (uint64_t)oy * d->dst_nb[1] +
						   (uint64_t)ox * d->dst_nb[0]) = acc;
				}
			}
		}
	}
	return 0;
}

static inline int32_t cmd_round_clamp_i32(float v, int32_t lo, int32_t hi)
{
	int32_t tmp[16] __attribute__((aligned(64)));
	int32_t q;

	if (v <= (float)lo) return lo;
	if (v >= (float)hi) return hi;
	/* Exact IEEE-754 round-to-nearest-even matching __float_to_int vector lanes & ONNX/ORT. */
	*(__int16_ptr)tmp = __float_to_int((float16)v);
	q = tmp[0];
	if (q < lo) q = lo;
	if (q > hi) q = hi;
	return q;
}

static void qlinear_store_elem(uint8_t *dst, uint64_t idx, int32_t acc_corr,
			       float a_scale, float b_scale, float inv_y_scale, int32_t y_zp,
			       const void *bias, uint32_t ch, uint32_t flags)
{
	int bias_i32 = (flags & C7X_CMD_FLAG_BIAS_I32) ? 1 : 0;
	int do_relu  = (flags & C7X_CMD_FLAG_RELU) ? 1 : 0;
	int y_f32    = (flags & C7X_CMD_FLAG_Y_F32) ? 1 : 0;
	int y_u8     = (flags & C7X_CMD_FLAG_Y_UINT8) ? 1 : 0;
	float val;

	if (bias && bias_i32) {
		acc_corr += ((const int32_t *)bias)[ch];
	}
	val = (float)acc_corr * (a_scale * b_scale);
	if (bias && !bias_i32) {
		val += ((const float *)bias)[ch];
	}
	if (do_relu && val < 0.0f) {
		val = 0.0f;
	}
	if (y_f32) {
		((float *)dst)[idx] = val;
	} else {
		float scaled = val * inv_y_scale + (float)y_zp;
		if (y_u8) {
			int32_t lo = do_relu ? (y_zp > 0 ? y_zp : 0) : 0;
			((uint8_t *)dst)[idx] = (uint8_t)cmd_round_clamp_i32(scaled, lo, 255);
		} else {
			int32_t lo = do_relu ? (y_zp > -128 ? y_zp : -128) : -128;
			((int8_t *)dst)[idx] = (int8_t)cmd_round_clamp_i32(scaled, lo, 127);
		}
	}
}

/*
 * 16-wide SIMD QLinear epilogue (zero libc rintf() calls, zero scalar divisions):
 * Writes n_valid elements starting at output channel n0 to either contiguous
 * row (dst_stride_elems == 1, QLinearMatMul) or strided NCHW channel layout
 * (dst_stride_elems == hw_out, QLinearConv2D).
 */
static void qlinear_epilogue_vec16(uint8_t *dst, uint64_t dst_base_idx, uint64_t dst_stride_elems,
				   const int32_t *cr, const int32_t *b_sum, const float *b_scales,
				   const void *bias, uint32_t n0, uint32_t n_valid,
				   int32_t a_zp, int32_t const_term,
				   float a_scale, float inv_y_scale, int32_t y_zp,
				   uint32_t per_ch, uint32_t flags)
{
	int bias_i32 = (flags & C7X_CMD_FLAG_BIAS_I32) ? 1 : 0;
	int do_relu  = (flags & C7X_CMD_FLAG_RELU) ? 1 : 0;
	int y_f32    = (flags & C7X_CMD_FLAG_Y_F32) ? 1 : 0;
	int y_u8     = (flags & C7X_CMD_FLAG_Y_UINT8) ? 1 : 0;
	int32_t lo   = y_u8 ? (do_relu ? (y_zp > 0 ? y_zp : 0) : 0)
			    : (do_relu ? (y_zp > -128 ? y_zp : -128) : -128);
	int32_t hi   = y_u8 ? 255 : 127;
	uint32_t n_vec = n_valid & ~15u;
	uint32_t ni = 0u, j;

	if (n_vec >= 16u) {
		int16   azp_v = (int16)a_zp;
		int16   cst_v = (int16)const_term;
		float16 asc_v = (float16)a_scale;
		float16 ab_v  = (float16)(a_scale * b_scales[0]);
		float16 iy_v  = (float16)inv_y_scale;
		float16 yz_v  = (float16)((float)y_zp);
		float16 z_v   = (float16)0.0f;
		int16   lo_v  = (int16)lo;
		int16   hi_v  = (int16)hi;
		int32_t qtmp[16] __attribute__((aligned(64)));
		float   ftmp[16] __attribute__((aligned(64)));

		for (; ni < n_vec; ni += 16u) {
			uint32_t col = n0 + ni;
			int16 cv   = *(__int16_ptr)(int32_t *)&cr[ni];
			int16 bsv  = *(__int16_ptr)(int32_t *)&b_sum[col];
			int16 corr = cv - azp_v * bsv + cst_v;
			float16 sc, val;

			if (bias && bias_i32) {
				corr += *(__int16_ptr)(int32_t *)&((const int32_t *)bias)[col];
			}
			sc  = per_ch ? (*(__float16_ptr)(float *)&b_scales[col] * asc_v) : ab_v;
			val = __int_to_float(corr) * sc;
			if (bias && !bias_i32) {
				val += *(__float16_ptr)(float *)&((const float *)bias)[col];
			}
			if (do_relu) {
				val = __max(val, z_v);
			}

			if (y_f32) {
				if (dst_stride_elems == 1ull) {
					float *out_p = &((float *)dst)[dst_base_idx + ni];
					if ((((uintptr_t)out_p) & 63u) == 0u) {
						*(__float16_ptr)out_p = val;
					} else {
						*(__float16_ptr)ftmp = val;
						memcpy(out_p, ftmp, 16u * sizeof(float));
					}
				} else {
					*(__float16_ptr)ftmp = val;
					for (j = 0u; j < 16u; j++) {
						((float *)dst)[dst_base_idx + (uint64_t)(ni + j) * dst_stride_elems] = ftmp[j];
					}
				}
			} else {
				float16 scaled = val * iy_v + yz_v;
				int16 qv = __min(__max(__float_to_int(scaled), lo_v), hi_v);
				*(__int16_ptr)qtmp = qv;
				if (y_u8) {
					if (dst_stride_elems == 1ull) {
						uint8_t *out = dst + dst_base_idx + ni;
						for (j = 0u; j < 16u; j++) out[j] = (uint8_t)qtmp[j];
					} else {
						for (j = 0u; j < 16u; j++) {
							dst[dst_base_idx + (uint64_t)(ni + j) * dst_stride_elems] = (uint8_t)qtmp[j];
						}
					}
				} else {
					if (dst_stride_elems == 1ull) {
						int8_t *out = (int8_t *)dst + dst_base_idx + ni;
						for (j = 0u; j < 16u; j++) out[j] = (int8_t)qtmp[j];
					} else {
						for (j = 0u; j < 16u; j++) {
							((int8_t *)dst)[dst_base_idx + (uint64_t)(ni + j) * dst_stride_elems] = (int8_t)qtmp[j];
						}
					}
				}
			}
		}
	}

	for (; ni < n_valid; ni++) {
		uint32_t col = n0 + ni;
		int32_t corr = cr[ni] - a_zp * b_sum[col] + const_term;
		float bs = per_ch ? b_scales[col] : b_scales[0];
		qlinear_store_elem(dst, dst_base_idx + (uint64_t)ni * dst_stride_elems, corr,
				   a_scale, bs, inv_y_scale, y_zp, bias, col, flags);
	}
}

static int exec_qlinear_matmul(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const int8_t  *b_tiled  = (const int8_t *)(uintptr_t)cmd->src0;
	const uint8_t *a_in     = (const uint8_t *)(uintptr_t)cmd->src1;
	const void    *bias     = cmd->src2 ? (const void *)(uintptr_t)cmd->src2 : NULL;
	uint8_t       *dst      = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_qlinear_matmul_desc *d =
		(const struct c7x_qlinear_matmul_desc *)(uintptr_t)cmd->src3;
	const int32_t *b_sum;
	const float   *b_scales;
	int a_u8 = (cmd->flags & C7X_CMD_FLAG_A_UINT8) ? 1 : 0;
	uint32_t max_bm, m_blk, n_chunk, m0, n0, mi, ki;
	int32_t k_zp_term;

	if (!b_tiled || !a_in || !dst || !d || !l2_scratch || l2_bytes < 8192u) {
		return -1;
	}
	if (d->M == 0u || d->K == 0u || d->N == 0u ||
	    (d->k_pad & 63u) != 0u || (d->n_pad & 63u) != 0u ||
	    d->k_pad < d->K || d->n_pad < d->N ||
	    !d->b_sum_addr || !d->b_scales_addr) {
		return -2;
	}
	b_sum    = (const int32_t *)(uintptr_t)d->b_sum_addr;
	b_scales = (const float *)(uintptr_t)d->b_scales_addr;
	k_zp_term = (int32_t)d->K * d->a_zp * d->b_zp;

#if defined(__C7X_HOSTEM__) || defined(_HOST_BUILD)
	max_bm = 16u;
#else
	max_bm = 60u;
#endif

	/* Choose n_chunk (multiple of 64) and m_blk (1..max_bm) that fit cleanly in L2 SRAM. */
	n_chunk = d->n_pad;
	if ((uint64_t)d->k_pad + (uint64_t)n_chunk * sizeof(int32_t) + 256u > (uint64_t)l2_bytes) {
		uint32_t max_n = (l2_bytes > d->k_pad + 256u)
				 ? ((l2_bytes - d->k_pad - 256u) / (uint32_t)sizeof(int32_t))
				 : 0u;
		n_chunk = max_n & ~63u;
		if (n_chunk == 0u) return -3;
	}
	{
		uint32_t per_row = d->k_pad + n_chunk * (uint32_t)sizeof(int32_t) + (uint32_t)sizeof(int32_t);
		m_blk = (l2_bytes - 256u) / per_row;
		if (m_blk == 0u) m_blk = 1u;
		if (m_blk > max_bm) m_blk = max_bm;
		if (m_blk > d->M) m_blk = d->M;
	}

	{
		uint32_t a_bytes    = (m_blk * d->k_pad + 63u) & ~63u;
		uint32_t asum_bytes = (m_blk * (uint32_t)sizeof(int32_t) + 63u) & ~63u;
		int8_t  *a_l2    = (int8_t *)l2_scratch;
		int32_t *a_sum   = (int32_t *)((uint8_t *)l2_scratch + a_bytes);
		int32_t *c_l2    = (int32_t *)((uint8_t *)a_sum + asum_bytes);
		uint32_t kt_cnt  = d->k_pad >> 6;

		for (m0 = 0u; m0 < d->M; m0 += m_blk) {
			uint32_t mc = (d->M - m0 < m_blk) ? (d->M - m0) : m_blk;

			/* Stage signed int8 A rows [mc][k_pad] into L2 SRAM & compute a_sum[mi]. */
			for (mi = 0u; mi < mc; mi++) {
				const uint8_t *ar = a_in + (uint64_t)(m0 + mi) * d->K;
				int8_t *ao = a_l2 + (uint64_t)mi * d->k_pad;
				int32_t s = 0;

				if (a_u8) {
					for (ki = 0u; ki < d->K; ki++) {
						int32_t v = (int32_t)ar[ki] - 128;
						ao[ki] = (int8_t)v;
						s += v;
					}
				} else {
					const int8_t *ars = (const int8_t *)ar;
					for (ki = 0u; ki < d->K; ki++) {
						int8_t v = ars[ki];
						ao[ki] = v;
						s += (int32_t)v;
					}
				}
				if (d->k_pad > d->K) {
					memset(ao + d->K, 0, d->k_pad - d->K);
				}
				a_sum[mi] = s;
			}

			for (n0 = 0u; n0 < d->n_pad; n0 += n_chunk) {
				uint32_t nc = (d->n_pad - n0 < n_chunk) ? (d->n_pad - n0) : n_chunk;
				uint32_t n_valid = (n0 < d->N) ? ((d->N - n0 < nc) ? (d->N - n0) : nc) : 0u;
				const int8_t *b_slice = b_tiled + (uint64_t)(n0 >> 6) * kt_cnt * 4096ull;

				if (n_valid == 0u) break;
				if (mc == 1u) {
					c7x_mma_gemv_i8_i32_tiled(a_l2, b_slice, c_l2, d->k_pad, nc);
				} else {
					c7x_mma_gemm_i8_i32_tiled(a_l2, b_slice, c_l2, mc, d->k_pad, nc, mc);
				}

				{
					float inv_y_scale = (d->y_scale > 0.0f) ? (1.0f / d->y_scale) : 0.0f;
					for (mi = 0u; mi < mc; mi++) {
						int32_t const_term = k_zp_term - d->b_zp * a_sum[mi];
						const int32_t *cr = c_l2 + (uint64_t)mi * nc;
						uint64_t row_off = (uint64_t)(m0 + mi) * d->N + n0;

						qlinear_epilogue_vec16(dst, row_off, 1ull, cr, b_sum, b_scales,
								       bias, n0, n_valid, d->a_zp, const_term,
								       d->a_scale, inv_y_scale, d->y_zp,
								       d->per_channel_b, cmd->flags);
					}
				}
			}
		}
	}

	return 0;
}

static int exec_qlinear_conv2d(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const int8_t  *b_data   = (const int8_t *)(uintptr_t)cmd->src0;
	const uint8_t *x_in     = (const uint8_t *)(uintptr_t)cmd->src1;
	const void    *bias     = cmd->src2 ? (const void *)(uintptr_t)cmd->src2 : NULL;
	uint8_t       *dst      = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_qlinear_conv2d_desc *d =
		(const struct c7x_qlinear_conv2d_desc *)(uintptr_t)cmd->src3;
	const int32_t *b_sum;
	const float   *b_scales;
	int a_u8 = (cmd->flags & C7X_CMD_FLAG_A_UINT8) ? 1 : 0;
	uint32_t k_col, max_bm, m_blk, n_chunk;
	uint64_t hw_in, hw_out, total_patches;

	if (!b_data || !x_in || !dst || !d) {
		return -1;
	}
	if (d->N == 0u || d->IC == 0u || d->OC == 0u ||
	    d->KH == 0u || d->KW == 0u || d->OH == 0u || d->OW == 0u ||
	    !d->b_sum_addr || !d->b_scales_addr) {
		return -2;
	}
	b_sum    = (const int32_t *)(uintptr_t)d->b_sum_addr;
	b_scales = (const float *)(uintptr_t)d->b_scales_addr;
	hw_in    = (uint64_t)d->IH * d->IW;
	hw_out   = (uint64_t)d->OH * d->OW;

	/* Depthwise QLinearConv2D (groups == IC == OC): weights are contiguous int8[OC][KH*KW]. */
	if (d->groups == d->IC && d->groups == d->OC && d->groups > 1u) {
		uint32_t in, c, oy, ox, ky, kx;
		int32_t k_zp_term = (int32_t)(d->KH * d->KW) * d->a_zp * d->b_zp;
		float inv_y_scale = (d->y_scale > 0.0f) ? (1.0f / d->y_scale) : 0.0f;

		for (in = 0u; in < d->N; in++) {
			for (c = 0u; c < d->OC; c++) {
				const uint8_t *xc = x_in + ((uint64_t)in * d->IC + c) * hw_in;
				const int8_t  *wc = b_data + (uint64_t)c * d->KH * d->KW;
				uint64_t dst_c = ((uint64_t)in * d->OC + c) * hw_out;
				float bs = d->per_channel_b ? b_scales[c] : b_scales[0];
				int32_t bsum_c = b_sum[c];

				for (oy = 0u; oy < d->OH; oy++) {
					for (ox = 0u; ox < d->OW; ox++) {
						int32_t acc = 0, asum = 0;
						uint32_t ki = 0u;

						for (ky = 0u; ky < d->KH; ky++) {
							int32_t sy = (int32_t)oy * d->s1 + (int32_t)ky * d->d1 - d->p1;
							int y_ok = (sy >= 0 && (uint32_t)sy < d->IH);
							for (kx = 0u; kx < d->KW; kx++) {
								int32_t sx = (int32_t)ox * d->s0 + (int32_t)kx * d->d0 - d->p0;
								int32_t av;
								if (!y_ok || sx < 0 || (uint32_t)sx >= d->IW) {
									av = d->a_zp;
								} else if (a_u8) {
									av = (int32_t)xc[(uint64_t)sy * d->IW + (uint32_t)sx] - 128;
								} else {
									av = (int32_t)((const int8_t *)xc)[(uint64_t)sy * d->IW + (uint32_t)sx];
								}
								acc += av * (int32_t)wc[ki++];
								asum += av;
							}
						}
						{
							int32_t corr = acc - d->a_zp * bsum_c - d->b_zp * asum + k_zp_term;
							qlinear_store_elem(dst, dst_c + (uint64_t)oy * d->OW + ox,
									   corr, d->a_scale, bs, inv_y_scale, d->y_zp,
									   bias, c, cmd->flags);
						}
					}
				}
			}
		}
		return 0;
	}

	if (d->groups != 1u) {
		return -3;
	}
	k_col = d->IC * d->KH * d->KW;
	if (!l2_scratch || l2_bytes < 8192u ||
	    (d->k_pad & 63u) != 0u || (d->n_pad & 63u) != 0u ||
	    d->k_pad < k_col || d->n_pad < d->OC) {
		return -4;
	}

#if defined(__C7X_HOSTEM__) || defined(_HOST_BUILD)
	max_bm = 16u;
#else
	max_bm = 60u;
#endif

	total_patches = (uint64_t)d->N * hw_out;
	n_chunk = d->n_pad;
	if ((uint64_t)d->k_pad + (uint64_t)n_chunk * sizeof(int32_t) + 256u > (uint64_t)l2_bytes) {
		uint32_t max_n = (l2_bytes > d->k_pad + 256u)
				 ? ((l2_bytes - d->k_pad - 256u) / (uint32_t)sizeof(int32_t))
				 : 0u;
		n_chunk = max_n & ~63u;
		if (n_chunk == 0u) return -5;
	}
	{
		uint32_t per_row = d->k_pad + n_chunk * (uint32_t)sizeof(int32_t) + (uint32_t)sizeof(int32_t);
		m_blk = (l2_bytes - 256u) / per_row;
		if (m_blk == 0u) m_blk = 1u;
		if (m_blk > max_bm) m_blk = max_bm;
		if ((uint64_t)m_blk > total_patches) m_blk = (uint32_t)total_patches;
	}

	/* L2-fused im2col -> MMA int8*int8->int32 -> zero-point + bias + ReLU + requant -> NCHW dst */
	{
		uint32_t a_bytes    = (m_blk * d->k_pad + 63u) & ~63u;
		uint32_t asum_bytes = (m_blk * (uint32_t)sizeof(int32_t) + 63u) & ~63u;
		int8_t  *a_l2    = (int8_t *)l2_scratch;
		int32_t *a_sum   = (int32_t *)((uint8_t *)l2_scratch + a_bytes);
		int32_t *c_l2    = (int32_t *)((uint8_t *)a_sum + asum_bytes);
		uint32_t kt_cnt  = d->k_pad >> 6;
		int32_t k_zp_term = (int32_t)k_col * d->a_zp * d->b_zp;
		uint64_t p0;

		for (p0 = 0ull; p0 < total_patches; p0 += m_blk) {
			uint32_t mc = (total_patches - p0 < (uint64_t)m_blk)
				      ? (uint32_t)(total_patches - p0)
				      : m_blk;
			uint32_t mi, n0;

			/* Build mc im2col patches directly into L2 SRAM with a_zp padding */
			for (mi = 0u; mi < mc; mi++) {
				uint64_t p = p0 + mi;
				uint32_t in = (uint32_t)(p / hw_out);
				uint32_t rem = (uint32_t)(p % hw_out);
				uint32_t oy = rem / d->OW;
				uint32_t ox = rem % d->OW;
				const uint8_t *xn = x_in + (uint64_t)in * d->IC * hw_in;
				int8_t *ao = a_l2 + (uint64_t)mi * d->k_pad;
				int32_t s = 0;
				uint32_t ic, ky, kx, ki = 0u;

				for (ic = 0u; ic < d->IC; ic++) {
					const uint8_t *xc = xn + (uint64_t)ic * hw_in;
					for (ky = 0u; ky < d->KH; ky++) {
						int32_t sy = (int32_t)oy * d->s1 + (int32_t)ky * d->d1 - d->p1;
						int y_ok = (sy >= 0 && (uint32_t)sy < d->IH);
						for (kx = 0u; kx < d->KW; kx++) {
							int32_t sx = (int32_t)ox * d->s0 + (int32_t)kx * d->d0 - d->p0;
							int32_t av;
							if (!y_ok || sx < 0 || (uint32_t)sx >= d->IW) {
								av = d->a_zp;
							} else if (a_u8) {
								av = (int32_t)xc[(uint64_t)sy * d->IW + (uint32_t)sx] - 128;
							} else {
								av = (int32_t)((const int8_t *)xc)[(uint64_t)sy * d->IW + (uint32_t)sx];
							}
							ao[ki++] = (int8_t)av;
							s += av;
						}
					}
				}
				if (d->k_pad > k_col) {
					memset(ao + k_col, 0, d->k_pad - k_col);
				}
				a_sum[mi] = s;
			}

			for (n0 = 0u; n0 < d->n_pad; n0 += n_chunk) {
				uint32_t nc = (d->n_pad - n0 < n_chunk) ? (d->n_pad - n0) : n_chunk;
				uint32_t n_valid = (n0 < d->OC) ? ((d->OC - n0 < nc) ? (d->OC - n0) : nc) : 0u;
				const int8_t *b_slice = b_data + (uint64_t)(n0 >> 6) * kt_cnt * 4096ull;

				if (n_valid == 0u) break;
				if (mc == 1u) {
					c7x_mma_gemv_i8_i32_tiled(a_l2, b_slice, c_l2, d->k_pad, nc);
				} else {
					c7x_mma_gemm_i8_i32_tiled(a_l2, b_slice, c_l2, mc, d->k_pad, nc, mc);
				}

				{
					float inv_y_scale = (d->y_scale > 0.0f) ? (1.0f / d->y_scale) : 0.0f;
					for (mi = 0u; mi < mc; mi++) {
						uint64_t p = p0 + mi;
						uint32_t in = (uint32_t)(p / hw_out);
						uint32_t rem = (uint32_t)(p % hw_out);
						int32_t const_term = k_zp_term - d->b_zp * a_sum[mi];
						const int32_t *cr = c_l2 + (uint64_t)mi * nc;
						uint64_t dst_base = (uint64_t)in * d->OC * hw_out + (uint64_t)n0 * hw_out + rem;

						qlinear_epilogue_vec16(dst, dst_base, hw_out, cr, b_sum, b_scales,
								       bias, n0, n_valid, d->a_zp, const_term,
								       d->a_scale, inv_y_scale, d->y_zp,
								       d->per_channel_b, cmd->flags);
					}
				}
			}
		}
	}

	return 0;
}

static int exec_dequantize_linear(const struct c7x_cmd *cmd)
{
	const uint8_t *src0   = (const uint8_t *)(uintptr_t)cmd->src0;
	const float   *scales = cmd->src1 ? (const float *)(uintptr_t)cmd->src1 : NULL;
	const uint8_t *zps    = cmd->src2 ? (const uint8_t *)(uintptr_t)cmd->src2 : NULL;
	float         *dst    = (float *)(uintptr_t)cmd->dst;
	const struct c7x_dequant_desc *d =
		(const struct c7x_dequant_desc *)(uintptr_t)cmd->src3;
	int is_u8 = (cmd->flags & C7X_CMD_FLAG_A_UINT8) ? 1 : 0;
	uint32_t o, a, i;

	if (!src0 || !dst || !d || d->outer == 0u || d->axis_dim == 0u || d->inner == 0u) {
		return -1;
	}
	if (d->axis_dim > 1u && !scales) {
		return -2;
	}
	if (d->axis_dim > 1u && d->has_zp && !zps) {
		return -3;
	}

	for (o = 0u; o < d->outer; o++) {
		for (a = 0u; a < d->axis_dim; a++) {
			float sc = (d->axis_dim == 1u) ? d->scalar_scale : scales[a];
			int32_t zp = 0;
			uint64_t base = ((uint64_t)o * d->axis_dim + a) * d->inner;
			const uint8_t *sp = src0 + base;
			float *dp = dst + base;
			uint32_t n_vec = d->inner & ~15u;
			float ftmp[16] __attribute__((aligned(64)));

			if (d->axis_dim == 1u) {
				zp = d->has_zp ? d->scalar_zp : 0;
			} else if (d->has_zp) {
				zp = is_u8 ? (int32_t)zps[a] : (int32_t)((const int8_t *)zps)[a];
			}

			i = 0u;
			if (n_vec >= 16u) {
				float16 sc_v = (float16)sc;
				int16   zp_v = (int16)zp;
				int32_t itmp[16] __attribute__((aligned(64)));
				uint32_t j;

				for (; i < n_vec; i += 16u) {
					if (is_u8) {
						for (j = 0u; j < 16u; j++) itmp[j] = (int32_t)sp[i + j];
					} else {
						const int8_t *sps = (const int8_t *)(sp + i);
						for (j = 0u; j < 16u; j++) itmp[j] = (int32_t)sps[j];
					}
					{
						float16 v = __int_to_float(*(__int16_ptr)itmp - zp_v) * sc_v;
						if ((((uintptr_t)&dp[i]) & 63u) == 0u) {
							*(__float16_ptr)&dp[i] = v;
						} else {
							*(__float16_ptr)ftmp = v;
							memcpy(&dp[i], ftmp, 16u * sizeof(float));
						}
					}
				}
			}
			for (; i < d->inner; i++) {
				int32_t x = is_u8 ? (int32_t)sp[i] : (int32_t)((const int8_t *)sp)[i];
				dp[i] = (float)(x - zp) * sc;
			}
		}
	}
	return 0;
}

static inline int fa_dtype_to_stage_fmt(uint32_t dt)
{
	if (dt == C7X_DTYPE_Q8_0) return 2;
	if (dt == C7X_DTYPE_F16)  return 1;
	return 0;
}

/* Accumulate acc[0..dv_pad16-1] = acc * alpha + sum_{c=0..bc-1} w[c] * v_l2[c][0..dv_pad16-1] */
static void fa_accum_v_l2(float *acc_l2, const float *v_l2, const float *w,
			  uint32_t bc, uint32_t dv_v, float alpha)
{
	uint32_t c, v;
	uint32_t dv_pad16 = dv_v << 4;

	if (dv_v == 4u) {
		float16 a0, a1, a2, a3;
		if (alpha == 0.0f) {
			a0 = a1 = a2 = a3 = (float16)0.0f;
		} else if (alpha == 1.0f) {
			a0 = *(__float16_ptr)(acc_l2);
			a1 = *(__float16_ptr)(acc_l2 + 16u);
			a2 = *(__float16_ptr)(acc_l2 + 32u);
			a3 = *(__float16_ptr)(acc_l2 + 48u);
		} else {
			float16 av = (float16)alpha;
			a0 = *(__float16_ptr)(acc_l2) * av;
			a1 = *(__float16_ptr)(acc_l2 + 16u) * av;
			a2 = *(__float16_ptr)(acc_l2 + 32u) * av;
			a3 = *(__float16_ptr)(acc_l2 + 48u) * av;
		}
		for (c = 0u; c < bc; c++) {
			float wc = w[c];
			if (wc != 0.0f) {
				float16 wv = (float16)wc;
				const float *vr = v_l2 + (uint64_t)c * 64u;
				a0 += wv * *(__float16_ptr)(float *)(vr);
				a1 += wv * *(__float16_ptr)(float *)(vr + 16u);
				a2 += wv * *(__float16_ptr)(float *)(vr + 32u);
				a3 += wv * *(__float16_ptr)(float *)(vr + 48u);
			}
		}
		*(__float16_ptr)(acc_l2)       = a0;
		*(__float16_ptr)(acc_l2 + 16u) = a1;
		*(__float16_ptr)(acc_l2 + 32u) = a2;
		*(__float16_ptr)(acc_l2 + 48u) = a3;
		return;
	}
	if (dv_v == 8u) {
		float16 a0, a1, a2, a3, a4, a5, a6, a7;
		if (alpha == 0.0f) {
			a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = (float16)0.0f;
		} else if (alpha == 1.0f) {
			a0 = *(__float16_ptr)(acc_l2);
			a1 = *(__float16_ptr)(acc_l2 + 16u);
			a2 = *(__float16_ptr)(acc_l2 + 32u);
			a3 = *(__float16_ptr)(acc_l2 + 48u);
			a4 = *(__float16_ptr)(acc_l2 + 64u);
			a5 = *(__float16_ptr)(acc_l2 + 80u);
			a6 = *(__float16_ptr)(acc_l2 + 96u);
			a7 = *(__float16_ptr)(acc_l2 + 112u);
		} else {
			float16 av = (float16)alpha;
			a0 = *(__float16_ptr)(acc_l2) * av;
			a1 = *(__float16_ptr)(acc_l2 + 16u) * av;
			a2 = *(__float16_ptr)(acc_l2 + 32u) * av;
			a3 = *(__float16_ptr)(acc_l2 + 48u) * av;
			a4 = *(__float16_ptr)(acc_l2 + 64u) * av;
			a5 = *(__float16_ptr)(acc_l2 + 80u) * av;
			a6 = *(__float16_ptr)(acc_l2 + 96u) * av;
			a7 = *(__float16_ptr)(acc_l2 + 112u) * av;
		}
		for (c = 0u; c < bc; c++) {
			float wc = w[c];
			if (wc != 0.0f) {
				float16 wv = (float16)wc;
				const float *vr = v_l2 + (uint64_t)c * 128u;
				a0 += wv * *(__float16_ptr)(float *)(vr);
				a1 += wv * *(__float16_ptr)(float *)(vr + 16u);
				a2 += wv * *(__float16_ptr)(float *)(vr + 32u);
				a3 += wv * *(__float16_ptr)(float *)(vr + 48u);
				a4 += wv * *(__float16_ptr)(float *)(vr + 64u);
				a5 += wv * *(__float16_ptr)(float *)(vr + 80u);
				a6 += wv * *(__float16_ptr)(float *)(vr + 96u);
				a7 += wv * *(__float16_ptr)(float *)(vr + 112u);
			}
		}
		*(__float16_ptr)(acc_l2)        = a0;
		*(__float16_ptr)(acc_l2 + 16u)  = a1;
		*(__float16_ptr)(acc_l2 + 32u)  = a2;
		*(__float16_ptr)(acc_l2 + 48u)  = a3;
		*(__float16_ptr)(acc_l2 + 64u)  = a4;
		*(__float16_ptr)(acc_l2 + 80u)  = a5;
		*(__float16_ptr)(acc_l2 + 96u)  = a6;
		*(__float16_ptr)(acc_l2 + 112u) = a7;
		return;
	}

	if (alpha == 0.0f) {
		for (v = 0u; v < dv_v; v++) {
			*(__float16_ptr)(acc_l2 + (v << 4)) = (float16)0.0f;
		}
	} else if (alpha != 1.0f) {
		float16 av = (float16)alpha;
		for (v = 0u; v < dv_v; v++) {
			*(__float16_ptr)(acc_l2 + (v << 4)) *= av;
		}
	}
	for (c = 0u; c < bc; c++) {
		float wc = w[c];
		if (wc != 0.0f) {
			float16 wv = (float16)wc;
			const float *vr = v_l2 + (uint64_t)c * dv_pad16;
			for (v = 0u; v < dv_v; v++) {
				*(__float16_ptr)(acc_l2 + (v << 4)) += wv * *(__float16_ptr)(float *)(vr + (v << 4));
			}
		}
	}
}

static int exec_flash_attn_ext(const struct c7x_cmd *cmd, void *l2_scratch, uint32_t l2_bytes)
{
	const uint8_t *q_base = (const uint8_t *)(uintptr_t)cmd->src0;
	const uint8_t *k_base = (const uint8_t *)(uintptr_t)cmd->src1;
	const uint8_t *v_base = (const uint8_t *)(uintptr_t)cmd->src2;
	uint8_t       *dst    = (uint8_t *)(uintptr_t)cmd->dst;
	const struct c7x_flash_attn_desc *d =
		(const struct c7x_flash_attn_desc *)(uintptr_t)cmd->src3;
	const uint8_t *mask_base  = d && d->mask_addr  ? (const uint8_t *)(uintptr_t)d->mask_addr  : NULL;
	const float   *sinks_base = d && d->sinks_addr ? (const float *)(uintptr_t)d->sinks_addr   : NULL;
	uint32_t DK, DV, N, n_kv, n_head, b3, rk2, rk3, rv2, rv3;
	uint32_t dk_pad16, dv_pad16, dk_v, dv_v;
	int k_fmt, v_fmt, mask_f16;
	float scale_eff;
	uint32_t b_kv, q_chunk;
	float *q_l2, *scores_l2, *acc_l2, *m_arr, *s_arr, *k_l2, *v_l2;
	uint32_t iq3, ik2;

	if (!q_base || !k_base || !v_base || !dst || !d || !l2_scratch) {
		return -1;
	}
	DK     = d->neq[0];
	N      = d->neq[1];
	n_head = d->neq[2];
	b3     = d->neq[3];
	n_kv   = d->nek[1];
	DV     = d->nev[0];
	if (DK == 0u || DV == 0u || N == 0u || n_head == 0u || b3 == 0u ||
	    d->nek[0] != DK || d->nev[1] != n_kv ||
	    d->nek[2] == 0u || d->nek[3] == 0u || d->nev[2] == 0u || d->nev[3] == 0u) {
		return -2;
	}
	if ((d->k_dtype == C7X_DTYPE_Q8_0 && (DK & 31u) != 0u) ||
	    (d->v_dtype == C7X_DTYPE_Q8_0 && (DV & 31u) != 0u)) {
		return -3;
	}

	rk2 = n_head / d->nek[2]; if (rk2 == 0u) rk2 = 1u;
	rk3 = b3     / d->nek[3]; if (rk3 == 0u) rk3 = 1u;
	rv2 = n_head / d->nev[2]; if (rv2 == 0u) rv2 = 1u;
	rv3 = b3     / d->nev[3]; if (rv3 == 0u) rv3 = 1u;
	if (rk2 != rv2 || rk3 != rv3 || (n_head % d->nek[2]) != 0u || (b3 % d->nek[3]) != 0u) {
		return -4;
	}

	k_fmt    = fa_dtype_to_stage_fmt(d->k_dtype);
	v_fmt    = fa_dtype_to_stage_fmt(d->v_dtype);
	mask_f16 = (d->mask_dtype == C7X_DTYPE_F16) ? 1 : 0;
	scale_eff = (d->logit_softcap != 0.0f) ? (d->scale / d->logit_softcap) : d->scale;

	dk_pad16 = (DK + 15u) & ~15u;
	dv_pad16 = (DV + 15u) & ~15u;
	dk_v     = dk_pad16 >> 4;
	dv_v     = dv_pad16 >> 4;

	/* Partition L2 scratchpad: q_l2 + acc_l2 + scores_l2 + M/S arrays + K/V tiles. */
	q_chunk = (N < 32u) ? N : 32u;
	while (q_chunk > 1u && (uint64_t)q_chunk * rk2 > 256u) {
		q_chunk >>= 1;
	}
	{
		uint32_t q_bytes   = CMD_ALIGN64(dk_pad16 * (uint32_t)sizeof(float));
		uint32_t acc_bytes = CMD_ALIGN64(dv_pad16 * (uint32_t)sizeof(float));
		uint32_t ms_cnt    = q_chunk * rk2;
		uint32_t ms_bytes  = CMD_ALIGN64(ms_cnt * (uint32_t)sizeof(float));
		uint32_t sc_bytes  = CMD_ALIGN64(256u * (uint32_t)sizeof(float));
		uint32_t fixed     = q_bytes + acc_bytes + 2u * ms_bytes + sc_bytes;
		uint32_t row_bytes = (dk_pad16 + dv_pad16) * (uint32_t)sizeof(float);

		if (l2_bytes <= fixed + row_bytes) {
			return -5;
		}
		b_kv = (l2_bytes - fixed) / row_bytes;
		if (b_kv > 256u) b_kv = 256u;
		if (b_kv == 0u) return -5;

		{
			uint8_t *p = (uint8_t *)l2_scratch;
			q_l2      = (float *)p; p += q_bytes;
			acc_l2    = (float *)p; p += acc_bytes;
			scores_l2 = (float *)p; p += sc_bytes;
			m_arr     = (float *)p; p += ms_bytes;
			s_arr     = (float *)p; p += ms_bytes;
			k_l2      = (float *)p; p += CMD_ALIGN64(b_kv * dk_pad16 * (uint32_t)sizeof(float));
			v_l2      = (float *)p;
		}
	}

	for (iq3 = 0u; iq3 < b3; iq3++) {
		uint32_t ik3 = iq3 / rk3;
		for (ik2 = 0u; ik2 < d->nek[2]; ik2++) {
			const uint8_t *k_head = k_base + (uint64_t)ik3 * d->nbk[3] + (uint64_t)ik2 * d->nbk[2];
			const uint8_t *v_head = v_base + (uint64_t)ik3 * d->nbv[3] + (uint64_t)ik2 * d->nbv[2];
			uint32_t q0;

			for (q0 = 0u; q0 < N; q0 += q_chunk) {
				uint32_t qc = (N - q0 < q_chunk) ? (N - q0) : q_chunk;
				uint32_t ic_min = 0u, ic_max = n_kv;
				uint32_t r_i2, qi, ic0;

				/* Zero running accumulators and init M = -INF, S = 0. */
				for (r_i2 = 0u; r_i2 < rk2; r_i2++) {
					uint32_t iq2 = ik2 * rk2 + r_i2;
					for (qi = 0u; qi < qc; qi++) {
						uint32_t iq1 = q0 + qi;
						uint32_t slot = r_i2 * qc + qi;
						float *dp = (float *)(dst + ((uint64_t)iq3 * N * n_head +
									     (uint64_t)iq2 + (uint64_t)iq1 * n_head) * d->nb_dst[1]);
						m_arr[slot] = -INFINITY;
						s_arr[slot] = 0.0f;
						memset(dp, 0, (size_t)DV * sizeof(float));
					}
				}

				/* Tighten active KV range [ic_min, ic_max) using mask if present! */
				if (mask_base && n_kv > 0u) {
					uint32_t found_min = n_kv, found_max = 0u;
					for (r_i2 = 0u; r_i2 < rk2; r_i2++) {
						uint32_t iq2 = ik2 * rk2 + r_i2;
						uint32_t m2 = d->nem[2] ? (iq2 % d->nem[2]) : 0u;
						uint32_t m3 = d->nem[3] ? (iq3 % d->nem[3]) : 0u;
						if (r_i2 > 0u && d->nem[2] <= 1u) break;
						for (qi = 0u; qi < qc; qi++) {
							uint32_t iq1 = q0 + qi;
							const uint8_t *mp = mask_base + (uint64_t)iq1 * d->nbm[1] +
									    (uint64_t)m2 * d->nbm[2] + (uint64_t)m3 * d->nbm[3];
							uint32_t c;
							if (mask_f16) {
								const uint16_t *m16 = (const uint16_t *)mp;
								for (c = 0u; c < found_min; c++) {
									if (m16[c] != 0xFC00u) { found_min = c; break; }
								}
								for (c = n_kv; c > found_max; c--) {
									if (m16[c - 1u] != 0xFC00u) { found_max = c; break; }
								}
							} else {
								const float *m32 = (const float *)mp;
								for (c = 0u; c < found_min; c++) {
									if (m32[c] != -INFINITY) { found_min = c; break; }
								}
								for (c = n_kv; c > found_max; c--) {
									if (m32[c - 1u] != -INFINITY) { found_max = c; break; }
								}
							}
						}
					}
					ic_min = found_min;
					ic_max = found_max;
				}

				/* Tile active KV tokens in blocks of b_kv, staging K & V into L2 SRAM ONCE per block! */
				for (ic0 = ic_min; ic0 < ic_max; ic0 += b_kv) {
					uint32_t bc = (ic_max - ic0 < b_kv) ? (ic_max - ic0) : b_kv;

					stage_src0_to_l2(k_head + (uint64_t)ic0 * d->nbk[1], d->nbk[1],
							 bc, DK, dk_pad16, DK, k_fmt, k_l2);
					stage_src0_to_l2(v_head + (uint64_t)ic0 * d->nbv[1], d->nbv[1],
							 bc, DV, dv_pad16, DV, v_fmt, v_l2);

					for (r_i2 = 0u; r_i2 < rk2; r_i2++) {
						uint32_t iq2 = ik2 * rk2 + r_i2;
						float slope = 1.0f;
						uint32_t m2 = d->nem[2] ? (iq2 % d->nem[2]) : 0u;
						uint32_t m3 = d->nem[3] ? (iq3 % d->nem[3]) : 0u;

						if (d->max_bias > 0.0f) {
							float base = (iq2 < d->n_head_log2) ? d->m0 : d->m1;
							int32_t exp_i = (iq2 < d->n_head_log2)
									? (int32_t)(iq2 + 1u)
									: (int32_t)(2u * (iq2 - d->n_head_log2) + 1u);
							slope = powf(base, (float)exp_i);
						}

						for (qi = 0u; qi < qc; qi++) {
							uint32_t iq1 = q0 + qi;
							uint32_t slot = r_i2 * qc + qi;
							const float *pq = (const float *)(q_base + (uint64_t)iq1 * d->nbq[1] +
											  (uint64_t)iq2 * d->nbq[2] +
											  (uint64_t)iq3 * d->nbq[3]);
							float *dp = (float *)(dst + ((uint64_t)iq3 * N * n_head +
										     (uint64_t)iq2 + (uint64_t)iq1 * n_head) * d->nb_dst[1]);
							const uint8_t *mp = mask_base
									    ? (mask_base + (uint64_t)iq1 * d->nbm[1] +
									       (uint64_t)m2 * d->nbm[2] + (uint64_t)m3 * d->nbm[3])
									    : NULL;
							uint32_t c, bc_vec;
							float tile_max = -INFINITY;
							float m_old, m_new, alpha, s_tile;

							memcpy(q_l2, pq, (size_t)DK * sizeof(float));
							for (c = DK; c < dk_pad16; c++) q_l2[c] = 0.0f;

							l2_matvec_f32(q_l2, k_l2, dk_v, bc, scores_l2);

							/* Apply scale, optional softcap, optional mask, and find tile_max. */
							for (c = 0u; c < bc; c++) {
								float mv = 0.0f;
								if (mp) {
									float raw_m = mask_f16
										      ? cmd_f16_to_f32(((const uint16_t *)mp)[ic0 + c])
										      : ((const float *)mp)[ic0 + c];
									mv = (raw_m == -INFINITY) ? -INFINITY : (slope * raw_m);
								}
								if (mv == -INFINITY) {
									scores_l2[c] = -INFINITY;
								} else {
									float sc = scores_l2[c] * scale_eff;
									if (d->logit_softcap != 0.0f) {
										sc = d->logit_softcap * tanhf(sc);
									}
									sc += mv;
									scores_l2[c] = sc;
									if (sc > tile_max) tile_max = sc;
								}
							}

							if (tile_max == -INFINITY) {
								continue;
							}

							m_old = m_arr[slot];
							m_new = (tile_max > m_old) ? tile_max : m_old;
							if (m_old == -INFINITY) {
								alpha = 0.0f;
							} else if (m_new == m_old) {
								alpha = 1.0f;
							} else {
								alpha = cmd_expf_scalar(m_old - m_new);
							}

							/* Vectorized exp(scores - m_new) and sum S_tile. */
							bc_vec = bc & ~15u;
							s_tile = 0.0f;
							c = 0u;
							if (bc_vec >= 16u) {
								float16 mv_v = (float16)m_new;
								float16 sum_v = (float16)0.0f;
								for (; c < bc_vec; c += 16u) {
									float16 sv = *(__float16_ptr)&scores_l2[c];
									float16 ev = cmd_vexpf_f16(sv - mv_v);
									*(__float16_ptr)&scores_l2[c] = ev;
									sum_v += ev;
								}
								s_tile = cmd_hsum_f16(sum_v);
							}
							for (; c < bc; c++) {
								float sv = scores_l2[c];
								float ev = (sv == -INFINITY) ? 0.0f : cmd_expf_scalar(sv - m_new);
								scores_l2[c] = ev;
								s_tile += ev;
							}

							if (alpha != 0.0f) {
								memcpy(acc_l2, dp, (size_t)DV * sizeof(float));
								for (c = DV; c < dv_pad16; c++) acc_l2[c] = 0.0f;
							}
							fa_accum_v_l2(acc_l2, v_l2, scores_l2, bc, dv_v, alpha);
							memcpy(dp, acc_l2, (size_t)DV * sizeof(float));

							m_arr[slot] = m_new;
							s_arr[slot] = s_arr[slot] * alpha + s_tile;
						}
					}
				}

				/* Finalize sinks and divide by S for every query in this chunk. */
				for (r_i2 = 0u; r_i2 < rk2; r_i2++) {
					uint32_t iq2 = ik2 * rk2 + r_i2;
					for (qi = 0u; qi < qc; qi++) {
						uint32_t iq1 = q0 + qi;
						uint32_t slot = r_i2 * qc + qi;
						float *dp = (float *)(dst + ((uint64_t)iq3 * N * n_head +
									     (uint64_t)iq2 + (uint64_t)iq1 * n_head) * d->nb_dst[1]);
						float M = m_arr[slot];
						float S = s_arr[slot];
						float scale_out;
						uint32_t c;

						if (sinks_base) {
							float sv = sinks_base[iq2];
							float ms = 1.0f, vs = 1.0f;
							if (sv > M) {
								ms = (M == -INFINITY) ? 0.0f : cmd_expf_scalar(M - sv);
								M = sv;
								for (c = 0u; c < DV; c++) dp[c] *= ms;
							} else {
								vs = cmd_expf_scalar(sv - M);
							}
							S = S * ms + vs;
						}
						scale_out = (S == 0.0f) ? 0.0f : (1.0f / S);
						for (c = 0u; c < DV; c++) {
							dp[c] *= scale_out;
						}
					}
				}
			}
		}
	}
	return 0;
}

/* -------------------------------------------------------------------------- */
/* Command-list entry point                                                   */
/* -------------------------------------------------------------------------- */

int c7x_cmd_exec_list(const struct c7x_cmd *cmds, uint32_t n_cmds,
		      void *l2_scratch, uint32_t l2_bytes,
		      uint32_t *done_out, uint32_t *fail_idx_out)
{
	uint32_t i;
	int l2_dirty = 0;

	if (done_out) *done_out = 0u;
	if (fail_idx_out) *fail_idx_out = 0u;
	if (n_cmds > 0u && !cmds) {
		return -1;
	}

	for (i = 0u; i < n_cmds; i++) {
		const struct c7x_cmd *cmd = &cmds[i];
		int rc = 0;

		switch (cmd->op) {
		case C7X_CMD_NOP:
			rc = 0;
			break;
		case C7X_CMD_MATMUL_W4A8_G64:
			/*
			 * Only flush dirty L1D lines covering l2_scratch right before DRU0 DMA
			 * is actually going to write into l2_scratch (DDR weights >= 1 MiB for m=1
			 * or >= 128 KiB for m>1). SE0 snoops L1D, so non-DRU commands need zero flush.
			 */
			if (l2_dirty && (cmd->flags & C7X_CMD_FLAG_USE_DRU) && cmd->src0 >= 0x80000000ull) {
				uint64_t kp = CMD_ALIGN64(cmd->p1);
				uint64_t np = CMD_ALIGN64(cmd->p2);
				uint64_t wb = (np >> 6) * (kp >> 6) * 2048ull;
				uint64_t min_dru = (cmd->p0 == 1u) ? (1024ull * 1024ull) : (128ull * 1024ull);
				if (wb >= min_dru) {
					cmd_l1d_wbinv_after_l2_scratch();
					l2_dirty = 0;
				}
			}
			rc = exec_matmul_w4a8_g64(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_MATMUL_W8A8:
			rc = exec_matmul_w8a8(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_MATMUL_F32:
			rc = exec_matmul_f32(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_RMS_NORM_F32:
			rc = exec_rms_norm_f32(cmd);
			break;
		case C7X_CMD_ADD_F32:
			rc = exec_binop_f32(cmd, 0);
			break;
		case C7X_CMD_MUL_F32:
			rc = exec_binop_f32(cmd, 1);
			break;
		case C7X_CMD_SCALE_F32:
			rc = exec_scale_f32(cmd);
			break;
		case C7X_CMD_UNARY_F32:
			rc = exec_unary_f32(cmd);
			break;
		case C7X_CMD_GLU_F32:
			rc = exec_glu_f32(cmd);
			break;
		case C7X_CMD_SOFT_MAX_F32:
			rc = exec_soft_max_f32(cmd);
			break;
		case C7X_CMD_ROPE_F32:
			rc = exec_rope_f32(cmd);
			break;
		case C7X_CMD_GET_ROWS:
			rc = exec_get_rows(cmd);
			break;
		case C7X_CMD_CPY:
			rc = exec_cpy(cmd);
			break;
		case C7X_CMD_SET_ROWS:
			rc = exec_set_rows(cmd);
			break;
		case C7X_CMD_NORM_F32:
			rc = exec_norm_f32(cmd);
			break;
		case C7X_CMD_IM2COL:
			rc = exec_im2col(cmd);
			break;
		case C7X_CMD_POOL_2D_F32:
			rc = exec_pool_2d_f32(cmd);
			break;
		case C7X_CMD_CONV_2D_F32:
			rc = exec_conv_2d_f32(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_QLINEAR_MATMUL:
			rc = exec_qlinear_matmul(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_QLINEAR_CONV2D:
			rc = exec_qlinear_conv2d(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		case C7X_CMD_DEQUANTIZE_LIN:
			rc = exec_dequantize_linear(cmd);
			break;
		case C7X_CMD_FLASH_ATTN_EXT:
			rc = exec_flash_attn_ext(cmd, l2_scratch, l2_bytes);
			l2_dirty = 1;
			break;
		default:
			rc = -100;
			break;
		}

		if (rc != 0) {
			if (l2_dirty) {
				cmd_l1d_wbinv_after_l2_scratch();
			}
			if (done_out) *done_out = i;
			if (fail_idx_out) *fail_idx_out = i;
			return rc;
		}
	}

	if (l2_dirty) {
		cmd_l1d_wbinv_after_l2_scratch();
	}
	if (done_out) *done_out = n_cmds;
	return 0;
}
