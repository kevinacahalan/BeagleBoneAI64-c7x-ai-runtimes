/* SPDX-License-Identifier: MIT */
/*
 * C7x 512-bit FP32 vector-unit kernels for transformer inference.
 *
 * Shared between TI C7000 CGT (cl7x) and x86_64 C7000 Host Emulation
 * (g++ -D__C7X_HOSTEM__).
 */

#ifdef __C7X_HOSTEM__
#include "c7x_host_emulation.h"
#else
#include <c7x.h>
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "vec_kernels.h"

static inline __STRM_TEMPLATE vec_strm_zero(void)
{
#ifdef __C7X_HOSTEM__
	return __uint16(0);
#else
	return (__uint16)(0);
#endif
}

/* Open SE on a 1D 32-bit (float/int32) stream (n_elems multiple of 16). */
static inline void se_open_1d_32b(int se, const void *p, uint32_t n_elems)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = vec_strm_zero();

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

/* Open SE on a 2D 32-bit stream: icnt0 elems wide (mult of 16), icnt1 rows, dim1 elems stride. */
static inline void se_open_2d_32b(int se, const void *p,
				  uint32_t icnt0, uint32_t icnt1, int32_t dim1)
{
	__SE_FLAGS flags = __SE_FLAGS_default();
	__STRM_TEMPLATE params = vec_strm_zero();

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

static inline float hsum_float16(float16 v)
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

static inline float hmax_float16(float16 v)
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

/*
 * 16-wide FP32 exp(x) on the C7x vector unit:
 *   Cody-Waite range reduction x = k*ln(2) + r (|r| <= ln(2)/2)
 *   Degree-6 Taylor polynomial on r, scaled by 2^k via IEEE-754 exponent shift.
 *   Max relative error < 2e-7 across [-87, +88].
 */
static inline float16 vexpf_f16(float16 x)
{
	const float16 c_lo    = (float16)(-87.33654475f);
	const float16 c_hi    = (float16)(88.72283905f);
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
	return p * pow2k;
}

/* 16-wide full-precision FP32 reciprocal (VRCPSP + 2 Newton-Raphson steps). */
static inline float16 vrecip_f16(float16 d)
{
	const float16 two = (float16)2.0f;
	float16 r = __recip(d);

	r = r * (two - d * r);
	r = r * (two - d * r);
	return r;
}

void c7x_rmsnorm_f32(float *out, const float *x, const float *weight,
		     uint32_t n, float eps)
{
	float16 ss_v = (float16)0.0f;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;
	float ss, inv;
	float16 inv_v;

	if (n_vec > 0u) {
		se_open_1d_32b(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 xv = __SE0ADV(float16);

			ss_v += xv * xv;
		}
		__SE0_CLOSE();
	}
	ss = hsum_float16(ss_v);
	for (; i < n; i++) {
		ss += x[i] * x[i];
	}

	inv = 1.0f / sqrtf(ss / (float)n + eps);
	inv_v = (float16)inv;

	i = 0u;
	if (n_vec > 0u) {
		se_open_1d_32b(0, x, n_vec);
		se_open_1d_32b(1, weight, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 xv = __SE0ADV(float16);
			float16 wv = __SE1ADV(float16);

			*(__float16_ptr)&out[i] = (xv * inv_v) * wv;
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
	}
	for (; i < n; i++) {
		out[i] = (x[i] * inv) * weight[i];
	}
}

void c7x_quantize_f32_i8(int8_t *out_q, const float *x,
			 uint32_t n, uint32_t n_pad, float *scale_out)
{
	float16 vmax = (float16)0.0f;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;
	float max_abs, scale, inv;

	if (n_vec > 0u) {
		se_open_1d_32b(0, x, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 xv = __abs(__SE0ADV(float16));

			vmax = __max(vmax, xv);
		}
		__SE0_CLOSE();
	}
	max_abs = hmax_float16(vmax);
	for (; i < n; i++) {
		float a = x[i] < 0.0f ? -x[i] : x[i];

		if (a > max_abs) {
			max_abs = a;
		}
	}

	scale = max_abs / 127.0f;
	*scale_out = scale;

	if (scale <= 0.0f) {
		memset(out_q, 0, n_pad);
		return;
	}

	inv = 1.0f / scale;
	{
		float16 inv_v = (float16)inv;
		float16 lo_v = (float16)(-127.0f);
		float16 hi_v = (float16)(127.0f);

		i = 0u;
		if (n_vec > 0u) {
			se_open_1d_32b(0, x, n_vec);
			for (; i < n_vec; i += 16u) {
				float16 xv = __SE0ADV(float16);
				float16 sv = __min(__max(xv * inv_v, lo_v), hi_v);
				int16 qv = __float_to_int(sv);
				int32_t tmp[16] __attribute__((aligned(64)));
				uint32_t k;

				*(__int16_ptr)tmp = qv;
				for (k = 0u; k < 16u; k++) {
					out_q[i + k] = (int8_t)tmp[k];
				}
			}
			__SE0_CLOSE();
		}
		for (; i < n; i++) {
			float s = x[i] * inv;
			int32_t q;

			if (s > 127.0f) {
				s = 127.0f;
			} else if (s < -127.0f) {
				s = -127.0f;
			}
			q = (int32_t)rintf(s);
			out_q[i] = (int8_t)q;
		}
	}

	for (i = n; i < n_pad; i++) {
		out_q[i] = 0;
	}
}

void c7x_dequant_i32_f32(float *out, const int32_t *y_i32,
			 const float *w_scale, float x_scale, uint32_t n)
{
	float16 xs_v = (float16)x_scale;
	uint32_t i = 0u;

	for (; i + 16u <= n; i += 16u) {
		int16 yv = *(__int16_ptr)(int32_t *)&y_i32[i];
		float16 ws = *(__float16_ptr)(float *)&w_scale[i];

		*(__float16_ptr)&out[i] = __int_to_float(yv) * (ws * xs_v);
	}
	for (; i < n; i++) {
		out[i] = (float)y_i32[i] * (w_scale[i] * x_scale);
	}
}

void c7x_dequant_add_i32_f32(float *acc, const int32_t *y_i32,
			     const float *w_scale, float x_scale, uint32_t n)
{
	float16 xs_v = (float16)x_scale;
	uint32_t i = 0u;

	for (; i + 16u <= n; i += 16u) {
		int16 yv = *(__int16_ptr)(int32_t *)&y_i32[i];
		float16 ws = *(__float16_ptr)(float *)&w_scale[i];
		float16 av = *(__float16_ptr)&acc[i];

		*(__float16_ptr)&acc[i] = av + __int_to_float(yv) * (ws * xs_v);
	}
	for (; i < n; i++) {
		acc[i] += (float)y_i32[i] * (w_scale[i] * x_scale);
	}
}

uint32_t c7x_dequant_argmax_i32_f32(float *out, const int32_t *y_i32,
				    const float *w_scale, float x_scale, uint32_t n)
{
	const uint32_t BLOCK_ELEMS = 1024u; /* 64 float16 vectors per block */
	float16 xs_v = (float16)x_scale;
	uint32_t best_block = 0u;
	float best_block_max = -3.4e38f;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;
	uint32_t best_idx = 0u;
	float best_val = -3.4e38f;
	uint32_t scan_end;

	/*
	 * Single-pass SE0+SE1 hardware-prefetched block-max scan:
	 * streams y_i32 (L2) via SE0 and w_scale (DDR) via SE1 sequentially so
	 * DDR prefetch eliminates per-line L2 cache misses, with only n/1024
	 * horizontal max reductions.
	 */
	if (n_vec > 0u) {
		se_open_1d_32b(0, y_i32, n_vec);
		se_open_1d_32b(1, w_scale, n_vec);
		while (i < n_vec) {
			uint32_t blk_start = i;
			uint32_t blk_end = blk_start + BLOCK_ELEMS;
			float16 bmax = (float16)(-3.4e38f);
			float bm;

			if (blk_end > n_vec) {
				blk_end = n_vec;
			}
			for (; i < blk_end; i += 16u) {
				int16 yv = __SE0ADV(int16);
				float16 ws = __SE1ADV(float16);
				float16 val = __int_to_float(yv) * (ws * xs_v);

				if (out) {
					*(__float16_ptr)&out[i] = val;
				}
				bmax = __max(bmax, val);
			}
			bm = hmax_float16(bmax);
			if (bm > best_block_max) {
				best_block_max = bm;
				best_block = blk_start;
			}
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
	}

	/* Scalar tail (n % 16) */
	for (; i < n; i++) {
		float val = (float)y_i32[i] * (w_scale[i] * x_scale);

		if (out) {
			out[i] = val;
		}
		if (val > best_block_max) {
			best_block_max = val;
			best_block = i;
		}
	}

	/* Locate exact argmax inside the single winning block (<= 1024 elements). */
	scan_end = best_block + BLOCK_ELEMS;
	if (scan_end > n) {
		scan_end = n;
	}
	for (i = best_block; i < scan_end; i++) {
		float val = out ? out[i] : ((float)y_i32[i] * (w_scale[i] * x_scale));

		if (val > best_val) {
			best_val = val;
			best_idx = i;
		}
	}
	return best_idx;
}

void c7x_dequant_i8_f32(float *out, const int8_t *row_q, float row_scale, uint32_t n)
{
	float16 s_v = (float16)row_scale;
	uint32_t i = 0u;

	for (; i + 16u <= n; i += 16u) {
		char16 qv = *(__char16_ptr)(int8_t *)&row_q[i];
		int16 iv = __convert_int16(qv);

		*(__float16_ptr)&out[i] = __int_to_float(iv) * s_v;
	}
	for (; i < n; i++) {
		out[i] = (float)row_q[i] * row_scale;
	}
}

void c7x_add_bias_f32(float *x, const float *bias, uint32_t n)
{
	uint32_t i = 0u;

	for (; i + 16u <= n; i += 16u) {
		float16 xv = *(__float16_ptr)&x[i];
		float16 bv = *(__float16_ptr)(float *)&bias[i];

		*(__float16_ptr)&x[i] = xv + bv;
	}
	for (; i < n; i++) {
		x[i] += bias[i];
	}
}

static inline void rope_apply_heads(float *vec, const float *cos_row, const float *sin_row,
				    uint32_t n_heads, uint32_t head_size, int neox)
{
	uint32_t half = head_size >> 1;
	uint32_t h, i;

	if (!neox) {
		for (h = 0u; h < n_heads; h++) {
			float *vh = vec + h * head_size;

			for (i = 0u; i < half; i++) {
				float v0 = vh[2u * i];
				float v1 = vh[2u * i + 1u];
				float c = cos_row[i];
				float s = sin_row[i];

				vh[2u * i]      = v0 * c - v1 * s;
				vh[2u * i + 1u] = v0 * s + v1 * c;
			}
		}
	} else {
		for (h = 0u; h < n_heads; h++) {
			float *v0p = vec + h * head_size;
			float *v1p = v0p + half;

			i = 0u;
			for (; i + 16u <= half; i += 16u) {
				float16 v0 = *(__float16_ptr)&v0p[i];
				float16 v1 = *(__float16_ptr)&v1p[i];
				float16 cv = *(__float16_ptr)(float *)&cos_row[i];
				float16 sv = *(__float16_ptr)(float *)&sin_row[i];

				*(__float16_ptr)&v0p[i] = v0 * cv - v1 * sv;
				*(__float16_ptr)&v1p[i] = v0 * sv + v1 * cv;
			}
			for (; i < half; i++) {
				float v0 = v0p[i];
				float v1 = v1p[i];
				float c = cos_row[i];
				float s = sin_row[i];

				v0p[i] = v0 * c - v1 * s;
				v1p[i] = v0 * s + v1 * c;
			}
		}
	}
}

void c7x_rope_f32(float *q, float *k, const float *cos_row, const float *sin_row,
		  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_size, int neox)
{
	rope_apply_heads(q, cos_row, sin_row, n_heads, head_size, neox);
	rope_apply_heads(k, cos_row, sin_row, n_kv_heads, head_size, neox);
}

void c7x_softmax_f32(float *x, uint32_t n)
{
	uint32_t i = 0u;
	float max_val, sum = 0.0f, inv_sum;

	if (n == 0u) {
		return;
	}
	if (n >= 16u) {
		float16 vmax = *(__float16_ptr)&x[0];

		for (i = 16u; i + 16u <= n; i += 16u) {
			vmax = __max(vmax, *(__float16_ptr)&x[i]);
		}
		max_val = hmax_float16(vmax);
		for (; i < n; i++) {
			if (x[i] > max_val) {
				max_val = x[i];
			}
		}
	} else {
		max_val = x[0];
		for (i = 1u; i < n; i++) {
			if (x[i] > max_val) {
				max_val = x[i];
			}
		}
	}

	{
		float16 m_v = (float16)max_val;
		float16 sum_v = (float16)0.0f;

		i = 0u;
		for (; i + 16u <= n; i += 16u) {
			float16 ev = vexpf_f16(*(__float16_ptr)&x[i] - m_v);

			*(__float16_ptr)&x[i] = ev;
			sum_v += ev;
		}
		sum = hsum_float16(sum_v);

		/* Vectorize partial tail (<16 elements) via one vexpf_f16 (zero libc expf calls). */
		if (i < n) {
			float tmp[16] __attribute__((aligned(64)));
			uint32_t rem = n - i;
			uint32_t k;
			float16 ev;

			for (k = 0u; k < rem; k++) {
				tmp[k] = x[i + k] - max_val;
			}
			for (; k < 16u; k++) {
				tmp[k] = 0.0f;
			}
			ev = vexpf_f16(*(__float16_ptr)tmp);
			*(__float16_ptr)tmp = ev;
			for (k = 0u; k < rem; k++) {
				x[i + k] = tmp[k];
				sum += tmp[k];
			}
		}
	}

	inv_sum = 1.0f / sum;
	i = 0u;
	{
		float16 inv_v = (float16)inv_sum;

		for (; i + 16u <= n; i += 16u) {
			float16 xv = *(__float16_ptr)&x[i];

			*(__float16_ptr)&x[i] = xv * inv_v;
		}
	}
	for (; i < n; i++) {
		x[i] *= inv_sum;
	}
}

void c7x_swiglu_f32(float *out, const float *hb, const float *hb2, uint32_t n)
{
	const float16 one = (float16)1.0f;
	uint32_t n_vec = n & ~15u;
	uint32_t i = 0u;

	if (n_vec > 0u) {
		se_open_1d_32b(0, hb, n_vec);
		se_open_1d_32b(1, hb2, n_vec);
		for (; i < n_vec; i += 16u) {
			float16 v  = __SE0ADV(float16);
			float16 v2 = __SE1ADV(float16);
			float16 denom = one + vexpf_f16(-v);
			float16 silu  = v * vrecip_f16(denom);

			*(__float16_ptr)&out[i] = silu * v2;
		}
		__SE0_CLOSE();
		__SE1_CLOSE();
	}
	for (; i < n; i++) {
		float v = hb[i];
		float silu = v / (1.0f + expf(-v));

		out[i] = silu * hb2[i];
	}
}

void c7x_attn_mha_f32(float *out, const float *q,
		      const float *k_cache_layer, const float *v_cache_layer,
		      float *att_buf, uint32_t pos,
		      uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_size)
{
	uint32_t kv_dim = n_kv_heads * head_size;
	uint32_t kv_mul = n_heads / n_kv_heads;
	float inv_sqrt_hd = 1.0f / sqrtf((float)head_size);
	uint32_t n_pos = pos + 1u;
	uint32_t h, t, i;

	/* Fast hardware-prefetched 2D SE0/SE1 path for head_size == 48 (stories15M). */
	if (head_size == 48u) {
		for (h = 0u; h < n_heads; h++) {
			const float *qh = q + h * 48u;
			float *oh = out + h * 48u;
			uint32_t kvh_off = (h / kv_mul) * 48u;
			float16 q0 = *(__float16_ptr)(float *)&qh[0];
			float16 q1 = *(__float16_ptr)(float *)&qh[16];
			float16 q2 = *(__float16_ptr)(float *)&qh[32];
			float16 o0 = (float16)0.0f;
			float16 o1 = (float16)0.0f;
			float16 o2 = (float16)0.0f;

			se_open_2d_32b(0, k_cache_layer + kvh_off, 48u, n_pos, (int32_t)kv_dim);
			se_open_2d_32b(1, v_cache_layer + kvh_off, 48u, n_pos, (int32_t)kv_dim);

			for (t = 0u; t < n_pos; t++) {
				float16 acc = q0 * __SE0ADV(float16);

				acc += q1 * __SE0ADV(float16);
				acc += q2 * __SE0ADV(float16);
				att_buf[t] = hsum_float16(acc) * inv_sqrt_hd;
			}
			__SE0_CLOSE();

			c7x_softmax_f32(att_buf, n_pos);

			for (t = 0u; t < n_pos; t++) {
				float16 av = (float16)att_buf[t];

				o0 += av * __SE1ADV(float16);
				o1 += av * __SE1ADV(float16);
				o2 += av * __SE1ADV(float16);
			}
			__SE1_CLOSE();

			*(__float16_ptr)&oh[0]  = o0;
			*(__float16_ptr)&oh[16] = o1;
			*(__float16_ptr)&oh[32] = o2;
		}
		return;
	}

	/* Fast hardware-prefetched 2D SE0/SE1 path for head_size == 64 (SmolLM2, Qwen2.5, Llama). */
	if (head_size == 64u) {
		for (h = 0u; h < n_heads; h++) {
			const float *qh = q + h * 64u;
			float *oh = out + h * 64u;
			uint32_t kvh_off = (h / kv_mul) * 64u;
			float16 q0 = *(__float16_ptr)(float *)&qh[0];
			float16 q1 = *(__float16_ptr)(float *)&qh[16];
			float16 q2 = *(__float16_ptr)(float *)&qh[32];
			float16 q3 = *(__float16_ptr)(float *)&qh[48];
			float16 o0 = (float16)0.0f;
			float16 o1 = (float16)0.0f;
			float16 o2 = (float16)0.0f;
			float16 o3 = (float16)0.0f;

			se_open_2d_32b(0, k_cache_layer + kvh_off, 64u, n_pos, (int32_t)kv_dim);
			se_open_2d_32b(1, v_cache_layer + kvh_off, 64u, n_pos, (int32_t)kv_dim);

			for (t = 0u; t < n_pos; t++) {
				float16 acc = q0 * __SE0ADV(float16);

				acc += q1 * __SE0ADV(float16);
				acc += q2 * __SE0ADV(float16);
				acc += q3 * __SE0ADV(float16);
				att_buf[t] = hsum_float16(acc) * inv_sqrt_hd;
			}
			__SE0_CLOSE();

			c7x_softmax_f32(att_buf, n_pos);

			for (t = 0u; t < n_pos; t++) {
				float16 av = (float16)att_buf[t];

				o0 += av * __SE1ADV(float16);
				o1 += av * __SE1ADV(float16);
				o2 += av * __SE1ADV(float16);
				o3 += av * __SE1ADV(float16);
			}
			__SE1_CLOSE();

			*(__float16_ptr)&oh[0]  = o0;
			*(__float16_ptr)&oh[16] = o1;
			*(__float16_ptr)&oh[32] = o2;
			*(__float16_ptr)&oh[48] = o3;
		}
		return;
	}

	/* Hardware-prefetched 2D SE0/SE1 path for any head_size multiple of 16 (e.g. 96, 128). */
	if ((head_size % 16u) == 0u && head_size <= 256u) {
		uint32_t n_v = head_size >> 4;

		for (h = 0u; h < n_heads; h++) {
			const float *qh = q + h * head_size;
			float *oh = out + h * head_size;
			uint32_t kvh_off = (h / kv_mul) * head_size;
			float16 o_regs[16];
			uint32_t v;

			for (v = 0u; v < n_v; v++) {
				o_regs[v] = (float16)0.0f;
			}

			se_open_2d_32b(0, k_cache_layer + kvh_off, head_size, n_pos, (int32_t)kv_dim);
			se_open_2d_32b(1, v_cache_layer + kvh_off, head_size, n_pos, (int32_t)kv_dim);

			for (t = 0u; t < n_pos; t++) {
				float16 acc = (float16)0.0f;

				for (v = 0u; v < n_v; v++) {
					float16 qv = *(__float16_ptr)(float *)&qh[v << 4];

					acc += qv * __SE0ADV(float16);
				}
				att_buf[t] = hsum_float16(acc) * inv_sqrt_hd;
			}
			__SE0_CLOSE();

			c7x_softmax_f32(att_buf, n_pos);

			for (t = 0u; t < n_pos; t++) {
				float16 av = (float16)att_buf[t];

				for (v = 0u; v < n_v; v++) {
					o_regs[v] += av * __SE1ADV(float16);
				}
			}
			__SE1_CLOSE();

			for (v = 0u; v < n_v; v++) {
				*(__float16_ptr)&oh[v << 4] = o_regs[v];
			}
		}
		return;
	}

	/* Generic path for any head_size. */
	for (h = 0u; h < n_heads; h++) {
		const float *qh = q + h * head_size;
		float *oh = out + h * head_size;
		uint32_t kvh_off = (h / kv_mul) * head_size;

		for (t = 0u; t < n_pos; t++) {
			const float *kt = k_cache_layer + t * kv_dim + kvh_off;
			float16 acc_v = (float16)0.0f;
			float score;

			i = 0u;
			for (; i + 16u <= head_size; i += 16u) {
				float16 qv = *(__float16_ptr)(float *)&qh[i];
				float16 kv = *(__float16_ptr)(float *)&kt[i];

				acc_v += qv * kv;
			}
			score = hsum_float16(acc_v);
			for (; i < head_size; i++) {
				score += qh[i] * kt[i];
			}
			att_buf[t] = score * inv_sqrt_hd;
		}

		c7x_softmax_f32(att_buf, n_pos);

		memset(oh, 0, head_size * sizeof(float));
		for (t = 0u; t < n_pos; t++) {
			const float *vt = v_cache_layer + t * kv_dim + kvh_off;
			float a = att_buf[t];
			float16 av = (float16)a;

			i = 0u;
			for (; i + 16u <= head_size; i += 16u) {
				float16 ov = *(__float16_ptr)&oh[i];
				float16 vv = *(__float16_ptr)(float *)&vt[i];

				*(__float16_ptr)&oh[i] = ov + av * vv;
			}
			for (; i < head_size; i++) {
				oh[i] += a * vt[i];
			}
		}
	}
}

int c7x_vec_selftest(float *max_err_out)
{
	static float x[288] __attribute__((aligned(64)));
	static float w[288] __attribute__((aligned(64)));
	static float y[288] __attribute__((aligned(64)));
	static float y_ref[288] __attribute__((aligned(64)));
	static int8_t xq[320] __attribute__((aligned(64)));
	static int32_t yi32[288] __attribute__((aligned(64)));
	float max_err = 0.0f;
	float scale = 0.0f;
	uint32_t i;

	for (i = 0u; i < 288u; i++) {
		int32_t v = (int32_t)((i * 37u + 13u) % 201u) - 100;
		x[i] = (float)v * 0.025f;
		w[i] = 0.5f + (float)((i * 19u) % 100u) * 0.01f;
	}

	/* 1. RMSNorm */
	{
		float ss = 0.0f, inv;

		c7x_rmsnorm_f32(y, x, w, 288u, 1e-5f);
		for (i = 0u; i < 288u; i++) {
			ss += x[i] * x[i];
		}
		inv = 1.0f / sqrtf(ss / 288.0f + 1e-5f);
		for (i = 0u; i < 288u; i++) {
			float ref = x[i] * inv * w[i];
			float e = fabsf(y[i] - ref);

			if (e > max_err) {
				max_err = e;
			}
		}
	}

	/* 2. Quantize f32 -> i8 + zero pad */
	{
		float ref_max = 0.0f, ref_scale, ref_inv;

		c7x_quantize_f32_i8(xq, x, 288u, 320u, &scale);
		for (i = 0u; i < 288u; i++) {
			float a = fabsf(x[i]);

			if (a > ref_max) {
				ref_max = a;
			}
		}
		ref_scale = ref_max / 127.0f;
		ref_inv = 1.0f / ref_scale;
		if (fabsf(scale - ref_scale) > 1e-6f) {
			if (max_err_out) {
				*max_err_out = fabsf(scale - ref_scale);
			}
			return -1;
		}
		for (i = 0u; i < 288u; i++) {
			int32_t ref_q = (int32_t)rintf(x[i] * ref_inv);

			if (ref_q > 127) {
				ref_q = 127;
			}
			if (ref_q < -127) {
				ref_q = -127;
			}
			if ((int32_t)xq[i] != ref_q) {
				if (max_err_out) {
					*max_err_out = (float)fabsf((float)xq[i] - (float)ref_q);
				}
				return -2;
			}
		}
		for (i = 288u; i < 320u; i++) {
			if (xq[i] != 0) {
				return -3;
			}
		}
	}

	/* 3. Dequant i32 -> f32 and dequant_add */
	for (i = 0u; i < 288u; i++) {
		yi32[i] = (int32_t)((i * 113u) % 4001u) - 2000;
		y_ref[i] = x[i];
	}
	c7x_dequant_i32_f32(y, yi32, w, scale, 288u);
	for (i = 0u; i < 288u; i++) {
		float ref = (float)yi32[i] * (w[i] * scale);
		float e = fabsf(y[i] - ref);

		if (e > max_err) {
			max_err = e;
		}
	}
	c7x_dequant_add_i32_f32(y_ref, yi32, w, scale, 288u);
	for (i = 0u; i < 288u; i++) {
		float ref = x[i] + (float)yi32[i] * (w[i] * scale);
		float e = fabsf(y_ref[i] - ref);

		if (e > max_err) {
			max_err = e;
		}
	}

	/* 4. Dequant i8 -> f32 (embedding lookup) */
	c7x_dequant_i8_f32(y, xq, scale, 288u);
	for (i = 0u; i < 288u; i++) {
		float ref = (float)xq[i] * scale;
		float e = fabsf(y[i] - ref);

		if (e > max_err) {
			max_err = e;
		}
	}

	/* 5. Softmax */
	for (i = 0u; i < 48u; i++) {
		y[i] = x[i];
	}
	c7x_softmax_f32(y, 48u);
	{
		float s = 0.0f;

		for (i = 0u; i < 48u; i++) {
			s += y[i];
		}
		if (fabsf(s - 1.0f) > 1e-5f) {
			if (max_err_out) {
				*max_err_out = fabsf(s - 1.0f);
			}
			return -4;
		}
	}

	/* 6. Vector SwiGLU (exercises vexpf_f16 and vrecip_f16) */
	c7x_swiglu_f32(y, x, w, 288u);
	for (i = 0u; i < 288u; i++) {
		float ref = (x[i] / (1.0f + expf(-x[i]))) * w[i];
		float e = fabsf(y[i] - ref);

		if (e > max_err) {
			max_err = e;
		}
	}

	/* 7. Fused dequant + argmax */
	{
		uint32_t got_idx = c7x_dequant_argmax_i32_f32(NULL, yi32, w, scale, 288u);
		uint32_t want_idx = 0u;
		float best_v = -3.4e38f;

		for (i = 0u; i < 288u; i++) {
			float val = (float)yi32[i] * (w[i] * scale);

			if (val > best_v) {
				best_v = val;
				want_idx = i;
			}
		}
		if (got_idx != want_idx) {
			return -6;
		}
	}

	if (max_err_out) {
		*max_err_out = max_err;
	}
	return (max_err < 1e-4f) ? 0 : -5;
}
