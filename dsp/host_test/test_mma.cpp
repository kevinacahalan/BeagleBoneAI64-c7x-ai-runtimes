/* SPDX-License-Identifier: MIT */
/*
 * x86_64 C7000 Host Emulation verification suite for C7x MMA and vector kernels.
 *
 * Every kernel is tested bit-exact against a plain C reference before running
 * on live TDA4VM silicon.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mma_kernels.h"
#include "vec_kernels.h"

static uint32_t g_rng = 0x12345678u;

static int8_t rand_i8(void)
{
	g_rng = g_rng * 1664525u + 1013904223u;
	return (int8_t)((g_rng >> 24) & 0xffu);
}

static uint8_t rand_u8(void)
{
	g_rng = g_rng * 1664525u + 1013904223u;
	return (uint8_t)((g_rng >> 24) & 0xffu);
}

static void *alloc_aligned64(size_t bytes)
{
	void *p = NULL;
	size_t padded = (bytes + 63u) & ~((size_t)63u);

	if (posix_memalign(&p, 64, padded ? padded : 64) != 0) {
		fprintf(stderr, "OOM (%zu bytes)\n", bytes);
		exit(1);
	}
	memset(p, 0, padded ? padded : 64);
	return p;
}

static int test_cbank(uint32_t depth)
{
	int8_t *a = (int8_t *)alloc_aligned64(depth * 64u);
	int8_t *b = (int8_t *)alloc_aligned64(64u * 64u);
	int32_t *got = (int32_t *)alloc_aligned64(depth * 64u * sizeof(int32_t));
	int32_t *want = (int32_t *)alloc_aligned64(depth * 64u * sizeof(int32_t));
	uint32_t i, bad = 0;

	for (i = 0; i < depth * 64u; i++) {
		a[i] = rand_i8();
	}
	for (i = 0; i < 64u * 64u; i++) {
		b[i] = rand_i8();
	}

	c7x_ref_gemm_i8_i32(a, b, want, depth, 64u, 64u, 0u);
	c7x_mma_probe_cbank(a, b, got, depth);

	for (i = 0; i < depth * 64u; i++) {
		if (got[i] != want[i]) {
			if (bad < 5u) {
				fprintf(stderr, "  cbank depth=%u mismatch at [%u,%u]: got=%d want=%d\n",
					depth, i / 64u, i % 64u, got[i], want[i]);
			}
			bad++;
		}
	}

	free(a);
	free(b);
	free(got);
	free(want);

	printf("[host-emu] probe_cbank depth=%2u: %s (%u mismatches)\n",
	       depth, bad ? "FAIL" : "PASS", bad);
	return bad ? 1 : 0;
}

static int test_gemm(uint32_t m, uint32_t k, uint32_t n, uint32_t b_trans, uint32_t block_m)
{
	int8_t *a = (int8_t *)alloc_aligned64((size_t)m * k);
	int8_t *b = (int8_t *)alloc_aligned64((size_t)k * n);
	int32_t *got = (int32_t *)alloc_aligned64((size_t)m * n * sizeof(int32_t));
	int32_t *want = (int32_t *)alloc_aligned64((size_t)m * n * sizeof(int32_t));
	uint32_t i, bad = 0;

	for (i = 0; i < m * k; i++) {
		a[i] = rand_i8();
	}
	for (i = 0; i < k * n; i++) {
		b[i] = rand_i8();
	}

	c7x_ref_gemm_i8_i32(a, b, want, m, k, n, b_trans);
	c7x_mma_gemm_i8_i32(a, b, got, m, k, n, b_trans, block_m);

	for (i = 0; i < m * n; i++) {
		if (got[i] != want[i]) {
			if (bad < 5u) {
				fprintf(stderr, "  gemm %ux%ux%u trans=%u bm=%u mismatch at [%u,%u]: got=%d want=%d\n",
					m, k, n, b_trans, block_m, i / n, i % n, got[i], want[i]);
			}
			bad++;
		}
	}

	free(a);
	free(b);
	free(got);
	free(want);

	printf("[host-emu] gemm %3ux%3ux%3u B_ORDER=%s block_m=%2u: %s (%u mismatches)\n",
	       m, k, n, b_trans ? "COL" : "ROW", block_m, bad ? "FAIL" : "PASS", bad);
	return bad ? 1 : 0;
}

static int test_w4a8(uint32_t k, uint32_t n)
{
	int8_t *a = (int8_t *)alloc_aligned64(k);
	uint8_t *w = (uint8_t *)alloc_aligned64((size_t)n * (k / 2u));
	int32_t *got = (int32_t *)alloc_aligned64((size_t)n * sizeof(int32_t));
	int32_t *want = (int32_t *)alloc_aligned64((size_t)n * sizeof(int32_t));
	uint32_t i, bad = 0;

	for (i = 0; i < k; i++) {
		a[i] = rand_i8();
	}
	for (i = 0; i < n * (k / 2u); i++) {
		w[i] = rand_u8();
	}

	c7x_ref_gemv_w4a8(a, w, want, k, n);
	c7x_mma_gemv_w4a8(a, w, got, k, n);

	for (i = 0; i < n; i++) {
		if (got[i] != want[i]) {
			if (bad < 5u) {
				fprintf(stderr, "  w4a8 %ux%u mismatch at [%u]: got=%d want=%d\n",
					k, n, i, got[i], want[i]);
			}
			bad++;
		}
	}

	free(a);
	free(w);
	free(got);
	free(want);
	printf("[host-emu] gemv_w4a8 K=%u N=%u: %s (%u mismatches)\n",
	       k, n, bad ? "FAIL" : "PASS", bad);
	return bad ? 1 : 0;
}

static int test_vec_kernels(void)
{
	float max_err = 0.0f;
	int rc = c7x_vec_selftest(&max_err);

	printf("[host-emu] vec_selftest: %s (rc=%d, max_err=%.6g)\n",
	       rc == 0 ? "PASS" : "FAIL", rc, max_err);
	if (rc != 0) {
		return 1;
	}

	/* Extra checks: RoPE, SwiGLU, and MHA attention vs scalar reference. */
	const uint32_t n_heads = 6u, n_kv = 6u, hd = 48u, dim = n_heads * hd, pos = 7u;
	float *q = (float *)alloc_aligned64(dim * sizeof(float));
	float *k = (float *)alloc_aligned64(dim * sizeof(float));
	float *q_ref = (float *)alloc_aligned64(dim * sizeof(float));
	float *k_ref = (float *)alloc_aligned64(dim * sizeof(float));
	float *cos_r = (float *)alloc_aligned64((hd / 2u) * sizeof(float));
	float *sin_r = (float *)alloc_aligned64((hd / 2u) * sizeof(float));
	float *kc = (float *)alloc_aligned64((pos + 1u) * dim * sizeof(float));
	float *vc = (float *)alloc_aligned64((pos + 1u) * dim * sizeof(float));
	float *att_out = (float *)alloc_aligned64(dim * sizeof(float));
	float *att_ref = (float *)alloc_aligned64(dim * sizeof(float));
	float *att_buf = (float *)alloc_aligned64((pos + 1u) * sizeof(float));
	float *hb = (float *)alloc_aligned64(768u * sizeof(float));
	float *hb2 = (float *)alloc_aligned64(768u * sizeof(float));
	float *sw_out = (float *)alloc_aligned64(768u * sizeof(float));

	for (uint32_t i = 0; i < dim; i++) {
		q[i] = q_ref[i] = (float)rand_i8() * 0.02f;
		k[i] = k_ref[i] = (float)rand_i8() * 0.02f;
	}
	for (uint32_t i = 0; i < hd / 2u; i++) {
		float freq = 1.0f / powf(10000.0f, (2.0f * (float)i) / (float)hd);
		cos_r[i] = cosf((float)pos * freq);
		sin_r[i] = sinf((float)pos * freq);
	}
	c7x_rope_f32(q, k, cos_r, sin_r, n_heads, n_kv, hd, 0);
	for (uint32_t h = 0; h < n_heads; h++) {
		for (uint32_t i = 0; i < hd / 2u; i++) {
			float q0 = q_ref[h * hd + 2u * i], q1 = q_ref[h * hd + 2u * i + 1u];
			float k0 = k_ref[h * hd + 2u * i], k1 = k_ref[h * hd + 2u * i + 1u];
			q_ref[h * hd + 2u * i] = q0 * cos_r[i] - q1 * sin_r[i];
			q_ref[h * hd + 2u * i + 1u] = q0 * sin_r[i] + q1 * cos_r[i];
			k_ref[h * hd + 2u * i] = k0 * cos_r[i] - k1 * sin_r[i];
			k_ref[h * hd + 2u * i + 1u] = k0 * sin_r[i] + k1 * cos_r[i];
		}
	}
	float rope_err = 0.0f;
	for (uint32_t i = 0; i < dim; i++) {
		float e1 = fabsf(q[i] - q_ref[i]);
		float e2 = fabsf(k[i] - k_ref[i]);
		if (e1 > rope_err) rope_err = e1;
		if (e2 > rope_err) rope_err = e2;
	}

	/* MHA attention check */
	for (uint32_t i = 0; i < (pos + 1u) * dim; i++) {
		kc[i] = (float)rand_i8() * 0.02f;
		vc[i] = (float)rand_i8() * 0.02f;
	}
	c7x_attn_mha_f32(att_out, q, kc, vc, att_buf, pos, n_heads, n_kv, hd);
	float inv_s = 1.0f / sqrtf((float)hd);
	for (uint32_t h = 0; h < n_heads; h++) {
		float scores[16];
		float m = -1e30f, s = 0.0f;
		for (uint32_t t = 0; t <= pos; t++) {
			float d = 0.0f;
			for (uint32_t i = 0; i < hd; i++) {
				d += q[h * hd + i] * kc[t * dim + h * hd + i];
			}
			scores[t] = d * inv_s;
			if (scores[t] > m) m = scores[t];
		}
		for (uint32_t t = 0; t <= pos; t++) {
			scores[t] = expf(scores[t] - m);
			s += scores[t];
		}
		for (uint32_t i = 0; i < hd; i++) {
			float acc = 0.0f;
			for (uint32_t t = 0; t <= pos; t++) {
				acc += (scores[t] / s) * vc[t * dim + h * hd + i];
			}
			att_ref[h * hd + i] = acc;
		}
	}
	float attn_err = 0.0f;
	for (uint32_t i = 0; i < dim; i++) {
		float e = fabsf(att_out[i] - att_ref[i]);
		if (e > attn_err) attn_err = e;
	}

	/* SwiGLU check */
	for (uint32_t i = 0; i < 768u; i++) {
		hb[i] = (float)rand_i8() * 0.05f;
		hb2[i] = (float)rand_i8() * 0.05f;
	}
	c7x_swiglu_f32(sw_out, hb, hb2, 768u);
	float sw_err = 0.0f;
	for (uint32_t i = 0; i < 768u; i++) {
		float ref = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
		float e = fabsf(sw_out[i] - ref);
		if (e > sw_err) sw_err = e;
	}

	free(q); free(k); free(q_ref); free(k_ref);
	free(cos_r); free(sin_r); free(kc); free(vc);
	free(att_out); free(att_ref); free(att_buf);
	free(hb); free(hb2); free(sw_out);

	int bad = (rope_err > 1e-5f || attn_err > 1e-5f || sw_err > 1e-5f);
	printf("[host-emu] vec rope/attn/swiglu: %s (rope=%.2g, attn=%.2g, swiglu=%.2g)\n",
	       bad ? "FAIL" : "PASS", rope_err, attn_err, sw_err);

	/* Test NeoX RoPE & head_size=64 2D SE attention */
	{
		const uint32_t hd64 = 64u, nh = 4u, nkv = 2u, d64 = nh * hd64, kvd64 = nkv * hd64, p64 = 5u;
		float *q64 = (float *)alloc_aligned64(d64 * sizeof(float));
		float *k64 = (float *)alloc_aligned64(kvd64 * sizeof(float));
		float *q64_ref = (float *)alloc_aligned64(d64 * sizeof(float));
		float *cos64 = (float *)alloc_aligned64((hd64 / 2u) * sizeof(float));
		float *sin64 = (float *)alloc_aligned64((hd64 / 2u) * sizeof(float));
		float *kc64 = (float *)alloc_aligned64((p64 + 1u) * kvd64 * sizeof(float));
		float *vc64 = (float *)alloc_aligned64((p64 + 1u) * kvd64 * sizeof(float));
		float *ao64 = (float *)alloc_aligned64(d64 * sizeof(float));
		float *ab64 = (float *)alloc_aligned64((p64 + 1u) * sizeof(float));

		for (uint32_t i = 0; i < d64; i++) q64[i] = q64_ref[i] = (float)rand_i8() * 0.02f;
		for (uint32_t i = 0; i < kvd64; i++) k64[i] = (float)rand_i8() * 0.02f;
		for (uint32_t i = 0; i < hd64 / 2u; i++) {
			float f = 1.0f / powf(10000.0f, (2.0f * (float)i) / (float)hd64);
			cos64[i] = cosf((float)p64 * f);
			sin64[i] = sinf((float)p64 * f);
		}
		c7x_rope_f32(q64, k64, cos64, sin64, nh, nkv, hd64, 1);
		float neox_err = 0.0f;
		for (uint32_t h = 0; h < nh; h++) {
			for (uint32_t i = 0; i < hd64 / 2u; i++) {
				float a = q64_ref[h * hd64 + i], b = q64_ref[h * hd64 + hd64 / 2u + i];
				float e0 = fabsf(q64[h * hd64 + i] - (a * cos64[i] - b * sin64[i]));
				float e1 = fabsf(q64[h * hd64 + hd64 / 2u + i] - (a * sin64[i] + b * cos64[i]));
				if (e0 > neox_err) neox_err = e0;
				if (e1 > neox_err) neox_err = e1;
			}
		}
		for (uint32_t i = 0; i < (p64 + 1u) * kvd64; i++) {
			kc64[i] = (float)rand_i8() * 0.02f;
			vc64[i] = (float)rand_i8() * 0.02f;
		}
		c7x_attn_mha_f32(ao64, q64, kc64, vc64, ab64, p64, nh, nkv, hd64);
		float gqa_err = 0.0f;
		float inv64 = 1.0f / sqrtf(64.0f);
		for (uint32_t h = 0; h < nh; h++) {
			float sc[16], mx = -1e30f, sm = 0.0f;
			uint32_t kvh = (h / (nh / nkv)) * hd64;
			for (uint32_t t = 0; t <= p64; t++) {
				float d = 0.0f;
				for (uint32_t i = 0; i < hd64; i++) d += q64[h * hd64 + i] * kc64[t * kvd64 + kvh + i];
				sc[t] = d * inv64;
				if (sc[t] > mx) mx = sc[t];
			}
			for (uint32_t t = 0; t <= p64; t++) { sc[t] = expf(sc[t] - mx); sm += sc[t]; }
			for (uint32_t i = 0; i < hd64; i++) {
				float acc = 0.0f;
				for (uint32_t t = 0; t <= p64; t++) acc += (sc[t] / sm) * vc64[t * kvd64 + kvh + i];
				float e = fabsf(ao64[h * hd64 + i] - acc);
				if (e > gqa_err) gqa_err = e;
			}
		}
		int bad64 = (neox_err > 1e-5f || gqa_err > 1e-5f);
		printf("[host-emu] vec neox-rope/gqa-hd64: %s (neox=%.2g, gqa64=%.2g)\n",
		       bad64 ? "FAIL" : "PASS", neox_err, gqa_err);
		bad |= bad64;
		free(q64); free(k64); free(q64_ref); free(cos64); free(sin64);
		free(kc64); free(vc64); free(ao64); free(ab64);
	}
	return bad;
}

int main(void)
{
	int fail = 0;

	fail |= test_cbank(16u);
	printf("[host-emu] probe_cbank depth=32 (expect wrap at row 16 because int32 C-bank = 16 rows x 256 B = 4 KiB):\n");
	(void)test_cbank(32u);

	/* Single-tile and multi-tile GEMV & GEMM, both ROW and COL (transposed) B. */
	fail |= test_gemm(1u,   64u,  64u, 0u, 16u);
	fail |= test_gemm(1u,  256u, 128u, 1u, 16u);
	fail |= test_gemm(7u,  128u, 192u, 1u, 16u);
	fail |= test_gemm(16u,  64u,  64u, 0u, 16u);
	fail |= test_gemm(16u, 256u, 128u, 0u, 16u);
	fail |= test_gemm(16u, 256u, 128u, 1u, 16u);
	fail |= test_gemm(32u, 192u, 128u, 1u, 16u);
	fail |= test_gemm(64u, 128u, 192u, 0u, 16u);
	fail |= test_gemm(64u, 128u, 192u, 1u, 16u);
	fail |= test_gemm(80u, 128u, 128u, 1u, 16u);

	fail |= test_w4a8(128u, 64u);
	fail |= test_w4a8(256u, 128u);

	/* Test tile-major group-64 W4A8 MMA kernel (M=1, M=12, odd M=19 k_tiles=1, multi-block M=37). */
	{
		const uint32_t ms[5] = { 1u, 1u, 12u, 19u, 37u };
		const uint32_t ks[5] = { 320u, 768u, 192u, 64u, 192u };
		const uint32_t ns[5] = { 320u, 256u, 128u, 128u, 192u };
		const int adds[5]    = { 0, 1, 0, 1, 0 };
		for (int tc = 0; tc < 5; tc++) {
			uint32_t m = ms[tc], k = ks[tc], n = ns[tc], out_dim = n - 12u;
			uint32_t kt = k / 64u, nt = n / 64u;
			int add = adds[tc];
			int8_t *a = (int8_t *)alloc_aligned64((size_t)m * k);
			float *as = (float *)alloc_aligned64(m * sizeof(float));
			uint8_t *wp = (uint8_t *)alloc_aligned64((size_t)nt * kt * 2048u);
			float *ws = (float *)alloc_aligned64((size_t)nt * kt * 64u * sizeof(float));
			float *got = (float *)alloc_aligned64((size_t)m * out_dim * sizeof(float));
			float *want = (float *)alloc_aligned64((size_t)m * out_dim * sizeof(float));
			uint32_t arg_got = 0u, arg_want = 0u;

			for (uint32_t i = 0; i < m * k; i++) a[i] = rand_i8();
			for (uint32_t i = 0; i < m; i++) as[i] = 0.015f + (float)i * 0.002f;
			for (uint32_t i = 0; i < nt * kt * 2048u; i++) wp[i] = rand_u8();
			for (uint32_t i = 0; i < nt * kt * 64u; i++) ws[i] = 0.005f + (float)(rand_u8() % 100u) * 0.0005f;
			for (uint32_t i = 0; i < m * out_dim; i++) got[i] = want[i] = (float)rand_i8() * 0.01f;

			c7x_ref_gemm_w4a8_g64_f32(want, a, as, wp, ws, m, k, n, out_dim, add, (m == 1u) ? &arg_want : NULL);
			c7x_mma_gemm_w4a8_g64_f32(got, a, as, wp, ws, m, k, n, out_dim, 16u, add, (m == 1u) ? &arg_got : NULL);
			float max_e = 0.0f;
			for (uint32_t i = 0; i < m * out_dim; i++) {
				float e = fabsf(got[i] - want[i]);
				if (e > max_e) max_e = e;
			}
			int bad = (max_e > 1e-4f || (m == 1u && arg_got != arg_want));
			printf("[host-emu] mma_w4a8_g64 %ux%ux%u add=%d: %s (max_err=%.2g, argmax=%u/%u)\n",
			       m, k, n, add, bad ? "FAIL" : "PASS", max_e, arg_got, arg_want);
			if (bad) fail = 1;
			free(a); free(as); free(wp); free(ws); free(got); free(want);
		}
	}

	/* Test 1D-streamed tile-major W8A8 MMA GEMM (M=1 & M=12). */
	{
		const uint32_t m = 12u, k = 192u, n = 128u;
		const uint32_t kt_n = k / 64u, nt_n = n / 64u;
		int8_t *a = (int8_t *)alloc_aligned64((size_t)m * k);
		int8_t *b_row = (int8_t *)alloc_aligned64((size_t)n * k);
		int8_t *b_til = (int8_t *)alloc_aligned64((size_t)n * k);
		int32_t *got = (int32_t *)alloc_aligned64((size_t)m * n * sizeof(int32_t));
		int32_t *want = (int32_t *)alloc_aligned64((size_t)m * n * sizeof(int32_t));
		uint32_t bad = 0u;

		for (uint32_t i = 0; i < m * k; i++) a[i] = rand_i8();
		for (uint32_t i = 0; i < n * k; i++) b_row[i] = rand_i8();
		for (uint32_t nt = 0; nt < nt_n; nt++) {
			for (uint32_t kt = 0; kt < kt_n; kt++) {
				for (uint32_t r = 0; r < 64u; r++) {
					memcpy(b_til + ((size_t)nt * kt_n + kt) * 4096u + r * 64u,
					       b_row + (size_t)(nt * 64u + r) * k + kt * 64u, 64u);
				}
			}
		}
		c7x_ref_gemm_i8_i32(a, b_row, want, m, k, n, 1u);
		c7x_mma_gemm_i8_i32_tiled(a, b_til, got, m, k, n, 16u);
		for (uint32_t i = 0; i < m * n; i++) {
			if (got[i] != want[i]) bad++;
		}
		printf("[host-emu] mma_w8_tiled %ux%ux%u: %s (%u mismatches)\n",
		       m, k, n, bad ? "FAIL" : "PASS", bad);
		if (bad) fail = 1;
		free(a); free(b_row); free(b_til); free(got); free(want);
	}

	/* Test 1D-streamed tile-major MMA GEMV against reference. */
	{
		const uint32_t k = 320u, n = 256u;
		const uint32_t kt_n = k / 64u, nt_n = n / 64u;
		int8_t *a = (int8_t *)alloc_aligned64(k);
		int8_t *b_row = (int8_t *)alloc_aligned64((size_t)n * k);
		int8_t *b_til = (int8_t *)alloc_aligned64((size_t)n * k);
		int32_t *got = (int32_t *)alloc_aligned64(n * sizeof(int32_t));
		int32_t *want = (int32_t *)alloc_aligned64(n * sizeof(int32_t));
		uint32_t bad = 0u;

		for (uint32_t i = 0; i < k; i++) a[i] = rand_i8();
		for (uint32_t i = 0; i < n * k; i++) b_row[i] = rand_i8();
		for (uint32_t nt = 0; nt < nt_n; nt++) {
			for (uint32_t kt = 0; kt < kt_n; kt++) {
				for (uint32_t r = 0; r < 64u; r++) {
					memcpy(b_til + ((size_t)nt * kt_n + kt) * 4096u + r * 64u,
					       b_row + (size_t)(nt * 64u + r) * k + kt * 64u, 64u);
				}
			}
		}
		c7x_ref_gemm_i8_i32(a, b_row, want, 1u, k, n, 1u);
		c7x_mma_gemv_i8_i32_tiled(a, b_til, got, k, n);
		for (uint32_t i = 0; i < n; i++) {
			if (got[i] != want[i]) bad++;
		}
		printf("[host-emu] gemv_tiled 1x%ux%u: %s (%u mismatches)\n",
		       k, n, bad ? "FAIL" : "PASS", bad);
		if (bad) fail = 1;
		free(a); free(b_row); free(b_til); free(got); free(want);
	}

	fail |= test_vec_kernels();

	printf("[host-emu] %s\n", fail ? "FAILED" : "ALL TESTS PASSED");
	return fail;
}
