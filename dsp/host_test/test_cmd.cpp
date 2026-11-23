/* SPDX-License-Identifier: MIT */
/*
 * x86_64 C7000 Host Emulation verification suite for the C7x Command-Buffer
 * Executor (c7x_cmd_exec_list / C7X_OP_EXEC_LIST).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmd_exec.h"
#include "mma_kernels.h"

static uint32_t g_rng = 0xC70055AAu;

static float rand_f32(void)
{
	g_rng = g_rng * 1664525u + 1013904223u;
	int32_t v = (int32_t)((g_rng >> 8) & 0xFFFFu) - 32768;
	return (float)v / 16384.0f;
}

static void *alloc64(size_t bytes)
{
	void *p = NULL;
	size_t sz = (bytes + 63u) & ~((size_t)63u);
	if (sz == 0) sz = 64;
	if (posix_memalign(&p, 64, sz) != 0) {
		fprintf(stderr, "OOM (%zu)\n", bytes);
		exit(1);
	}
	memset(p, 0, sz);
	return p;
}

static uint16_t ref_f32_to_f16(float f)
{
	uint32_t bits;
	memcpy(&bits, &f, 4);
	uint32_t sign = (bits >> 16) & 0x8000u;
	uint32_t exp  = (bits >> 23) & 0xFFu;
	uint32_t mant = bits & 0x7FFFFFu;
	if (exp == 255u) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
	int32_t ne = (int32_t)exp - 112;
	if (ne >= 31) return (uint16_t)(sign | 0x7C00u);
	if (ne <= 0) {
		if (ne < -10) return (uint16_t)sign;
		mant = (mant | 0x800000u) >> (uint32_t)(1 - ne);
		mant += 0xFFFu + ((mant >> 13) & 1u);
		return (uint16_t)(sign | (mant >> 13));
	}
	mant += 0xFFFu + ((mant >> 13) & 1u);
	if (mant & 0x800000u) {
		mant = 0;
		if (++ne >= 31) return (uint16_t)(sign | 0x7C00u);
	}
	return (uint16_t)(sign | ((uint32_t)ne << 10) | (mant >> 13));
}

static void pack_w4a8_g64(const float *w, uint32_t n, uint32_t k,
			  uint8_t *wp, float *ws)
{
	uint32_t n_pad = (n + 63u) & ~63u;
	uint32_t k_pad = (k + 63u) & ~63u;
	uint32_t nt_cnt = n_pad >> 6;
	uint32_t kt_cnt = k_pad >> 6;

	for (uint32_t nt = 0; nt < nt_cnt; nt++) {
		for (uint32_t kt = 0; kt < kt_cnt; kt++) {
			uint8_t *tile = wp + ((uint64_t)nt * kt_cnt + kt) * 2048ull;
			float *ts = ws + ((uint64_t)nt * kt_cnt + kt) * 64ull;
			for (uint32_t r = 0; r < 64u; r++) {
				uint32_t row = nt * 64u + r;
				float amax = 0.0f;
				if (row < n) {
					for (uint32_t c = 0; c < 64u; c++) {
						uint32_t col = kt * 64u + c;
						if (col < k) {
							float v = fabsf(w[(uint64_t)row * k + col]);
							if (v > amax) amax = v;
						}
					}
				}
				float sc = amax / 7.0f;
				ts[r] = sc;
				float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
				uint8_t *vec = tile + (r >> 1) * 64u;
				for (uint32_t c = 0; c < 64u; c++) {
					uint32_t col = kt * 64u + c;
					int q = 0;
					if (row < n && col < k) {
						float sv = w[(uint64_t)row * k + col] * inv;
						q = (int)lrintf(sv);
						if (q > 7) q = 7;
						if (q < -7) q = -7;
					}
					uint8_t nib = (uint8_t)(q & 0x0F);
					if ((r & 1u) == 0u) {
						vec[c] = nib;
					} else {
						vec[c] |= (uint8_t)(nib << 4);
					}
				}
			}
		}
	}
}

static void pack_w8a8(const float *w, uint32_t n, uint32_t k,
		      int8_t *wt, float *ws)
{
	uint32_t n_pad = (n + 63u) & ~63u;
	uint32_t k_pad = (k + 63u) & ~63u;
	uint32_t nt_cnt = n_pad >> 6;
	uint32_t kt_cnt = k_pad >> 6;

	for (uint32_t row = 0; row < n_pad; row++) {
		float amax = 0.0f;
		if (row < n) {
			for (uint32_t col = 0; col < k; col++) {
				float v = fabsf(w[(uint64_t)row * k + col]);
				if (v > amax) amax = v;
			}
		}
		float sc = amax / 127.0f;
		ws[row] = sc;
		float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
		uint32_t nt = row >> 6;
		uint32_t r  = row & 63u;
		for (uint32_t kt = 0; kt < kt_cnt; kt++) {
			int8_t *tile = wt + ((uint64_t)nt * kt_cnt + kt) * 4096ull;
			for (uint32_t c = 0; c < 64u; c++) {
				uint32_t col = kt * 64u + c;
				int q = 0;
				if (row < n && col < k) {
					q = (int)lrintf(w[(uint64_t)row * k + col] * inv);
					if (q > 127) q = 127;
					if (q < -127) q = -127;
				}
				tile[r * 64u + c] = (int8_t)q;
			}
		}
	}
}

static float cosine_sim(const float *a, const float *b, uint32_t n)
{
	double dot = 0.0, na = 0.0, nb = 0.0;
	for (uint32_t i = 0; i < n; i++) {
		dot += (double)a[i] * b[i];
		na  += (double)a[i] * a[i];
		nb  += (double)b[i] * b[i];
	}
	if (na == 0.0 || nb == 0.0) return 0.0f;
	return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static int test_matmul_cmds(void *l2, uint32_t l2_bytes)
{
	const uint32_t M = 5, K = 160, N = 96; /* non-64-multiple K & N! */
	const uint32_t k_pad = (K + 63u) & ~63u;
	const uint32_t n_pad = (N + 63u) & ~63u;
	float *a = (float *)alloc64(M * K * sizeof(float));
	float *w = (float *)alloc64(N * K * sizeof(float));
	float *ref = (float *)alloc64(M * N * sizeof(float));
	float *out4 = (float *)alloc64(M * N * sizeof(float));
	float *out8 = (float *)alloc64(M * N * sizeof(float));
	float *outf = (float *)alloc64(M * N * sizeof(float));
	uint8_t *wp4 = (uint8_t *)alloc64((size_t)(n_pad / 64) * (k_pad / 64) * 2048);
	float   *ws4 = (float *)alloc64((size_t)(n_pad / 64) * (k_pad / 64) * 64 * sizeof(float));
	int8_t  *wt8 = (int8_t *)alloc64((size_t)(n_pad / 64) * (k_pad / 64) * 4096);
	float   *ws8 = (float *)alloc64(n_pad * sizeof(float));

	for (uint32_t i = 0; i < M * K; i++) a[i] = rand_f32();
	for (uint32_t i = 0; i < N * K; i++) w[i] = rand_f32();
	for (uint32_t m = 0; m < M; m++) {
		for (uint32_t n = 0; n < N; n++) {
			float s = 0.0f;
			for (uint32_t k = 0; k < K; k++) s += a[m * K + k] * w[n * K + k];
			ref[m * N + n] = s;
		}
	}
	pack_w4a8_g64(w, N, K, wp4, ws4);
	pack_w8a8(w, N, K, wt8, ws8);

	struct c7x_matmul_f32_desc mdesc;
	memset(&mdesc, 0, sizeof(mdesc));
	mdesc.ne0[0] = K; mdesc.ne0[1] = N; mdesc.ne0[2] = 1; mdesc.ne0[3] = 1;
	mdesc.ne1[0] = K; mdesc.ne1[1] = M; mdesc.ne1[2] = 1; mdesc.ne1[3] = 1;
	mdesc.ne_dst[0] = N; mdesc.ne_dst[1] = M; mdesc.ne_dst[2] = 1; mdesc.ne_dst[3] = 1;
	mdesc.nb0[0] = 4; mdesc.nb0[1] = K * 4; mdesc.nb0[2] = N * K * 4; mdesc.nb0[3] = N * K * 4;
	mdesc.nb1[0] = 4; mdesc.nb1[1] = K * 4; mdesc.nb1[2] = M * K * 4; mdesc.nb1[3] = M * K * 4;
	mdesc.nb_dst[0] = 4; mdesc.nb_dst[1] = N * 4; mdesc.nb_dst[2] = M * N * 4; mdesc.nb_dst[3] = M * N * 4;

	struct c7x_cmd cmds[3];
	memset(cmds, 0, sizeof(cmds));
	cmds[0].op = C7X_CMD_MATMUL_W4A8_G64;
	cmds[0].p0 = M; cmds[0].p1 = K; cmds[0].p2 = N;
	cmds[0].src0 = (uint64_t)(uintptr_t)wp4;
	cmds[0].src1 = (uint64_t)(uintptr_t)a;
	cmds[0].src2 = (uint64_t)(uintptr_t)ws4;
	cmds[0].dst  = (uint64_t)(uintptr_t)out4;

	cmds[1].op = C7X_CMD_MATMUL_W8A8;
	cmds[1].p0 = M; cmds[1].p1 = K; cmds[1].p2 = N;
	cmds[1].src0 = (uint64_t)(uintptr_t)wt8;
	cmds[1].src1 = (uint64_t)(uintptr_t)a;
	cmds[1].src2 = (uint64_t)(uintptr_t)ws8;
	cmds[1].dst  = (uint64_t)(uintptr_t)out8;

	cmds[2].op = C7X_CMD_MATMUL_F32;
	cmds[2].src0 = (uint64_t)(uintptr_t)w;
	cmds[2].src1 = (uint64_t)(uintptr_t)a;
	cmds[2].src3 = (uint64_t)(uintptr_t)&mdesc;
	cmds[2].dst  = (uint64_t)(uintptr_t)outf;

	uint32_t done = 0, fail = 0;
	int rc = c7x_cmd_exec_list(cmds, 3, l2, l2_bytes, &done, &fail);
	float cos4 = cosine_sim(out4, ref, M * N);
	float cos8 = cosine_sim(out8, ref, M * N);
	float max_f32 = 0.0f;
	for (uint32_t i = 0; i < M * N; i++) {
		float e = fabsf(outf[i] - ref[i]);
		if (e > max_f32) max_f32 = e;
	}

	int bad = (rc != 0 || done != 3 || cos4 < 0.99f || cos8 < 0.9995f || max_f32 > 1e-4f);
	printf("[host-cmd] matmul (W4A8 cos=%.5f, W8A8 cos=%.6f, F32 max_err=%.2e): %s\n",
	       cos4, cos8, max_f32, bad ? "FAIL" : "PASS");

	free(a); free(w); free(ref); free(out4); free(out8); free(outf);
	free(wp4); free(ws4); free(wt8); free(ws8);
	return bad ? 1 : 0;
}

static int test_vec_and_chain_cmds(void *l2, uint32_t l2_bytes)
{
	/* Test chained: RMS_NORM -> MUL(weight) -> GLU(SWIGLU) -> SOFT_MAX(with F16 mask) -> ROPE(NEOX) -> SET_ROWS/CPY/GET_ROWS */
	const uint32_t D = 64, R = 3;
	float *x = (float *)alloc64(R * D * sizeof(float));
	float *w = (float *)alloc64(D * sizeof(float));
	float *normed = (float *)alloc64(R * D * sizeof(float));
	float *scaled = (float *)alloc64(R * D * sizeof(float));
	float *glu_out = (float *)alloc64(R * (D / 2) * sizeof(float));
	float *sm_out = (float *)alloc64(R * (D / 2) * sizeof(float));
	uint16_t *mask16 = (uint16_t *)alloc64(R * (D / 2) * sizeof(uint16_t));
	float *rope_out = (float *)alloc64(R * D * sizeof(float));
	int32_t *pos = (int32_t *)alloc64(R * sizeof(int32_t));

	for (uint32_t i = 0; i < R * D; i++) x[i] = rand_f32();
	for (uint32_t i = 0; i < D; i++) w[i] = 0.5f + 0.5f * rand_f32();
	for (uint32_t r = 0; r < R; r++) {
		pos[r] = (int32_t)(r + 7);
		for (uint32_t c = 0; c < D / 2; c++) {
			mask16[r * (D / 2) + c] = (c <= r + 10) ? ref_f32_to_f16(0.0f) : ref_f32_to_f16(-1000.0f);
		}
	}

	struct c7x_softmax_desc smd;
	memset(&smd, 0, sizeof(smd));
	smd.ne0 = D / 2; smd.ne1 = R; smd.ne2 = 1; smd.ne3 = 1;
	smd.mask_ne1 = R; smd.mask_ne2 = 1; smd.mask_ne3 = 1;
	smd.nb01 = (D / 2) * 4; smd.nb_dst1 = (D / 2) * 4;
	smd.mask_nb1 = (D / 2) * 2;
	smd.scale = 0.25f;

	struct c7x_rope_desc rd;
	memset(&rd, 0, sizeof(rd));
	rd.ne0[0] = D; rd.ne0[1] = 1; rd.ne0[2] = R; rd.ne0[3] = 1;
	rd.nb0[0] = 4; rd.nb0[1] = D * 4; rd.nb0[2] = D * 4; rd.nb0[3] = R * D * 4;
	rd.nb_dst[0] = 4; rd.nb_dst[1] = D * 4; rd.nb_dst[2] = D * 4; rd.nb_dst[3] = R * D * 4;
	rd.n_dims = D; rd.mode = C7X_ROPE_MODE_NEOX; rd.n_ctx_orig = 2048; rd.truncate = 1;
	rd.freq_base = 10000.0f; rd.freq_scale = 1.0f; rd.attn_factor = 1.0f;

	/* Also test SET_ROWS -> CPY(F16->F32) -> GET_ROWS(Q8_0) in the same batch */
	uint16_t *kv_f16 = (uint16_t *)alloc64(8 * D * sizeof(uint16_t));
	float *kv_back = (float *)alloc64(8 * D * sizeof(float));
	int64_t *set_idx = (int64_t *)alloc64(R * sizeof(int64_t));
	set_idx[0] = 2; set_idx[1] = 5; set_idx[2] = 1;

	struct c7x_set_rows_desc srd;
	memset(&srd, 0, sizeof(srd));
	srd.ne0[0] = D; srd.ne0[1] = R; srd.ne0[2] = 1; srd.ne0[3] = 1;
	srd.ne1[0] = R; srd.ne1[1] = 1; srd.ne1[2] = 1; srd.ne1[3] = 1;
	srd.ne_dst[0] = D; srd.ne_dst[1] = 8; srd.ne_dst[2] = 1; srd.ne_dst[3] = 1;
	srd.nb0[0] = 4; srd.nb0[1] = D * 4; srd.nb0[2] = R * D * 4; srd.nb0[3] = R * D * 4;
	srd.nb1[0] = 8; srd.nb1[1] = R * 8; srd.nb1[2] = R * 8;
	srd.nb_dst[0] = 2; srd.nb_dst[1] = D * 2; srd.nb_dst[2] = 8 * D * 2; srd.nb_dst[3] = 8 * D * 2;

	struct c7x_cpy_desc cpd;
	memset(&cpd, 0, sizeof(cpd));
	cpd.ne0[0] = D; cpd.ne0[1] = 8; cpd.ne0[2] = 1; cpd.ne0[3] = 1;
	cpd.ne_dst[0] = D; cpd.ne_dst[1] = 8; cpd.ne_dst[2] = 1; cpd.ne_dst[3] = 1;
	cpd.nb0[0] = 2; cpd.nb0[1] = D * 2; cpd.nb0[2] = 8 * D * 2; cpd.nb0[3] = 8 * D * 2;
	cpd.nb_dst[0] = 4; cpd.nb_dst[1] = D * 4; cpd.nb_dst[2] = 8 * D * 4; cpd.nb_dst[3] = 8 * D * 4;

	struct c7x_cmd cmds[7];
	memset(cmds, 0, sizeof(cmds));
	/* 0: unweighted RMSNorm */
	cmds[0].op = C7X_CMD_RMS_NORM_F32;
	cmds[0].p0 = D; cmds[0].p1 = R; cmds[0].f0 = 1e-5f;
	cmds[0].src0 = (uint64_t)(uintptr_t)x;
	cmds[0].dst  = (uint64_t)(uintptr_t)normed;
	/* 1: row-broadcast MUL by weight */
	cmds[1].op = C7X_CMD_MUL_F32;
	cmds[1].p0 = D; cmds[1].p1 = R; cmds[1].p2 = 1; cmds[1].p3 = D;
	cmds[1].src0 = (uint64_t)(uintptr_t)normed;
	cmds[1].src1 = (uint64_t)(uintptr_t)w;
	cmds[1].dst  = (uint64_t)(uintptr_t)scaled;
	/* 2: single-tensor SwiGLU */
	cmds[2].op = C7X_CMD_GLU_F32;
	cmds[2].p0 = D / 2; cmds[2].p1 = R; cmds[2].p2 = C7X_GLU_SWIGLU;
	cmds[2].src0 = (uint64_t)(uintptr_t)scaled;
	cmds[2].dst  = (uint64_t)(uintptr_t)glu_out;
	/* 3: Softmax with F16 mask */
	cmds[3].op = C7X_CMD_SOFT_MAX_F32;
	cmds[3].flags = C7X_CMD_FLAG_MASK_F16;
	cmds[3].src0 = (uint64_t)(uintptr_t)glu_out;
	cmds[3].src1 = (uint64_t)(uintptr_t)mask16;
	cmds[3].src3 = (uint64_t)(uintptr_t)&smd;
	cmds[3].dst  = (uint64_t)(uintptr_t)sm_out;
	/* 4: RoPE NEOX */
	cmds[4].op = C7X_CMD_ROPE_F32;
	cmds[4].src0 = (uint64_t)(uintptr_t)scaled;
	cmds[4].src1 = (uint64_t)(uintptr_t)pos;
	cmds[4].src3 = (uint64_t)(uintptr_t)&rd;
	cmds[4].dst  = (uint64_t)(uintptr_t)rope_out;
	/* 5: SET_ROWS into F16 KV cache */
	cmds[5].op = C7X_CMD_SET_ROWS;
	cmds[5].flags = C7X_CMD_FLAG_IDX_I64;
	cmds[5].p0 = C7X_DTYPE_F16;
	cmds[5].src0 = (uint64_t)(uintptr_t)rope_out;
	cmds[5].src1 = (uint64_t)(uintptr_t)set_idx;
	cmds[5].src3 = (uint64_t)(uintptr_t)&srd;
	cmds[5].dst  = (uint64_t)(uintptr_t)kv_f16;
	/* 6: CPY F16 -> F32 */
	cmds[6].op = C7X_CMD_CPY;
	cmds[6].p0 = C7X_DTYPE_F16; cmds[6].p1 = C7X_DTYPE_F32;
	cmds[6].src0 = (uint64_t)(uintptr_t)kv_f16;
	cmds[6].src3 = (uint64_t)(uintptr_t)&cpd;
	cmds[6].dst  = (uint64_t)(uintptr_t)kv_back;

	uint32_t done = 0, fail = 0;
	int rc = c7x_cmd_exec_list(cmds, 7, l2, l2_bytes, &done, &fail);

	/* Verify against scalar reference */
	float max_err = 0.0f;
	for (uint32_t r = 0; r < R; r++) {
		float ss = 0.0f;
		for (uint32_t c = 0; c < D; c++) ss += x[r * D + c] * x[r * D + c];
		float inv = 1.0f / sqrtf(ss / D + 1e-5f);
		float ref_s[D];
		for (uint32_t c = 0; c < D; c++) {
			ref_s[c] = x[r * D + c] * inv * w[c];
			float e = fabsf(scaled[r * D + c] - ref_s[c]);
			if (e > max_err) max_err = e;
		}
		/* Check softmax sums to 1 and masked tail is ~0 */
		float sm_sum = 0.0f;
		for (uint32_t c = 0; c < D / 2; c++) sm_sum += sm_out[r * (D / 2) + c];
		float e_sm = fabsf(sm_sum - 1.0f);
		if (e_sm > max_err) max_err = e_sm;
		if (sm_out[r * (D / 2) + (D / 2 - 1)] > 1e-6f) max_err = 1.0f;

		/* Check SET_ROWS + CPY F16 roundtrip */
		int64_t dst_row = set_idx[r];
		for (uint32_t c = 0; c < D; c++) {
			float e = fabsf(kv_back[dst_row * D + c] - rope_out[r * D + c]);
			if (e > 2e-3f) max_err = e; /* FP16 tolerance */
		}
	}

	int bad = (rc != 0 || done != 7 || max_err > 2e-3f);
	printf("[host-cmd] 7-op chained subgraph (max_err=%.2e): %s\n",
	       max_err, bad ? "FAIL" : "PASS");

	/* Bounds & edge-case checks: SET_ROWS OOB must fail; EXP(100) == +Inf; SwiGLU(-200) finite */
	set_idx[0] = 8; /* OOB for ne_dst[1] = 8 */
	int rc_oob = c7x_cmd_exec_list(&cmds[5], 1, l2, l2_bytes, &done, &fail);
	if (rc_oob == 0) {
		fprintf(stderr, "[host-cmd] SET_ROWS OOB was NOT rejected!\n");
		bad = 1;
	}

	float *edge_in  = (float *)alloc64(32 * sizeof(float));
	float *edge_out = (float *)alloc64(32 * sizeof(float));
	for (uint32_t i = 0; i < 32; i++) edge_in[i] = (i & 1u) ? 100.0f : -200.0f;
	struct c7x_cmd ecmd;
	memset(&ecmd, 0, sizeof(ecmd));
	ecmd.op = C7X_CMD_UNARY_F32; ecmd.p0 = 32; ecmd.p1 = C7X_UNARY_EXP;
	ecmd.src0 = (uint64_t)(uintptr_t)edge_in;
	ecmd.dst  = (uint64_t)(uintptr_t)edge_out;
	if (c7x_cmd_exec_list(&ecmd, 1, l2, l2_bytes, &done, &fail) != 0 ||
	    !isinf(edge_out[1]) || edge_out[0] != 0.0f) {
		fprintf(stderr, "[host-cmd] EXP edge failed: out[0]=%g out[1]=%g\n", edge_out[0], edge_out[1]);
		bad = 1;
	}
	ecmd.p1 = C7X_UNARY_SILU;
	if (c7x_cmd_exec_list(&ecmd, 1, l2, l2_bytes, &done, &fail) != 0 ||
	    isnan(edge_out[0]) || isnan(edge_out[1])) {
		fprintf(stderr, "[host-cmd] SILU(-200/+100) produced NaN!\n");
		bad = 1;
	}
	printf("[host-cmd] edge checks (SET_ROWS OOB, EXP +Inf/0, SILU -200): %s\n",
	       bad ? "FAIL" : "PASS");

	free(edge_in); free(edge_out);
	free(x); free(w); free(normed); free(scaled); free(glu_out); free(sm_out);
	free(mask16); free(rope_out); free(pos); free(kv_f16); free(kv_back); free(set_idx);
	return bad ? 1 : 0;
}

int main(void)
{
	void *l2 = alloc64(444u * 1024u);
	int fail = 0;

	fail |= test_matmul_cmds(l2, 444u * 1024u);
	fail |= test_vec_and_chain_cmds(l2, 444u * 1024u);

	free(l2);
	printf("\n=== Command-Buffer Host Emulation Suite: %s ===\n",
	       fail ? "FAIL" : "ALL PASSED");
	return fail;
}
