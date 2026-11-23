/* SPDX-License-Identifier: MIT */
/*
 * Unit tests for libc7x: coalescing arena allocator, tile-major weight packers
 * (FP32 / Q4_0 / Q8_0 / F16), and command-buffer execution via hostemu.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c7x_rt.h"

static uint32_t g_rng = 0x51A7C700u;

static float rand_f32(void)
{
	g_rng = g_rng * 1664525u + 1013904223u;
	return ((float)((int32_t)((g_rng >> 8) & 0xFFFFu) - 32768)) / 16384.0f;
}

static uint16_t f32_to_f16(float f)
{
	uint32_t bits;
	uint32_t sign, exp, mant;
	int32_t ne;

	memcpy(&bits, &f, 4);
	sign = (bits >> 16) & 0x8000u;
	exp  = (bits >> 23) & 0xFFu;
	mant = bits & 0x7FFFFFu;
	if (exp == 255u) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
	ne = (int32_t)exp - 112;
	if (ne >= 31) return (uint16_t)(sign | 0x7C00u);
	if (ne <= 0) {
		if (ne < -10) return (uint16_t)sign;
		mant = (mant | 0x800000u) >> (uint32_t)(1 - ne);
		mant += 0xFFFu + ((mant >> 13) & 1u);
		return (uint16_t)(sign | (mant >> 13));
	}
	mant += 0xFFFu + ((mant >> 13) & 1u);
	if (mant & 0x800000u) {
		mant = 0u;
		if (++ne >= 31) return (uint16_t)(sign | 0x7C00u);
	}
	return (uint16_t)(sign | ((uint32_t)ne << 10) | (mant >> 13));
}

static int test_allocator(c7x_ctx_t *ctx)
{
	struct c7x_rt_info info0, info1;
	c7x_buf_t *a, *b, *c, *d, *m;

	c7x_get_info(ctx, &info0);
	a = c7x_alloc(ctx, 4096u, C7X_MEM_DDR);
	b = c7x_alloc(ctx, 8192u, C7X_MEM_DDR);
	c = c7x_alloc(ctx, 4096u, C7X_MEM_DDR);
	m = c7x_alloc(ctx, 16384u, C7X_MEM_MSMC);
	if (!a || !b || !c || !m) {
		fprintf(stderr, "alloc failed\n");
		return 1;
	}
	if ((c7x_buf_phys(a) & 63u) || (c7x_buf_phys(b) & 63u) || (c7x_buf_phys(m) & 63u)) {
		fprintf(stderr, "alignment failed\n");
		return 1;
	}
	if (c7x_host_to_phys(ctx, (uint8_t *)c7x_buf_host_ptr(b) + 128u) != c7x_buf_phys(b) + 128u) {
		fprintf(stderr, "host_to_phys failed\n");
		return 1;
	}

	/* Free a and b -> should coalesce into 12 KiB hole that fits d(10 KiB) */
	c7x_free(ctx, a);
	c7x_free(ctx, b);
	d = c7x_alloc(ctx, 10240u, C7X_MEM_DDR);
	if (!d || c7x_buf_phys(d) != info0.ddr_base) {
		fprintf(stderr, "coalescing failed\n");
		return 1;
	}
	c7x_free(ctx, d);
	c7x_free(ctx, c);
	c7x_free(ctx, m);

	c7x_get_info(ctx, &info1);
	if (info1.ddr_free != info0.ddr_free || info1.msmc_free != info0.msmc_free) {
		fprintf(stderr, "free bytes mismatch after full free: ddr %zu vs %zu\n",
			info1.ddr_free, info0.ddr_free);
		return 1;
	}
	printf("[libc7x] allocator coalescing & address translation: PASS\n");
	return 0;
}

static int test_q4_0_and_q8_0_cmdbuf(c7x_ctx_t *ctx)
{
	const uint32_t M = 4, K = 128, N = 96;
	float *w_f32 = (float *)calloc((size_t)N * K, sizeof(float));
	float *a_f32 = (float *)calloc((size_t)M * K, sizeof(float));
	float *ref   = (float *)calloc((size_t)M * N, sizeof(float));
	uint8_t *q4_buf = (uint8_t *)calloc((size_t)N * (K / 32u) * 18u, 1);
	uint8_t *q8_buf = (uint8_t *)calloc((size_t)N * (K / 32u) * 34u, 1);
	size_t s4_off = 0, s8_off = 0;
	size_t sz4 = c7x_w4a8_g64_packed_bytes(N, K, &s4_off);
	size_t sz8 = c7x_w8a8_packed_bytes(N, K, &s8_off);
	c7x_buf_t *b_w4, *b_w8, *b_a, *b_o4, *b_o8;
	uint8_t *tmp4 = (uint8_t *)calloc(1, sz4);
	uint8_t *tmp8 = (uint8_t *)calloc(1, sz8);
	c7x_cmdbuf_t *cb;
	struct c7x_cmd cmd;
	struct c7x_exec_stats stats;
	float *out4 = (float *)calloc((size_t)M * N, sizeof(float));
	float *out8 = (float *)calloc((size_t)M * N, sizeof(float));
	uint32_t i, m, n, k, b, j;
	double dot4 = 0, na4 = 0, nb4 = 0;
	double dot8 = 0, na8 = 0, nb8 = 0;
	float cos4, cos8;
	int bad;

	for (i = 0; i < N * K; i++) w_f32[i] = rand_f32();
	for (i = 0; i < M * K; i++) a_f32[i] = rand_f32();

	/* Encode w_f32 to GGML Q4_0 and Q8_0 */
	for (n = 0; n < N; n++) {
		for (b = 0; b < K / 32u; b++) {
			const float *src = w_f32 + n * K + b * 32u;
			uint8_t *b4 = q4_buf + (n * (K / 32u) + b) * 18u;
			uint8_t *b8 = q8_buf + (n * (K / 32u) + b) * 34u;
			float amax = 0.0f;
			uint16_t d4h, d8h;
			float d4, id4, d8, id8;

			for (j = 0; j < 32u; j++) {
				if (fabsf(src[j]) > amax) amax = fabsf(src[j]);
			}
			d4 = amax / 7.0f;
			id4 = d4 > 0.0f ? (1.0f / d4) : 0.0f;
			d4h = f32_to_f16(d4);
			b4[0] = (uint8_t)(d4h & 0xFFu);
			b4[1] = (uint8_t)(d4h >> 8);
			for (j = 0; j < 16u; j++) {
				int q0 = (int)lrintf(src[j] * id4) + 8;
				int q1 = (int)lrintf(src[j + 16u] * id4) + 8;
				if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
				if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
				b4[2u + j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
			}

			d8 = amax / 127.0f;
			id8 = d8 > 0.0f ? (1.0f / d8) : 0.0f;
			d8h = f32_to_f16(d8);
			b8[0] = (uint8_t)(d8h & 0xFFu);
			b8[1] = (uint8_t)(d8h >> 8);
			for (j = 0; j < 32u; j++) {
				int q = (int)lrintf(src[j] * id8);
				if (q < -127) q = -127; if (q > 127) q = 127;
				((int8_t *)(b8 + 2u))[j] = (int8_t)q;
			}
		}
	}

	for (m = 0; m < M; m++) {
		for (n = 0; n < N; n++) {
			float s = 0.0f;
			for (k = 0; k < K; k++) s += a_f32[m * K + k] * w_f32[n * K + k];
			ref[m * N + n] = s;
		}
	}

	c7x_pack_q4_0_to_w4a8_g64(q4_buf, N, K, tmp4, (float *)(tmp4 + s4_off));
	c7x_pack_q8_0_to_w8a8(q8_buf, N, K, (int8_t *)tmp8, (float *)(tmp8 + s8_off));

	b_w4 = c7x_alloc(ctx, sz4, C7X_MEM_MSMC);
	b_w8 = c7x_alloc(ctx, sz8, C7X_MEM_DDR);
	b_a  = c7x_alloc(ctx, M * K * sizeof(float), C7X_MEM_DDR);
	b_o4 = c7x_alloc(ctx, M * N * sizeof(float), C7X_MEM_DDR);
	b_o8 = c7x_alloc(ctx, M * N * sizeof(float), C7X_MEM_DDR);
	c7x_write(ctx, b_w4, 0, tmp4, sz4);
	c7x_write(ctx, b_w8, 0, tmp8, sz8);
	c7x_write(ctx, b_a, 0, a_f32, M * K * sizeof(float));

	cb = c7x_cmdbuf_create(ctx, 16, 4096);
	memset(&cmd, 0, sizeof(cmd));
	cmd.op   = C7X_CMD_MATMUL_W4A8_G64;
	cmd.p0   = M; cmd.p1 = K; cmd.p2 = N;
	cmd.src0 = c7x_buf_phys(b_w4);
	cmd.src1 = c7x_buf_phys(b_a);
	cmd.src2 = c7x_buf_phys(b_w4) + s4_off;
	cmd.dst  = c7x_buf_phys(b_o4);
	c7x_cmdbuf_push(cb, &cmd);

	memset(&cmd, 0, sizeof(cmd));
	cmd.op   = C7X_CMD_MATMUL_W8A8;
	cmd.p0   = M; cmd.p1 = K; cmd.p2 = N;
	cmd.src0 = c7x_buf_phys(b_w8);
	cmd.src1 = c7x_buf_phys(b_a);
	cmd.src2 = c7x_buf_phys(b_w8) + s8_off;
	cmd.dst  = c7x_buf_phys(b_o8);
	c7x_cmdbuf_push(cb, &cmd);

	bad = (c7x_cmdbuf_submit(ctx, cb, &stats) != 0 || stats.cmds_done != 2u);
	c7x_read(ctx, out4, b_o4, 0, M * N * sizeof(float));
	c7x_read(ctx, out8, b_o8, 0, M * N * sizeof(float));

	for (i = 0; i < M * N; i++) {
		dot4 += (double)out4[i] * ref[i]; na4 += (double)out4[i] * out4[i]; nb4 += (double)ref[i] * ref[i];
		dot8 += (double)out8[i] * ref[i]; na8 += (double)out8[i] * out8[i]; nb8 += (double)ref[i] * ref[i];
	}
	cos4 = (float)(dot4 / (sqrt(na4) * sqrt(nb4)));
	cos8 = (float)(dot8 / (sqrt(na8) * sqrt(nb8)));
	if (cos4 < 0.99f || cos8 < 0.999f) bad = 1;

	printf("[libc7x] Q4_0->W4A8(MSMC) cos=%.5f, Q8_0->W8A8(DDR) cos=%.6f: %s\n",
	       cos4, cos8, bad ? "FAIL" : "PASS");

	c7x_cmdbuf_destroy(cb);
	c7x_free(ctx, b_w4); c7x_free(ctx, b_w8); c7x_free(ctx, b_a);
	c7x_free(ctx, b_o4); c7x_free(ctx, b_o8);
	free(w_f32); free(a_f32); free(ref); free(q4_buf); free(q8_buf);
	free(tmp4); free(tmp8); free(out4); free(out8);
	return bad;
}

int main(void)
{
	struct c7x_open_opts opts;
	c7x_ctx_t *ctx;
	double ping_us = 0.0;
	int fail = 0;

	memset(&opts, 0, sizeof(opts));
	opts.transport = C7X_TRANSPORT_HOSTEMU;
	opts.emu_ddr_bytes = 32u * 1024u * 1024u;

	ctx = c7x_open(&opts);
	if (!ctx) {
		fprintf(stderr, "c7x_open failed\n");
		return 1;
	}
	if (c7x_ping(ctx, &ping_us) != 0) {
		fprintf(stderr, "c7x_ping failed\n");
		c7x_close(ctx);
		return 1;
	}
	printf("[libc7x] hostemu ping: %.1f us\n", ping_us);

	fail |= test_allocator(ctx);
	fail |= test_q4_0_and_q8_0_cmdbuf(ctx);

	c7x_close(ctx);
	printf("\n=== libc7x Host Emulation Suite: %s ===\n", fail ? "FAIL" : "ALL PASSED");
	return fail;
}
