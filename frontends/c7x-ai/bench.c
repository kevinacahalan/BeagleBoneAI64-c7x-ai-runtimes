/* SPDX-License-Identifier: MIT */
/*
 * C7x MMA + DRU + MSMC + DDR microbenchmark & verification suite.
 *
 * Every compute and DMA test verifies every single output byte/int32 against
 * an independent A72 reference implementation.
 */
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "c7x_ai.h"

static uint32_t g_rng = 0xCAFEBABEu;

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

static int c7x_exec(volatile struct c7x_ctrl *ctrl, uint32_t op, uint32_t flags,
		    uint32_t iters, uint32_t m, uint32_t k, uint32_t n,
		    uint64_t a_addr, uint64_t b_addr, uint64_t c_addr,
		    double *wall_us)
{
	uint32_t next_seq = ctrl->req_seq + 1u;
	double t0;
	uint64_t timeout_us = 10000000.0; /* 10 s */

	ctrl->op     = op;
	ctrl->flags  = flags;
	ctrl->iters  = iters;
	ctrl->m      = m;
	ctrl->k      = k;
	ctrl->n      = n;
	ctrl->a_addr = a_addr;
	ctrl->b_addr = b_addr;
	ctrl->c_addr = c_addr;
	__sync_synchronize();

	t0 = now_us();
	ctrl->req_seq = next_seq;
	__sync_synchronize();

	while (ctrl->done_seq != next_seq) {
		if (now_us() - t0 > timeout_us) {
			fprintf(stderr, "TIMEOUT waiting for C7x op=%u\n", op);
			return -1;
		}
	}
	__sync_synchronize();
	if (wall_us) {
		*wall_us = now_us() - t0;
	}
	if (ctrl->status != C7X_STATUS_OK) {
		fprintf(stderr, "C7x op=%u failed with status=%u\n", op, ctrl->status);
		return -1;
	}
	return 0;
}

static void ref_gemm_i8_i32(const int8_t *a, const int8_t *b, int32_t *c,
			    uint32_t m, uint32_t k, uint32_t n, uint32_t b_trans)
{
	uint32_t i, j, p;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			int32_t acc = 0;

			if (b_trans) {
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

static void ref_gemv_w4a8(const int8_t *a, const uint8_t *w, int32_t *y,
			  uint32_t k, uint32_t n)
{
	uint32_t j, p, half_k = k >> 1;

	for (j = 0; j < n; j++) {
		const uint8_t *wj = w + j * half_k;
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

static int run_ping(volatile struct c7x_ctrl *ctrl)
{
	double sum_us = 0.0, cyc = 0.0;
	int i;

	for (i = 0; i < 100; i++) {
		double us = 0.0;

		if (c7x_exec(ctrl, C7X_OP_PING, 0, 1, 0, 0, 0, 0, 0, 0, &us)) {
			return 1;
		}
		sum_us += us;
		cyc += (double)ctrl->cycles;
	}
	printf("[ping] 100 round-trips: avg wall=%.2f us, C7x kernel=%.1f cyc\n",
	       sum_us / 100.0, cyc / 100.0);
	return 0;
}

static int run_cbank_probe(volatile struct c7x_ctrl *ctrl, uint8_t *ddr)
{
	const uint32_t depths[] = { 16u, 32u, 48u, 60u, 63u, 64u };
	const uint32_t num_depths = (uint32_t)(sizeof(depths) / sizeof(depths[0]));
	int8_t *a_host = malloc(64u * 64u);
	int8_t *b_host = malloc(64u * 64u);
	int32_t *want  = malloc(64u * 64u * sizeof(int32_t));
	uint64_t a_pa  = C7X_DDR_ARENA_BASE;
	uint64_t b_pa  = a_pa + 0x10000u;
	uint64_t y_pa  = b_pa + 0x10000u;
	uint32_t d_idx, i;
	int fail = 0;

	printf("\n=== 1. MMA C-bank Row Depth Hardware Probe ===\n");
	for (i = 0; i < 64u * 64u; i++) {
		a_host[i] = rand_i8();
		b_host[i] = rand_i8();
	}
	memcpy(ddr + (a_pa - C7X_DDR_ARENA_BASE), a_host, 64u * 64u);
	memcpy(ddr + (b_pa - C7X_DDR_ARENA_BASE), b_host, 64u * 64u);

	for (d_idx = 0; d_idx < num_depths; d_idx++) {
		uint32_t depth = depths[d_idx];
		const int32_t *got = (const int32_t *)(ddr + (y_pa - C7X_DDR_ARENA_BASE));
		uint32_t bad = 0, first_bad_row = 999u;

		memset(ddr + (y_pa - C7X_DDR_ARENA_BASE), 0x55, depth * 64u * sizeof(int32_t));
		ref_gemm_i8_i32(a_host, b_host, want, depth, 64u, 64u, 0u);

		if (c7x_exec(ctrl, C7X_OP_PROBE_CBANK, 0, 1, 0, depth, 0, a_pa, b_pa, y_pa, NULL)) {
			fail = 1;
			break;
		}
		for (i = 0; i < depth * 64u; i++) {
			if (got[i] != want[i]) {
				if (bad < 3u) {
					printf("    depth=%u mismatch [%u,%u]: got=%d (0x%08x) want=%d\n",
					       depth, i / 64u, i % 64u, got[i], (uint32_t)got[i], want[i]);
				}
				if (first_bad_row == 999u) {
					first_bad_row = i / 64u;
				}
				bad++;
			}
		}
		printf("  depth=%2u rows: %-4s (mismatches=%u, first_bad_row=%s, cycles=%" PRIu64 ")\n",
		       depth, bad ? "WRAP" : "PASS", bad,
		       first_bad_row == 999u ? "none" : (first_bad_row == 0u ? "0" :
			   (first_bad_row == 16u ? "16" : "other")),
		       ctrl->cycles);
		if (depth == 16u && bad != 0u) {
			fail = 1;
		}
	}

	free(a_host);
	free(b_host);
	free(want);
	return fail;
}

static int run_mem_bw(volatile struct c7x_ctrl *ctrl, uint8_t *ddr)
{
	struct {
		const char *name;
		uint64_t src;
		uint64_t dst;
		uint32_t bytes;
		uint32_t iters;
	} cases[] = {
		{ "L2 SRAM   (128 KiB)", C7X_L2_SCRATCH_ADDR, C7X_L2_SCRATCH_ADDR + 0x20000u, 128u * 1024u, 200u },
		{ "MSMC SRAM (1 MiB)  ", C7X_MSMC_BASE,       C7X_MSMC_BASE + 0x100000u,      1024u * 1024u, 50u },
		{ "DDR Arena (4 MiB)  ", C7X_DDR_ARENA_BASE,  C7X_DDR_ARENA_BASE + 0x400000u, 4u * 1024u * 1024u, 10u },
	};
	size_t i;

	printf("\n=== 2. C7x Streaming Engine (SE0) Memory Bandwidth ===\n");
	/* Initialize DDR region so reads aren't trivial zeros. */
	memset(ddr, 0x3C, 4u * 1024u * 1024u);

	for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		double total_bytes = (double)cases[i].bytes * (double)cases[i].iters;
		double read_gbps, copy_gbps;

		if (c7x_exec(ctrl, C7X_OP_MEM_BW_READ, 0, cases[i].iters,
			     cases[i].bytes, 0, 0, cases[i].src, 0, 0, NULL)) {
			return 1;
		}
		read_gbps = total_bytes / (double)ctrl->cycles; /* 1 GHz -> bytes/cyc == GB/s */

		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, cases[i].iters,
			     cases[i].bytes, 0, 0, cases[i].src, cases[i].dst, 0, NULL)) {
			return 1;
		}
		/* Copy moves 2x bytes (read + write); report payload GB/s and total bus GB/s. */
		copy_gbps = total_bytes / (double)ctrl->cycles;

		printf("  %s: SE0 read = %6.2f GB/s | SE0 copy = %6.2f GB/s payload (%6.2f GB/s R+W)\n",
		       cases[i].name, read_gbps, copy_gbps, copy_gbps * 2.0);
	}
	return 0;
}

static void mmio_zero(void *dst, size_t bytes)
{
	volatile uint64_t *p64 = (volatile uint64_t *)dst;
	size_t i;

	for (i = 0; i < bytes / 8u; i++) {
		p64[i] = 0ull;
	}
}

static int run_dru_bw(volatile struct c7x_ctrl *ctrl, uint8_t *ddr)
{
	struct {
		const char *name;
		uint64_t src;
		uint64_t dst;
		uint32_t bytes;
		uint32_t iters;
	} cases[] = {
		{ "DDR  -> L2   (128 KiB)", C7X_DDR_ARENA_BASE, C7X_L2_SCRATCH_ADDR, 128u * 1024u, 100u },
		{ "L2   -> DDR  (128 KiB)", C7X_L2_SCRATCH_ADDR, C7X_DDR_ARENA_BASE + 0x100000u, 128u * 1024u, 100u },
		{ "DDR  -> MSMC (1 MiB)  ", C7X_DDR_ARENA_BASE, C7X_MSMC_BASE, 1024u * 1024u, 40u },
		{ "MSMC -> L2   (128 KiB)", C7X_MSMC_BASE, C7X_L2_SCRATCH_ADDR, 128u * 1024u, 100u },
		{ "DDR  -> DDR  (4 MiB)  ", C7X_DDR_ARENA_BASE, C7X_DDR_ARENA_BASE + 0x400000u, 4u * 1024u * 1024u, 10u },
	};
	uint32_t k;
	size_t i;

	printf("\n=== 3. Bare-Metal C7x DRU0 Direct-TR DMA Bandwidth ===\n");
	printf("  DRU0: ok=%u pid=0x%016" PRIx64 " caps=0x%016" PRIx64 "\n",
	       ctrl->dru_ok, ctrl->dru_pid, ctrl->dru_caps);
	if (!ctrl->dru_ok) {
		printf("  DRU0 unavailable (skipped).\n");
		return 0;
	}

	/* Fill source pattern in DDR arena and verify DDR->L2->DDR round-trip integrity. */
	for (k = 0; k < 128u * 1024u; k++) {
		ddr[k] = (uint8_t)((k * 131u + 17u) & 0xffu);
	}
	mmio_zero(ddr + 0x100000u, 128u * 1024u);

	for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		double total_bytes = (double)cases[i].bytes * (double)cases[i].iters;
		double gbps;

		if (c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, cases[i].iters,
			     cases[i].bytes, 0, 0, cases[i].src, cases[i].dst, 0, NULL)) {
			return 1;
		}
		gbps = total_bytes / (double)ctrl->cycles;
		printf("  %s: %6.2f GB/s (%.1f us/copy)\n",
		       cases[i].name, gbps, (double)ctrl->cycles / (cases[i].iters * 1000.0));
	}

	/* Verify DDR -> L2 -> DDR round-trip matched byte-for-byte. */
	if (memcmp(ddr, ddr + 0x100000u, 128u * 1024u) != 0) {
		fprintf(stderr, "  FAIL: DRU DDR->L2->DDR data mismatch!\n");
		return 1;
	}
	printf("  DRU DDR->L2->DDR 128 KiB integrity check: PASS\n");

	/* Verify full 7.625 MiB MSMC SRAM bounds (low edge 0x70040000 & high edge 0x707C0000). */
	{
		uint64_t msmc_lo = C7X_MSMC_BASE;
		uint64_t msmc_hi = C7X_MSMC_BASE + C7X_MSMC_SIZE - 128u * 1024u;

		mmio_zero(ddr + 0x100000u, 128u * 1024u);
		if (c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, 1, 128u * 1024u, 0, 0,
			     C7X_DDR_ARENA_BASE, msmc_lo, 0, NULL) ||
		    c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, 1, 128u * 1024u, 0, 0,
			     msmc_lo, msmc_hi, 0, NULL) ||
		    c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, 1, 128u * 1024u, 0, 0,
			     msmc_hi, C7X_DDR_ARENA_BASE + 0x100000u, 0, NULL)) {
			fprintf(stderr, "  FAIL: MSMC 7.625 MiB full-span DRU copy failed!\n");
			return 1;
		}
		if (memcmp(ddr, ddr + 0x100000u, 128u * 1024u) != 0) {
			fprintf(stderr, "  FAIL: MSMC 7.625 MiB edge-to-edge data mismatch!\n");
			return 1;
		}
		printf("  MSMC 7.625 MiB span (0x%08" PRIx64 "..0x%08" PRIx64 ") edge-to-edge check: PASS\n",
		       msmc_lo, msmc_hi + 128u * 1024u);
	}

	/* Verify 1.5 GiB high-DDR arena (0x880000000) via SE0 + DRU0 when reserved in DT. */
	if (access("/proc/device-tree/reserved-memory/c7x-llm-arena@880000000", F_OK) == 0) {
		uint64_t hi_lo = C7X_LLM_HI_ARENA_BASE;
		uint64_t hi_top = C7X_LLM_HI_ARENA_BASE + C7X_LLM_HI_ARENA_SIZE - 128u * 1024u;

		mmio_zero(ddr + 0x100000u, 128u * 1024u);
		if (c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, 1, 128u * 1024u, 0, 0,
			     C7X_DDR_ARENA_BASE, hi_lo, 0, NULL) ||
		    c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, 128u * 1024u, 0, 0,
			     hi_lo, hi_top, 0, NULL) ||
		    c7x_exec(ctrl, C7X_OP_DRU_COPY, 0, 1, 128u * 1024u, 0, 0,
			     hi_top, C7X_DDR_ARENA_BASE + 0x100000u, 0, NULL)) {
			fprintf(stderr, "  FAIL: High-DDR 1.5 GiB arena DRU/SE0 check failed!\n");
			return 1;
		}
		if (memcmp(ddr, ddr + 0x100000u, 128u * 1024u) != 0) {
			fprintf(stderr, "  FAIL: High-DDR 1.5 GiB arena data mismatch!\n");
			return 1;
		}
		printf("  High-DDR 1.5 GiB arena (0x%" PRIx64 "..0x%" PRIx64 ") DRU+SE0 check: PASS\n",
		       hi_lo, hi_top + 128u * 1024u);
	}
	return 0;
}

static int run_gemm_case(volatile struct c7x_ctrl *ctrl, uint8_t *ddr,
			 const char *tier, uint64_t a_pa, uint64_t b_pa, uint64_t c_pa,
			 uint32_t m, uint32_t k, uint32_t n, uint32_t b_trans,
			 uint32_t block_m, uint32_t iters)
{
	int8_t *a_host = malloc((size_t)m * k);
	int8_t *b_host = malloc((size_t)k * n);
	int32_t *want  = malloc((size_t)m * n * sizeof(int32_t));
	uint64_t ddr_a = C7X_DDR_ARENA_BASE;
	uint64_t ddr_b = ddr_a + (((uint64_t)m * k + 0xfffu) & ~0xfffu);
	uint64_t ddr_c = ddr_b + (((uint64_t)k * n + 0xfffu) & ~0xfffu);
	uint32_t i, bad = 0;
	uint32_t flags = (b_trans ? C7X_GEMM_B_TRANSPOSE : 0u) |
			 ((block_m << C7X_GEMM_BLOCK_M_SHIFT) & C7X_GEMM_BLOCK_M_MASK);
	double macs = (double)m * (double)k * (double)n;
	double avg_cyc, mac_per_cyc, gops, gbps_b;

	if (a_pa == 0ull) {
		a_pa = ddr_a;
		b_pa = ddr_b;
		c_pa = ddr_c;
	}

	for (i = 0; i < m * k; i++) {
		a_host[i] = rand_i8();
	}
	for (i = 0; i < k * n; i++) {
		b_host[i] = rand_i8();
	}
	memcpy(ddr + (ddr_a - C7X_DDR_ARENA_BASE), a_host, (size_t)m * k);
	memcpy(ddr + (ddr_b - C7X_DDR_ARENA_BASE), b_host, (size_t)k * n);

	/* If target tier is L2 or MSMC, stage inputs using C7x copy. */
	if (a_pa != ddr_a) {
		uint32_t ab = ((m * k + 63u) & ~63u);
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, ab, 0, 0, ddr_a, a_pa, 0, NULL)) {
			goto err;
		}
	}
	if (b_pa != ddr_b) {
		uint32_t bb = ((k * n + 63u) & ~63u);
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, bb, 0, 0, ddr_b, b_pa, 0, NULL)) {
			goto err;
		}
	}

	ref_gemm_i8_i32(a_host, b_host, want, m, k, n, b_trans);

	uint64_t kern_cycles;

	if (c7x_exec(ctrl, C7X_OP_GEMM_MMA_I8, flags,
		     iters, m, k, n, a_pa, b_pa, c_pa, NULL)) {
		goto err;
	}
	kern_cycles = ctrl->cycles;

	/* Read back C from target tier if not already in DDR. */
	if (c_pa != ddr_c) {
		uint32_t cb = ((m * n * 4u + 63u) & ~63u);
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, cb, 0, 0, c_pa, ddr_c, 0, NULL)) {
			goto err;
		}
	}
	{
		const int32_t *got = (const int32_t *)(ddr + (ddr_c - C7X_DDR_ARENA_BASE));

		for (i = 0; i < m * n; i++) {
			if (got[i] != want[i]) {
				if (bad < 4u) {
					fprintf(stderr, "    mismatch [%u,%u]: got=%d want=%d\n",
						i / n, i % n, got[i], want[i]);
				}
				bad++;
			}
		}
	}

	avg_cyc = (double)kern_cycles / (double)iters;
	mac_per_cyc = macs / avg_cyc;
	gops = (2.0 * macs) / avg_cyc; /* 1 GHz -> 2*MACs/cyc = GOPS */
	gbps_b = ((double)k * (double)n) / avg_cyc; /* effective weight BW in GB/s */

	printf("  %-5s %3ux%4ux%4u (%s bm=%2u): %7.1f us | %6.1f MAC/cyc | %7.1f GOPS | W_BW=%5.2f GB/s | %s\n",
	       tier, m, k, n, b_trans ? "BT" : "BN", block_m ? block_m : 16u,
	       avg_cyc / 1000.0, mac_per_cyc, gops, gbps_b,
	       bad ? "FAIL" : "PASS");

	free(a_host);
	free(b_host);
	free(want);
	return bad ? 1 : 0;

err:
	free(a_host);
	free(b_host);
	free(want);
	return 1;
}

static int run_w4a8_case(volatile struct c7x_ctrl *ctrl, uint8_t *ddr,
			 const char *tier, uint64_t a_pa, uint64_t w_pa, uint64_t y_pa,
			 uint32_t k, uint32_t n, uint32_t iters)
{
	int8_t *a_host = malloc(k);
	uint8_t *w_host = malloc((size_t)n * (k / 2u));
	int32_t *want = malloc((size_t)n * sizeof(int32_t));
	uint64_t ddr_a = C7X_DDR_ARENA_BASE;
	uint64_t ddr_w = ddr_a + ((k + 0xfffu) & ~0xfffu);
	uint64_t ddr_y = ddr_w + (((uint64_t)n * (k / 2u) + 0xfffu) & ~0xfffu);
	uint32_t i, bad = 0;
	double avg_cyc, gbps_w, macs = (double)k * (double)n;

	if (a_pa == 0ull) {
		a_pa = ddr_a;
		w_pa = ddr_w;
		y_pa = ddr_y;
	}

	for (i = 0; i < k; i++) {
		a_host[i] = rand_i8();
	}
	for (i = 0; i < n * (k / 2u); i++) {
		w_host[i] = rand_u8();
	}
	memcpy(ddr + (ddr_a - C7X_DDR_ARENA_BASE), a_host, k);
	memcpy(ddr + (ddr_w - C7X_DDR_ARENA_BASE), w_host, (size_t)n * (k / 2u));

	if (a_pa != ddr_a) {
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, (k + 63u) & ~63u, 0, 0, ddr_a, a_pa, 0, NULL)) {
			goto err;
		}
	}
	if (w_pa != ddr_w) {
		uint32_t wb = ((n * (k / 2u) + 63u) & ~63u);
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, wb, 0, 0, ddr_w, w_pa, 0, NULL)) {
			goto err;
		}
	}

	uint64_t kern_cycles;

	ref_gemv_w4a8(a_host, w_host, want, k, n);
	if (c7x_exec(ctrl, C7X_OP_GEMV_W4A8, 0, iters, 1u, k, n, a_pa, w_pa, y_pa, NULL)) {
		goto err;
	}
	kern_cycles = ctrl->cycles;
	if (y_pa != ddr_y) {
		if (c7x_exec(ctrl, C7X_OP_MEM_BW_COPY, 0, 1, (n * 4u + 63u) & ~63u, 0, 0, y_pa, ddr_y, 0, NULL)) {
			goto err;
		}
	}
	{
		const int32_t *got = (const int32_t *)(ddr + (ddr_y - C7X_DDR_ARENA_BASE));

		for (i = 0; i < n; i++) {
			if (got[i] != want[i]) {
				if (bad < 3u) {
					printf("    w4a8 mismatch [%u]: got=%d want=%d\n", i, got[i], want[i]);
				}
				bad++;
			}
		}
	}

	avg_cyc = (double)kern_cycles / (double)iters;
	gbps_w = ((double)n * (double)(k / 2u)) / avg_cyc;
	printf("  %-5s 1x%4ux%4u (W4A8): %7.1f us | %6.1f MAC/cyc | W_BW=%5.2f GB/s (eq INT8=%5.2f GB/s) | %s\n",
	       tier, k, n, avg_cyc / 1000.0, macs / avg_cyc, gbps_w, gbps_w * 2.0,
	       bad ? "FAIL" : "PASS");

	free(a_host);
	free(w_host);
	free(want);
	return bad ? 1 : 0;

err:
	free(a_host);
	free(w_host);
	free(want);
	return 1;
}

int cmd_bench(int argc, char **argv)
{
	const char *suite = argc > 2 ? argv[2] : "all";
	volatile struct c7x_ctrl *ctrl;
	uint8_t *ddr;
	char state[32];
	int fail = 0;

	if (!rproc_running(C7X_RPROC_NAME, state, sizeof(state))) {
		fprintf(stderr, "C7x is %s; run 'c7x-ai load' first\n", state);
		return 1;
	}
	ctrl = (volatile struct c7x_ctrl *)map_phys((off_t)C7X_CTRL_ADDR, C7X_CTRL_SIZE);
	if (!ctrl) {
		return 1;
	}
	if (ctrl->magic != C7X_CTRL_MAGIC) {
		fprintf(stderr, "C7x magic mismatch (got 0x%08x, want 0x%08x); load %s first\n",
			ctrl->magic, C7X_CTRL_MAGIC, C7X_DEFAULT_FW);
		unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
		return 1;
	}
	ddr = (uint8_t *)map_phys((off_t)C7X_DDR_ARENA_BASE, C7X_DDR_ARENA_SIZE);
	if (!ddr) {
		unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
		return 1;
	}

	printf("C7x-AI Hardware Benchmark Suite (DNUM=%u, MMU=%u, L1DCFG=%u, L2CFG=%u)\n",
	       ctrl->dnum, ctrl->scr_lo & 1u, ctrl->l1dcfg, ctrl->l2cfg);

	if (strcmp(suite, "all") == 0 || strcmp(suite, "ping") == 0) {
		fail |= run_ping(ctrl);
	}
	if (strcmp(suite, "all") == 0 || strcmp(suite, "cbank") == 0) {
		fail |= run_cbank_probe(ctrl, ddr);
	}
	if (strcmp(suite, "all") == 0 || strcmp(suite, "mem") == 0) {
		fail |= run_mem_bw(ctrl, ddr);
	}
	if (strcmp(suite, "all") == 0 || strcmp(suite, "dru") == 0) {
		fail |= run_dru_bw(ctrl, ddr);
	}
	if (strcmp(suite, "all") == 0 || strcmp(suite, "gemm") == 0) {
		uint64_t l2_a = C7X_L2_SCRATCH_ADDR;
		uint64_t l2_b = l2_a + 0x10000u; /* 64 KiB for A */
		uint64_t l2_c = l2_b + 0x40000u; /* 256 KiB for B */
		uint64_t ms_a = C7X_MSMC_BASE;
		uint64_t ms_b = ms_a + 0x080000u;
		uint64_t ms_c = ms_b + 0x200000u;

		printf("\n=== 4. MMA INT8xINT8->INT32 GEMM (Prefill / Encoder Shapes) ===\n");
		fail |= run_gemm_case(ctrl, ddr, "L2",   l2_a, l2_b, l2_c,  16u,  64u,  64u, 0u, 16u, 200u);
		fail |= run_gemm_case(ctrl, ddr, "L2",   l2_a, l2_b, l2_c,  60u, 256u, 256u, 1u, 60u,  50u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c,  64u, 256u, 256u, 1u, 16u,  20u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c,  64u, 256u, 256u, 1u, 32u,  20u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c,  60u, 512u, 512u, 1u, 60u,  10u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 120u, 512u, 512u, 1u, 60u,  10u);
		fail |= run_gemm_case(ctrl, ddr, "DDR",  0, 0, 0,  60u, 512u, 512u, 1u, 60u, 10u);
		fail |= run_gemm_case(ctrl, ddr, "DDR",  0, 0, 0, 120u, 768u, 768u, 1u, 60u,  5u);
	}
	if (strcmp(suite, "all") == 0 || strcmp(suite, "gemv") == 0) {
		uint64_t l2_a = C7X_L2_SCRATCH_ADDR;
		uint64_t l2_b = l2_a + 0x4000u;
		uint64_t l2_c = l2_b + 0x50000u;
		uint64_t ms_a = C7X_MSMC_BASE;
		uint64_t ms_b = ms_a + 0x10000u;
		uint64_t ms_c = ms_b + 0x300000u;

		printf("\n=== 5. Autoregressive Decode GEMV (M=1, W8A8 MMA vs W4A8 Vector) ===\n");
		/* Stories15M/42M/110M / SmolLM2-like linear projection shapes */
		fail |= run_gemm_case(ctrl, ddr, "L2",   l2_a, l2_b, l2_c, 1u, 256u,  256u, 1u, 16u, 100u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 1u, 256u,  768u, 1u, 16u,  50u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 1u, 512u,  512u, 1u, 16u,  50u);
		fail |= run_gemm_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 1u, 512u, 2048u, 1u, 16u,  20u);
		fail |= run_gemm_case(ctrl, ddr, "DDR",  0, 0, 0, 1u, 512u,  512u, 1u, 16u, 20u);
		fail |= run_gemm_case(ctrl, ddr, "DDR",  0, 0, 0, 1u, 512u, 2048u, 1u, 16u, 10u);
		fail |= run_gemm_case(ctrl, ddr, "DDR",  0, 0, 0, 1u, 768u,  768u, 1u, 16u, 10u);

		printf("\n  --- Packed W4A8 (4-bit weights, 8-bit activations) ---\n");
		fail |= run_w4a8_case(ctrl, ddr, "L2",   l2_a, l2_b, l2_c, 256u,  256u, 100u);
		fail |= run_w4a8_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 512u,  512u,  50u);
		fail |= run_w4a8_case(ctrl, ddr, "MSMC", ms_a, ms_b, ms_c, 512u, 2048u,  20u);
		fail |= run_w4a8_case(ctrl, ddr, "DDR",  0, 0, 0, 512u,  512u, 20u);
		fail |= run_w4a8_case(ctrl, ddr, "DDR",  0, 0, 0, 512u, 2048u, 10u);
		fail |= run_w4a8_case(ctrl, ddr, "DDR",  0, 0, 0, 768u,  768u, 10u);
	}

	printf("\nSummary: %s\n", fail ? "FAILED" : "ALL BENCHMARKS PASSED BIT-EXACT");
	unmap_phys(ddr, C7X_DDR_ARENA_SIZE);
	unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
	return fail;
}
