/* SPDX-License-Identifier: MIT */
/*
 * x86_64 C7000 Host Emulation verification for the full C7x + MMA transformer:
 *   1. Single decoder layer (layer 0) vs Python W8A8 golden trace
 *   2. Full 6-layer model (pos = 0) logits & layer outputs vs Python golden trace
 *   3. Multi-step greedy autoregressive generation vs Python golden token IDs
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c7x_llama_fmt.h"
#include "transformer.h"

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

static uint8_t *read_file_aligned(const char *path, size_t *out_size)
{
	FILE *f = fopen(path, "rb");
	long sz;
	uint8_t *buf;

	if (!f) {
		perror(path);
		exit(1);
	}
	fseek(f, 0, SEEK_END);
	sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	buf = (uint8_t *)alloc_aligned64((size_t)sz);
	if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
		fprintf(stderr, "short read on %s\n", path);
		exit(1);
	}
	fclose(f);
	*out_size = (size_t)sz;
	return buf;
}

static void compare_vec(const char *label, const float *got, const float *want,
			uint32_t n, float *max_err_out, double *cos_out)
{
	float max_err = 0.0f;
	double dot = 0.0, ng = 0.0, nw = 0.0;

	for (uint32_t i = 0; i < n; i++) {
		float e = fabsf(got[i] - want[i]);
		if (e > max_err) {
			max_err = e;
		}
		dot += (double)got[i] * (double)want[i];
		ng += (double)got[i] * (double)got[i];
		nw += (double)want[i] * (double)want[i];
	}
	double cos_sim = dot / (sqrt(ng) * sqrt(nw) + 1e-12);
	printf("[host-llama] %-14s max_abs_err=%.6g  cosine=%.7f\n", label, max_err, cos_sim);
	if (max_err_out) *max_err_out = max_err;
	if (cos_out) *cos_out = cos_sim;
}

static uint32_t argmax_f32(const float *v, uint32_t n)
{
	uint32_t best = 0u;
	float best_v = v[0];

	for (uint32_t i = 1u; i < n; i++) {
		if (v[i] > best_v) {
			best_v = v[i];
			best = i;
		}
	}
	return best;
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: %s <model.c7m> <golden.bin> [steps]\n", argv[0]);
		return 2;
	}
	const char *model_path = argv[1];
	const char *golden_path = argv[2];
	uint32_t steps = argc > 3 ? (uint32_t)atoi(argv[3]) : 6u;

	size_t model_sz = 0, golden_sz = 0;
	uint8_t *model_buf = read_file_aligned(model_path, &model_sz);
	uint8_t *golden_buf = read_file_aligned(golden_path, &golden_sz);

	const struct c7m_header *hdr = (const struct c7m_header *)model_buf;
	if (c7x_llama_validate(hdr) != 0) {
		fprintf(stderr, "invalid .c7m header\n");
		return 1;
	}

	const uint32_t *gh = (const uint32_t *)golden_buf;
	if (gh[0] != 0x31444C47u) {
		fprintf(stderr, "invalid golden magic 0x%08x\n", gh[0]);
		return 1;
	}
	uint32_t g_dim = gh[1];
	uint32_t g_vocab = gh[2];
	uint32_t g_steps = gh[3];
	uint32_t start_tok = gh[4];
	if (g_dim != hdr->dim || g_vocab != hdr->vocab_size) {
		fprintf(stderr, "golden dimensions mismatch\n");
		return 1;
	}
	if (steps > g_steps) {
		steps = g_steps;
	}

	const uint32_t *g_tokens = (const uint32_t *)(golden_buf + 20u);
	const float *g_l0 = (const float *)(g_tokens + g_steps);
	const float *g_last = g_l0 + g_dim;
	const float *g_fnorm = g_last + g_dim;
	const float *g_logits = g_fnorm + g_dim;

	size_t kv_bytes = (size_t)hdr->n_layers * hdr->seq_len * hdr->kv_dim * sizeof(float);
	float *k_cache = (float *)alloc_aligned64(kv_bytes);
	float *v_cache = (float *)alloc_aligned64(kv_bytes);
	float *logits = (float *)alloc_aligned64(hdr->vocab_size * sizeof(float));
	float *dbg_x = (float *)alloc_aligned64(hdr->dim * sizeof(float));
	void *scratch = alloc_aligned64(444u * 1024u);

	int fail = 0;
	float err = 0.0f;
	double cos_sim = 0.0;

	/* 1. Single decoder layer verification (layer 0, pos = 0) */
	if (c7x_llama_forward_token(hdr, start_tok, 0u, k_cache, v_cache, NULL, NULL,
				    scratch, 444u * 1024u, NULL, 0u, NULL, dbg_x, 0, 1u) != 0) {
		fprintf(stderr, "single-layer forward failed\n");
		return 1;
	}
	compare_vec("layer0_out", dbg_x, g_l0, hdr->dim, &err, &cos_sim);
	if (err > 1e-4f || cos_sim < 0.999999) {
		fprintf(stderr, "FAIL: layer0_out diverged\n");
		fail = 1;
	}

	/* 2. Full model forward pass (pos = 0..steps-1) */
	memset(k_cache, 0, kv_bytes);
	memset(v_cache, 0, kv_bytes);
	uint32_t tok = start_tok;
	uint32_t tok_matches = 0u;
	for (uint32_t pos = 0u; pos < steps; pos++) {
		uint32_t argmax = 0u;
		int rc = c7x_llama_forward_token(hdr, tok, pos, k_cache, v_cache,
						 (pos == 0u) ? logits : NULL, &argmax,
						 scratch, 444u * 1024u, NULL, 0u, NULL,
						 (pos == 0u) ? dbg_x : NULL, -1, 0u);
		if (rc != 0) {
			fprintf(stderr, "forward failed at pos=%u (rc=%d)\n", pos, rc);
			return 1;
		}
		if (pos == 0u) {
			compare_vec("final_norm", dbg_x, g_fnorm, hdr->dim, &err, &cos_sim);
			if (err > 5e-4f || cos_sim < 0.999999) {
				fprintf(stderr, "FAIL: final_norm diverged\n");
				fail = 1;
			}
			compare_vec("pos0_logits", logits, g_logits, hdr->vocab_size, &err, &cos_sim);
			if (err > 5e-3f || cos_sim < 0.999999) {
				fprintf(stderr, "FAIL: pos0_logits diverged\n");
				fail = 1;
			}
			if (argmax != argmax_f32(logits, hdr->vocab_size)) {
				fprintf(stderr, "FAIL: fused argmax mismatch\n");
				fail = 1;
			}
		}
		uint32_t nxt = argmax;
		if (nxt == g_tokens[pos]) {
			tok_matches++;
		} else {
			fprintf(stderr, "FAIL: pos=%u token got=%u want=%u\n",
				pos, nxt, g_tokens[pos]);
			fail = 1;
		}
		printf("[host-llama] pos=%u in=%u -> out=%u (want=%u)\n",
		       pos, tok, nxt, g_tokens[pos]);
		tok = nxt;
	}

	printf("[host-llama] greedy token match: %u/%u -> %s\n",
	       tok_matches, steps, fail ? "FAILED" : "PASS");

	/* 3. Batched prefill verification (matches token-by-token next token) */
	if (steps >= 4u) {
		uint32_t pf_toks[16];
		uint32_t pf_n = steps < 8u ? steps : 8u;
		uint32_t pf_argmax = 0u;
		void *pf_buf = alloc_aligned64(2u * 1024u * 1024u);

		pf_toks[0] = start_tok;
		for (uint32_t i = 1u; i < pf_n; i++) {
			pf_toks[i] = g_tokens[i - 1u];
		}
		memset(k_cache, 0, kv_bytes);
		memset(v_cache, 0, kv_bytes);
		int rc = c7x_llama_prefill_tokens(hdr, pf_toks, pf_n, 0u, k_cache, v_cache,
						  NULL, &pf_argmax, scratch, 444u * 1024u,
						  pf_buf, 2u * 1024u * 1024u, NULL, 0u, NULL);
		int pf_ok = (rc == 0 && pf_argmax == g_tokens[pf_n - 1u]);
		printf("[host-llama] batched prefill (%u tok): rc=%d next=%u want=%u -> %s\n",
		       pf_n, rc, pf_argmax, g_tokens[pf_n - 1u], pf_ok ? "PASS" : "FAIL");
		if (!pf_ok) {
			fail = 1;
		}
		free(pf_buf);
	}

	free(model_buf);
	free(golden_buf);
	free(k_cache);
	free(v_cache);
	free(logits);
	free(dbg_x);
	free(scratch);
	return fail;
}
