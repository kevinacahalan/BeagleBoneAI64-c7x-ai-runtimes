/* SPDX-License-Identifier: MIT */
/*
 * Linux A72 host runner & verifier for the C7x + MMA transformer runtime.
 *
 * Subcommands:
 *   c7x-ai llama verify <model.c7m> [tokenizer.bin] <golden.bin> [steps]
 *   c7x-ai llama run    <model.c7m> [tokenizer.bin] [-p prompt] [-n steps] [-t temp] [-P topp] [-s seed]
 */
#define _GNU_SOURCE

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "c7x_ai.h"
#include "c7x_llama_fmt.h"

#define C7T_MAGIC 0x31543743u /* "C7T1" */

/* -------------------------------------------------------------------------- */
/* Safe O_SYNC /dev/mem helpers (avoid ARM64 DC ZVA SIGBUS from glibc memset) */
/* -------------------------------------------------------------------------- */

static void mmio_zero(void *dst, size_t n)
{
	volatile uint64_t *p64 = (volatile uint64_t *)dst;
	size_t i = 0;

	for (; i + 8u <= n; i += 8u) {
		p64[i / 8u] = 0ull;
	}
	for (; i < n; i++) {
		((volatile uint8_t *)dst)[i] = 0u;
	}
}

static void mmio_write(void *dst, const void *src, size_t n)
{
	volatile uint64_t *d64 = (volatile uint64_t *)dst;
	const uint64_t *s64 = (const uint64_t *)src;
	size_t i = 0;

	if ((((uintptr_t)dst | (uintptr_t)src) & 7u) == 0u) {
		for (; i + 8u <= n; i += 8u) {
			d64[i / 8u] = s64[i / 8u];
		}
	}
	for (; i < n; i++) {
		((volatile uint8_t *)dst)[i] = ((const uint8_t *)src)[i];
	}
}

static void mmio_read(void *dst, const void *src, size_t n)
{
	uint64_t *d64 = (uint64_t *)dst;
	const volatile uint64_t *s64 = (const volatile uint64_t *)src;
	size_t i = 0;

	if ((((uintptr_t)dst | (uintptr_t)src) & 7u) == 0u) {
		for (; i + 8u <= n; i += 8u) {
			d64[i / 8u] = s64[i / 8u];
		}
	}
	for (; i < n; i++) {
		((uint8_t *)dst)[i] = ((const volatile uint8_t *)src)[i];
	}
}

static inline size_t align64_sz(size_t n)
{
	return (n + 63u) & ~((size_t)63u);
}

static int c7x_exec_op(volatile struct c7x_ctrl *ctrl, uint32_t op,
		       uint64_t a_addr, uint64_t b_addr, uint64_t c_addr,
		       double *wall_us)
{
	uint32_t next_seq = ctrl->req_seq + 1u;
	double t0;
	const double timeout_us = 30000000.0; /* 30 s */

	ctrl->op     = op;
	ctrl->flags  = 0u;
	ctrl->iters  = 1u;
	ctrl->m      = 0u;
	ctrl->k      = 0u;
	ctrl->n      = 0u;
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
		fprintf(stderr, "C7x op=%u failed with status=%u (aux0=0x%" PRIx64 ", aux1=0x%" PRIx64 ")\n",
			op, ctrl->status, ctrl->aux0, ctrl->aux1);
		return -1;
	}
	return 0;
}

/* -------------------------------------------------------------------------- */
/* Universal tokenizer: supports both embedded C7T1 (HF BPE) and llama2.c bin */
/* -------------------------------------------------------------------------- */

struct c7t_special {
	uint32_t id;
	uint32_t str_off;
	uint32_t str_len;
};

struct c7t_merge {
	uint64_t key; /* ((uint64_t)left << 32) | right */
	uint32_t rank;
	uint32_t merged_id;
};

struct tok_index {
	char *str;
	int id;
};

struct tokenizer {
	int is_c7t;

	/* External llama2.c tokenizer.bin fields */
	char **vocab;
	float *vocab_scores;
	struct tok_index *sorted_vocab;
	int vocab_size;
	unsigned int max_token_length;
	unsigned char byte_pieces[512];

	/* Embedded C7T1 tokenizer fields (owned buffer) */
	uint8_t *c7t_buf;
	uint32_t c7t_vocab;
	uint32_t c7t_n_special;
	uint32_t c7t_n_merges;
	uint32_t c7t_flags;
	const int32_t *c7t_byte_to_id;
	const uint32_t *c7t_piece_offs;
	const struct c7t_special *c7t_specials;
	const struct c7t_merge *c7t_merges;
	const uint8_t *c7t_pieces;
	char decode_tmp[1024];
};

static int cmp_c7t_merge_key(const void *a, const void *b)
{
	uint64_t ka = ((const struct c7t_merge *)a)->key;
	uint64_t kb = ((const struct c7t_merge *)b)->key;

	if (ka < kb) return -1;
	if (ka > kb) return 1;
	return 0;
}

static int tokenizer_load_c7t(struct tokenizer *t, const uint8_t *blob, size_t bytes)
{
	const uint32_t *hdr;

	if (bytes < 32u) {
		return -1;
	}
	memset(t, 0, sizeof(*t));
	t->c7t_buf = malloc(bytes);
	if (!t->c7t_buf) {
		return -1;
	}
	memcpy(t->c7t_buf, blob, bytes);
	hdr = (const uint32_t *)t->c7t_buf;
	if (hdr[0] != C7T_MAGIC || hdr[1] != 1u) {
		free(t->c7t_buf);
		t->c7t_buf = NULL;
		return -1;
	}
	t->is_c7t         = 1;
	t->c7t_vocab      = hdr[2];
	t->vocab_size     = (int)hdr[2];
	t->c7t_n_special  = hdr[3];
	t->c7t_n_merges   = hdr[4];
	t->c7t_flags      = hdr[5];
	t->c7t_byte_to_id = (const int32_t *)(t->c7t_buf + 32u);
	t->c7t_piece_offs = (const uint32_t *)(t->c7t_buf + 32u + 256u * 4u);
	t->c7t_specials   = (const struct c7t_special *)(t->c7t_piece_offs + (t->c7t_vocab + 1u));
	t->c7t_merges     = (const struct c7t_merge *)(t->c7t_specials + t->c7t_n_special);
	t->c7t_pieces     = t->c7t_buf + hdr[6];
	return 0;
}

static int compare_tok(const void *a, const void *b)
{
	return strcmp(((const struct tok_index *)a)->str, ((const struct tok_index *)b)->str);
}

static int tokenizer_load(struct tokenizer *t, const char *path, int vocab_size)
{
	FILE *f = fopen(path, "rb");
	int i;

	if (!f) {
		perror(path);
		return -1;
	}
	memset(t, 0, sizeof(*t));
	t->vocab_size = vocab_size;
	t->vocab = calloc((size_t)vocab_size, sizeof(char *));
	t->vocab_scores = calloc((size_t)vocab_size, sizeof(float));
	if (!t->vocab || !t->vocab_scores) {
		fclose(f);
		return -1;
	}
	for (i = 0; i < 256; i++) {
		t->byte_pieces[i * 2] = (unsigned char)i;
		t->byte_pieces[i * 2 + 1] = '\0';
	}
	if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
		fclose(f);
		return -1;
	}
	for (i = 0; i < vocab_size; i++) {
		int len = 0;

		if (fread(&t->vocab_scores[i], sizeof(float), 1, f) != 1 ||
		    fread(&len, sizeof(int), 1, f) != 1 || len < 0 || len > 4096) {
			fclose(f);
			return -1;
		}
		t->vocab[i] = malloc((size_t)len + 1u);
		if (!t->vocab[i] || fread(t->vocab[i], 1, (size_t)len, f) != (size_t)len) {
			fclose(f);
			return -1;
		}
		t->vocab[i][len] = '\0';
	}
	fclose(f);
	return 0;
}

static void tokenizer_free(struct tokenizer *t)
{
	int i;

	if (t->vocab) {
		for (i = 0; i < t->vocab_size; i++) {
			free(t->vocab[i]);
		}
		free(t->vocab);
	}
	free(t->vocab_scores);
	free(t->sorted_vocab);
	free(t->c7t_buf);
	memset(t, 0, sizeof(*t));
}

static const char *tokenizer_decode(struct tokenizer *t, int prev_token, int token)
{
	if (t->is_c7t) {
		if (token < 0 || (uint32_t)token >= t->c7t_vocab) {
			return "";
		}
		uint32_t off = t->c7t_piece_offs[token];
		uint32_t len = t->c7t_piece_offs[token + 1] - off;
		if (len >= sizeof(t->decode_tmp)) {
			len = sizeof(t->decode_tmp) - 1u;
		}
		memcpy(t->decode_tmp, t->c7t_pieces + off, len);
		t->decode_tmp[len] = '\0';
		return t->decode_tmp;
	}

	const char *piece = t->vocab[token];
	unsigned char byte_val = 0;

	if (prev_token == 1 && piece[0] == ' ') {
		piece++;
	}
	if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
		piece = (const char *)&t->byte_pieces[byte_val * 2];
	}
	return piece;
}

static int str_lookup(const char *str, struct tok_index *sorted, int vocab_size)
{
	struct tok_index key = { (char *)str, 0 };
	struct tok_index *res = bsearch(&key, sorted, (size_t)vocab_size,
					sizeof(struct tok_index), compare_tok);

	return res ? res->id : -1;
}

static int c7t_bpe_chunk(struct tokenizer *t, const uint8_t *chunk, size_t len,
			 int *out, int max_out)
{
	int ids[1024];
	int n = 0;
	size_t i;

	if (len == 0u) {
		return 0;
	}
	for (i = 0u; i < len && n < 1024; i++) {
		int id = t->c7t_byte_to_id[chunk[i]];
		if (id >= 0) {
			ids[n++] = id;
		}
	}
	while (n >= 2) {
		uint32_t best_rank = 0xffffffffu;
		int best_pos = -1;
		int best_mid = -1;
		int j;

		for (j = 0; j < n - 1; j++) {
			struct c7t_merge key = { ((uint64_t)(uint32_t)ids[j] << 32) | (uint32_t)ids[j + 1], 0u, 0u };
			struct c7t_merge *res = bsearch(&key, t->c7t_merges, t->c7t_n_merges,
							sizeof(struct c7t_merge), cmp_c7t_merge_key);
			if (res && res->rank < best_rank) {
				best_rank = res->rank;
				best_mid = (int)res->merged_id;
				best_pos = j;
			}
		}
		if (best_pos < 0) {
			break;
		}
		ids[best_pos] = best_mid;
		for (j = best_pos + 1; j < n - 1; j++) {
			ids[j] = ids[j + 1];
		}
		n--;
	}
	int emit = (n < max_out) ? n : max_out;
	for (int j = 0; j < emit; j++) {
		out[j] = ids[j];
	}
	return emit;
}

static int tokenizer_encode(struct tokenizer *t, const char *text, int bos, int eos,
			    int *tokens, int max_tokens)
{
	if (t->is_c7t) {
		const uint8_t *raw = (const uint8_t *)(text ? text : "");
		size_t rlen = strlen((const char *)raw);
		size_t pos = 0u, chunk_start = 0u;
		int n_tokens = 0;

		while (pos < rlen && n_tokens < max_tokens) {
			int matched = 0;
			for (uint32_t s = 0u; s < t->c7t_n_special; s++) {
				uint32_t sl = t->c7t_specials[s].str_len;
				if (sl > 0u && pos + sl <= rlen &&
				    memcmp(raw + pos, t->c7t_pieces + t->c7t_specials[s].str_off, sl) == 0) {
					if (pos > chunk_start) {
						n_tokens += c7t_bpe_chunk(t, raw + chunk_start, pos - chunk_start,
									  tokens + n_tokens, max_tokens - n_tokens);
					}
					if (n_tokens < max_tokens) {
						tokens[n_tokens++] = (int)t->c7t_specials[s].id;
					}
					pos += sl;
					chunk_start = pos;
					matched = 1;
					break;
				}
			}
			if (!matched) {
				pos++;
			}
		}
		if (pos > chunk_start && n_tokens < max_tokens) {
			n_tokens += c7t_bpe_chunk(t, raw + chunk_start, pos - chunk_start,
						  tokens + n_tokens, max_tokens - n_tokens);
		}
		return n_tokens;
	}

	char *str_buf;
	size_t str_len = 0;
	int n_tokens = 0;
	int i;

	if (!t->sorted_vocab) {
		t->sorted_vocab = malloc((size_t)t->vocab_size * sizeof(struct tok_index));
		for (i = 0; i < t->vocab_size; i++) {
			t->sorted_vocab[i].str = t->vocab[i];
			t->sorted_vocab[i].id = i;
		}
		qsort(t->sorted_vocab, (size_t)t->vocab_size,
		      sizeof(struct tok_index), compare_tok);
	}

	str_buf = malloc((size_t)t->max_token_length * 2u + 4u);
	if (bos && n_tokens < max_tokens) {
		tokens[n_tokens++] = 1;
	}
	if (text && text[0] != '\0') {
		int dummy = str_lookup(" ", t->sorted_vocab, t->vocab_size);

		if (dummy >= 0 && n_tokens < max_tokens) {
			tokens[n_tokens++] = dummy;
		}
	}

	for (const char *c = text; c && *c != '\0'; c++) {
		if ((*c & 0xC0) != 0x80) {
			str_len = 0;
		}
		str_buf[str_len++] = *c;
		str_buf[str_len] = '\0';
		if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
			continue;
		}
		int id = str_lookup(str_buf, t->sorted_vocab, t->vocab_size);

		if (id != -1) {
			if (n_tokens < max_tokens) {
				tokens[n_tokens++] = id;
			}
		} else {
			for (size_t b = 0; b < str_len; b++) {
				if (n_tokens < max_tokens) {
					tokens[n_tokens++] = (unsigned char)str_buf[b] + 3;
				}
			}
		}
		str_len = 0;
	}

	while (1) {
		float best_score = -1e10f;
		int best_id = -1;
		int best_idx = -1;

		for (i = 0; i < n_tokens - 1; i++) {
			snprintf(str_buf, (size_t)t->max_token_length * 2u + 4u,
				 "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
			int id = str_lookup(str_buf, t->sorted_vocab, t->vocab_size);

			if (id != -1 && t->vocab_scores[id] > best_score) {
				best_score = t->vocab_scores[id];
				best_id = id;
				best_idx = i;
			}
		}
		if (best_idx == -1) {
			break;
		}
		tokens[best_idx] = best_id;
		for (i = best_idx + 1; i < n_tokens - 1; i++) {
			tokens[i] = tokens[i + 1];
		}
		n_tokens--;
	}

	if (eos && n_tokens < max_tokens) {
		tokens[n_tokens++] = 2;
	}
	free(str_buf);
	return n_tokens;
}

/* -------------------------------------------------------------------------- */
/* DDR arena layout & model loading                                           */
/* -------------------------------------------------------------------------- */

struct llama_arena {
	volatile struct c7x_ctrl *ctrl;
	uint8_t *ddr;
	uint8_t *msmc;
	uint64_t arena_base;
	size_t arena_size;
	struct c7m_header hdr;

	size_t off_model;
	size_t off_k_cache;
	size_t off_v_cache;
	size_t off_logits;
	size_t off_debug_x;
	size_t off_prof;
	size_t off_req;
	size_t off_tokens;
	size_t off_prefill_buf;
	size_t prefill_buf_bytes;
	size_t kv_bytes;
	size_t total_used;
	uint32_t msmc_n_layers;

	uint8_t *embedded_tok;
	size_t embedded_tok_size;
};

static int arena_load_model(struct llama_arena *a, const char *model_path)
{
	FILE *f;
	long fsz;
	uint8_t *host_buf;
	size_t off = 0;

	memset(a, 0, sizeof(*a));

	/* Prefer 512 MiB LLM arena (0xC0000000) when reserved in DT, else 64 MiB (0xAC000000). */
	if (access("/proc/device-tree/reserved-memory/c7x-llm-arena@c0000000", F_OK) == 0) {
		a->arena_base = C7X_LLM_ARENA_BASE;
		a->arena_size = C7X_LLM_ARENA_SIZE;
	} else {
		a->arena_base = C7X_DDR_ARENA_BASE;
		a->arena_size = C7X_DDR_ARENA_SIZE;
	}

	a->ctrl = (volatile struct c7x_ctrl *)map_phys((off_t)C7X_CTRL_ADDR, C7X_CTRL_SIZE);
	a->ddr  = (uint8_t *)map_phys((off_t)a->arena_base, a->arena_size);
	a->msmc = (uint8_t *)map_phys((off_t)C7X_MSMC_BASE, C7X_MSMC_SIZE);
	if (!a->ctrl || !a->ddr) {
		fprintf(stderr, "failed to map C7x control block or DDR arena 0x%08" PRIx64 "\n",
			a->arena_base);
		return -1;
	}
	if (a->ctrl->magic != C7X_CTRL_MAGIC) {
		fprintf(stderr, "C7x firmware not running (magic=0x%08x). Run 'c7x-ai load' first.\n",
			a->ctrl->magic);
		return -1;
	}

	f = fopen(model_path, "rb");
	if (!f) {
		perror(model_path);
		return -1;
	}
	fseek(f, 0, SEEK_END);
	fsz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (fsz < (long)sizeof(struct c7m_header) || (size_t)fsz > a->arena_size) {
		fprintf(stderr, "model size %.2f MiB exceeds %.0f MiB DDR arena\n",
			(double)fsz / (1024.0 * 1024.0), (double)a->arena_size / (1024.0 * 1024.0));
		fclose(f);
		return -1;
	}

	host_buf = malloc((size_t)fsz);
	if (!host_buf || fread(host_buf, 1, (size_t)fsz, f) != (size_t)fsz) {
		fprintf(stderr, "failed reading %s\n", model_path);
		free(host_buf);
		fclose(f);
		return -1;
	}
	fclose(f);

	memcpy(&a->hdr, host_buf, sizeof(a->hdr));
	if (a->hdr.magic != C7M_MAGIC || a->hdr.version != C7M_VERSION) {
		fprintf(stderr, "invalid .c7m header magic=0x%08x ver=%u\n",
			a->hdr.magic, a->hdr.version);
		free(host_buf);
		return -1;
	}

	a->off_model = off;
	off += align64_sz((size_t)fsz);

	a->kv_bytes = (size_t)a->hdr.n_layers * a->hdr.seq_len * a->hdr.kv_dim * sizeof(float);
	a->off_k_cache = off;
	off += align64_sz(a->kv_bytes);

	a->off_v_cache = off;
	off += align64_sz(a->kv_bytes);

	a->off_logits = off;
	off += align64_sz((size_t)a->hdr.vocab_size * sizeof(float));

	a->off_debug_x = off;
	off += align64_sz((size_t)a->hdr.dim * sizeof(float));

	a->off_prof = off;
	off += align64_sz(sizeof(struct c7x_llama_prof));

	a->off_req = off;
	off += align64_sz(sizeof(struct c7x_llama_req));

	a->off_tokens = off;
	off += align64_sz(4096u); /* up to 1024 prompt token IDs */

	a->off_prefill_buf = off;
	a->prefill_buf_bytes = 8u * 1024u * 1024u; /* up to 8 MiB batched prefill workspace */
	if (off + a->prefill_buf_bytes > a->arena_size) {
		a->prefill_buf_bytes = (a->arena_size > off)
				       ? ((a->arena_size - off) & ~((size_t)63u))
				       : 0u;
	}
	off += a->prefill_buf_bytes;
	a->total_used = off;

	if (a->total_used > a->arena_size) {
		fprintf(stderr, "model + KV cache (%.2f MiB) exceeds %.0f MiB DDR arena\n",
			(double)a->total_used / (1024.0 * 1024.0),
			(double)a->arena_size / (1024.0 * 1024.0));
		free(host_buf);
		return -1;
	}

	/* Keep host copy of embedded C7T1 tokenizer blob if present */
	if (a->hdr.off_tokenizer != 0u && a->hdr.tokenizer_size > 0u &&
	    (size_t)a->hdr.off_tokenizer + a->hdr.tokenizer_size <= (size_t)fsz) {
		a->embedded_tok = malloc(a->hdr.tokenizer_size);
		if (a->embedded_tok) {
			memcpy(a->embedded_tok, host_buf + a->hdr.off_tokenizer, a->hdr.tokenizer_size);
			a->embedded_tok_size = a->hdr.tokenizer_size;
		}
	}

	double t0 = now_us();
	mmio_write(a->ddr + a->off_model, host_buf, (size_t)fsz);
	mmio_zero(a->ddr + a->off_k_cache, a->kv_bytes);
	mmio_zero(a->ddr + a->off_v_cache, a->kv_bytes);

	/* Stage as many leading decoder layers as fit into on-chip 4 MiB MSMC SRAM. */
	if (a->msmc && a->hdr.off_layers_base != 0u && a->hdr.layer_stride != 0u) {
		uint32_t fit = C7X_MSMC_SIZE / a->hdr.layer_stride;
		if (fit > a->hdr.n_layers) {
			fit = a->hdr.n_layers;
		}
		if (fit > 0u) {
			size_t copy_bytes = (size_t)fit * a->hdr.layer_stride;
			if (a->hdr.off_layers_base + copy_bytes <= (size_t)fsz) {
				mmio_write(a->msmc, host_buf + a->hdr.off_layers_base, copy_bytes);
				a->msmc_n_layers = fit;
			}
		}
	}
	free(host_buf);
	{
		const char *mode_str = "w8a8-row";
		if ((a->hdr.flags & C7M_FLAG_W4A8_LAYERS) && (a->hdr.flags & C7M_FLAG_W4A8_WCLS)) {
			mode_str = "w4a8-g64";
		} else if (a->hdr.flags & C7M_FLAG_W4A8_WCLS) {
			mode_str = "hybrid-w8/w4cls";
		} else if (a->hdr.flags & C7M_FLAG_TILED_WEIGHTS) {
			mode_str = "w8a8-tiled";
		}
		printf("[llama] loaded %s (%.2f MiB, %s) into DDR 0x%08" PRIx64 " (used %.2f/%.0f MiB), staged %u/%u layers (%.2f MiB) in MSMC SRAM in %.1f ms\n",
		       model_path, (double)fsz / (1024.0 * 1024.0), mode_str,
		       a->arena_base,
		       (double)a->total_used / (1024.0 * 1024.0),
		       (double)a->arena_size / (1024.0 * 1024.0),
		       a->msmc_n_layers, a->hdr.n_layers,
		       (double)a->msmc_n_layers * a->hdr.layer_stride / (1024.0 * 1024.0),
		       (now_us() - t0) / 1000.0);
	}
	printf("[llama] config: dim=%u (pad %u), hidden=%u (pad %u), layers=%u, heads=%u/%u, vocab=%u, seq=%u\n",
	       a->hdr.dim, a->hdr.dim_pad, a->hdr.hidden_dim, a->hdr.hidden_pad,
	       a->hdr.n_layers, a->hdr.n_heads, a->hdr.n_kv_heads,
	       a->hdr.vocab_size, a->hdr.seq_len);
	return 0;
}

static void arena_close(struct llama_arena *a)
{
	if (a->ctrl) {
		unmap_phys((void *)a->ctrl, C7X_CTRL_SIZE);
	}
	if (a->ddr) {
		unmap_phys(a->ddr, a->arena_size);
	}
	if (a->msmc) {
		unmap_phys(a->msmc, C7X_MSMC_SIZE);
	}
	free(a->embedded_tok);
	memset(a, 0, sizeof(*a));
}

static int arena_forward_token(struct llama_arena *a, uint32_t token, uint32_t pos,
			       float *logits_host, uint32_t *argmax_out,
			       float *debug_x_host,
			       int32_t debug_layer, uint32_t stop_after_layer,
			       struct c7x_llama_prof *prof_host, double *wall_us)
{
	struct c7x_llama_req req;

	memset(&req, 0, sizeof(req));
	req.model_addr       = a->arena_base + a->off_model;
	req.k_cache_addr     = a->arena_base + a->off_k_cache;
	req.v_cache_addr     = a->arena_base + a->off_v_cache;
	req.logits_addr      = logits_host ? (a->arena_base + a->off_logits) : 0ull;
	req.prof_addr        = prof_host ? (a->arena_base + a->off_prof) : 0ull;
	req.debug_x_addr     = debug_x_host ? (a->arena_base + a->off_debug_x) : 0ull;
	req.msmc_layers_addr = a->msmc_n_layers ? C7X_MSMC_BASE : 0ull;
	req.msmc_n_layers    = a->msmc_n_layers;
	req.token            = token;
	req.pos              = pos;
	req.debug_layer      = debug_layer;
	req.stop_after_layer = stop_after_layer;

	mmio_write(a->ddr + a->off_req, &req, sizeof(req));

	if (c7x_exec_op(a->ctrl, C7X_OP_LLAMA_FORWARD,
			a->arena_base + a->off_req, 0ull, 0ull, wall_us) != 0) {
		return -1;
	}

	if (argmax_out) {
		*argmax_out = (uint32_t)a->ctrl->aux1;
	}
	if (logits_host && stop_after_layer == 0u) {
		mmio_read(logits_host, a->ddr + a->off_logits,
			  (size_t)a->hdr.vocab_size * sizeof(float));
	}
	if (debug_x_host) {
		mmio_read(debug_x_host, a->ddr + a->off_debug_x,
			  (size_t)a->hdr.dim * sizeof(float));
	}
	if (prof_host) {
		mmio_read(prof_host, a->ddr + a->off_prof, sizeof(*prof_host));
	}
	return 0;
}

static int arena_prefill_tokens(struct llama_arena *a, const uint32_t *tokens,
				uint32_t n_tokens, uint32_t start_pos,
				float *logits_host, uint32_t *argmax_out,
				struct c7x_llama_prof *prof_host, double *wall_us)
{
	struct c7x_llama_req req;

	if (n_tokens == 0u || n_tokens > 1024u) {
		return -1;
	}
	mmio_write(a->ddr + a->off_tokens, tokens, (size_t)n_tokens * sizeof(uint32_t));

	memset(&req, 0, sizeof(req));
	req.model_addr        = a->arena_base + a->off_model;
	req.k_cache_addr      = a->arena_base + a->off_k_cache;
	req.v_cache_addr      = a->arena_base + a->off_v_cache;
	req.logits_addr       = logits_host ? (a->arena_base + a->off_logits) : 0ull;
	req.prof_addr         = prof_host ? (a->arena_base + a->off_prof) : 0ull;
	req.msmc_layers_addr  = a->msmc_n_layers ? C7X_MSMC_BASE : 0ull;
	req.msmc_n_layers     = a->msmc_n_layers;
	req.pos               = start_pos;
	req.n_tokens          = n_tokens;
	req.tokens_addr       = a->arena_base + a->off_tokens;
	req.prefill_buf_addr  = a->arena_base + a->off_prefill_buf;
	req.prefill_buf_bytes = (uint32_t)a->prefill_buf_bytes;

	mmio_write(a->ddr + a->off_req, &req, sizeof(req));

	if (c7x_exec_op(a->ctrl, C7X_OP_LLAMA_PREFILL,
			a->arena_base + a->off_req, 0ull, 0ull, wall_us) != 0) {
		return -1;
	}
	if (argmax_out) {
		*argmax_out = (uint32_t)a->ctrl->aux1;
	}
	if (logits_host) {
		mmio_read(logits_host, a->ddr + a->off_logits,
			  (size_t)a->hdr.vocab_size * sizeof(float));
	}
	if (prof_host) {
		mmio_read(prof_host, a->ddr + a->off_prof, sizeof(*prof_host));
	}
	return 0;
}

static uint32_t argmax_f32(const float *v, uint32_t n)
{
	uint32_t best = 0u, i;
	float best_v = v[0];

	for (i = 1u; i < n; i++) {
		if (v[i] > best_v) {
			best_v = v[i];
			best = i;
		}
	}
	return best;
}

struct prob_index {
	float prob;
	uint32_t idx;
};

static int cmp_prob_desc(const void *a, const void *b)
{
	float pa = ((const struct prob_index *)a)->prob;
	float pb = ((const struct prob_index *)b)->prob;

	if (pa > pb) return -1;
	if (pa < pb) return 1;
	return 0;
}

static uint64_t g_sample_rng = 0x123456789abcdef0ull;

static float rand_f32_unit(void)
{
	g_sample_rng ^= g_sample_rng >> 12;
	g_sample_rng ^= g_sample_rng << 25;
	g_sample_rng ^= g_sample_rng >> 27;
	return (float)((g_sample_rng * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

static uint32_t sample_topp(float *logits, uint32_t n, float temp, float topp)
{
	float max_l = logits[0], sum = 0.0f;
	uint32_t i;

	if (temp <= 0.0f) {
		return argmax_f32(logits, n);
	}
	for (i = 1u; i < n; i++) {
		if (logits[i] > max_l) max_l = logits[i];
	}
	for (i = 0u; i < n; i++) {
		logits[i] = expf((logits[i] - max_l) / temp);
		sum += logits[i];
	}
	float inv = 1.0f / sum;
	for (i = 0u; i < n; i++) {
		logits[i] *= inv;
	}

	if (topp <= 0.0f || topp >= 1.0f) {
		float r = rand_f32_unit(), cdf = 0.0f;
		for (i = 0u; i < n; i++) {
			cdf += logits[i];
			if (r < cdf) return i;
		}
		return n - 1u;
	}

	struct prob_index *pi = malloc(n * sizeof(*pi));
	uint32_t n0 = 0u;
	float cutoff = (1.0f - topp) / (float)(n - 1u);

	for (i = 0u; i < n; i++) {
		if (logits[i] >= cutoff) {
			pi[n0].idx = i;
			pi[n0].prob = logits[i];
			n0++;
		}
	}
	qsort(pi, n0, sizeof(*pi), cmp_prob_desc);

	float cum = 0.0f;
	uint32_t last = n0 - 1u;
	for (i = 0u; i < n0; i++) {
		cum += pi[i].prob;
		if (cum > topp) {
			last = i;
			break;
		}
	}
	float r = rand_f32_unit() * cum, cdf = 0.0f;
	uint32_t chosen = pi[last].idx;
	for (i = 0u; i <= last; i++) {
		cdf += pi[i].prob;
		if (r < cdf) {
			chosen = pi[i].idx;
			break;
		}
	}
	free(pi);
	return chosen;
}

/* -------------------------------------------------------------------------- */
/* Independent Cortex-A72 pure-C W8A8 CPU reference (for head-to-head perf)   */
/* -------------------------------------------------------------------------- */

static void cpu_rmsnorm(float *out, const float *x, const float *w, uint32_t n)
{
	float ss = 0.0f;
	uint32_t i;

	for (i = 0u; i < n; i++) {
		ss += x[i] * x[i];
	}
	float inv = 1.0f / sqrtf(ss / (float)n + 1e-5f);
	for (i = 0u; i < n; i++) {
		out[i] = (x[i] * inv) * w[i];
	}
}

static void cpu_quantize(int8_t *xq, const float *x, uint32_t n, uint32_t n_pad, float *scale_out)
{
	float max_abs = 0.0f;
	uint32_t i;

	for (i = 0u; i < n; i++) {
		float a = fabsf(x[i]);
		if (a > max_abs) max_abs = a;
	}
	float s = max_abs / 127.0f;
	*scale_out = s;
	if (s <= 0.0f) {
		memset(xq, 0, n_pad);
		return;
	}
	float inv = 1.0f / s;
	for (i = 0u; i < n; i++) {
		float v = x[i] * inv;
		if (v > 127.0f) v = 127.0f;
		if (v < -127.0f) v = -127.0f;
		xq[i] = (int8_t)(int32_t)rintf(v);
	}
	for (; i < n_pad; i++) {
		xq[i] = 0;
	}
}

static void cpu_linear_w8a8(float *out, const int8_t *xq, float xs,
			    const int8_t *wq, const float *ws,
			    uint32_t k_pad, uint32_t out_dim, int add, int tiled)
{
	uint32_t j, p;

	if (!tiled) {
		for (j = 0u; j < out_dim; j++) {
			const int8_t *row = wq + (size_t)j * k_pad;
			int32_t acc = 0;

			for (p = 0u; p < k_pad; p++) {
				acc += (int32_t)row[p] * (int32_t)xq[p];
			}
			float val = (float)acc * (ws[j] * xs);
			out[j] = add ? (out[j] + val) : val;
		}
	} else {
		uint32_t k_tiles = k_pad >> 6;

		for (j = 0u; j < out_dim; j++) {
			uint32_t nt = j >> 6;
			uint32_t r = j & 63u;
			int32_t acc = 0;

			for (uint32_t kt = 0u; kt < k_tiles; kt++) {
				const int8_t *seg =
					wq + ((size_t)nt * k_tiles + kt) * 4096u + r * 64u;
				const int8_t *xseg = xq + (kt << 6);

				for (p = 0u; p < 64u; p++) {
					acc += (int32_t)seg[p] * (int32_t)xseg[p];
				}
			}
			float val = (float)acc * (ws[j] * xs);
			out[j] = add ? (out[j] + val) : val;
		}
	}
}

static double run_a72_cpu_benchmark(const char *model_path, uint32_t start_tok,
				    uint32_t steps, uint32_t *out_tokens)
{
	FILE *f = fopen(model_path, "rb");
	if (!f) return 0.0;
	fseek(f, 0, SEEK_END);
	long fsz = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *buf = malloc((size_t)fsz);
	if (!buf || fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
		free(buf);
		fclose(f);
		return 0.0;
	}
	fclose(f);

	const struct c7m_header *h = (const struct c7m_header *)buf;
	const struct c7m_layer_offs *layers = (const struct c7m_layer_offs *)(buf + h->off_layer_table);
	const float *rope_cos = (const float *)(buf + h->off_rope_cos);
	const float *rope_sin = (const float *)(buf + h->off_rope_sin);
	const float *rms_final = (const float *)(buf + h->off_rms_final);
	const int8_t *wcls_q = (const int8_t *)(buf + h->off_wcls_q);
	const float *wcls_s = (const float *)(buf + h->off_wcls_s);

	size_t kv_n = (size_t)h->n_layers * h->seq_len * h->kv_dim;
	float *kc = calloc(kv_n, sizeof(float));
	float *vc = calloc(kv_n, sizeof(float));
	float *x = malloc(h->dim_pad * sizeof(float));
	float *xb = malloc(h->dim_pad * sizeof(float));
	float *q = malloc(h->dim_pad * sizeof(float));
	float *k = malloc(h->dim_pad * sizeof(float));
	float *v = malloc(h->dim_pad * sizeof(float));
	float *hb = malloc(h->hidden_pad * sizeof(float));
	float *hb2 = malloc(h->hidden_pad * sizeof(float));
	float *att = malloc(h->seq_len * sizeof(float));
	float *logits = malloc(h->vocab_size * sizeof(float));
	int8_t *xq = malloc((h->hidden_pad > h->dim_pad ? h->hidden_pad : h->dim_pad));

	double t0 = now_us();
	uint32_t tok = start_tok;
	float inv_hd = 1.0f / sqrtf((float)h->head_size);
	uint32_t kv_mul = h->n_heads / h->n_kv_heads;
	int tiled = (h->flags & C7M_FLAG_TILED_WEIGHTS) ? 1 : 0;

	for (uint32_t pos = 0u; pos < steps; pos++) {
		const float *cr = rope_cos + pos * (h->head_size >> 1);
		const float *sr = rope_sin + pos * (h->head_size >> 1);
		float xs = 0.0f;

		if (!tiled) {
			for (uint32_t i = 0u; i < h->dim; i++) {
				x[i] = (float)wcls_q[(size_t)tok * h->dim_pad + i] * wcls_s[tok];
			}
		} else {
			uint32_t k_tiles = h->dim_pad >> 6;
			uint32_t nt = tok >> 6;
			uint32_t r = tok & 63u;
			for (uint32_t i = 0u; i < h->dim; i++) {
				uint32_t kt = i >> 6;
				uint32_t c = i & 63u;
				int8_t qv = wcls_q[((size_t)nt * k_tiles + kt) * 4096u + r * 64u + c];
				x[i] = (float)qv * wcls_s[tok];
			}
		}
		for (uint32_t l = 0u; l < h->n_layers; l++) {
			const struct c7m_layer_offs *lo = &layers[l];
			float *kl = kc + (size_t)l * h->seq_len * h->kv_dim;
			float *vl = vc + (size_t)l * h->seq_len * h->kv_dim;

			cpu_rmsnorm(xb, x, (const float *)(buf + lo->rms_att), h->dim);
			cpu_quantize(xq, xb, h->dim, h->dim_pad, &xs);
			cpu_linear_w8a8(q, xq, xs, (const int8_t *)(buf + lo->wq_q),
					(const float *)(buf + lo->wq_s), h->dim_pad, h->dim, 0, tiled);
			cpu_linear_w8a8(k, xq, xs, (const int8_t *)(buf + lo->wk_q),
					(const float *)(buf + lo->wk_s), h->dim_pad, h->kv_dim, 0, tiled);
			cpu_linear_w8a8(v, xq, xs, (const int8_t *)(buf + lo->wv_q),
					(const float *)(buf + lo->wv_s), h->dim_pad, h->kv_dim, 0, tiled);

			for (uint32_t hd = 0u; hd < h->n_heads; hd++) {
				for (uint32_t i = 0u; i < (h->head_size >> 1); i++) {
					float v0 = q[hd * h->head_size + 2 * i];
					float v1 = q[hd * h->head_size + 2 * i + 1];
					q[hd * h->head_size + 2 * i]     = v0 * cr[i] - v1 * sr[i];
					q[hd * h->head_size + 2 * i + 1] = v0 * sr[i] + v1 * cr[i];
				}
			}
			for (uint32_t hd = 0u; hd < h->n_kv_heads; hd++) {
				for (uint32_t i = 0u; i < (h->head_size >> 1); i++) {
					float v0 = k[hd * h->head_size + 2 * i];
					float v1 = k[hd * h->head_size + 2 * i + 1];
					k[hd * h->head_size + 2 * i]     = v0 * cr[i] - v1 * sr[i];
					k[hd * h->head_size + 2 * i + 1] = v0 * sr[i] + v1 * cr[i];
				}
			}
			memcpy(kl + (size_t)pos * h->kv_dim, k, h->kv_dim * sizeof(float));
			memcpy(vl + (size_t)pos * h->kv_dim, v, h->kv_dim * sizeof(float));

			for (uint32_t hd = 0u; hd < h->n_heads; hd++) {
				const float *qh = q + hd * h->head_size;
				float *oh = xb + hd * h->head_size;
				uint32_t kvh = (hd / kv_mul) * h->head_size;
				float mx = -1e30f, sum = 0.0f;

				for (uint32_t t = 0u; t <= pos; t++) {
					const float *kt = kl + (size_t)t * h->kv_dim + kvh;
					float d = 0.0f;
					for (uint32_t i = 0u; i < h->head_size; i++) d += qh[i] * kt[i];
					att[t] = d * inv_hd;
					if (att[t] > mx) mx = att[t];
				}
				for (uint32_t t = 0u; t <= pos; t++) {
					att[t] = expf(att[t] - mx);
					sum += att[t];
				}
				memset(oh, 0, h->head_size * sizeof(float));
				for (uint32_t t = 0u; t <= pos; t++) {
					const float *vt = vl + (size_t)t * h->kv_dim + kvh;
					float a = att[t] / sum;
					for (uint32_t i = 0u; i < h->head_size; i++) oh[i] += a * vt[i];
				}
			}
			cpu_quantize(xq, xb, h->dim, h->dim_pad, &xs);
			cpu_linear_w8a8(x, xq, xs, (const int8_t *)(buf + lo->wo_q),
					(const float *)(buf + lo->wo_s), h->dim_pad, h->dim, 1, tiled);

			cpu_rmsnorm(xb, x, (const float *)(buf + lo->rms_ffn), h->dim);
			cpu_quantize(xq, xb, h->dim, h->dim_pad, &xs);
			cpu_linear_w8a8(hb, xq, xs, (const int8_t *)(buf + lo->w1_q),
					(const float *)(buf + lo->w1_s), h->dim_pad, h->hidden_dim, 0, tiled);
			cpu_linear_w8a8(hb2, xq, xs, (const int8_t *)(buf + lo->w3_q),
					(const float *)(buf + lo->w3_s), h->dim_pad, h->hidden_dim, 0, tiled);
			for (uint32_t i = 0u; i < h->hidden_dim; i++) {
				hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i];
			}
			cpu_quantize(xq, hb, h->hidden_dim, h->hidden_pad, &xs);
			cpu_linear_w8a8(x, xq, xs, (const int8_t *)(buf + lo->w2_q),
					(const float *)(buf + lo->w2_s), h->hidden_pad, h->dim, 1, tiled);
		}
		cpu_rmsnorm(xb, x, rms_final, h->dim);
		cpu_quantize(xq, xb, h->dim, h->dim_pad, &xs);
		cpu_linear_w8a8(logits, xq, xs, wcls_q, wcls_s, h->dim_pad, h->vocab_size, 0, tiled);
		tok = argmax_f32(logits, h->vocab_size);
		if (out_tokens) {
			out_tokens[pos] = tok;
		}
	}
	double elapsed_s = (now_us() - t0) / 1e6;

	free(buf); free(kc); free(vc); free(x); free(xb);
	free(q); free(k); free(v); free(hb); free(hb2);
	free(att); free(logits); free(xq);
	return (double)steps / elapsed_s;
}

static void print_profile(const struct c7x_llama_prof *acc, uint32_t n_layers, uint32_t n_tokens)
{
	double inv_tok = 1.0 / (double)n_tokens;
	double tot = (double)acc->cyc_total * inv_tok;
	double ms_per_cyc = 1000.0 / (double)C7X_CLOCK_HZ;
	uint32_t l;

	printf("\n=== C7x + MMA Per-Token Profile (avg over %u tokens, 1.0 GHz C7x) ===\n", n_tokens);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "TOTAL (C7x)",
	       tot, tot * ms_per_cyc, 100.0);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  Embedding",
	       acc->cyc_emb * inv_tok, acc->cyc_emb * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_emb * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  RMSNorm",
	       acc->cyc_rmsnorm * inv_tok, acc->cyc_rmsnorm * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_rmsnorm * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  Quant f32->i8",
	       acc->cyc_quant * inv_tok, acc->cyc_quant * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_quant * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  QKV MMA",
	       acc->cyc_qkv_mma * inv_tok, acc->cyc_qkv_mma * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_qkv_mma * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  RoPE + KV write",
	       acc->cyc_rope * inv_tok, acc->cyc_rope * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_rope * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  MHA Attention",
	       acc->cyc_attn * inv_tok, acc->cyc_attn * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_attn * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  Wo MMA",
	       acc->cyc_wo_mma * inv_tok, acc->cyc_wo_mma * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_wo_mma * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  FFN W1+W3 MMA",
	       acc->cyc_ffn_up_mma * inv_tok, acc->cyc_ffn_up_mma * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_ffn_up_mma * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  SwiGLU",
	       acc->cyc_swiglu * inv_tok, acc->cyc_swiglu * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_swiglu * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  FFN W2 MMA",
	       acc->cyc_ffn_down_mma * inv_tok, acc->cyc_ffn_down_mma * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_ffn_down_mma * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  Classifier MMA",
	       acc->cyc_cls_mma * inv_tok, acc->cyc_cls_mma * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_cls_mma * inv_tok / tot);
	printf("  %-18s %10.0f cyc  %7.3f ms  (%5.1f%%)\n", "  Dequant i32->f32",
	       acc->cyc_dequant * inv_tok, acc->cyc_dequant * inv_tok * ms_per_cyc,
	       100.0 * acc->cyc_dequant * inv_tok / tot);

	printf("\n  Per-layer avg cycles:\n");
	for (l = 0u; l < n_layers; l++) {
		double lc = (double)acc->cyc_layer[l] * inv_tok;
		printf("    layer %u: %10.0f cyc  (%.3f ms)\n", l, lc, lc * ms_per_cyc);
	}
}

static void prof_add(struct c7x_llama_prof *acc, const struct c7x_llama_prof *p, uint32_t n_layers)
{
	uint32_t l;

	acc->cyc_total        += p->cyc_total;
	acc->cyc_emb          += p->cyc_emb;
	acc->cyc_rmsnorm      += p->cyc_rmsnorm;
	acc->cyc_quant        += p->cyc_quant;
	acc->cyc_qkv_mma      += p->cyc_qkv_mma;
	acc->cyc_rope         += p->cyc_rope;
	acc->cyc_attn         += p->cyc_attn;
	acc->cyc_wo_mma       += p->cyc_wo_mma;
	acc->cyc_ffn_up_mma   += p->cyc_ffn_up_mma;
	acc->cyc_swiglu       += p->cyc_swiglu;
	acc->cyc_ffn_down_mma += p->cyc_ffn_down_mma;
	acc->cyc_cls_mma      += p->cyc_cls_mma;
	acc->cyc_dequant      += p->cyc_dequant;
	for (l = 0u; l < n_layers; l++) {
		acc->cyc_layer[l] += p->cyc_layer[l];
	}
}

static void compare_vec(const char *label, const float *got, const float *want,
			uint32_t n, float *max_err_out, double *cos_out)
{
	float max_err = 0.0f;
	double dot = 0.0, ng = 0.0, nw = 0.0;
	uint32_t i;

	for (i = 0u; i < n; i++) {
		float e = fabsf(got[i] - want[i]);
		if (e > max_err) {
			max_err = e;
		}
		dot += (double)got[i] * (double)want[i];
		ng += (double)got[i] * (double)got[i];
		nw += (double)want[i] * (double)want[i];
	}
	double cos_sim = dot / (sqrt(ng) * sqrt(nw) + 1e-12);
	printf("[verify] %-14s max_abs_err=%.6g  cosine=%.7f\n", label, max_err, cos_sim);
	if (max_err_out) *max_err_out = max_err;
	if (cos_out) *cos_out = cos_sim;
}

/* -------------------------------------------------------------------------- */
/* Subcommand: c7x-ai llama verify                                            */
/* -------------------------------------------------------------------------- */

static int cmd_llama_verify(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr, "usage: c7x-ai llama verify <model.c7m> <tokenizer.bin> <golden.bin> [steps]\n");
		return 2;
	}
	const char *model_path = argv[3];
	const char *tok_path   = argv[4];
	const char *gold_path  = argv[5];
	uint32_t steps = argc > 6 ? (uint32_t)atoi(argv[6]) : 40u;

	struct llama_arena a;
	struct tokenizer tok;
	FILE *gf;
	long gsz;
	uint8_t *gbuf;
	int fail = 0;

	if (arena_load_model(&a, model_path) != 0) {
		return 1;
	}

	/* Step 1: Run C7x silicon FP32 vector kernel self-test */
	{
		double us = 0.0;
		float max_err = 0.0f;
		uint32_t err_bits;

		if (c7x_exec_op(a.ctrl, C7X_OP_VEC_SELFTEST, 0, 0, 0, &us) != 0) {
			fprintf(stderr, "[verify] C7x silicon vec_selftest FAILED\n");
			arena_close(&a);
			return 1;
		}
		err_bits = (uint32_t)a.ctrl->aux0;
		memcpy(&max_err, &err_bits, sizeof(max_err));
		printf("[verify] silicon vec_selftest: PASS (max_err=%.6g, %" PRIu64 " cyc, %.1f us)\n",
		       max_err, a.ctrl->cycles, us);
	}

	if (a.embedded_tok) {
		if (tokenizer_load_c7t(&tok, a.embedded_tok, a.embedded_tok_size) != 0) {
			fprintf(stderr, "failed loading embedded C7T1 tokenizer\n");
			arena_close(&a);
			return 1;
		}
	} else if (tokenizer_load(&tok, tok_path, (int)a.hdr.vocab_size) != 0) {
		arena_close(&a);
		return 1;
	}

	gf = fopen(gold_path, "rb");
	if (!gf) {
		perror(gold_path);
		tokenizer_free(&tok);
		arena_close(&a);
		return 1;
	}
	fseek(gf, 0, SEEK_END);
	gsz = ftell(gf);
	fseek(gf, 0, SEEK_SET);
	gbuf = malloc((size_t)gsz);
	if (!gbuf || fread(gbuf, 1, (size_t)gsz, gf) != (size_t)gsz) {
		fprintf(stderr, "short read on %s\n", gold_path);
		free(gbuf);
		fclose(gf);
		tokenizer_free(&tok);
		arena_close(&a);
		return 1;
	}
	fclose(gf);

	const uint32_t *gh = (const uint32_t *)gbuf;
	int is_gld2 = (gh[0] == 0x32444C47u);
	if ((gh[0] != 0x31444C47u && !is_gld2) || gh[1] != a.hdr.dim || gh[2] != a.hdr.vocab_size) {
		fprintf(stderr, "golden file header mismatch\n");
		free(gbuf);
		tokenizer_free(&tok);
		arena_close(&a);
		return 1;
	}
	uint32_t g_steps = gh[3];
	uint32_t start_tok = gh[4];
	uint32_t g_nprompt = is_gld2 ? gh[5] : 1u;
	if (steps > g_steps) {
		steps = g_steps;
	}

	const uint32_t *g_tokens = (const uint32_t *)(gbuf + (is_gld2 ? 24u : 20u));
	const float *g_l0     = (const float *)(g_tokens + g_steps);
	const float *g_last   = g_l0 + a.hdr.dim;
	const float *g_fnorm  = g_last + a.hdr.dim;
	const float *g_logits = g_fnorm + a.hdr.dim;

	float *logits = malloc((size_t)a.hdr.vocab_size * sizeof(float));
	float *dbg_x  = malloc((size_t)a.hdr.dim * sizeof(float));
	struct c7x_llama_prof prof, prof_acc;
	float err = 0.0f;
	double cos_sim = 0.0;
	double wall_us = 0.0;

	/* Step 2: Single decoder layer verification on live C7x silicon */
	if (arena_forward_token(&a, start_tok, 0u, NULL, NULL, dbg_x, 0, 1u, &prof, &wall_us) != 0) {
		fail = 1;
		goto out;
	}
	compare_vec("silicon_l0_out", dbg_x, g_l0, a.hdr.dim, &err, &cos_sim);
	printf("[verify] layer0 single-layer latency: %.3f ms (%" PRIu64 " C7x cyc)\n",
	       (double)prof.cyc_total / 1e6, prof.cyc_total);
	if (err > 1e-4f || cos_sim < 0.999999) {
		fprintf(stderr, "FAIL: silicon layer0_out diverged from Python reference!\n");
		fail = 1;
	}

	/* Step 3: Full 6-layer greedy generation on live C7x silicon */
	mmio_zero(a.ddr + a.off_k_cache, a.kv_bytes);
	mmio_zero(a.ddr + a.off_v_cache, a.kv_bytes);
	memset(&prof_acc, 0, sizeof(prof_acc));

	uint32_t token = start_tok;
	uint32_t matches = 0u;
	uint32_t gen_checked = 0u;
	double sum_wall_us = 0.0;
	double t_gen0 = now_us();

	printf("\n--- Live C7x + MMA Silicon Output (%u tokens, greedy temp=0) ---\n", steps);
	for (uint32_t pos = 0u; pos < steps; pos++) {
		uint32_t nxt = 0u;

		if (arena_forward_token(&a, token, pos,
					(pos == 0u) ? logits : NULL, &nxt,
					(pos == 0u) ? dbg_x : NULL, -1, 0u,
					&prof, &wall_us) != 0) {
			fail = 1;
			goto out;
		}
		sum_wall_us += wall_us;
		prof_add(&prof_acc, &prof, a.hdr.n_layers);

		if (pos == 0u) {
			printf("\n");
			compare_vec("silicon_fnorm", dbg_x, g_fnorm, a.hdr.dim, &err, &cos_sim);
			if (err > 5e-4f || cos_sim < 0.999999) {
				fail = 1;
			}
			compare_vec("silicon_logits", logits, g_logits, a.hdr.vocab_size, &err, &cos_sim);
			if (err > 5e-3f || cos_sim < 0.999999) {
				fail = 1;
			}
			if (nxt != argmax_f32(logits, a.hdr.vocab_size)) {
				fprintf(stderr, "FAIL: C7x fused argmax mismatch at pos=0\n");
				fail = 1;
			}
			printf("\nText: ");
			if (is_gld2) {
				fputs(tokenizer_decode(&tok, 0, (int)start_tok), stdout);
			}
		}

		if (pos + 1u < g_nprompt) {
			/* Teacher-force prompt token */
			nxt = g_tokens[pos];
		} else {
			gen_checked++;
			if (nxt == g_tokens[pos]) {
				matches++;
			} else {
				fprintf(stderr, "\n[MISMATCH] pos=%u got=%u want=%u\n",
					pos, nxt, g_tokens[pos]);
				fail = 1;
			}
		}
		fputs(tokenizer_decode(&tok, (int)token, (int)nxt), stdout);
		fflush(stdout);
		token = nxt;
	}
	double total_gen_s = (now_us() - t_gen0) / 1e6;
	double avg_c7x_ms = (double)prof_acc.cyc_total / (double)steps / 1e6;
	double avg_wall_ms = sum_wall_us / (double)steps / 1000.0;

	printf("\n\n[verify] greedy token match vs Python reference: %u/%u -> %s\n",
	       matches, gen_checked, (matches == gen_checked && !fail) ? "PASS" : "FAIL");
	printf("[perf]   C7x compute: %.3f ms/tok (%.2f tok/s) | C7x+IPC wall: %.3f ms/tok (%.2f tok/s) | End-to-end (incl. A72): %.2f tok/s\n",
	       avg_c7x_ms, 1000.0 / avg_c7x_ms,
	       avg_wall_ms, 1000.0 / avg_wall_ms,
	       (double)steps / total_gen_s);

	/* Head-to-head single-core Cortex-A72 CPU baseline (W8A8 models) */
	if (!(a.hdr.flags & (C7M_FLAG_W4A8_LAYERS | C7M_FLAG_W4A8_WCLS))) {
		uint32_t cpu_n = steps < 10u ? steps : 10u;
		uint32_t cpu_toks[10];
		double cpu_tps = run_a72_cpu_benchmark(model_path, start_tok, cpu_n, cpu_toks);
		uint32_t cpu_match = 0u;
		for (uint32_t i = 0u; i < cpu_n; i++) {
			if (cpu_toks[i] == g_tokens[i]) cpu_match++;
		}
		printf("[a72-cpu] 1-core A72 W8A8 baseline (%u tok): %.2f tok/s (match %u/%u) -> C7x+MMA is %.1fx faster (DSP) / %.1fx faster (e2e)\n",
		       cpu_n, cpu_tps, cpu_match, cpu_n,
		       (1000.0 / avg_c7x_ms) / cpu_tps,
		       ((double)steps / total_gen_s) / cpu_tps);
	}

	/* Step 4: Verify & benchmark batched multi-token MMA prefill */
	uint32_t pf_n = is_gld2 ? g_nprompt : (steps >= 32u ? 32u : 0u);
	if (pf_n >= 2u && pf_n <= 60u && pf_n <= steps) {
		uint32_t pf_toks[60];
		uint32_t pf_argmax = 0u;
		double pf_wall_us = 0.0;
		struct c7x_llama_prof pf_prof;

		pf_toks[0] = start_tok;
		for (uint32_t i = 1u; i < pf_n; i++) {
			pf_toks[i] = g_tokens[i - 1u];
		}
		mmio_zero(a.ddr + a.off_k_cache, a.kv_bytes);
		mmio_zero(a.ddr + a.off_v_cache, a.kv_bytes);

		if (arena_prefill_tokens(&a, pf_toks, pf_n, 0u, NULL, &pf_argmax,
					 &pf_prof, &pf_wall_us) != 0) {
			fprintf(stderr, "[verify] batched prefill FAILED\n");
			fail = 1;
		} else {
			double pf_dsp_ms = (double)pf_prof.cyc_total / 1e6;
			double pf_tps = (double)pf_n / (pf_dsp_ms / 1000.0);
			int ok = (pf_argmax == g_tokens[pf_n - 1u]);
			printf("[prefill] batched MMA prefill (%u tokens): %.3f ms (%.1f tok/s DSP, %.1f tok/s wall) | next_tok=%u want=%u -> %s\n",
			       pf_n, pf_dsp_ms, pf_tps, (double)pf_n / (pf_wall_us / 1e6),
			       pf_argmax, g_tokens[pf_n - 1u], ok ? "PASS" : "FAIL");
			if (!ok) {
				fail = 1;
			}
		}
	}

	print_profile(&prof_acc, a.hdr.n_layers, steps);

out:
	free(logits);
	free(dbg_x);
	free(gbuf);
	tokenizer_free(&tok);
	arena_close(&a);
	return fail;
}

/* -------------------------------------------------------------------------- */
/* Subcommand: c7x-ai llama run                                               */
/* -------------------------------------------------------------------------- */

static int cmd_llama_run(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: c7x-ai llama run <model.c7m> [tokenizer.bin] [-p prompt] [-n steps] [-t temp] [-P topp] [-s seed]\n");
		return 2;
	}
	const char *model_path = argv[3];
	const char *tok_path   = NULL;
	const char *prompt     = "";
	uint32_t steps         = 128u;
	float temp             = 0.0f;
	float topp             = 0.9f;
	int i                  = 4;

	if (i < argc && argv[i][0] != '-') {
		tok_path = argv[i++];
	}
	for (; i < argc; i++) {
		if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
			prompt = argv[++i];
		} else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
			steps = (uint32_t)atoi(argv[++i]);
		} else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
			temp = (float)atof(argv[++i]);
		} else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
			topp = (float)atof(argv[++i]);
		} else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
			g_sample_rng = (uint64_t)strtoull(argv[++i], NULL, 10) | 1ull;
		}
	}

	struct llama_arena a;
	struct tokenizer tok;

	if (arena_load_model(&a, model_path) != 0) {
		return 1;
	}
	if (a.embedded_tok) {
		if (tokenizer_load_c7t(&tok, a.embedded_tok, a.embedded_tok_size) != 0) {
			fprintf(stderr, "failed loading embedded C7T1 tokenizer\n");
			arena_close(&a);
			return 1;
		}
	} else if (tok_path) {
		if (tokenizer_load(&tok, tok_path, (int)a.hdr.vocab_size) != 0) {
			arena_close(&a);
			return 1;
		}
	} else {
		fprintf(stderr, "model has no embedded tokenizer; pass tokenizer.bin\n");
		arena_close(&a);
		return 1;
	}
	if (steps > a.hdr.seq_len) {
		steps = a.hdr.seq_len;
	}

	int prompt_tokens[512];
	uint32_t pf_toks[512];
	int n_prompt = tokenizer_encode(&tok, prompt, 1, 0, prompt_tokens, 512);
	if (n_prompt <= 0) {
		prompt_tokens[0] = (int)(a.hdr.bos_token_id ? a.hdr.bos_token_id : 1u);
		n_prompt = 1;
	}
	float *logits = malloc((size_t)a.hdr.vocab_size * sizeof(float));
	struct c7x_llama_prof prof, prof_acc;
	memset(&prof_acc, 0, sizeof(prof_acc));

	for (i = 0; i < n_prompt; i++) {
		pf_toks[i] = (uint32_t)prompt_tokens[i];
	}

	double t0 = now_us();
	uint32_t decoded = 0u;
	uint32_t token = pf_toks[0];
	uint32_t pos = 0u;

	/* Fast batched MMA prefill when prompt has >1 tokens */
	if (n_prompt > 1) {
		double pf_wall_us = 0.0;
		uint32_t argmax = 0u;
		int need_logits = (temp > 0.0f);

		if (tok.is_c7t) {
			for (i = 0; i < n_prompt; i++) {
				fputs(tokenizer_decode(&tok, 0, (int)pf_toks[i]), stdout);
			}
		} else {
			for (i = 1; i < n_prompt; i++) {
				fputs(tokenizer_decode(&tok, (int)pf_toks[i - 1], (int)pf_toks[i]), stdout);
			}
		}
		fflush(stdout);

		if (arena_prefill_tokens(&a, pf_toks, (uint32_t)n_prompt, 0u,
					 need_logits ? logits : NULL, &argmax,
					 &prof, &pf_wall_us) != 0) {
			free(logits);
			tokenizer_free(&tok);
			arena_close(&a);
			return 1;
		}
		fprintf(stderr, "\n[prefill] %d prompt tokens in %.2f ms (%.1f tok/s DSP, %.1f tok/s wall)\n",
			n_prompt, (double)prof.cyc_total / 1e6,
			(double)n_prompt / ((double)prof.cyc_total / 1e9),
			(double)n_prompt / (pf_wall_us / 1e6));
		uint32_t nxt = (temp > 0.0f)
			       ? sample_topp(logits, a.hdr.vocab_size, temp, topp)
			       : argmax;
		if (nxt != (a.hdr.eos_token_id ? a.hdr.eos_token_id : 2u)) {
			fputs(tokenizer_decode(&tok, (int)pf_toks[n_prompt - 1], (int)nxt), stdout);
			fflush(stdout);
		}
		token = nxt;
		pos = (uint32_t)n_prompt;
		t0 = now_us();
	}

	uint32_t max_pos = pos + steps;
	if (max_pos > a.hdr.seq_len) {
		max_pos = a.hdr.seq_len;
	}
	for (; pos < max_pos; pos++) {
		double wall_us = 0.0;
		uint32_t argmax = 0u;
		int need_logits = (temp > 0.0f);

		if (arena_forward_token(&a, token, pos,
					need_logits ? logits : NULL, &argmax,
					NULL, -1, 0u, &prof, &wall_us) != 0) {
			free(logits);
			tokenizer_free(&tok);
			arena_close(&a);
			return 1;
		}
		prof_add(&prof_acc, &prof, a.hdr.n_layers);
		decoded++;

		uint32_t nxt = (temp > 0.0f)
			       ? sample_topp(logits, a.hdr.vocab_size, temp, topp)
			       : argmax;
		if (nxt == (a.hdr.eos_token_id ? a.hdr.eos_token_id : 2u) ||
		    (!tok.is_c7t && nxt == 1u)) {
			break;
		}
		fputs(tokenizer_decode(&tok, (int)token, (int)nxt), stdout);
		fflush(stdout);
		token = nxt;
	}
	double elapsed_s = (now_us() - t0) / 1e6;
	double avg_c7x_ms = decoded ? ((double)prof_acc.cyc_total / (double)decoded / 1e6) : 0.0;

	printf("\n\n[perf] %u decoded tokens in %.3f s -> %.2f tok/s end-to-end (C7x DSP: %.3f ms/tok = %.2f tok/s)\n",
	       decoded, elapsed_s, decoded ? ((double)decoded / elapsed_s) : 0.0,
	       avg_c7x_ms, avg_c7x_ms > 0.0 ? (1000.0 / avg_c7x_ms) : 0.0);
	if (decoded > 0u) {
		print_profile(&prof_acc, a.hdr.n_layers, decoded);
	}

	free(logits);
	tokenizer_free(&tok);
	arena_close(&a);
	return 0;
}

int cmd_llama(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr,
			"usage:\n"
			"  c7x-ai llama verify <model.c7m> [tokenizer.bin] <golden.bin> [steps]\n"
			"  c7x-ai llama run    <model.c7m> [tokenizer.bin] [-p prompt] [-n steps] [-t temp] [-P topp] [-s seed]\n");
		return 2;
	}
	if (strcmp(argv[2], "verify") == 0) {
		return cmd_llama_verify(argc, argv);
	}
	if (strcmp(argv[2], "run") == 0) {
		return cmd_llama_run(argc, argv);
	}
	fprintf(stderr, "unknown llama subcommand '%s'\n", argv[2]);
	return 2;
}
