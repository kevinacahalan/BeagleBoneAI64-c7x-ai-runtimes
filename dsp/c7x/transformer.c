/* SPDX-License-Identifier: MIT */
/*
 * C7x + MMA Universal Transformer inference engine.
 *
 * Supports:
 *   - W8A8 MMA (row-major or 1D-streamed 64x64 tile-major)
 *   - Group-64 row-pair W4A8 (dual SE0/SE1 vector dotp4_ext + fused dequant/argmax)
 *   - Optional QKV biases (Qwen2/2.5), untied embeddings, NeoX or interleaved RoPE,
 *     configurable rms_norm_eps, GQA, and MSMC SRAM hot-layer staging.
 */

#ifdef __C7X_HOSTEM__
#include "c7x_host_emulation.h"
static inline uint64_t read_tsc(void) { return 0ull; }
#else
#include <c7x.h>
static inline uint64_t read_tsc(void) { return (uint64_t)__TSC; }
#endif

#include <stdint.h>
#include <string.h>

#include "mma_kernels.h"
#include "transformer.h"
#include "vec_kernels.h"

static inline uint32_t align64_u32(uint32_t n)
{
	return (n + 63u) & ~63u;
}

static inline const void *model_ptr(const struct c7m_header *hdr, uint32_t off)
{
	return (const uint8_t *)hdr + off;
}

static inline const void *layer_ptr(const struct c7m_header *hdr,
				    const uint8_t *msmc_base, uint32_t msmc_n_layers,
				    uint32_t layer_idx, uint32_t off)
{
	if (msmc_base && layer_idx < msmc_n_layers &&
	    hdr->off_layers_base != 0u && off >= hdr->off_layers_base) {
		return msmc_base + (off - hdr->off_layers_base);
	}
	return (const uint8_t *)hdr + off;
}

static inline void mma_linear_w8(const int8_t *xq, const int8_t *wq, int32_t *y_i32,
				 uint32_t k_pad, uint32_t n_pad, int tiled)
{
	if (tiled) {
		c7x_mma_gemv_i8_i32_tiled(xq, wq, y_i32, k_pad, n_pad);
	} else {
		c7x_mma_gemm_i8_i32(xq, wq, y_i32, 1u, k_pad, n_pad, 1u, 1u);
	}
}

static void emb_lookup(float *out, const struct c7m_header *hdr, uint32_t token)
{
	uint32_t dim = hdr->dim;
	uint32_t dim_pad = hdr->dim_pad;

	if (hdr->off_emb_q != 0u && hdr->off_emb_s != 0u) {
		/* Separate row-major int8 embedding table */
		const int8_t *emb_q = (const int8_t *)model_ptr(hdr, hdr->off_emb_q);
		const float *emb_s  = (const float *)model_ptr(hdr, hdr->off_emb_s);

		c7x_dequant_i8_f32(out, emb_q + (uint64_t)token * dim_pad, emb_s[token], dim);
		return;
	}

	if (hdr->flags & C7M_FLAG_W4A8_WCLS) {
		/* Unpack single token row from tile-major group-64 W4A8 wcls */
		const uint8_t *wq = (const uint8_t *)model_ptr(hdr, hdr->off_wcls_q);
		const float *ws   = (const float *)model_ptr(hdr, hdr->off_wcls_s);
		uint32_t k_tiles = dim_pad >> 6;
		uint32_t nt = token >> 6;
		uint32_t r = token & 63u;
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
		return;
	}

	/* Tied W8A8 wcls (tiled or row-major) */
	{
		const int8_t *wcls_q = (const int8_t *)model_ptr(hdr, hdr->off_wcls_q);
		const float *wcls_s  = (const float *)model_ptr(hdr, hdr->off_wcls_s);
		int tiled = (hdr->flags & C7M_FLAG_TILED_WEIGHTS) ? 1 : 0;

		if (!tiled) {
			c7x_dequant_i8_f32(out, wcls_q + (uint64_t)token * dim_pad,
					   wcls_s[token], dim);
		} else {
			uint32_t k_tiles = dim_pad >> 6;
			uint32_t nt = token >> 6;
			uint32_t row = token & 63u;
			uint32_t kt;

			for (kt = 0u; kt < k_tiles; kt++) {
				uint32_t col0 = kt << 6;
				uint32_t rem = (dim > col0) ? (dim - col0) : 0u;
				uint32_t chunk = rem > 64u ? 64u : rem;

				if (chunk > 0u) {
					const int8_t *src =
						wcls_q + ((uint64_t)nt * k_tiles + kt) * 4096ull +
						row * 64u;
					c7x_dequant_i8_f32(out + col0, src, wcls_s[token], chunk);
				}
			}
		}
	}
}

int c7x_llama_validate(const struct c7m_header *hdr)
{
	if (!hdr) {
		return -1;
	}
	if (hdr->magic != C7M_MAGIC || hdr->version != C7M_VERSION) {
		return -2;
	}
	if (hdr->dim == 0u || hdr->hidden_dim == 0u || hdr->n_layers == 0u ||
	    hdr->n_layers > C7M_MAX_LAYERS || hdr->n_heads == 0u || hdr->n_kv_heads == 0u ||
	    hdr->vocab_size == 0u || hdr->seq_len == 0u) {
		return -3;
	}
	if ((hdr->dim_pad % 64u) != 0u || (hdr->hidden_pad % 64u) != 0u ||
	    (hdr->vocab_pad % 64u) != 0u) {
		return -4;
	}
	if (hdr->dim_pad < hdr->dim || hdr->hidden_pad < hdr->hidden_dim ||
	    hdr->vocab_pad < hdr->vocab_size) {
		return -5;
	}
	return 0;
}

struct scratch_arena {
	float   *x;
	float   *xb;
	float   *xb2;
	float   *q;
	float   *k;
	float   *v;
	float   *hb;
	float   *hb2;
	float   *att;
	int8_t  *xq;
	int32_t *y_i32;
};

static int carve_scratch(const struct c7m_header *hdr, void *buf, uint32_t bytes,
			 struct scratch_arena *a, uint32_t *used_out)
{
	uint8_t *p = (uint8_t *)buf;
	uint32_t off = 0u;
	uint32_t dim_b = align64_u32(hdr->dim_pad * (uint32_t)sizeof(float));
	uint32_t hid_b = align64_u32(hdr->hidden_pad * (uint32_t)sizeof(float));
	uint32_t att_b = align64_u32(hdr->seq_len * (uint32_t)sizeof(float));
	uint32_t max_k = hdr->hidden_pad > hdr->dim_pad ? hdr->hidden_pad : hdr->dim_pad;
	uint32_t xq_b  = align64_u32(max_k);
	/* W4A8 wcls skips y_i32 staging completely, allowing 152k+ vocab inside 444 KiB L2! */
	uint32_t max_n = (hdr->flags & C7M_FLAG_W4A8_WCLS)
			 ? max_k
			 : (hdr->vocab_pad > max_k ? hdr->vocab_pad : max_k);
	uint32_t y32_b = align64_u32(max_n * (uint32_t)sizeof(int32_t));

#define TAKE(field, type, sz)                             \
	do {                                              \
		if (off + (sz) > bytes) {                 \
			return -1;                        \
		}                                         \
		(field) = (type *)(p + off);              \
		off += (sz);                              \
	} while (0)

	TAKE(a->x,     float,   dim_b);
	TAKE(a->xb,    float,   dim_b);
	TAKE(a->xb2,   float,   dim_b);
	TAKE(a->q,     float,   dim_b);
	TAKE(a->k,     float,   dim_b);
	TAKE(a->v,     float,   dim_b);
	TAKE(a->hb,    float,   hid_b);
	TAKE(a->hb2,   float,   hid_b);
	TAKE(a->att,   float,   att_b);
	TAKE(a->xq,    int8_t,  xq_b);
	TAKE(a->y_i32, int32_t, y32_b);

#undef TAKE
	if (used_out) {
		*used_out = off;
	}
	return 0;
}

static inline void proj_linear(float *out, const int8_t *xq, float x_scale,
			       const void *wq, const float *ws, int32_t *y_i32,
			       uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
			       int add, int is_w4, int tiled,
			       uint64_t *cyc_mm, uint64_t *cyc_dq)
{
	uint64_t t0 = read_tsc();

	if (is_w4) {
		c7x_mma_gemm_w4a8_g64_f32(out, xq, &x_scale,
					  (const uint8_t *)wq, ws,
					  1u, k_pad, n_pad, out_dim, 1u, add, NULL);
		if (cyc_mm) {
			*cyc_mm += (read_tsc() - t0);
		}
	} else {
		uint64_t t1;

		mma_linear_w8(xq, (const int8_t *)wq, y_i32, k_pad, n_pad, tiled);
		t1 = read_tsc();
		if (cyc_mm) {
			*cyc_mm += (t1 - t0);
		}
		if (add) {
			c7x_dequant_add_i32_f32(out, y_i32, ws, x_scale, out_dim);
		} else {
			c7x_dequant_i32_f32(out, y_i32, ws, x_scale, out_dim);
		}
		if (cyc_dq) {
			*cyc_dq += (read_tsc() - t1);
		}
	}
}

int c7x_llama_forward_token(const struct c7m_header *hdr,
			    uint32_t token, uint32_t pos,
			    float *k_cache, float *v_cache,
			    float *logits_out, uint32_t *argmax_out,
			    void *scratch, uint32_t scratch_bytes,
			    const void *msmc_layers, uint32_t msmc_n_layers,
			    struct c7x_llama_prof *prof,
			    float *debug_x_out, int32_t debug_layer,
			    uint32_t stop_after_layer)
{
	struct scratch_arena s;
	const struct c7m_layer_offs *layers;
	const struct c7m_layer_bias_offs *biases = NULL;
	const uint8_t *msmc_base = (const uint8_t *)msmc_layers;
	const float *rope_cos;
	const float *rope_sin;
	const float *rms_final;
	const void *wcls_q;
	const float *wcls_s;
	uint32_t dim, dim_pad, hidden_dim, hidden_pad, kv_dim, kv_dim_pad;
	uint32_t n_layers_run, l, argmax = 0u;
	uint64_t t_start, t0, t1, t_layer0;
	float x_scale = 0.0f;
	float eps;
	int tiled, w4_layers, w4_wcls, neox;

	if (c7x_llama_validate(hdr) != 0) {
		return -1;
	}
	if (token >= hdr->vocab_size || pos >= hdr->seq_len || !k_cache || !v_cache || !scratch) {
		return -2;
	}
	if (stop_after_layer == 0u && !logits_out && !argmax_out) {
		return -3;
	}
	uint32_t scratch_used = 0u;

	if (carve_scratch(hdr, scratch, scratch_bytes, &s, &scratch_used) != 0) {
		return -4;
	}
	if (scratch_bytes > scratch_used) {
		c7x_mma_set_dru_l2_buf((uint8_t *)scratch + scratch_used, scratch_bytes - scratch_used);
	} else {
		c7x_mma_set_dru_l2_buf(NULL, 0u);
	}

	if (prof) {
		memset(prof, 0, sizeof(*prof));
	}

	t_start = read_tsc();

	dim        = hdr->dim;
	dim_pad    = hdr->dim_pad;
	hidden_dim = hdr->hidden_dim;
	hidden_pad = hdr->hidden_pad;
	kv_dim     = hdr->kv_dim;
	kv_dim_pad = align64_u32(kv_dim);
	tiled      = (hdr->flags & C7M_FLAG_TILED_WEIGHTS) ? 1 : 0;
	w4_layers  = (hdr->flags & C7M_FLAG_W4A8_LAYERS) ? 1 : 0;
	w4_wcls    = (hdr->flags & C7M_FLAG_W4A8_WCLS) ? 1 : 0;
	neox       = (hdr->flags & C7M_FLAG_ROPE_NEOX) ? 1 : 0;
	eps        = (hdr->rms_norm_eps > 0.0f) ? hdr->rms_norm_eps : 1e-5f;
	n_layers_run = (stop_after_layer > 0u && stop_after_layer <= hdr->n_layers)
		       ? stop_after_layer
		       : hdr->n_layers;

	layers    = (const struct c7m_layer_offs *)model_ptr(hdr, hdr->off_layer_table);
	if ((hdr->flags & C7M_FLAG_HAS_QKV_BIAS) && hdr->off_bias_table != 0u) {
		biases = (const struct c7m_layer_bias_offs *)model_ptr(hdr, hdr->off_bias_table);
	}
	rope_cos  = (const float *)model_ptr(hdr, hdr->off_rope_cos) + pos * (hdr->head_size >> 1);
	rope_sin  = (const float *)model_ptr(hdr, hdr->off_rope_sin) + pos * (hdr->head_size >> 1);
	rms_final = (const float *)model_ptr(hdr, hdr->off_rms_final);
	wcls_q    = model_ptr(hdr, hdr->off_wcls_q);
	wcls_s    = (const float *)model_ptr(hdr, hdr->off_wcls_s);

	/* 1. Token embedding lookup into L2 s.x */
	t0 = read_tsc();
	emb_lookup(s.x, hdr, token);
	t1 = read_tsc();
	if (prof) {
		prof->cyc_emb += (t1 - t0);
	}

	/* 2. Transformer decoder layers */
	for (l = 0u; l < n_layers_run; l++) {
		const struct c7m_layer_offs *lo = &layers[l];
		const float *rms_att = (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->rms_att);
		const float *rms_ffn = (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->rms_ffn);
		float *k_layer = k_cache + (uint64_t)l * hdr->seq_len * kv_dim;
		float *v_layer = v_cache + (uint64_t)l * hdr->seq_len * kv_dim;

		t_layer0 = read_tsc();

		/* 2a. Attention RMSNorm */
		t0 = read_tsc();
		c7x_rmsnorm_f32(s.xb, s.x, rms_att, dim, eps);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_rmsnorm += (t1 - t0);
		}

		/* 2b. Quantize xb once and run Q, K, V projections */
		t0 = read_tsc();
		c7x_quantize_f32_i8(s.xq, s.xb, dim, dim_pad, &x_scale);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_quant += (t1 - t0);
		}

		proj_linear(s.q, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wq_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wq_s),
			    s.y_i32, dim_pad, dim_pad, dim, 0, w4_layers, tiled,
			    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		proj_linear(s.k, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wk_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wk_s),
			    s.y_i32, dim_pad, kv_dim_pad, kv_dim, 0, w4_layers, tiled,
			    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		proj_linear(s.v, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wv_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wv_s),
			    s.y_i32, dim_pad, kv_dim_pad, kv_dim, 0, w4_layers, tiled,
			    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		/* Optional QKV biases (Qwen2/2.5) */
		if (biases) {
			const struct c7m_layer_bias_offs *bo = &biases[l];

			if (bo->bq) {
				c7x_add_bias_f32(s.q, (const float *)model_ptr(hdr, bo->bq), dim);
			}
			if (bo->bk) {
				c7x_add_bias_f32(s.k, (const float *)model_ptr(hdr, bo->bk), kv_dim);
			}
			if (bo->bv) {
				c7x_add_bias_f32(s.v, (const float *)model_ptr(hdr, bo->bv), kv_dim);
			}
		}

		/* 2c. RoPE + store K, V into KV cache */
		t0 = read_tsc();
		c7x_rope_f32(s.q, s.k, rope_cos, rope_sin,
			     hdr->n_heads, hdr->n_kv_heads, hdr->head_size, neox);
		memcpy(k_layer + (uint64_t)pos * kv_dim, s.k, kv_dim * sizeof(float));
		memcpy(v_layer + (uint64_t)pos * kv_dim, s.v, kv_dim * sizeof(float));
		t1 = read_tsc();
		if (prof) {
			prof->cyc_rope += (t1 - t0);
		}

		/* 2d. Multi-head causal attention -> s.xb */
		t0 = read_tsc();
		c7x_attn_mha_f32(s.xb, s.q, k_layer, v_layer, s.att, pos,
				 hdr->n_heads, hdr->n_kv_heads, hdr->head_size);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_attn += (t1 - t0);
		}

		/* 2e. Output projection Wo + residual add into s.x */
		t0 = read_tsc();
		c7x_quantize_f32_i8(s.xq, s.xb, dim, dim_pad, &x_scale);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_quant += (t1 - t0);
		}

		proj_linear(s.x, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wo_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wo_s),
			    s.y_i32, dim_pad, dim_pad, dim, 1, w4_layers, tiled,
			    prof ? &prof->cyc_wo_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		/* 2f. FFN RMSNorm */
		t0 = read_tsc();
		c7x_rmsnorm_f32(s.xb, s.x, rms_ffn, dim, eps);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_rmsnorm += (t1 - t0);
		}

		/* 2g. Quantize xb once and run W1, W3 projections */
		t0 = read_tsc();
		c7x_quantize_f32_i8(s.xq, s.xb, dim, dim_pad, &x_scale);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_quant += (t1 - t0);
		}

		proj_linear(s.hb, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w1_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w1_s),
			    s.y_i32, dim_pad, hidden_pad, hidden_dim, 0, w4_layers, tiled,
			    prof ? &prof->cyc_ffn_up_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		proj_linear(s.hb2, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w3_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w3_s),
			    s.y_i32, dim_pad, hidden_pad, hidden_dim, 0, w4_layers, tiled,
			    prof ? &prof->cyc_ffn_up_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		/* 2h. SwiGLU: hb = SiLU(hb) * hb2 */
		t0 = read_tsc();
		c7x_swiglu_f32(s.hb, s.hb, s.hb2, hidden_dim);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_swiglu += (t1 - t0);
		}

		/* 2i. Down projection W2 + residual add into s.x */
		t0 = read_tsc();
		c7x_quantize_f32_i8(s.xq, s.hb, hidden_dim, hidden_pad, &x_scale);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_quant += (t1 - t0);
		}

		proj_linear(s.x, s.xq, x_scale,
			    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w2_q),
			    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w2_s),
			    s.y_i32, hidden_pad, dim_pad, dim, 1, w4_layers, tiled,
			    prof ? &prof->cyc_ffn_down_mma : NULL, prof ? &prof->cyc_dequant : NULL);

		if (prof) {
			prof->cyc_layer[l] = read_tsc() - t_layer0;
		}
		if (debug_x_out && debug_layer == (int32_t)l) {
			memcpy(debug_x_out, s.x, dim * sizeof(float));
		}
	}

	if (stop_after_layer > 0u) {
		c7x_mma_set_dru_l2_buf(NULL, 0u);
		if (prof) {
			prof->cyc_total = read_tsc() - t_start;
		}
		return 0;
	}

	/* 3. Final RMSNorm */
	t0 = read_tsc();
	c7x_rmsnorm_f32(s.xb, s.x, rms_final, dim, eps);
	t1 = read_tsc();
	if (prof) {
		prof->cyc_rmsnorm += (t1 - t0);
	}
	if (debug_x_out && debug_layer < 0) {
		memcpy(debug_x_out, s.xb, dim * sizeof(float));
	}

	/* 4. Final classifier projection (wcls) -> argmax (+ optional logits_out) */
	t0 = read_tsc();
	c7x_quantize_f32_i8(s.xq, s.xb, dim, dim_pad, &x_scale);
	t1 = read_tsc();
	if (prof) {
		prof->cyc_quant += (t1 - t0);
	}

	if (w4_wcls) {
		t0 = read_tsc();
		c7x_mma_gemm_w4a8_g64_f32(logits_out, s.xq, &x_scale,
					  (const uint8_t *)wcls_q, wcls_s,
					  1u, dim_pad, hdr->vocab_pad, hdr->vocab_size,
					  1u, 0, &argmax);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_cls_mma += (t1 - t0);
		}
	} else {
		t0 = read_tsc();
		mma_linear_w8(s.xq, (const int8_t *)wcls_q, s.y_i32,
			      dim_pad, hdr->vocab_pad, tiled);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_cls_mma += (t1 - t0);
		}

		t0 = read_tsc();
		argmax = c7x_dequant_argmax_i32_f32(logits_out, s.y_i32, wcls_s,
						    x_scale, hdr->vocab_size);
		t1 = read_tsc();
		if (prof) {
			prof->cyc_dequant += (t1 - t0);
		}
	}

	c7x_mma_set_dru_l2_buf(NULL, 0u);
	if (argmax_out) {
		*argmax_out = argmax;
	}
	if (prof) {
		prof->cyc_total = read_tsc() - t_start;
	}

	return 0;
}

static void proj_linear_batched(float *out, const int8_t *xq, const float *x_scales,
				const void *wq, const float *ws, int32_t *y_i32,
				uint32_t m, uint32_t k_pad, uint32_t n_pad, uint32_t out_dim,
				uint32_t block_m, int add, int is_w4, int tiled,
				uint64_t *cyc_mm, uint64_t *cyc_dq)
{
	uint64_t t0 = read_tsc();

	if (is_w4) {
		c7x_mma_gemm_w4a8_g64_f32(out, xq, x_scales, (const uint8_t *)wq, ws,
					  m, k_pad, n_pad, out_dim, block_m, add, NULL);
		if (cyc_mm) {
			*cyc_mm += (read_tsc() - t0);
		}
	} else {
		uint64_t t1;
		uint32_t r;

		if (tiled) {
			c7x_mma_gemm_i8_i32_tiled(xq, (const int8_t *)wq, y_i32,
						  m, k_pad, n_pad, block_m);
		} else {
			c7x_mma_gemm_i8_i32(xq, (const int8_t *)wq, y_i32,
					    m, k_pad, n_pad, 1u, block_m);
		}
		t1 = read_tsc();
		if (cyc_mm) {
			*cyc_mm += (t1 - t0);
		}
		for (r = 0u; r < m; r++) {
			if (add) {
				c7x_dequant_add_i32_f32(out + (size_t)r * out_dim,
							y_i32 + (size_t)r * n_pad,
							ws, x_scales[r], out_dim);
			} else {
				c7x_dequant_i32_f32(out + (size_t)r * out_dim,
						    y_i32 + (size_t)r * n_pad,
						    ws, x_scales[r], out_dim);
			}
		}
		if (cyc_dq) {
			*cyc_dq += (read_tsc() - t1);
		}
	}
}

int c7x_llama_prefill_tokens(const struct c7m_header *hdr,
			     const uint32_t *tokens, uint32_t n_tokens, uint32_t start_pos,
			     float *k_cache, float *v_cache,
			     float *logits_out, uint32_t *argmax_out,
			     void *l2_scratch, uint32_t l2_bytes,
			     void *prefill_buf, uint32_t prefill_bytes,
			     const void *msmc_layers, uint32_t msmc_n_layers,
			     struct c7x_llama_prof *prof)
{
	const struct c7m_layer_offs *layers;
	const struct c7m_layer_bias_offs *biases = NULL;
	const uint8_t *msmc_base = (const uint8_t *)msmc_layers;
	uint32_t dim, dim_pad, hidden_dim, hidden_pad, kv_dim, kv_dim_pad, max_k, max_n;
	uint32_t max_chunk, done = 0u, argmax = 0u;
	uint64_t t_start, t0, t1, t_layer0;
	float eps;
	int tiled, w4_layers, w4_wcls, neox;
	struct scratch_arena l2s;

	if (c7x_llama_validate(hdr) != 0) {
		return -1;
	}
	if (!tokens || n_tokens == 0u || start_pos + n_tokens > hdr->seq_len ||
	    !k_cache || !v_cache || !l2_scratch || !prefill_buf) {
		return -2;
	}
	if (n_tokens == 1u) {
		return c7x_llama_forward_token(hdr, tokens[0], start_pos, k_cache, v_cache,
					       logits_out, argmax_out, l2_scratch, l2_bytes,
					       msmc_layers, msmc_n_layers, prof, NULL, -1, 0u);
	}
	{
		uint32_t l2_used = 0u;
		if (carve_scratch(hdr, l2_scratch, l2_bytes, &l2s, &l2_used) != 0) {
			return -3;
		}
		if (l2_bytes > l2_used) {
			c7x_mma_set_dru_l2_buf((uint8_t *)l2_scratch + l2_used, l2_bytes - l2_used);
		} else {
			c7x_mma_set_dru_l2_buf(NULL, 0u);
		}
	}
	if (prof) {
		memset(prof, 0, sizeof(*prof));
	}

	t_start = read_tsc();

	dim        = hdr->dim;
	dim_pad    = hdr->dim_pad;
	hidden_dim = hdr->hidden_dim;
	hidden_pad = hdr->hidden_pad;
	kv_dim     = hdr->kv_dim;
	kv_dim_pad = align64_u32(kv_dim);
	max_k      = hidden_pad > dim_pad ? hidden_pad : dim_pad;
	max_n      = hidden_pad > dim_pad ? hidden_pad : dim_pad;
	tiled      = (hdr->flags & C7M_FLAG_TILED_WEIGHTS) ? 1 : 0;
	w4_layers  = (hdr->flags & C7M_FLAG_W4A8_LAYERS) ? 1 : 0;
	w4_wcls    = (hdr->flags & C7M_FLAG_W4A8_WCLS) ? 1 : 0;
	neox       = (hdr->flags & C7M_FLAG_ROPE_NEOX) ? 1 : 0;
	eps        = (hdr->rms_norm_eps > 0.0f) ? hdr->rms_norm_eps : 1e-5f;

	layers = (const struct c7m_layer_offs *)model_ptr(hdr, hdr->off_layer_table);
	if ((hdr->flags & C7M_FLAG_HAS_QKV_BIAS) && hdr->off_bias_table != 0u) {
		biases = (const struct c7m_layer_bias_offs *)model_ptr(hdr, hdr->off_bias_table);
	}

	/* Choose maximum chunk size (up to 60 on silicon, 16 in host emulation) that fits prefill_buf. */
#ifdef __C7X_HOSTEM__
	max_chunk = 16u;
#else
	max_chunk = 60u;
#endif
	{
		size_t per_tok =
			(size_t)dim * 3u * sizeof(float) +          /* X, XB, Q */
			(size_t)kv_dim * 2u * sizeof(float) +       /* K, V */
			(size_t)hidden_dim * 2u * sizeof(float) +   /* HB, HB2 */
			(size_t)max_k * sizeof(int8_t) +            /* XQ */
			(size_t)sizeof(float) +                     /* XS */
			(w4_layers ? 0u : ((size_t)max_n * sizeof(int32_t))) + /* Y32 */
			512u;                                       /* alignment slack */
		uint32_t fit = (uint32_t)(prefill_bytes / per_tok);

		if (fit == 0u) {
			return -4;
		}
		if (fit < max_chunk) {
			max_chunk = fit;
		}
	}

	while (done < n_tokens) {
		uint32_t m = (n_tokens - done < max_chunk) ? (n_tokens - done) : max_chunk;
		uint32_t cur_pos = start_pos + done;
		uint8_t *bp = (uint8_t *)prefill_buf;
		size_t boff = 0u;
		float *X, *XB, *Q, *K, *V, *HB, *HB2, *XS;
		int8_t *XQ;
		int32_t *Y32 = NULL;
		uint32_t t, l;

#define PTAKE(var, type, sz_bytes)                                \
		do {                                              \
			size_t _asz = ((sz_bytes) + 63u) & ~63u;  \
			if (boff + _asz > prefill_bytes) {        \
				return -5;                        \
			}                                         \
			(var) = (type *)(bp + boff);              \
			boff += _asz;                             \
		} while (0)

		PTAKE(X,   float,  (size_t)m * dim * sizeof(float));
		PTAKE(XB,  float,  (size_t)m * dim * sizeof(float));
		PTAKE(Q,   float,  (size_t)m * dim * sizeof(float));
		PTAKE(K,   float,  (size_t)m * kv_dim * sizeof(float));
		PTAKE(V,   float,  (size_t)m * kv_dim * sizeof(float));
		PTAKE(HB,  float,  (size_t)m * hidden_dim * sizeof(float));
		PTAKE(HB2, float,  (size_t)m * hidden_dim * sizeof(float));
		PTAKE(XQ,  int8_t, (size_t)m * max_k);
		PTAKE(XS,  float,  (size_t)m * sizeof(float));
		if (!w4_layers) {
			PTAKE(Y32, int32_t, (size_t)m * max_n * sizeof(int32_t));
		}
#undef PTAKE

		/* 1. Embeddings for all m tokens in chunk */
		t0 = read_tsc();
		for (t = 0u; t < m; t++) {
			if (tokens[done + t] >= hdr->vocab_size) {
				return -6;
			}
			emb_lookup(X + (size_t)t * dim, hdr, tokens[done + t]);
		}
		if (prof) {
			prof->cyc_emb += (read_tsc() - t0);
		}

		/* 2. All decoder layers (batched MMA projections!) */
		for (l = 0u; l < hdr->n_layers; l++) {
			const struct c7m_layer_offs *lo = &layers[l];
			const float *rms_att = (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->rms_att);
			const float *rms_ffn = (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->rms_ffn);
			float *k_layer = k_cache + (uint64_t)l * hdr->seq_len * kv_dim;
			float *v_layer = v_cache + (uint64_t)l * hdr->seq_len * kv_dim;

			t_layer0 = read_tsc();

			/* Attention RMSNorm + quantize all m rows */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_rmsnorm_f32(XB + (size_t)t * dim, X + (size_t)t * dim,
						rms_att, dim, eps);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_rmsnorm += (t1 - t0);

			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_quantize_f32_i8(XQ + (size_t)t * dim_pad, XB + (size_t)t * dim,
						    dim, dim_pad, &XS[t]);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_quant += (t1 - t0);

			/* Batched Q, K, V on MMA */
			proj_linear_batched(Q, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wq_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wq_s),
					    Y32, m, dim_pad, dim_pad, dim, max_chunk, 0, w4_layers, tiled,
					    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			proj_linear_batched(K, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wk_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wk_s),
					    Y32, m, dim_pad, kv_dim_pad, kv_dim, max_chunk, 0, w4_layers, tiled,
					    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			proj_linear_batched(V, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wv_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wv_s),
					    Y32, m, dim_pad, kv_dim_pad, kv_dim, max_chunk, 0, w4_layers, tiled,
					    prof ? &prof->cyc_qkv_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			/* Optional QKV biases + RoPE + KV cache populate */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				uint32_t p = cur_pos + t;
				float *qt = Q + (size_t)t * dim;
				float *kt = K + (size_t)t * kv_dim;
				float *vt = V + (size_t)t * kv_dim;
				const float *rc = (const float *)model_ptr(hdr, hdr->off_rope_cos) +
						  p * (hdr->head_size >> 1);
				const float *rs = (const float *)model_ptr(hdr, hdr->off_rope_sin) +
						  p * (hdr->head_size >> 1);

				if (biases) {
					const struct c7m_layer_bias_offs *bo = &biases[l];
					if (bo->bq) c7x_add_bias_f32(qt, (const float *)model_ptr(hdr, bo->bq), dim);
					if (bo->bk) c7x_add_bias_f32(kt, (const float *)model_ptr(hdr, bo->bk), kv_dim);
					if (bo->bv) c7x_add_bias_f32(vt, (const float *)model_ptr(hdr, bo->bv), kv_dim);
				}
				c7x_rope_f32(qt, kt, rc, rs, hdr->n_heads, hdr->n_kv_heads, hdr->head_size, neox);
				memcpy(k_layer + (uint64_t)p * kv_dim, kt, kv_dim * sizeof(float));
				memcpy(v_layer + (uint64_t)p * kv_dim, vt, kv_dim * sizeof(float));
			}
			t1 = read_tsc();
			if (prof) prof->cyc_rope += (t1 - t0);

			/* Causal MHA attention for each position in chunk */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_attn_mha_f32(XB + (size_t)t * dim, Q + (size_t)t * dim,
						 k_layer, v_layer, l2s.att, cur_pos + t,
						 hdr->n_heads, hdr->n_kv_heads, hdr->head_size);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_attn += (t1 - t0);

			/* Quantize XB + batched Wo + residual add */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_quantize_f32_i8(XQ + (size_t)t * dim_pad, XB + (size_t)t * dim,
						    dim, dim_pad, &XS[t]);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_quant += (t1 - t0);

			proj_linear_batched(X, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wo_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->wo_s),
					    Y32, m, dim_pad, dim_pad, dim, max_chunk, 1, w4_layers, tiled,
					    prof ? &prof->cyc_wo_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			/* FFN RMSNorm + quantize */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_rmsnorm_f32(XB + (size_t)t * dim, X + (size_t)t * dim,
						rms_ffn, dim, eps);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_rmsnorm += (t1 - t0);

			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_quantize_f32_i8(XQ + (size_t)t * dim_pad, XB + (size_t)t * dim,
						    dim, dim_pad, &XS[t]);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_quant += (t1 - t0);

			/* Batched W1, W3 on MMA */
			proj_linear_batched(HB, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w1_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w1_s),
					    Y32, m, dim_pad, hidden_pad, hidden_dim, max_chunk, 0, w4_layers, tiled,
					    prof ? &prof->cyc_ffn_up_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			proj_linear_batched(HB2, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w3_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w3_s),
					    Y32, m, dim_pad, hidden_pad, hidden_dim, max_chunk, 0, w4_layers, tiled,
					    prof ? &prof->cyc_ffn_up_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			/* SwiGLU + quantize HB */
			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_swiglu_f32(HB + (size_t)t * hidden_dim,
					       HB + (size_t)t * hidden_dim,
					       HB2 + (size_t)t * hidden_dim, hidden_dim);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_swiglu += (t1 - t0);

			t0 = read_tsc();
			for (t = 0u; t < m; t++) {
				c7x_quantize_f32_i8(XQ + (size_t)t * hidden_pad,
						    HB + (size_t)t * hidden_dim,
						    hidden_dim, hidden_pad, &XS[t]);
			}
			t1 = read_tsc();
			if (prof) prof->cyc_quant += (t1 - t0);

			/* Batched W2 + residual add into X */
			proj_linear_batched(X, XQ, XS,
					    layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w2_q),
					    (const float *)layer_ptr(hdr, msmc_base, msmc_n_layers, l, lo->w2_s),
					    Y32, m, hidden_pad, dim_pad, dim, max_chunk, 1, w4_layers, tiled,
					    prof ? &prof->cyc_ffn_down_mma : NULL, prof ? &prof->cyc_dequant : NULL);

			if (prof) {
				prof->cyc_layer[l] += (read_tsc() - t_layer0);
			}
		}

		done += m;

		/* 3. Classifier ONLY for the very last token of the prompt! */
		if (done == n_tokens) {
			float last_scale = 0.0f;
			const float *rms_final = (const float *)model_ptr(hdr, hdr->off_rms_final);
			const void *wcls_q     = model_ptr(hdr, hdr->off_wcls_q);
			const float *wcls_s    = (const float *)model_ptr(hdr, hdr->off_wcls_s);

			t0 = read_tsc();
			c7x_rmsnorm_f32(l2s.xb, X + (size_t)(m - 1u) * dim, rms_final, dim, eps);
			if (prof) prof->cyc_rmsnorm += (read_tsc() - t0);

			t0 = read_tsc();
			c7x_quantize_f32_i8(l2s.xq, l2s.xb, dim, dim_pad, &last_scale);
			if (prof) prof->cyc_quant += (read_tsc() - t0);

			if (w4_wcls) {
				t0 = read_tsc();
				c7x_mma_gemm_w4a8_g64_f32(logits_out, l2s.xq, &last_scale,
							  (const uint8_t *)wcls_q, wcls_s,
							  1u, dim_pad, hdr->vocab_pad, hdr->vocab_size,
							  1u, 0, &argmax);
				if (prof) prof->cyc_cls_mma += (read_tsc() - t0);
			} else {
				t0 = read_tsc();
				mma_linear_w8(l2s.xq, (const int8_t *)wcls_q, l2s.y_i32,
					      dim_pad, hdr->vocab_pad, tiled);
				t1 = read_tsc();
				if (prof) prof->cyc_cls_mma += (t1 - t0);
				argmax = c7x_dequant_argmax_i32_f32(logits_out, l2s.y_i32, wcls_s,
								    last_scale, hdr->vocab_size);
				if (prof) prof->cyc_dequant += (read_tsc() - t1);
			}
		}
	}

	c7x_mma_set_dru_l2_buf(NULL, 0u);
	if (argmax_out) {
		*argmax_out = argmax;
	}
	if (prof) {
		prof->cyc_total = read_tsc() - t_start;
	}
	return 0;
}
