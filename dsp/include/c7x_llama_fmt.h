/* SPDX-License-Identifier: MIT */
/*
 * Shared .c7m quantized transformer binary format & profiling structures.
 *
 * Used by:
 *   - tools/llama_ref.py & tools/convert_hf.py (packers & Python reference)
 *   - dsp/c7x/transformer.c (C7x DSP + MMA runtime)
 *   - dsp/host_test/test_llama.cpp (x86_64 host emulation verifier)
 *   - linux-app/llama.c (A72 Linux host loader, tokenizer & sampler)
 */
#ifndef C7X_LLAMA_FMT_H_
#define C7X_LLAMA_FMT_H_

#include <stdint.h>

#define C7M_MAGIC       0x314D3743u /* "C7M1" */
#define C7M_VERSION     1u
#define C7M_HEADER_SIZE 256u
#define C7M_MAX_LAYERS  48u

#define C7M_FLAG_TILED_WEIGHTS (1u << 0) /* W8A8 weights in [N/64][K/64][64][64] MMA tile-major */
#define C7M_FLAG_W4A8_LAYERS   (1u << 1) /* Decoder layer weights in group-64 row-pair W4A8 */
#define C7M_FLAG_W4A8_WCLS     (1u << 2) /* Output classifier (wcls) in group-64 row-pair W4A8 */
#define C7M_FLAG_HAS_QKV_BIAS  (1u << 3) /* Layers include float32 bq/bk/bv biases (Qwen2) */
#define C7M_FLAG_UNTIED_EMB    (1u << 4) /* Separate int8 token_embedding_table at off_emb_q/s */
#define C7M_FLAG_ROPE_NEOX     (1u << 5) /* Half-split NeoX RoPE (HF Llama/SmolLM/Qwen2/3) */
#define C7M_FLAG_HAS_QK_NORM   (1u << 6) /* Layers include per-head float32 q_norm/k_norm (Qwen3) */

/*
 * 256-byte (64B-aligned) header at offset 0 of a .c7m file.
 * All tensor offsets are byte offsets from the start of the .c7m buffer and
 * are 64-byte aligned.
 */
struct c7m_header {
	uint32_t magic;
	uint32_t version;
	uint32_t total_size;

	uint32_t dim;
	uint32_t dim_pad;
	uint32_t hidden_dim;
	uint32_t hidden_pad;
	uint32_t n_layers;
	uint32_t n_heads;
	uint32_t n_kv_heads;
	uint32_t head_size;
	uint32_t kv_dim;
	uint32_t vocab_size;
	uint32_t vocab_pad;
	uint32_t seq_len;

	uint32_t off_rope_cos;     /* float32[seq_len][head_size/2] */
	uint32_t off_rope_sin;     /* float32[seq_len][head_size/2] */
	uint32_t off_rms_final;    /* float32[dim] */
	uint32_t off_wcls_q;       /* W8A8 or W4A8 classifier weights */
	uint32_t off_wcls_s;       /* Classifier scales (per-row or row-pair group-64) */
	uint32_t off_layer_table;  /* struct c7m_layer_offs[n_layers] */

	uint32_t flags;            /* C7M_FLAG_* */
	uint32_t off_layers_base;  /* Start byte offset of contiguous layer blob */
	uint32_t layer_stride;     /* Contiguous byte size per layer (for MSMC staging) */

	/* Universal LLM fields (0 = v1 defaults for backward compatibility) */
	float    rms_norm_eps;     /* 0.0f -> default 1e-5f */
	uint32_t w4_group_size;    /* 64 when W4A8 flags are set */
	uint32_t off_emb_q;        /* Row-major int8[vocab_size][dim_pad] (if UNTIED or W4A8_WCLS) */
	uint32_t off_emb_s;        /* float32[vocab_size] */
	uint32_t off_bias_table;   /* struct c7m_layer_bias_offs[n_layers] (if HAS_QKV_BIAS) */
	uint32_t off_tokenizer;    /* Optional embedded tokenizer blob (0 = external) */
	uint32_t tokenizer_size;   /* Byte length of embedded tokenizer blob */
	uint32_t bos_token_id;     /* Model BOS token ID (0 -> default 1) */
	uint32_t eos_token_id;     /* Model EOS token ID (0 -> default 2) */
	uint32_t off_qknorm_table; /* struct c7m_layer_qknorm_offs[n_layers] (if HAS_QK_NORM) */

	uint32_t reserved[30];     /* Zero-padded to 256 bytes */
};

/*
 * 64-byte table entry per transformer decoder layer.
 * For W8A8: *_q is int8[out_pad][in_pad] (tiled if TILED_WEIGHTS), *_s is float32[out_pad].
 * For W4A8: *_q is uint8[out_pad/2][in_pad/64][64], *_s is float32[out_pad/2][in_pad/64][2].
 */
struct c7m_layer_offs {
	uint32_t rms_att;
	uint32_t rms_ffn;
	uint32_t wq_q;
	uint32_t wq_s;
	uint32_t wk_q;
	uint32_t wk_s;
	uint32_t wv_q;
	uint32_t wv_s;
	uint32_t wo_q;
	uint32_t wo_s;
	uint32_t w1_q;
	uint32_t w1_s;
	uint32_t w2_q;
	uint32_t w2_s;
	uint32_t w3_q;
	uint32_t w3_s;
};

/* Optional 16-byte bias offset entry per layer (when C7M_FLAG_HAS_QKV_BIAS is set). */
struct c7m_layer_bias_offs {
	uint32_t bq; /* float32[q_dim] (0 = none) */
	uint32_t bk; /* float32[kv_dim] (0 = none) */
	uint32_t bv; /* float32[kv_dim] (0 = none) */
	uint32_t reserved;
};

/* Optional 8-byte QK-RMSNorm offset entry per layer (when C7M_FLAG_HAS_QK_NORM is set). */
struct c7m_layer_qknorm_offs {
	uint32_t q_norm; /* float32[head_size] (0 = none) */
	uint32_t k_norm; /* float32[head_size] (0 = none) */
};

/*
 * Per-token cycle breakdown written by the C7x DSP after a forward pass.
 */
struct c7x_llama_prof {
	uint64_t cyc_total;
	uint64_t cyc_emb;
	uint64_t cyc_rmsnorm;
	uint64_t cyc_quant;
	uint64_t cyc_qkv_mma;
	uint64_t cyc_rope;
	uint64_t cyc_attn;
	uint64_t cyc_wo_mma;
	uint64_t cyc_ffn_up_mma;
	uint64_t cyc_swiglu;
	uint64_t cyc_ffn_down_mma;
	uint64_t cyc_cls_mma;
	uint64_t cyc_dequant;
	uint64_t cyc_layer[C7M_MAX_LAYERS];
};

/* Request flags for struct c7x_llama_req.flags */
#define C7X_LLAMA_REQ_KV_INT8    (1u << 0) /* K/V caches are per-head-row symmetric INT8 + scales */
#define C7X_LLAMA_REQ_ALL_LOGITS (1u << 1) /* Prefill: write logits for ALL n_tokens (speculative) */
#define C7X_LLAMA_REQ_ALL_ARGMAX (1u << 2) /* Prefill: write uint32_t argmax[n_tokens] into logits_addr */

/*
 * Request descriptor for C7X_OP_LLAMA_FORWARD / C7X_OP_LLAMA_PREFILL.
 */
struct c7x_llama_req {
	uint64_t model_addr;       /* Physical address of .c7m buffer in DDR */
	uint64_t k_cache_addr;     /* float32 or int8_t[n_layers][seq_len][kv_dim] */
	uint64_t v_cache_addr;     /* float32 or int8_t[n_layers][seq_len][kv_dim] */
	uint64_t logits_addr;      /* float32[vocab_size] (or [n_tokens][vocab_size] if ALL_LOGITS) */
	uint64_t prof_addr;        /* Optional physical address of struct c7x_llama_prof */
	uint64_t debug_x_addr;     /* Optional physical address of float32[dim] layer output */
	uint64_t msmc_layers_addr; /* Optional MSMC copy of layers 0..msmc_n_layers-1 */
	uint32_t msmc_n_layers;    /* Number of leading layers staged in MSMC SRAM */
	uint32_t token;            /* Input token ID (for single-token decode) */
	uint32_t pos;              /* Sequence position (or start pos for prefill) */
	int32_t  debug_layer;      /* Layer index to snapshot into debug_x_addr (-1 = final x) */
	uint32_t stop_after_layer; /* 0 = full model; >0 = stop after N layers */
	uint32_t n_tokens;         /* For prefill: number of prompt tokens in tokens_addr */
	uint64_t tokens_addr;      /* For prefill: uint32_t[n_tokens] in DDR */
	uint64_t prefill_buf_addr; /* For prefill: DDR workspace for batched activations */
	uint32_t prefill_buf_bytes;/* Byte size of prefill workspace */
	uint32_t flags;            /* C7X_LLAMA_REQ_* */
	uint64_t k_scale_addr;     /* float32[n_layers][seq_len][n_kv_heads] (if KV_INT8) */
	uint64_t v_scale_addr;     /* float32[n_layers][seq_len][n_kv_heads] (if KV_INT8) */
};

#endif /* C7X_LLAMA_FMT_H_ */
