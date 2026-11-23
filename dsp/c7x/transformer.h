/* SPDX-License-Identifier: MIT */
/*
 * C7x + MMA W8A8 Transformer (llama2.c / stories15M) inference engine.
 *
 * Compiles identically for TI C7000 CGT (cl7x) and x86_64 C7000 Host Emulation.
 */
#ifndef DSP_C7X_TRANSFORMER_H_
#define DSP_C7X_TRANSFORMER_H_

#include <stdint.h>

#include "c7x_llama_fmt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimum L2/heap scratch bytes required for stories15M-class models (<= 32000 vocab). */
#define C7X_LLAMA_MIN_SCRATCH_BYTES (256u * 1024u)

/* Validate a loaded .c7m header. Returns 0 on valid, negative error code otherwise. */
int c7x_llama_validate(const struct c7m_header *hdr);

/*
 * Run a single token forward pass through the quantized transformer using:
 *   - C7x MMA (c7x_mma_gemm_i8_i32) for all linear projections (wq/wk/wv/wo/w1/w2/w3/wcls)
 *   - C7x 512-bit FP32 vector unit for RMSNorm, RoPE, MHA, SwiGLU, quant/dequant, residuals
 *
 * Parameters:
 *   hdr:              Loaded .c7m model header (followed by tensors in contiguous memory)
 *   token:            Input token ID (0..vocab_size-1)
 *   pos:              Sequence position (0..seq_len-1)
 *   k_cache:          float32[n_layers][seq_len][kv_dim]
 *   v_cache:          float32[n_layers][seq_len][kv_dim]
 *   logits_out:       float32[vocab_size] (ignored if stop_after_layer > 0)
 *   scratch:          64-byte aligned scratchpad (>= C7X_LLAMA_MIN_SCRATCH_BYTES; L2 SRAM on C7x)
 *   scratch_bytes:    Size of scratch buffer in bytes
 *   prof:             Optional profiling output (may be NULL)
 *   debug_x_out:      Optional float32[dim] snapshot of layer `debug_layer` (or final norm if -1)
 *   debug_layer:      Layer index to snapshot (0..n_layers-1), or -1 for final RMSNorm output
 *   stop_after_layer: 0 = full model + logits; N (1..n_layers) = stop after N layers
 *
 * Returns 0 on success, negative error code on invalid parameters.
 */
int c7x_llama_forward_token(const struct c7m_header *hdr,
			    uint32_t token, uint32_t pos,
			    void *k_cache, void *v_cache,
			    float *k_scale, float *v_scale, uint32_t req_flags,
			    float *logits_out, uint32_t *argmax_out,
			    void *scratch, uint32_t scratch_bytes,
			    const void *msmc_layers, uint32_t msmc_n_layers,
			    struct c7x_llama_prof *prof,
			    float *debug_x_out, int32_t debug_layer,
			    uint32_t stop_after_layer);

/*
 * Batched multi-token MMA prefill:
 *   Processes tokens[0..n_tokens-1] at positions start_pos..start_pos+n_tokens-1
 *   in chunks of up to 240 tokens (B-stationary W4A8 reuse), running the classifier
 *   for the last token (or all tokens if C7X_LLAMA_REQ_ALL_LOGITS is set).
 */
int c7x_llama_prefill_tokens(const struct c7m_header *hdr,
			     const uint32_t *tokens, uint32_t n_tokens, uint32_t start_pos,
			     void *k_cache, void *v_cache,
			     float *k_scale, float *v_scale, uint32_t req_flags,
			     float *logits_out, uint32_t *argmax_out,
			     void *l2_scratch, uint32_t l2_bytes,
			     void *prefill_buf, uint32_t prefill_bytes,
			     const void *msmc_layers, uint32_t msmc_n_layers,
			     struct c7x_llama_prof *prof);

#ifdef __cplusplus
}
#endif

#endif /* DSP_C7X_TRANSFORMER_H_ */
