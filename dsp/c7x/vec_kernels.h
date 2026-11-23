/* SPDX-License-Identifier: MIT */
/*
 * C7x 512-bit FP32 vector-unit kernels for transformer inference.
 *
 * Compiles identically for TI C7000 CGT (cl7x) and x86_64 C7000 Host Emulation.
 */
#ifndef DSP_C7X_VEC_KERNELS_H_
#define DSP_C7X_VEC_KERNELS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps) */
void c7x_rmsnorm_f32(float *out, const float *x, const float *weight,
		     uint32_t n, float eps);

/* Per-head in-place RMSNorm over vec[n_heads][head_size] using weight[head_size] (Qwen3 QK-norm). */
void c7x_rmsnorm_heads_f32(float *vec, const float *weight,
			   uint32_t n_heads, uint32_t head_size, float eps);

/* Quantize one KV position [n_kv_heads][head_size] into symmetric per-head INT8 + float32 scales. */
void c7x_kv_quant_store_i8(int8_t *dst_q, float *dst_s, const float *src,
			   uint32_t n_kv_heads, uint32_t head_size);

/*
 * Dynamic per-vector symmetric INT8 quantization:
 *   scale = max(|x[0..n-1]|) / 127.0f
 *   out_q[i] = clamp(rint(x[i] / scale), -127, 127) for i < n
 *   out_q[i] = 0 for n <= i < n_pad
 */
void c7x_quantize_f32_i8(int8_t *out_q, const float *x,
			 uint32_t n, uint32_t n_pad, float *scale_out);

/* out[i] = (float)y_i32[i] * (w_scale[i] * x_scale) */
void c7x_dequant_i32_f32(float *out, const int32_t *y_i32,
			 const float *w_scale, float x_scale, uint32_t n);

/* acc[i] += (float)y_i32[i] * (w_scale[i] * x_scale) */
void c7x_dequant_add_i32_f32(float *acc, const int32_t *y_i32,
			     const float *w_scale, float x_scale, uint32_t n);

/*
 * Dequantize y_i32[0..n-1] with (w_scale[i] * x_scale) and return argmax index.
 * If out != NULL, also stores all n dequantized floats to out[0..n-1].
 */
uint32_t c7x_dequant_argmax_i32_f32(float *out, const int32_t *y_i32,
				    const float *w_scale, float x_scale, uint32_t n);

/* out[i] = (float)row_q[i] * row_scale (embedding row lookup) */
void c7x_dequant_i8_f32(float *out, const int8_t *row_q, float row_scale, uint32_t n);

/* x[i] += bias[i] (for models with QKV projection biases such as Qwen2/2.5). */
void c7x_add_bias_f32(float *x, const float *bias, uint32_t n);

/*
 * In-place RoPE on q[n_heads * head_size] and k[n_kv_heads * head_size].
 *   neox == 0: interleaved pairs (2*i, 2*i+1) [llama2.c]
 *   neox != 0: half-split pairs (i, i + head_size/2) [HuggingFace Llama/SmolLM/Qwen2]
 */
void c7x_rope_f32(float *q, float *k, const float *cos_row, const float *sin_row,
		  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_size, int neox);

/* In-place numerically stable softmax over x[0..n-1]. */
void c7x_softmax_f32(float *x, uint32_t n);

/* out[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i] */
void c7x_swiglu_f32(float *out, const float *hb, const float *hb2, uint32_t n);

/*
 * Causal multi-head attention for a single decode token at position `pos`:
 *   q:             float32[n_heads * head_size]
 *   k_cache_layer: float32[seq_len][kv_dim] (entries 0..pos already populated)
 *   v_cache_layer: float32[seq_len][kv_dim] (entries 0..pos already populated)
 *   att_buf:       float32[pos + 1] scratch
 *   out:           float32[n_heads * head_size]
 */
void c7x_attn_mha_f32(float *out, const float *q,
		      const float *k_cache_layer, const float *v_cache_layer,
		      float *att_buf, uint32_t pos,
		      uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_size);

/*
 * Causal multi-head attention against per-head-row symmetric INT8 KV cache:
 *   k_q_layer, v_q_layer: int8_t[seq_len][kv_dim]
 *   k_s_layer, v_s_layer: float32[seq_len][n_kv_heads]
 */
void c7x_attn_mha_i8_f32(float *out, const float *q,
			 const int8_t *k_q_layer, const float *k_s_layer,
			 const int8_t *v_q_layer, const float *v_s_layer,
			 float *att_buf, uint32_t pos,
			 uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_size);

/* Run built-in self-test of all vector kernels against scalar references. Returns 0 on pass. */
int c7x_vec_selftest(float *max_err_out);

#ifdef __cplusplus
}
#endif

#endif /* DSP_C7X_VEC_KERNELS_H_ */
