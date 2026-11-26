/* SPDX-License-Identifier: MIT */
/*
 * Shared C7x Command-Buffer ABI (C7X_OP_EXEC_LIST).
 *
 * Included by:
 *   - C7x DSP firmware (cl7x -mv7100 -O3)
 *   - x86_64 C7000 Host Emulation (g++ -D__C7X_HOSTEM__)
 *   - Linux host runtime (libc7x) and frontends (ggml-c7x, future ORT EP)
 *
 * All structures use explicit fixed-width little-endian fields with natural
 * alignment and zero host/DSP pointer types (addresses are 64-bit physical /
 * arena addresses cast to pointers on the DSP).
 */
#ifndef C7X_CMD_PROTO_H_
#define C7X_CMD_PROTO_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C7X_CMD_ABI_VERSION 1u

/* Command opcodes executed sequentially inside C7X_OP_EXEC_LIST. */
enum c7x_cmd_op {
	C7X_CMD_NOP             = 0u,
	C7X_CMD_MATMUL_W4A8_G64 = 1u,  /* FP32 act x tile-major W4A8-g64 weights -> FP32 */
	C7X_CMD_MATMUL_W8A8     = 2u,  /* FP32 act x tile-major W8A8 weights -> FP32 */
	C7X_CMD_MATMUL_F32      = 3u,  /* General 4D batched FP32/F16 src0 x FP32 src1 -> FP32 */
	C7X_CMD_RMS_NORM_F32    = 4u,  /* Row-wise RMSNorm (optional fused weight multiply) */
	C7X_CMD_ADD_F32         = 5u,  /* 4D broadcasting elementwise dst = src0 + src1 */
	C7X_CMD_MUL_F32         = 6u,  /* 4D broadcasting elementwise dst = src0 * src1 */
	C7X_CMD_SCALE_F32       = 7u,  /* dst = src0 * f0 + f1 */
	C7X_CMD_UNARY_F32       = 8u,  /* Elementwise unary op (subop in p1) */
	C7X_CMD_GLU_F32         = 9u,  /* Fused SwiGLU / GeGLU / ReGLU (subop in p2) */
	C7X_CMD_SOFT_MAX_F32    = 10u, /* Row-wise softmax with scale, optional mask & ALiBi */
	C7X_CMD_ROPE_F32        = 11u, /* GGML-compatible NORMAL & NEOX RoPE (+ YaRN) */
	C7X_CMD_GET_ROWS        = 12u, /* Gather rows (F32/F16/Q8_0/Q4_0/W4A8/W8A8 -> F32) */
	C7X_CMD_CPY             = 13u, /* 4D strided tensor copy / cast (F32/F16/Q8_0/I32) */
	C7X_CMD_SET_ROWS        = 14u, /* Scatter rows into dst (F32/F16/Q8_0) by I64/I32 index */
	C7X_CMD_NORM_F32        = 15u, /* Row-wise LayerNorm (optional fused gamma & beta) */
	C7X_CMD_IM2COL          = 16u, /* 1D / 2D im2col (F32/F16 -> F32/F16) */
	C7X_CMD_POOL_2D_F32     = 17u, /* 2D AvgPool / MaxPool (F32/F16 -> F32) */
	C7X_CMD_CONV_2D_F32     = 18u, /* 2D Conv (F32/F16 kernel x F32 input + optional bias) */
	C7X_CMD_QLINEAR_MATMUL  = 19u, /* Asymmetric INT8/UINT8 MMA MatMul + zero-points + requant */
	C7X_CMD_QLINEAR_CONV2D  = 20u, /* Asymmetric INT8/UINT8 L2-fused im2col + MMA Conv2D */
	C7X_CMD_DEQUANTIZE_LIN  = 21u, /* INT8/UINT8 -> FP32 DequantizeLinear (tensor or per-axis) */
	C7X_CMD_FLASH_ATTN_EXT  = 22u, /* Fused FlashAttention-2 (Q F32 x K/V F16/F32/Q8_0 + mask + sinks) */
};

/* Common command flags */
#define C7X_CMD_FLAG_ADD        (1u << 0) /* Accumulate into dst (+=) for MATMUL ops */
#define C7X_CMD_FLAG_USE_DRU    (1u << 1) /* Enable DRU0 L2 ping-pong when available */
#define C7X_CMD_FLAG_SRC0_F16   (1u << 2) /* src0 is IEEE-754 FP16 (MATMUL_F32/POOL/CONV) */
#define C7X_CMD_FLAG_MASK_F16   (1u << 3) /* Softmax mask is IEEE-754 FP16 */
#define C7X_CMD_FLAG_GLU_SWAP   (1u << 4) /* GLU swapped halves (up first, gate second) */
#define C7X_CMD_FLAG_IDX_I64    (1u << 5) /* Row indices are int64_t (else int32_t) */
#define C7X_CMD_FLAG_RELU       (1u << 6) /* Fused ReLU activation (CONV_2D / QLINEAR_*) */
#define C7X_CMD_FLAG_A_UINT8    (1u << 7) /* QLinear input A is uint8 (shifted by -128 for MMA) */
#define C7X_CMD_FLAG_B_UINT8    (1u << 8) /* QLinear original weight B was uint8 */
#define C7X_CMD_FLAG_Y_UINT8    (1u << 9) /* QLinear output Y is uint8 (else int8) */
#define C7X_CMD_FLAG_Y_F32      (1u << 10) /* QLinear output Y is FP32 (dequantized) */
#define C7X_CMD_FLAG_BIAS_I32   (1u << 11) /* QLinear bias in src2 is int32 (else float32) */
#define C7X_CMD_FLAG_SRC0_Q8_0  (1u << 12) /* src0 is GGML Q8_0 blocks (34B / 32 elems) for MATMUL_F32 */

/* Pooling modes for C7X_CMD_POOL_2D_F32 */
enum c7x_pool_op {
	C7X_POOL_AVG          = 0u, /* Divide by k0 * k1 (GGML / ONNX count_include_pad=1) */
	C7X_POOL_MAX          = 1u, /* Max over valid in-bounds elements */
	C7X_POOL_AVG_EXCL_PAD = 2u, /* Divide by count of valid in-bounds elements (ONNX default) */
};

/* Element data types used by GET_ROWS / CPY / SET_ROWS */
enum c7x_dtype {
	C7X_DTYPE_F32      = 0u,
	C7X_DTYPE_F16      = 1u,
	C7X_DTYPE_Q8_0     = 2u,
	C7X_DTYPE_Q4_0     = 3u,
	C7X_DTYPE_I32      = 4u,
	C7X_DTYPE_W4A8_G64 = 5u, /* Tile-major [n_pad/64][k_pad/64][32][64] + scales */
	C7X_DTYPE_W8A8     = 6u, /* Tile-major [n_pad/64][k_pad/64][64][64] + scales */
};

/* Sub-opcodes for C7X_CMD_UNARY_F32 (in cmd.p1) */
enum c7x_unary_op {
	C7X_UNARY_SILU       = 0u,
	C7X_UNARY_GELU       = 1u,
	C7X_UNARY_RELU       = 2u,
	C7X_UNARY_SIGMOID    = 3u,
	C7X_UNARY_TANH       = 4u,
	C7X_UNARY_NEG        = 5u,
	C7X_UNARY_EXP        = 6u,
	C7X_UNARY_ABS        = 7u,
	C7X_UNARY_GELU_QUICK = 8u,
	C7X_UNARY_SQR        = 9u,
	C7X_UNARY_SQRT       = 10u,
};

/* Sub-opcodes for C7X_CMD_GLU_F32 (in cmd.p2) */
enum c7x_glu_op {
	C7X_GLU_SWIGLU     = 0u,
	C7X_GLU_GEGLU      = 1u,
	C7X_GLU_REGLU      = 2u,
	C7X_GLU_GEGLU_QUICK = 3u,
};

/* RoPE mode flags (matching GGML_ROPE_TYPE_*) */
#define C7X_ROPE_MODE_NORMAL 0u
#define C7X_ROPE_MODE_NEOX   2u

/*
 * Fixed 96-byte command descriptor.
 *
 * Layout:
 *   offset  0..31: op, flags, p0..p5 (8 x uint32_t)
 *   offset 32..71: dst, src0, src1, src2, src3 (5 x uint64_t)
 *   offset 72..79: aux_u64 (1 x uint64_t)
 *   offset 80..95: f0, f1, f2, f3 (4 x float)
 */
struct c7x_cmd {
	uint32_t op;
	uint32_t flags;
	uint32_t p0;
	uint32_t p1;
	uint32_t p2;
	uint32_t p3;
	uint32_t p4;
	uint32_t p5;

	uint64_t dst;
	uint64_t src0;
	uint64_t src1;
	uint64_t src2;
	uint64_t src3;
	uint64_t aux_u64;

	float    f0;
	float    f1;
	float    f2;
	float    f3;
};

/*
 * Auxiliary 4D tensor geometry descriptor (stored in shared DDR arena and
 * referenced by cmd.src3 for strided / broadcast ops).
 */
struct c7x_4d_binop_desc {
	uint32_t ne0[4];
	uint32_t ne1[4];
	uint32_t ne_dst[4];
	uint64_t nb0[4];
	uint64_t nb1[4];
	uint64_t nb_dst[4];
};

struct c7x_matmul_f32_desc {
	uint32_t ne0[4]; /* src0: [K, N, B02, B03] (GGML convention) */
	uint32_t ne1[4]; /* src1: [K, M, B12, B13] */
	uint32_t ne_dst[4]; /* dst: [N, M, B12, B13] */
	uint64_t nb0[4];
	uint64_t nb1[4];
	uint64_t nb_dst[4];
};

struct c7x_softmax_desc {
	uint32_t ne0;       /* row length (n_cols) */
	uint32_t ne1;       /* rows per head */
	uint32_t ne2;       /* heads */
	uint32_t ne3;       /* batch */
	uint32_t mask_ne1;  /* 0 if no mask */
	uint32_t mask_ne2;
	uint32_t mask_ne3;
	uint32_t n_head_log2;
	uint64_t nb01;      /* src0/dst row byte stride (dim 1) */
	uint64_t nb02;      /* src0/dst dim 2 byte stride */
	uint64_t nb03;      /* src0/dst dim 3 byte stride */
	uint64_t nb_dst1;
	uint64_t nb_dst2;
	uint64_t nb_dst3;
	uint64_t mask_nb1;
	uint64_t mask_nb2;
	uint64_t mask_nb3;
	float    scale;
	float    max_bias;
	float    m0;
	float    m1;
};

struct c7x_rope_desc {
	uint32_t ne0[4];    /* [head_dim, n_heads, n_tokens, n_batch] */
	uint64_t nb0[4];    /* src0 byte strides */
	uint64_t nb_dst[4]; /* dst byte strides */
	uint32_t n_dims;    /* number of rotated dimensions */
	uint32_t mode;      /* C7X_ROPE_MODE_NORMAL (0) or C7X_ROPE_MODE_NEOX (2) */
	uint32_t n_ctx_orig;
	uint32_t has_freq_factors;
	uint32_t truncate;  /* 1 = floor/ceil YaRN corr dims (GGML default), 0 = continuous */
	uint32_t reserved0;
	float    freq_base;
	float    freq_scale;
	float    ext_factor;
	float    attn_factor;
	float    beta_fast;
	float    beta_slow;
};

struct c7x_get_rows_desc {
	uint32_t ne00;      /* row length (elements) */
	uint32_t ne01;      /* total rows in table */
	uint32_t ne02;      /* batch dim 2 of table */
	uint32_t ne10;      /* indices dim 0 */
	uint32_t ne11;      /* indices dim 1 */
	uint32_t ne12;      /* indices dim 2 */
	uint32_t k_pad;     /* padded K for W4A8_G64 / W8A8 tiled tables */
	uint32_t n_pad;     /* padded N for W4A8_G64 / W8A8 tiled tables */
	uint64_t nb01;      /* table row byte stride */
	uint64_t nb02;
	uint64_t nb03;
	uint64_t nb10;      /* indices byte strides */
	uint64_t nb11;
	uint64_t nb12;
	uint64_t nb_dst1;   /* dst byte strides */
	uint64_t nb_dst2;
	uint64_t nb_dst3;
};

struct c7x_cpy_desc {
	uint32_t ne0[4];    /* source logical shape */
	uint32_t ne_dst[4]; /* destination logical shape (same total nelements) */
	uint64_t nb0[4];    /* source byte strides */
	uint64_t nb_dst[4]; /* destination byte strides */
};

struct c7x_set_rows_desc {
	uint32_t ne0[4];    /* src0 shape [row_len, n_rows, n_heads, n_batch] */
	uint32_t ne1[4];    /* src1 (indices) shape [n_rows, idx_ne1, idx_ne2, 1] */
	uint32_t ne_dst[4]; /* dst shape [row_len, max_rows, n_heads, n_batch] (for bounds check) */
	uint32_t reserved[4];
	uint64_t nb0[4];    /* src0 byte strides */
	uint64_t nb1[4];    /* src1 byte strides */
	uint64_t nb_dst[4]; /* dst byte strides */
};

struct c7x_norm_desc {
	uint32_t ne0[4];    /* [row_len, d1, d2, d3] */
	uint64_t nb0[4];    /* src0 byte strides */
	uint64_t nb_dst[4]; /* dst byte strides */
	float    eps;
	uint32_t has_gamma; /* cmd->src1 is float[ne0[0]] */
	uint32_t has_beta;  /* cmd->src2 is float[ne0[0]] */
	uint32_t reserved0;
};

struct c7x_im2col_desc {
	uint32_t N;
	uint32_t IC;
	uint32_t IH;
	uint32_t IW;
	uint32_t KH;
	uint32_t KW;
	uint32_t OH;
	uint32_t OW;
	int32_t  s0;        /* stride along W */
	int32_t  s1;        /* stride along H */
	int32_t  p0;        /* pad along W */
	int32_t  p1;        /* pad along H */
	int32_t  d0;        /* dilation along W */
	int32_t  d1;        /* dilation along H */
	uint32_t is_2d;     /* 1 = 2D, 0 = 1D (IH=KH=OH=1) */
	uint32_t reserved0;
	uint64_t src_nb1;   /* row stride along H (bytes) */
	uint64_t src_nb_c;  /* channel stride along IC (bytes) */
	uint64_t src_nb_n;  /* batch stride along N (bytes) */
};

struct c7x_pool2d_desc {
	uint32_t op;        /* enum c7x_pool_op */
	uint32_t IW;
	uint32_t IH;
	uint32_t OW;
	uint32_t OH;
	uint32_t n_planes;  /* C * N */
	int32_t  k0;        /* kernel W */
	int32_t  k1;        /* kernel H */
	int32_t  s0;        /* stride W */
	int32_t  s1;        /* stride H */
	int32_t  p0;        /* pad W */
	int32_t  p1;        /* pad H */
	uint64_t src_nb1;   /* src row stride (bytes) */
	uint64_t src_nb2;   /* src plane stride (bytes) */
	uint64_t dst_nb1;   /* dst row stride (bytes) */
	uint64_t dst_nb2;   /* dst plane stride (bytes) */
};

struct c7x_conv2d_f32_desc {
	uint32_t N;
	uint32_t IC;
	uint32_t IH;
	uint32_t IW;
	uint32_t OC;
	uint32_t KH;
	uint32_t KW;
	uint32_t OH;
	uint32_t OW;
	uint32_t has_bias;  /* cmd->src2 is float[OC] */
	int32_t  s0;        /* stride W */
	int32_t  s1;        /* stride H */
	int32_t  p0;        /* pad W */
	int32_t  p1;        /* pad H */
	int32_t  d0;        /* dilation W */
	int32_t  d1;        /* dilation H */
	uint64_t src_nb[4]; /* [nb_w, nb_h, nb_c, nb_n] */
	uint64_t dst_nb[4]; /* [nb_ow, nb_oh, nb_oc, nb_n] */
};

struct c7x_qlinear_matmul_desc {
	uint32_t M;
	uint32_t K;
	uint32_t N;
	uint32_t k_pad;
	uint32_t n_pad;
	int32_t  a_zp;      /* signed zero-point (-128..127; if A_UINT8, already shifted by -128) */
	int32_t  b_zp;      /* signed zero-point (-128..127; if B_UINT8, already shifted by -128) */
	int32_t  y_zp;      /* output zero-point in output space (-128..127 or 0..255) */
	float    a_scale;
	float    y_scale;
	uint32_t per_channel_b; /* 0 = scalar b_scales[0], 1 = per-row b_scales[n] */
	uint32_t reserved0;
	uint64_t b_sum_addr;    /* const int32_t[n_pad]: sum_k b_s[n,k] */
	uint64_t b_scales_addr; /* const float[per_channel_b ? n_pad : 1] */
};

struct c7x_qlinear_conv2d_desc {
	uint32_t N;
	uint32_t IC;
	uint32_t IH;
	uint32_t IW;
	uint32_t OC;
	uint32_t KH;
	uint32_t KW;
	uint32_t OH;
	uint32_t OW;
	uint32_t groups;        /* 1 = standard MMA conv; IC==OC==groups = depthwise conv */
	int32_t  s0;
	int32_t  s1;
	int32_t  p0;
	int32_t  p1;
	int32_t  d0;
	int32_t  d1;
	uint32_t k_pad;         /* align64(IC * KH * KW) for groups == 1 */
	uint32_t n_pad;         /* align64(OC) for groups == 1 */
	int32_t  a_zp;
	int32_t  b_zp;
	int32_t  y_zp;
	uint32_t per_channel_b;
	float    a_scale;
	float    y_scale;
	uint64_t b_sum_addr;    /* const int32_t[n_pad or OC] */
	uint64_t b_scales_addr; /* const float[per_channel_b ? (n_pad or OC) : 1] */
};

struct c7x_dequant_desc {
	uint32_t outer;         /* product of dims before quant axis (1 for per-tensor) */
	uint32_t axis_dim;      /* size of quant axis (1 for per-tensor) */
	uint32_t inner;         /* product of dims after quant axis (n_elems for per-tensor) */
	uint32_t has_zp;        /* 1 if src2 holds zero_point[axis_dim] (else 0) */
	int32_t  scalar_zp;     /* used when axis_dim == 1 */
	float    scalar_scale;  /* used when axis_dim == 1 */
};

struct c7x_flash_attn_desc {
	uint32_t neq[4];        /* Q: [DK, N, n_head, b3] */
	uint32_t nek[4];        /* K: [DK, n_kv, n_head_k, b3_k] */
	uint32_t nev[4];        /* V: [DV, n_kv, n_head_v, b3_v] */
	uint32_t nem[4];        /* mask: [n_kv, N_m, ne_m2, ne_m3] */
	uint32_t k_dtype;       /* C7X_DTYPE_F32 / F16 / Q8_0 */
	uint32_t v_dtype;       /* C7X_DTYPE_F32 / F16 / Q8_0 */
	uint32_t mask_dtype;    /* C7X_DTYPE_F16 / F32 */
	uint32_t n_head_log2;   /* 1u << floor(log2(n_head)) for ALiBi */
	float    scale;
	float    max_bias;
	float    logit_softcap;
	float    m0;
	float    m1;
	uint32_t reserved0;
	uint64_t nbq[4];
	uint64_t nbk[4];
	uint64_t nbv[4];
	uint64_t nbm[4];
	uint64_t nb_dst[4];     /* dst: [DV, n_head, N, b3] */
	uint64_t mask_addr;
	uint64_t sinks_addr;
};

typedef char c7x_cmd_size_assert_[sizeof(struct c7x_cmd) == 96u ? 1 : -1];
typedef char c7x_binop_desc_size_assert_[sizeof(struct c7x_4d_binop_desc) == 144u ? 1 : -1];
typedef char c7x_mm_desc_size_assert_[sizeof(struct c7x_matmul_f32_desc) == 144u ? 1 : -1];
typedef char c7x_sm_desc_size_assert_[sizeof(struct c7x_softmax_desc) == 120u ? 1 : -1];
typedef char c7x_rope_desc_size_assert_[sizeof(struct c7x_rope_desc) == 128u ? 1 : -1];
typedef char c7x_get_rows_desc_size_assert_[sizeof(struct c7x_get_rows_desc) == 104u ? 1 : -1];
typedef char c7x_cpy_desc_size_assert_[sizeof(struct c7x_cpy_desc) == 96u ? 1 : -1];
typedef char c7x_set_rows_desc_size_assert_[sizeof(struct c7x_set_rows_desc) == 160u ? 1 : -1];
typedef char c7x_norm_desc_size_assert_[sizeof(struct c7x_norm_desc) == 96u ? 1 : -1];
typedef char c7x_im2col_desc_size_assert_[sizeof(struct c7x_im2col_desc) == 88u ? 1 : -1];
typedef char c7x_pool2d_desc_size_assert_[sizeof(struct c7x_pool2d_desc) == 80u ? 1 : -1];
typedef char c7x_conv2d_desc_size_assert_[sizeof(struct c7x_conv2d_f32_desc) == 128u ? 1 : -1];
typedef char c7x_qmm_desc_size_assert_[sizeof(struct c7x_qlinear_matmul_desc) == 64u ? 1 : -1];
typedef char c7x_qconv_desc_size_assert_[sizeof(struct c7x_qlinear_conv2d_desc) == 112u ? 1 : -1];
typedef char c7x_dequant_desc_size_assert_[sizeof(struct c7x_dequant_desc) == 24u ? 1 : -1];
typedef char c7x_fa_desc_size_assert_[sizeof(struct c7x_flash_attn_desc) == 280u ? 1 : -1];

#ifdef __cplusplus
}
#endif

#endif /* C7X_CMD_PROTO_H_ */
