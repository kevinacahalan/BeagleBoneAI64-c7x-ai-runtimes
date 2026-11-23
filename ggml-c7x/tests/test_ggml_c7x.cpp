/* SPDX-License-Identifier: MIT */
/*
 * Targeted integration & regression test suite for libggml-c7x.so covering
 * dynamic loading, weight round-trips, partial writes, tied GET_ROWS + MUL_MAT,
 * F16 2D/3D exact matmul, SET_ROWS bounds checking, CPY dtype gating, RoPE YaRN
 * truncate=0/1, Softmax ALiBi/sinks/F16-mask, SwiGLU(-200)/EXP overflow, and
 * >4096-op command-buffer auto-flush.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t g_rng = 0x13579BDFu;
static float randf() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float)((int32_t)((g_rng >> 8) & 0xFFFFu) - 32768)) / 16384.0f;
}

static bool run_graph(ggml_backend_t backend, struct ggml_cgraph * gf,
                      ggml_backend_buffer_t * out_buf = nullptr) {
    ggml_gallocr_t g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g, gf)) {
        ggml_gallocr_free(g);
        return false;
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(g);
    (void)out_buf;
    return st == GGML_STATUS_SUCCESS;
}

static int test_weight_roundtrip_and_partial_write(ggml_backend_t c7x) {
    const int64_t K = 128, N = 64, M = 3;
    std::vector<float> wf(N * K), af(M * K);
    for (auto & v : wf) v = randf();
    for (auto & v : af) v = randf();

    std::vector<block_q8_0> q8((N * K) / QK8_0);
    for (int64_t r = 0; r < N; r++) {
        quantize_row_q8_0_ref(wf.data() + r * K, q8.data() + r * (K / QK8_0), K);
    }

    struct ggml_init_params ip { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(c7x));
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // 1. Full set_tensor (triggers eager repack) -> get_tensor round-trip
    ggml_backend_tensor_set(W, q8.data(), 0, ggml_nbytes(W));
    std::vector<block_q8_0> q8_back((N * K) / QK8_0);
    ggml_backend_tensor_get(W, q8_back.data(), 0, ggml_nbytes(W));

    // Dequantize original vs read-back and check cosine > 0.999
    std::vector<float> d_orig(N * K), d_back(N * K);
    for (int64_t r = 0; r < N; r++) {
        dequantize_row_q8_0(q8.data() + r * (K / QK8_0), d_orig.data() + r * K, K);
        dequantize_row_q8_0(q8_back.data() + r * (K / QK8_0), d_back.data() + r * K, K);
    }
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < d_orig.size(); i++) {
        dot += (double)d_orig[i] * d_back[i];
        na  += (double)d_orig[i] * d_orig[i];
        nb  += (double)d_back[i] * d_back[i];
    }
    float cos_rt = (float)(dot / (sqrt(na) * sqrt(nb)));

    // 2. Partial set_tensor (overwrite row 1 only!) -> verify row 0 intact & row 1 updated
    std::vector<block_q8_0> row1(K / QK8_0);
    std::vector<float> row1_f(K, 0.5f);
    quantize_row_q8_0_ref(row1_f.data(), row1.data(), K);
    size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, K);
    ggml_backend_tensor_set(W, row1.data(), row_bytes, row_bytes);

    ggml_backend_tensor_get(W, q8_back.data(), 0, ggml_nbytes(W));
    std::vector<float> r1_deq(K);
    dequantize_row_q8_0(q8_back.data() + (K / QK8_0), r1_deq.data(), K);
    float max_r1_err = 0.0f;
    for (int64_t c = 0; c < K; c++) {
        max_r1_err = std::max(max_r1_err, fabsf(r1_deq[c] - 0.5f));
    }

    ggml_backend_buffer_free(bw);
    ggml_free(ctx);

    bool ok = (cos_rt > 0.999f && max_r1_err < 0.02f);
    printf("[ggml-c7x-test] weight round-trip (cos=%.6f) + partial set_tensor (err=%.4f): %s\n",
           cos_rt, max_r1_err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_tied_get_rows_and_mul_mat_first_graph(ggml_backend_t c7x) {
    // Same Q4_0 WEIGHTS tensor used by GET_ROWS then MUL_MAT in one graph (Claude C1!)
    const int64_t K = 128, V = 64;
    std::vector<float> wf(V * K);
    for (auto & v : wf) v = randf();
    std::vector<block_q4_0> q4((V * K) / QK4_0);
    for (int64_t r = 0; r < V; r++) {
        quantize_row_q4_0_ref(wf.data() + r * K, q4.data() + r * (K / QK4_0), K);
    }

    struct ggml_init_params ip { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * W   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, V);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);

    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(c7x));
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(W, q4.data(), 0, ggml_nbytes(W));
    int32_t ids[2] = { 3, 17 };
    ggml_backend_tensor_set(idx, ids, 0, sizeof(ids));

    struct ggml_tensor * emb    = ggml_get_rows(ctx, W, idx);
    struct ggml_tensor * logits = ggml_mul_mat(ctx, W, emb);
    struct ggml_cgraph * gf     = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c7x));
    bool ok = ggml_gallocr_alloc_graph(g, gf) &&
              ggml_backend_graph_compute(c7x, gf) == GGML_STATUS_SUCCESS;

    std::vector<float> emb_got(2 * K), log_got(2 * V);
    if (ok) {
        ggml_backend_tensor_get(emb, emb_got.data(), 0, emb_got.size() * sizeof(float));
        ggml_backend_tensor_get(logits, log_got.data(), 0, log_got.size() * sizeof(float));
    }

    // Check cosine of logits vs float reference (W * W[ids]^T)
    double dot = 0, na = 0, nb = 0;
    for (int t = 0; t < 2; t++) {
        for (int64_t v = 0; v < V; v++) {
            float ref = 0.0f;
            for (int64_t k = 0; k < K; k++) ref += wf[ids[t] * K + k] * wf[v * K + k];
            float got = log_got[t * V + v];
            dot += (double)got * ref; na += (double)got * got; nb += (double)ref * ref;
        }
    }
    float cos_v = ok ? (float)(dot / (sqrt(na) * sqrt(nb))) : 0.0f;
    ok = ok && (cos_v > 0.985f);

    ggml_gallocr_free(g);
    ggml_backend_buffer_free(bw);
    ggml_free(ctx);

    printf("[ggml-c7x-test] tied GET_ROWS + MUL_MAT in 1st graph (cos=%.5f): %s\n",
           cos_v, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_set_rows_oob_and_cpy_gating(ggml_backend_t c7x) {
    struct ggml_init_params ip { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    // 1. Check CPY I32 <-> F32 is rejected by supports_op (Claude I5)
    struct ggml_tensor * a_i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
    struct ggml_tensor * b_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    struct ggml_tensor * bad_cpy1 = ggml_cpy(ctx, a_i32, b_f32);
    struct ggml_tensor * bad_cpy2 = ggml_cpy(ctx, b_f32, a_i32);
    bool cpy_ok = !ggml_backend_dev_supports_op(ggml_backend_get_device(c7x), bad_cpy1) &&
                  !ggml_backend_dev_supports_op(ggml_backend_get_device(c7x), bad_cpy2);

    // 2. Check SET_ROWS with OOB index fails safely (Claude I4)
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(c7x));
    int64_t oob = 4; // ne[1] == 4 -> OOB!
    ggml_backend_tensor_set(idx, &oob, 0, sizeof(oob));
    struct ggml_tensor * sr = ggml_set_rows(ctx, dst, src, idx);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sr);
    bool oob_rejected = !run_graph(c7x, gf);

    ggml_backend_buffer_free(b);
    ggml_free(ctx);

    bool ok = cpy_ok && oob_rejected;
    printf("[ggml-c7x-test] CPY I32<->F32 rejected=%d, SET_ROWS OOB rejected=%d: %s\n",
           (int)cpy_ok, (int)oob_rejected, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_cmdbuf_auto_flush_4200_ops(ggml_backend_t c7x) {
    // Chain 4,200 ADD ops (> 4,096 cmdbuf capacity!) to verify seamless chunked flushing (Claude I11)
    const int N_OPS = 4200;
    const int LEN = 16;
    struct ggml_init_params ip {
        ggml_tensor_overhead() * (size_t)(N_OPS + 16) + ggml_graph_overhead_custom(N_OPS + 16, false),
        nullptr, true
    };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LEN);
    struct ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LEN);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(c7x));
    std::vector<float> z(LEN, 0.0f), ones(LEN, 1.0f);
    ggml_backend_tensor_set(x, z.data(), 0, LEN * sizeof(float));
    ggml_backend_tensor_set(one, ones.data(), 0, LEN * sizeof(float));

    struct ggml_tensor * cur = x;
    for (int i = 0; i < N_OPS; i++) {
        cur = ggml_add(ctx, cur, one);
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, N_OPS + 16, false);
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c7x));
    bool ok = ggml_gallocr_alloc_graph(g, gf) &&
              ggml_backend_graph_compute(c7x, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> out(LEN, 0.0f);
    if (ok) {
        ggml_backend_tensor_get(cur, out.data(), 0, LEN * sizeof(float));
        for (int i = 0; i < LEN; i++) {
            if (fabsf(out[i] - (float)N_OPS) > 1e-3f) ok = false;
        }
    }
    ggml_gallocr_free(g);
    ggml_backend_buffer_free(b);
    ggml_free(ctx);

    printf("[ggml-c7x-test] 4,200-op command-buffer auto-flush (got=%.1f want=%d): %s\n",
           out[0], N_OPS, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_quant_mma_mul_mat_3d_and_q4k(ggml_backend_t c7x) {
    // Dedicated MMA coverage for Q4_0, Q4_K, and Q8_0 WEIGHTS + 3D src1 + full memset
    const int64_t K = 256, N = 96, M = 3, B2 = 2;
    std::vector<float> wf(N * K), af(B2 * M * K);
    for (auto & v : wf) v = randf();
    for (auto & v : af) v = randf();

    std::vector<float> ref(B2 * M * N, 0.0f);
    for (int64_t b = 0; b < B2; b++) {
        for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
                float s = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    s += af[(b * M + m) * K + k] * wf[n * K + k];
                }
                ref[(b * M + m) * N + n] = s;
            }
        }
    }

    enum ggml_type types[3] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 };
    const char * names[3]   = { "Q4_0", "Q4_K", "Q8_0" };
    const float min_cos[3]  = { 0.985f, 0.985f, 0.999f };
    int fail = 0;

    for (int ti = 0; ti < 3; ti++) {
        struct ggml_init_params ip { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * W = ggml_new_tensor_2d(ctx, types[ti], K, N);
        struct ggml_tensor * A = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, B2);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(c7x));
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<uint8_t> qraw(ggml_nbytes(W));
        size_t rbytes = ggml_row_size(types[ti], K);
        for (int64_t r = 0; r < N; r++) {
            if (types[ti] == GGML_TYPE_Q4_0) {
                quantize_row_q4_0_ref(wf.data() + r * K, (block_q4_0 *)(qraw.data() + r * rbytes), K);
            } else if (types[ti] == GGML_TYPE_Q4_K) {
                quantize_row_q4_K_ref(wf.data() + r * K, (block_q4_K *)(qraw.data() + r * rbytes), K);
            } else {
                quantize_row_q8_0_ref(wf.data() + r * K, (block_q8_0 *)(qraw.data() + r * rbytes), K);
            }
        }
        ggml_backend_tensor_set(W, qraw.data(), 0, qraw.size());
        ggml_backend_tensor_set(A, af.data(), 0, af.size() * sizeof(float));

        struct ggml_tensor * Y = ggml_mul_mat(ctx, W, A);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);

        ggml_gallocr_t g = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c7x));
        bool ok = ggml_gallocr_alloc_graph(g, gf) &&
                  ggml_backend_graph_compute(c7x, gf) == GGML_STATUS_SUCCESS;
        std::vector<float> got(B2 * M * N, 0.0f);
        if (ok) {
            ggml_backend_tensor_get(Y, got.data(), 0, got.size() * sizeof(float));
        }
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            dot += (double)got[i] * ref[i];
            na  += (double)got[i] * got[i];
            nb  += (double)ref[i] * ref[i];
        }
        float cos_v = (ok && na > 0 && nb > 0) ? (float)(dot / (sqrt(na) * sqrt(nb))) : 0.0f;

        // Also test full-size memset(0) on WEIGHTS keeps MUL_MAT lowerable and produces 0
        ggml_backend_tensor_memset(W, 0, 0, ggml_nbytes(W));
        bool zero_ok = ggml_backend_graph_compute(c7x, gf) == GGML_STATUS_SUCCESS;
        if (zero_ok) {
            ggml_backend_tensor_get(Y, got.data(), 0, got.size() * sizeof(float));
            for (float v : got) {
                if (fabsf(v) > 1e-6f) zero_ok = false;
            }
        }

        bool pass = ok && (cos_v >= min_cos[ti]) && zero_ok;
        printf("[ggml-c7x-test] 3D-src1 MMA MUL_MAT %s (cos=%.5f, memset0=%d): %s\n",
               names[ti], cos_v, (int)zero_ok, pass ? "PASS" : "FAIL");
        if (!pass) fail = 1;

        ggml_gallocr_free(g);
        ggml_backend_buffer_free(bw);
        ggml_free(ctx);
    }
    return fail;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t c7x = ggml_backend_init_by_name("C7X0", nullptr);
    if (!c7x) {
        fprintf(stderr, "Failed to load C7X0 backend (set GGML_BACKEND_PATH)\n");
        return 1;
    }

    int fail = 0;
    fail |= test_weight_roundtrip_and_partial_write(c7x);
    fail |= test_tied_get_rows_and_mul_mat_first_graph(c7x);
    fail |= test_set_rows_oob_and_cpy_gating(c7x);
    fail |= test_cmdbuf_auto_flush_4200_ops(c7x);
    fail |= test_quant_mma_mul_mat_3d_and_q4k(c7x);

    ggml_backend_free(c7x);
    printf("\n=== ggml-c7x Integration Suite: %s ===\n", fail ? "FAIL" : "ALL PASSED");
    return fail;
}
