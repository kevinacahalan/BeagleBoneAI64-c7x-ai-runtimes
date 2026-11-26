/* SPDX-License-Identifier: MIT */
/*
 * ggml-c7x — Out-of-tree GGML / llama.cpp dynamic backend plugin for the
 * TI C71x DSP + MMA on BeagleBone AI-64 (TDA4VM / J721E) and x86_64 C7000
 * Host Emulation.
 *
 * Load into any GGML_BACKEND_DL=ON build of llama.cpp via:
 *   GGML_BACKEND_PATH=/path/to/libggml-c7x.so llama-cli -m model.gguf ...
 */

#include "ggml-c7x.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "c7x_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Weight repacking metadata (repacked eagerly at set_tensor for WEIGHTS)
// ---------------------------------------------------------------------------

enum c7x_weight_fmt {
    C7X_WFMT_RAW      = 0,
    C7X_WFMT_W4A8_G64 = 1,
    C7X_WFMT_W8A8     = 2,
};

struct c7x_repacked_meta {
    c7x_weight_fmt fmt;
    enum ggml_type type;
    int64_t        ne[4];
    size_t         nb[4];
    uint32_t       n;
    uint32_t       k;
    size_t         scales_off;
    size_t         packed_sz;
    uint64_t       w_phys;
    uint64_t       s_phys;
    c7x_buf_t    * msmc_buf; // Optional MSMC hot-weight mirror (owned by buffer)
};

struct ggml_backend_c7x_device_context {
    c7x_ctx_t       * rt = nullptr;
    c7x_rt_info       info {};
    std::string       name;
    std::string       desc;
    std::mutex        repack_mtx;
    bool              enable_msmc_pin = true;
};

struct ggml_backend_c7x_buffer_context {
    ggml_backend_c7x_device_context * dev_ctx = nullptr;
    c7x_buf_t                       * buf     = nullptr;
    std::unordered_map<const void *, c7x_repacked_meta> repacked;

    ~ggml_backend_c7x_buffer_context() {
        if (dev_ctx && dev_ctx->rt) {
            std::lock_guard<std::mutex> lk(dev_ctx->repack_mtx);
            for (auto & kv : repacked) {
                if (kv.second.msmc_buf) {
                    c7x_free(dev_ctx->rt, kv.second.msmc_buf);
                }
            }
            repacked.clear();
            if (buf) {
                c7x_free(dev_ctx->rt, buf);
            }
        }
    }
};

struct ggml_backend_c7x_context {
    ggml_backend_c7x_device_context * dev_ctx = nullptr;
    c7x_cmdbuf_t                    * cmdbuf  = nullptr;
    std::string                       name;
    std::vector<const struct ggml_tensor *> cmd_nodes;

    ~ggml_backend_c7x_context() {
        if (cmdbuf) {
            c7x_cmdbuf_destroy(cmdbuf);
        }
    }
};

// ---------------------------------------------------------------------------
// Helpers: eligibility, sizing, dtype mapping, repacking & unpacking
// ---------------------------------------------------------------------------

// Note: F16 is intentionally NOT repacked to int8 so FP16 matmuls remain exact.
static bool c7x_tensor_can_repack(const struct ggml_tensor * t) {
    if (!t || t->view_src != nullptr || t->op != GGML_OP_NONE) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    if (t->ne[0] < 64 || t->ne[1] < 1) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool c7x_prefer_w8a8() {
    static const bool pref = []() {
        const char * e = std::getenv("GGML_C7X_W8A8");
        if (!e || e[0] == '\0' || e[0] == '0' ||
            e[0] == 'f' || e[0] == 'F' || e[0] == 'n' || e[0] == 'N') {
            return false;
        }
        if ((e[0] == 'o' || e[0] == 'O') && (e[1] == 'f' || e[1] == 'F')) {
            return false;
        }
        return true;
    }();
    return pref;
}

static size_t c7x_tensor_packed_size(const struct ggml_tensor * t,
                                     c7x_weight_fmt * fmt_out,
                                     size_t * scales_off_out) {
    const uint32_t k = (uint32_t) t->ne[0];
    const uint32_t n = (uint32_t) t->ne[1];
    if ((t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_K) && !c7x_prefer_w8a8()) {
        if (fmt_out) *fmt_out = C7X_WFMT_W4A8_G64;
        return c7x_w4a8_g64_packed_bytes(n, k, scales_off_out);
    }
    if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_1 ||
        t->type == GGML_TYPE_Q5_0 || t->type == GGML_TYPE_Q5_K ||
        t->type == GGML_TYPE_Q6_K ||
        ((t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_K) && c7x_prefer_w8a8())) {
        if (fmt_out) *fmt_out = C7X_WFMT_W8A8;
        return c7x_w8a8_packed_bytes(n, k, scales_off_out);
    }
    if (fmt_out) *fmt_out = C7X_WFMT_RAW;
    if (scales_off_out) *scales_off_out = 0;
    return ggml_nbytes(t);
}

static bool ggml_type_to_c7x_dtype(enum ggml_type t, uint32_t * out) {
    switch (t) {
        case GGML_TYPE_F32:  *out = C7X_DTYPE_F32;  return true;
        case GGML_TYPE_F16:  *out = C7X_DTYPE_F16;  return true;
        case GGML_TYPE_Q8_0: *out = C7X_DTYPE_Q8_0; return true;
        case GGML_TYPE_Q4_0: *out = C7X_DTYPE_Q4_0; return true;
        case GGML_TYPE_I32:  *out = C7X_DTYPE_I32;  return true;
        default: return false;
    }
}

static bool ggml_unary_to_c7x(enum ggml_unary_op op, uint32_t * out) {
    switch (op) {
        case GGML_UNARY_OP_SILU:       *out = C7X_UNARY_SILU;       return true;
        case GGML_UNARY_OP_GELU:       *out = C7X_UNARY_GELU;       return true;
        case GGML_UNARY_OP_GELU_QUICK: *out = C7X_UNARY_GELU_QUICK; return true;
        case GGML_UNARY_OP_RELU:       *out = C7X_UNARY_RELU;       return true;
        case GGML_UNARY_OP_SIGMOID:    *out = C7X_UNARY_SIGMOID;    return true;
        case GGML_UNARY_OP_TANH:       *out = C7X_UNARY_TANH;       return true;
        case GGML_UNARY_OP_NEG:        *out = C7X_UNARY_NEG;        return true;
        case GGML_UNARY_OP_EXP:        *out = C7X_UNARY_EXP;        return true;
        case GGML_UNARY_OP_ABS:        *out = C7X_UNARY_ABS;        return true;
        default: return false;
    }
}

static bool ggml_glu_to_c7x(enum ggml_glu_op op, uint32_t * out) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:      *out = C7X_GLU_SWIGLU;      return true;
        case GGML_GLU_OP_GEGLU:       *out = C7X_GLU_GEGLU;       return true;
        case GGML_GLU_OP_REGLU:       *out = C7X_GLU_REGLU;       return true;
        case GGML_GLU_OP_GEGLU_QUICK: *out = C7X_GLU_GEGLU_QUICK; return true;
        default: return false;
    }
}

static bool c7x_is_own_buffer(ggml_backend_buffer_t buf, ggml_backend_dev_t dev) {
    return buf != nullptr && buf->buft != nullptr && buf->buft->device == dev;
}

static ggml_backend_c7x_buffer_context * c7x_buf_ctx_of(const struct ggml_tensor * t,
                                                        ggml_backend_dev_t dev) {
    if (!t || !c7x_is_own_buffer(t->buffer, dev)) {
        return nullptr;
    }
    return static_cast<ggml_backend_c7x_buffer_context *>(t->buffer->context);
}

static bool c7x_meta_matches_tensor(const c7x_repacked_meta & m, const struct ggml_tensor * t) {
    return m.type == t->type &&
           m.ne[0] == t->ne[0] && m.ne[1] == t->ne[1] &&
           m.ne[2] == t->ne[2] && m.ne[3] == t->ne[3] &&
           m.nb[0] == t->nb[0] && m.nb[1] == t->nb[1] &&
           m.nb[2] == t->nb[2] && m.nb[3] == t->nb[3];
}

// Thread-safe by-value lookup (never returns a dangling map pointer).
static bool c7x_lookup_repacked(const struct ggml_tensor * t,
                                ggml_backend_dev_t dev,
                                c7x_repacked_meta * out) {
    auto * bctx = c7x_buf_ctx_of(t, dev);
    if (!bctx || !t->data || !t->buffer) {
        return false;
    }
    if (t->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
    auto it = bctx->repacked.find(t->data);
    if (it == bctx->repacked.end() || !c7x_meta_matches_tensor(it->second, t)) {
        return false;
    }
    if (out) {
        *out = it->second;
    }
    return true;
}

// Unpack a repacked tensor back into raw GGML format (for get_tensor / partial writes).
static bool c7x_unpack_repacked_to_raw(ggml_backend_c7x_buffer_context * bctx,
                                       const struct ggml_tensor * tensor,
                                       const c7x_repacked_meta & meta,
                                       std::vector<uint8_t> & raw_out) {
    raw_out.assign(ggml_nbytes(tensor), 0);
    std::vector<uint8_t> packed(meta.packed_sz);
    c7x_read_ptr(bctx->dev_ctx->rt, packed.data(), tensor->data, meta.packed_sz);

    if (meta.type == GGML_TYPE_Q4_0 && meta.fmt == C7X_WFMT_W4A8_G64) {
        c7x_unpack_w4a8_g64_to_q4_0(packed.data(),
                                    reinterpret_cast<const float *>(packed.data() + meta.scales_off),
                                    meta.n, meta.k, raw_out.data());
        return true;
    }
    if (meta.type == GGML_TYPE_Q8_0 && meta.fmt == C7X_WFMT_W8A8) {
        c7x_unpack_w8a8_to_q8_0(reinterpret_cast<const int8_t *>(packed.data()),
                                reinterpret_cast<const float *>(packed.data() + meta.scales_off),
                                meta.n, meta.k, raw_out.data());
        return true;
    }
    if (meta.type == GGML_TYPE_Q4_K && meta.fmt == C7X_WFMT_W4A8_G64) {
        std::vector<float> row_f32(meta.k);
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, meta.k);
        const float * scales = reinterpret_cast<const float *>(packed.data() + meta.scales_off);
        for (uint32_t r = 0; r < meta.n; r++) {
            c7x_unpack_w4a8_g64_row_f32(packed.data(), scales, r, meta.k, row_f32.data());
            quantize_row_q4_K_ref(row_f32.data(),
                                  reinterpret_cast<block_q4_K *>(raw_out.data() + (size_t) r * row_bytes),
                                  meta.k);
        }
        return true;
    }
    if (meta.fmt == C7X_WFMT_W8A8 &&
        (meta.type == GGML_TYPE_Q4_0 || meta.type == GGML_TYPE_Q4_K ||
         meta.type == GGML_TYPE_Q4_1 || meta.type == GGML_TYPE_Q5_0 ||
         meta.type == GGML_TYPE_Q5_K || meta.type == GGML_TYPE_Q6_K)) {
        std::vector<float> row_f32(meta.k);
        const size_t row_bytes = ggml_row_size(meta.type, meta.k);
        const int8_t * w_tiled = reinterpret_cast<const int8_t *>(packed.data());
        const float * scales = reinterpret_cast<const float *>(packed.data() + meta.scales_off);
        for (uint32_t r = 0; r < meta.n; r++) {
            c7x_unpack_w8a8_row_f32(w_tiled, scales, r, meta.k, row_f32.data());
            uint8_t * dst_row = raw_out.data() + (size_t) r * row_bytes;
            if (meta.type == GGML_TYPE_Q4_0) {
                quantize_row_q4_0_ref(row_f32.data(), reinterpret_cast<block_q4_0 *>(dst_row), meta.k);
            } else if (meta.type == GGML_TYPE_Q4_K) {
                quantize_row_q4_K_ref(row_f32.data(), reinterpret_cast<block_q4_K *>(dst_row), meta.k);
            } else if (meta.type == GGML_TYPE_Q4_1) {
                quantize_row_q4_1_ref(row_f32.data(), reinterpret_cast<block_q4_1 *>(dst_row), meta.k);
            } else if (meta.type == GGML_TYPE_Q5_0) {
                quantize_row_q5_0_ref(row_f32.data(), reinterpret_cast<block_q5_0 *>(dst_row), meta.k);
            } else if (meta.type == GGML_TYPE_Q5_K) {
                quantize_row_q5_K_ref(row_f32.data(), reinterpret_cast<block_q5_K *>(dst_row), meta.k);
            } else {
                quantize_row_q6_K_ref(row_f32.data(), reinterpret_cast<block_q6_K *>(dst_row), meta.k);
            }
        }
        return true;
    }
    return false;
}

// Repack directly from caller's cached host buffer at set_tensor time (zero O_SYNC read-back).
static bool c7x_repack_from_host(ggml_backend_c7x_buffer_context * bctx,
                                 struct ggml_tensor * t,
                                 const void * raw_host_data) {
    c7x_weight_fmt fmt = C7X_WFMT_RAW;
    size_t scales_off = 0;
    const size_t packed_sz = c7x_tensor_packed_size(t, &fmt, &scales_off);
    const uint32_t k = (uint32_t) t->ne[0];
    const uint32_t n = (uint32_t) t->ne[1];

    const uint8_t * buf_base = (const uint8_t *) c7x_buf_host_ptr(bctx->buf);
    const uint8_t * t_ptr    = (const uint8_t *) t->data;
    if (!buf_base || t_ptr < buf_base ||
        (size_t)(t_ptr - buf_base) + packed_sz > c7x_buf_size(bctx->buf)) {
        return false;
    }

    std::vector<uint8_t> packed(packed_sz, 0);
    if (t->type == GGML_TYPE_Q4_0 && fmt == C7X_WFMT_W4A8_G64) {
        c7x_pack_q4_0_to_w4a8_g64(raw_host_data, n, k,
                                  packed.data(),
                                  reinterpret_cast<float *>(packed.data() + scales_off));
    } else if (t->type == GGML_TYPE_Q4_K && fmt == C7X_WFMT_W4A8_G64) {
        // Stream 64 rows at a time (~500 KiB temp instead of ~0.9 GB for huge vocab matrices!)
        const uint32_t nt_cnt = ((n + 63u) & ~63u) >> 6;
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, k);
        std::vector<float> tile_f32((size_t) 64u * k);
        const uint8_t * raw_bytes = (const uint8_t *) raw_host_data;
        float * scales = reinterpret_cast<float *>(packed.data() + scales_off);
        for (uint32_t nt = 0; nt < nt_cnt; nt++) {
            uint32_t row0 = nt * 64u;
            uint32_t valid = std::min<uint32_t>(64u, n - row0);
            for (uint32_t r = 0; r < valid; r++) {
                dequantize_row_q4_K(
                    reinterpret_cast<const block_q4_K *>(raw_bytes + (size_t)(row0 + r) * row_bytes),
                    tile_f32.data() + (size_t) r * k, k);
            }
            c7x_pack_f32_ntile_to_w4a8_g64(tile_f32.data(), valid, k, nt, packed.data(), scales);
        }
    } else if (t->type == GGML_TYPE_Q8_0) {
        c7x_pack_q8_0_to_w8a8(raw_host_data, n, k,
                              reinterpret_cast<int8_t *>(packed.data()),
                              reinterpret_cast<float *>(packed.data() + scales_off));
    } else if (fmt == C7X_WFMT_W8A8 &&
               (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_K ||
                t->type == GGML_TYPE_Q4_1 || t->type == GGML_TYPE_Q5_0 ||
                t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K)) {
        const uint32_t nt_cnt = ((n + 63u) & ~63u) >> 6;
        const size_t row_bytes = ggml_row_size(t->type, k);
        std::vector<float> tile_f32((size_t) 64u * k);
        const uint8_t * raw_bytes = (const uint8_t *) raw_host_data;
        int8_t * w_tiled = reinterpret_cast<int8_t *>(packed.data());
        float  * scales  = reinterpret_cast<float *>(packed.data() + scales_off);
        for (uint32_t nt = 0; nt < nt_cnt; nt++) {
            uint32_t row0 = nt * 64u;
            uint32_t valid = std::min<uint32_t>(64u, n - row0);
            for (uint32_t r = 0; r < valid; r++) {
                const uint8_t * src_row = raw_bytes + (size_t)(row0 + r) * row_bytes;
                float * dst_row = tile_f32.data() + (size_t) r * k;
                if (t->type == GGML_TYPE_Q4_0) {
                    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(src_row), dst_row, k);
                } else if (t->type == GGML_TYPE_Q4_K) {
                    dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(src_row), dst_row, k);
                } else if (t->type == GGML_TYPE_Q4_1) {
                    dequantize_row_q4_1(reinterpret_cast<const block_q4_1 *>(src_row), dst_row, k);
                } else if (t->type == GGML_TYPE_Q5_0) {
                    dequantize_row_q5_0(reinterpret_cast<const block_q5_0 *>(src_row), dst_row, k);
                } else if (t->type == GGML_TYPE_Q5_K) {
                    dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(src_row), dst_row, k);
                } else {
                    dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(src_row), dst_row, k);
                }
            }
            c7x_pack_f32_ntile_to_w8a8(tile_f32.data(), valid, k, nt, w_tiled, scales);
        }
    } else {
        return false;
    }

    c7x_write_ptr(bctx->dev_ctx->rt, t->data, packed.data(), packed_sz);

    c7x_repacked_meta meta {};
    meta.fmt        = fmt;
    meta.type       = t->type;
    for (int i = 0; i < 4; i++) {
        meta.ne[i] = t->ne[i];
        meta.nb[i] = t->nb[i];
    }
    meta.n          = n;
    meta.k          = k;
    meta.scales_off = scales_off;
    meta.packed_sz  = packed_sz;
    meta.w_phys     = c7x_host_to_phys(bctx->dev_ctx->rt, t->data);
    meta.s_phys     = meta.w_phys + scales_off;
    meta.msmc_buf   = nullptr;

    if (bctx->dev_ctx->enable_msmc_pin && packed_sz <= (2u * 1024u * 1024u)) {
        c7x_buf_t * mbuf = c7x_alloc(bctx->dev_ctx->rt, packed_sz, C7X_MEM_MSMC);
        if (mbuf) {
            c7x_write(bctx->dev_ctx->rt, mbuf, 0, packed.data(), packed_sz);
            meta.msmc_buf = mbuf;
            meta.w_phys   = c7x_buf_phys(mbuf);
            meta.s_phys   = meta.w_phys + scales_off;
        }
    }

    std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
    auto old = bctx->repacked.find(t->data);
    if (old != bctx->repacked.end()) {
        if (old->second.msmc_buf) {
            c7x_free(bctx->dev_ctx->rt, old->second.msmc_buf);
        }
        bctx->repacked.erase(old);
    }
    bctx->repacked.emplace(t->data, meta);
    return true;
}

// ---------------------------------------------------------------------------
// Backend buffer interface
// ---------------------------------------------------------------------------

static void ggml_backend_c7x_buffer_free(ggml_backend_buffer_t buffer) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    delete bctx;
}

static void * ggml_backend_c7x_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    return c7x_buf_host_ptr(bctx->buf);
}

static void ggml_backend_c7x_buffer_erase_repacked(ggml_backend_c7x_buffer_context * bctx,
                                                   const void * key,
                                                   c7x_repacked_meta * prev_out) {
    std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
    auto it = bctx->repacked.find(key);
    if (it != bctx->repacked.end()) {
        if (prev_out) {
            *prev_out = it->second;
            prev_out->msmc_buf = nullptr;
        }
        if (it->second.msmc_buf) {
            c7x_free(bctx->dev_ctx->rt, it->second.msmc_buf);
        }
        bctx->repacked.erase(it);
    } else if (prev_out) {
        prev_out->fmt = C7X_WFMT_RAW;
    }
}

static void ggml_backend_c7x_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value,
                                                  size_t offset,
                                                  size_t size) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    const size_t full_sz = ggml_nbytes(tensor);

    // If this is a partial memset of a repacked weight, unpack first so remaining bytes stay valid!
    c7x_repacked_meta prev {};
    ggml_backend_c7x_buffer_erase_repacked(bctx, tensor->data, &prev);
    if (prev.fmt != C7X_WFMT_RAW && (offset != 0 || size != full_sz)) {
        std::vector<uint8_t> raw;
        if (c7x_unpack_repacked_to_raw(bctx, tensor, prev, raw)) {
            memset(raw.data() + offset, value, size);
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && c7x_tensor_can_repack(tensor)) {
                c7x_repack_from_host(bctx, tensor, raw.data());
            } else {
                c7x_write_ptr(bctx->dev_ctx->rt, tensor->data, raw.data(), raw.size());
            }
            return;
        }
    }

    uint8_t * dst = (uint8_t *) tensor->data + offset;
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        c7x_tensor_can_repack(tensor) &&
        offset == 0 && size == full_sz) {
        std::vector<uint8_t> raw(full_sz, value);
        if (c7x_repack_from_host(bctx, tensor, raw.data())) {
            return;
        }
    }
    if (value == 0) {
        c7x_zero_ptr(bctx->dev_ctx->rt, dst, size);
    } else {
        std::vector<uint8_t> tmp(size, value);
        c7x_write_ptr(bctx->dev_ctx->rt, dst, tmp.data(), size);
    }
}

static void ggml_backend_c7x_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               struct ggml_tensor * tensor,
                                               const void * data,
                                               size_t offset,
                                               size_t size) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    const size_t full_sz = ggml_nbytes(tensor);

    // Eagerly repack eligible full-tensor WEIGHTS writes directly from caller's host RAM
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        c7x_tensor_can_repack(tensor) &&
        offset == 0 && size == full_sz) {
        if (c7x_repack_from_host(bctx, tensor, data)) {
            return;
        }
    }

    // Handle partial write onto a previously repacked weight cleanly
    c7x_repacked_meta prev {};
    ggml_backend_c7x_buffer_erase_repacked(bctx, tensor->data, &prev);
    if (prev.fmt != C7X_WFMT_RAW && (offset != 0 || size != full_sz)) {
        std::vector<uint8_t> raw;
        if (c7x_unpack_repacked_to_raw(bctx, tensor, prev, raw)) {
            memcpy(raw.data() + offset, data, size);
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && c7x_tensor_can_repack(tensor)) {
                c7x_repack_from_host(bctx, tensor, raw.data());
            } else {
                c7x_write_ptr(bctx->dev_ctx->rt, tensor->data, raw.data(), raw.size());
            }
            return;
        }
    }

    c7x_write_ptr(bctx->dev_ctx->rt, (uint8_t *) tensor->data + offset, data, size);
}

static void ggml_backend_c7x_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * tensor,
                                               void * data,
                                               size_t offset,
                                               size_t size) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    c7x_repacked_meta meta {};
    bool is_repacked = false;
    {
        std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
        auto it = bctx->repacked.find(tensor->data);
        if (it != bctx->repacked.end() && c7x_meta_matches_tensor(it->second, tensor)) {
            meta = it->second;
            is_repacked = true;
        }
    }
    if (is_repacked) {
        std::vector<uint8_t> raw;
        if (c7x_unpack_repacked_to_raw(bctx, tensor, meta, raw) && offset + size <= raw.size()) {
            memcpy(data, raw.data() + offset, size);
            return;
        }
    }
    c7x_read_ptr(bctx->dev_ctx->rt, data, (const uint8_t *) tensor->data + offset, size);
}

static bool ggml_backend_c7x_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_c7x_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_c7x_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    {
        std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
        for (auto & kv : bctx->repacked) {
            if (kv.second.msmc_buf) {
                c7x_free(bctx->dev_ctx->rt, kv.second.msmc_buf);
            }
        }
        bctx->repacked.clear();
    }
    if (value == 0) {
        c7x_zero(bctx->dev_ctx->rt, bctx->buf, 0, buffer->size);
    } else {
        std::vector<uint8_t> chunk(std::min<size_t>(buffer->size, 65536u), value);
        for (size_t off = 0; off < buffer->size; off += chunk.size()) {
            size_t n = std::min(chunk.size(), buffer->size - off);
            c7x_write(bctx->dev_ctx->rt, bctx->buf, off, chunk.data(), n);
        }
    }
}

static void ggml_backend_c7x_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * bctx = static_cast<ggml_backend_c7x_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lk(bctx->dev_ctx->repack_mtx);
    for (auto & kv : bctx->repacked) {
        if (kv.second.msmc_buf) {
            c7x_free(bctx->dev_ctx->rt, kv.second.msmc_buf);
        }
    }
    bctx->repacked.clear();
}

static const struct ggml_backend_buffer_i ggml_backend_c7x_buffer_iface = {
    /* .free_buffer   = */ ggml_backend_c7x_buffer_free,
    /* .get_base      = */ ggml_backend_c7x_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_c7x_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_c7x_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_c7x_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_c7x_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_c7x_buffer_clear,
    /* .reset         = */ ggml_backend_c7x_buffer_reset,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend buffer type interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_c7x_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "C7X";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_c7x_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev_ctx = static_cast<ggml_backend_c7x_device_context *>(buft->context);
    if (!dev_ctx || !dev_ctx->rt) {
        return nullptr;
    }
    c7x_buf_t * buf = c7x_alloc(dev_ctx->rt, size, C7X_MEM_DDR);
    if (!buf) {
        GGML_LOG_ERROR("ggml-c7x: failed to allocate %zu bytes from C7x DDR arena\n", size);
        return nullptr;
    }
    auto * bctx = new ggml_backend_c7x_buffer_context();
    bctx->dev_ctx = dev_ctx;
    bctx->buf     = buf;
    return ggml_backend_buffer_init(buft, ggml_backend_c7x_buffer_iface, bctx, size);
}

static size_t ggml_backend_c7x_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_c7x_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * dev_ctx = static_cast<ggml_backend_c7x_device_context *>(buft->context);
    return (dev_ctx && dev_ctx->rt) ? dev_ctx->info.ddr_total : 0;
}

static size_t ggml_backend_c7x_buft_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                   const struct ggml_tensor * tensor) {
    if (c7x_tensor_can_repack(tensor)) {
        return std::max(c7x_tensor_packed_size(tensor, nullptr, nullptr), ggml_nbytes(tensor));
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static bool ggml_backend_c7x_buft_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static const struct ggml_backend_buffer_type_i ggml_backend_c7x_buft_iface = {
    /* .get_name         = */ ggml_backend_c7x_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_c7x_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_c7x_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_c7x_buft_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_c7x_buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_backend_c7x_buft_is_host,
};

// ---------------------------------------------------------------------------
// Command-buffer flush + graph lowering -> C7X_OP_EXEC_LIST
// ---------------------------------------------------------------------------

static bool flush_cmdbuf(ggml_backend_c7x_context * ctx) {
    if (c7x_cmdbuf_count(ctx->cmdbuf) == 0u) {
        return true;
    }
    struct c7x_exec_stats stats {};
    if (c7x_cmdbuf_submit(ctx->dev_ctx->rt, ctx->cmdbuf, &stats) != 0) {
        const char * bad_name = (stats.fail_idx < ctx->cmd_nodes.size() && ctx->cmd_nodes[stats.fail_idx])
                                ? ctx->cmd_nodes[stats.fail_idx]->name
                                : "<unknown>";
        const char * bad_op   = (stats.fail_idx < ctx->cmd_nodes.size() && ctx->cmd_nodes[stats.fail_idx])
                                ? ggml_op_name(ctx->cmd_nodes[stats.fail_idx]->op)
                                : "?";
        GGML_LOG_ERROR("ggml-c7x: C7X_OP_EXEC_LIST failed at cmd %u (%s, op=%s, done=%u)\n",
                       stats.fail_idx, bad_name, bad_op, stats.cmds_done);
        c7x_cmdbuf_reset(ctx->cmdbuf);
        ctx->cmd_nodes.clear();
        return false;
    }
    static const int profile = []() {
        const char * e = std::getenv("GGML_C7X_PROFILE");
        return e ? std::atoi(e) : 0;
    }();
    if (profile == 1) {
        fprintf(stderr, "ggml-c7x: EXEC_LIST cmds=%u dsp=%.2f ms total_dsp=%.2f ms wall=%.2f ms\n",
                stats.cmds_done,
                (double) stats.cycles / 1e6,
                (double) stats.total_cycles / 1e6,
                stats.wall_us / 1e3);
    }
    c7x_cmdbuf_reset(ctx->cmdbuf);
    ctx->cmd_nodes.clear();
    return true;
}

static bool push_cmd(ggml_backend_c7x_context * ctx,
                     const struct ggml_tensor * node,
                     const struct c7x_cmd & cmd,
                     const void * desc = nullptr,
                     size_t desc_bytes = 0) {
    struct c7x_cmd c = cmd;
    if (desc && desc_bytes > 0) {
        uint64_t dp = c7x_cmdbuf_push_desc(ctx->cmdbuf, desc, desc_bytes);
        if (!dp) {
            if (!flush_cmdbuf(ctx)) return false;
            dp = c7x_cmdbuf_push_desc(ctx->cmdbuf, desc, desc_bytes);
            if (!dp) return false;
        }
        c.src3 = dp;
    }
    if (c7x_cmdbuf_push(ctx->cmdbuf, &c) != 0) {
        if (!flush_cmdbuf(ctx)) return false;
        if (desc && desc_bytes > 0) {
            uint64_t dp = c7x_cmdbuf_push_desc(ctx->cmdbuf, desc, desc_bytes);
            if (!dp) return false;
            c.src3 = dp;
        }
        if (c7x_cmdbuf_push(ctx->cmdbuf, &c) != 0) return false;
    }
    ctx->cmd_nodes.push_back(node);
    return true;
}

static bool lower_node(ggml_backend_c7x_context * ctx,
                       ggml_backend_dev_t dev,
                       struct ggml_tensor * node) {
    c7x_ctx_t * rt = ctx->dev_ctx->rt;
    struct c7x_cmd cmd {};

    auto phys = [rt](const struct ggml_tensor * t) -> uint64_t {
        return (t && t->data) ? c7x_host_to_phys(rt, t->data) : 0ull;
    };

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * src1 = node->src[1];
            const uint64_t p_dst  = phys(node);
            const uint64_t p_src1 = phys(src1);
            if (!p_dst || !p_src1) return false;

            // Fast MMA path for repacked 2D weights (handles 2D, 3D, or 4D contiguous src1!)
            c7x_repacked_meta rm {};
            if (c7x_lookup_repacked(src0, dev, &rm)) {
                if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(node)) {
                    return false;
                }
                const uint64_t m_total = (uint64_t) src1->ne[1] * src1->ne[2] * src1->ne[3];
                // Always hand L2 scratch to mma_kernels when DRU is available: mma_kernels places
                // tile_acc in L2 SRAM and enforces its own >=1 MiB (m=1) / >=128 KiB (m>1) / DDR-only gates.
                cmd.op    = (rm.fmt == C7X_WFMT_W4A8_G64) ? C7X_CMD_MATMUL_W4A8_G64 : C7X_CMD_MATMUL_W8A8;
                cmd.flags = ctx->dev_ctx->info.dru_ok ? C7X_CMD_FLAG_USE_DRU : 0u;
                cmd.p0    = (uint32_t) m_total;
                cmd.p1    = (uint32_t) src0->ne[0]; // K
                cmd.p2    = (uint32_t) src0->ne[1]; // N
                cmd.p3    = 0u;
                cmd.src0  = rm.w_phys;
                cmd.src1  = p_src1;
                cmd.src2  = rm.s_phys;
                cmd.dst   = p_dst;
                return push_cmd(ctx, node, cmd);
            }

            // General exact 4D batched F32 / F16 / Q8_0 (KV cache) matmul
            if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                 (src0->type == GGML_TYPE_Q8_0 && src0->nb[0] == sizeof(block_q8_0) && (src0->ne[0] % 32) == 0)) &&
                src1->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) {
                const uint64_t p_src0 = phys(src0);
                if (!p_src0) return false;
                struct c7x_matmul_f32_desc md {};
                for (int i = 0; i < 4; i++) {
                    md.ne0[i]    = (uint32_t) src0->ne[i];
                    md.ne1[i]    = (uint32_t) src1->ne[i];
                    md.ne_dst[i] = (uint32_t) node->ne[i];
                    md.nb0[i]    = (uint64_t) src0->nb[i];
                    md.nb1[i]    = (uint64_t) src1->nb[i];
                    md.nb_dst[i] = (uint64_t) node->nb[i];
                }
                cmd.op    = C7X_CMD_MATMUL_F32;
                cmd.flags = (src0->type == GGML_TYPE_F16)  ? C7X_CMD_FLAG_SRC0_F16  :
                            (src0->type == GGML_TYPE_Q8_0) ? C7X_CMD_FLAG_SRC0_Q8_0 : 0u;
                cmd.src0  = p_src0;
                cmd.src1  = p_src1;
                cmd.dst   = p_dst;
                return push_cmd(ctx, node, cmd, &md, sizeof(md));
            }
            return false;
        }

        case GGML_OP_RMS_NORM: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            if (!p_src0 || !p_dst) return false;
            float eps = 0.0f;
            memcpy(&eps, node->op_params, sizeof(float));
            cmd.op   = C7X_CMD_RMS_NORM_F32;
            cmd.p0   = (uint32_t) src0->ne[0];
            cmd.p1   = (uint32_t) ggml_nrows(src0);
            cmd.p2   = (uint32_t) (src0->nb[1] / sizeof(float));
            cmd.p3   = (uint32_t) (node->nb[1] / sizeof(float));
            cmd.f0   = eps;
            cmd.src0 = p_src0;
            cmd.src1 = 0ull;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd);
        }

        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * src1 = node->src[1];
            const uint64_t p_src0 = phys(src0), p_src1 = phys(src1), p_dst = phys(node);
            if (!p_src0 || !p_src1 || !p_dst) return false;
            const uint32_t op = (node->op == GGML_OP_MUL) ? C7X_CMD_MUL_F32 : C7X_CMD_ADD_F32;

            if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(node) &&
                src0->ne[0] == node->ne[0] &&
                (src1->ne[2] == 1 && src1->ne[3] == 1) &&
                (src1->ne[1] == 1 || (src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[1] == src0->ne[1]))) {
                cmd.op   = op;
                cmd.p0   = (uint32_t) node->ne[0];
                cmd.p1   = (uint32_t) ggml_nrows(node);
                cmd.p2   = (uint32_t) src1->ne[1];
                cmd.p3   = (uint32_t) src1->ne[0];
                cmd.src0 = p_src0;
                cmd.src1 = p_src1;
                cmd.dst  = p_dst;
                return push_cmd(ctx, node, cmd);
            }

            struct c7x_4d_binop_desc bd {};
            for (int i = 0; i < 4; i++) {
                bd.ne0[i]    = (uint32_t) src0->ne[i];
                bd.ne1[i]    = (uint32_t) src1->ne[i];
                bd.ne_dst[i] = (uint32_t) node->ne[i];
                bd.nb0[i]    = (uint64_t) src0->nb[i];
                bd.nb1[i]    = (uint64_t) src1->nb[i];
                bd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            cmd.op   = op;
            cmd.src0 = p_src0;
            cmd.src1 = p_src1;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &bd, sizeof(bd));
        }

        case GGML_OP_SCALE: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            if (!p_src0 || !p_dst) return false;
            float s = 1.0f, b = 0.0f;
            memcpy(&s, (const float *) node->op_params + 0, sizeof(float));
            memcpy(&b, (const float *) node->op_params + 1, sizeof(float));
            cmd.op   = C7X_CMD_SCALE_F32;
            cmd.p0   = (uint32_t) ggml_nelements(node);
            cmd.p1   = 1u;
            cmd.f0   = s;
            cmd.f1   = b;
            cmd.src0 = p_src0;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd);
        }

        case GGML_OP_UNARY: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            uint32_t subop = 0;
            if (!p_src0 || !p_dst || !ggml_unary_to_c7x(ggml_get_unary_op(node), &subop)) {
                return false;
            }
            cmd.op   = C7X_CMD_UNARY_F32;
            cmd.p0   = (uint32_t) ggml_nelements(node);
            cmd.p1   = subop;
            cmd.src0 = p_src0;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd);
        }

        case GGML_OP_GLU: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * src1 = node->src[1];
            const uint64_t p_src0 = phys(src0), p_src1 = phys(src1), p_dst = phys(node);
            uint32_t subop = 0;
            if (!p_src0 || !p_dst || (src1 && !p_src1) ||
                !ggml_glu_to_c7x(ggml_get_glu_op(node), &subop)) {
                return false;
            }
            const int32_t swapped = ggml_get_op_params_i32(node, 1);
            cmd.op    = C7X_CMD_GLU_F32;
            cmd.flags = swapped ? C7X_CMD_FLAG_GLU_SWAP : 0u;
            cmd.p0    = (uint32_t) node->ne[0];
            cmd.p1    = (uint32_t) ggml_nrows(node);
            cmd.p2    = subop;
            cmd.p4    = (uint32_t) (src0->nb[1] / sizeof(float));
            cmd.p5    = src1 ? (uint32_t) (src1->nb[1] / sizeof(float)) : 0u;
            cmd.src0  = p_src0;
            cmd.src1  = p_src1;
            cmd.dst   = p_dst;
            return push_cmd(ctx, node, cmd);
        }

        case GGML_OP_SOFT_MAX: {
            const struct ggml_tensor * src0  = node->src[0];
            const struct ggml_tensor * mask  = node->src[1];
            const struct ggml_tensor * sinks = node->src[2];
            const uint64_t p_src0 = phys(src0), p_mask = phys(mask), p_sinks = phys(sinks), p_dst = phys(node);
            if (!p_src0 || !p_dst || (mask && !p_mask) || (sinks && !p_sinks)) return false;

            float scale = 1.0f, max_bias = 0.0f;
            memcpy(&scale,    (const float *) node->op_params + 0, sizeof(float));
            memcpy(&max_bias, (const float *) node->op_params + 1, sizeof(float));

            struct c7x_softmax_desc sd {};
            sd.ne0     = (uint32_t) src0->ne[0];
            sd.ne1     = (uint32_t) src0->ne[1];
            sd.ne2     = (uint32_t) src0->ne[2];
            sd.ne3     = (uint32_t) src0->ne[3];
            sd.nb01    = (uint64_t) src0->nb[1];
            sd.nb02    = (uint64_t) src0->nb[2];
            sd.nb03    = (uint64_t) src0->nb[3];
            sd.nb_dst1 = (uint64_t) node->nb[1];
            sd.nb_dst2 = (uint64_t) node->nb[2];
            sd.nb_dst3 = (uint64_t) node->nb[3];
            sd.scale    = scale;
            sd.max_bias = max_bias;
            if (max_bias > 0.0f) {
                const uint32_t n_head = (uint32_t) src0->ne[2];
                const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((double) n_head));
                sd.n_head_log2 = n_head_log2;
                sd.m0 = powf(2.0f, -(max_bias) / (float) n_head_log2);
                sd.m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);
            }
            if (mask) {
                sd.mask_ne1 = (uint32_t) mask->ne[1];
                sd.mask_ne2 = (uint32_t) mask->ne[2];
                sd.mask_ne3 = (uint32_t) mask->ne[3];
                sd.mask_nb1 = (uint64_t) mask->nb[1];
                sd.mask_nb2 = (uint64_t) mask->nb[2];
                sd.mask_nb3 = (uint64_t) mask->nb[3];
            }

            cmd.op    = C7X_CMD_SOFT_MAX_F32;
            cmd.flags = (mask && mask->type == GGML_TYPE_F16) ? C7X_CMD_FLAG_MASK_F16 : 0u;
            cmd.src0  = p_src0;
            cmd.src1  = p_mask;
            cmd.src2  = p_sinks;
            cmd.dst   = p_dst;
            return push_cmd(ctx, node, cmd, &sd, sizeof(sd));
        }

        case GGML_OP_ROPE: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * pos  = node->src[1];
            const struct ggml_tensor * ff   = node->src[2];
            const uint64_t p_src0 = phys(src0), p_pos = phys(pos), p_ff = phys(ff), p_dst = phys(node);
            if (!p_src0 || !p_pos || !p_dst || (ff && !p_ff)) return false;

            struct c7x_rope_desc rd {};
            for (int i = 0; i < 4; i++) {
                rd.ne0[i]    = (uint32_t) src0->ne[i];
                rd.nb0[i]    = (uint64_t) src0->nb[i];
                rd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            rd.n_dims           = (uint32_t) ((const int32_t *) node->op_params)[1];
            rd.mode             = (uint32_t) ((const int32_t *) node->op_params)[2];
            rd.truncate         = (((const int32_t *) node->op_params)[3] == 0) ? 1u : 0u;
            rd.n_ctx_orig       = (uint32_t) ((const int32_t *) node->op_params)[4];
            rd.has_freq_factors = (ff != nullptr) ? 1u : 0u;
            memcpy(&rd.freq_base,   (const int32_t *) node->op_params +  5, sizeof(float));
            memcpy(&rd.freq_scale,  (const int32_t *) node->op_params +  6, sizeof(float));
            memcpy(&rd.ext_factor,  (const int32_t *) node->op_params +  7, sizeof(float));
            memcpy(&rd.attn_factor, (const int32_t *) node->op_params +  8, sizeof(float));
            memcpy(&rd.beta_fast,   (const int32_t *) node->op_params +  9, sizeof(float));
            memcpy(&rd.beta_slow,   (const int32_t *) node->op_params + 10, sizeof(float));

            cmd.op   = C7X_CMD_ROPE_F32;
            cmd.src0 = p_src0;
            cmd.src1 = p_pos;
            cmd.src2 = p_ff;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &rd, sizeof(rd));
        }

        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * idx  = node->src[1];
            const uint64_t p_idx = phys(idx), p_dst = phys(node);
            if (!p_idx || !p_dst) return false;

            struct c7x_get_rows_desc gd {};
            gd.ne00    = (uint32_t) src0->ne[0];
            gd.ne01    = (uint32_t) src0->ne[1];
            gd.ne02    = (uint32_t) src0->ne[2];
            gd.ne10    = (uint32_t) idx->ne[0];
            gd.ne11    = (uint32_t) idx->ne[1];
            gd.ne12    = (uint32_t) idx->ne[2];
            gd.nb01    = (uint64_t) src0->nb[1];
            gd.nb02    = (uint64_t) src0->nb[2];
            gd.nb03    = (uint64_t) src0->nb[3];
            gd.nb10    = (uint64_t) idx->nb[0];
            gd.nb11    = (uint64_t) idx->nb[1];
            gd.nb12    = (uint64_t) idx->nb[2];
            gd.nb_dst1 = (uint64_t) node->nb[1];
            gd.nb_dst2 = (uint64_t) node->nb[2];
            gd.nb_dst3 = (uint64_t) node->nb[3];

            c7x_repacked_meta rm {};
            uint32_t dtype = 0;
            uint64_t w_phys = phys(src0);
            uint64_t s_phys = 0ull;
            if (c7x_lookup_repacked(src0, dev, &rm)) {
                dtype    = (rm.fmt == C7X_WFMT_W4A8_G64) ? C7X_DTYPE_W4A8_G64 : C7X_DTYPE_W8A8;
                gd.k_pad = (rm.k + 63u) & ~63u;
                gd.n_pad = (rm.n + 63u) & ~63u;
                w_phys   = rm.w_phys;
                s_phys   = rm.s_phys;
            } else if (!w_phys || !ggml_type_to_c7x_dtype(src0->type, &dtype)) {
                return false;
            }

            cmd.op   = C7X_CMD_GET_ROWS;
            cmd.p4   = dtype;
            cmd.src0 = w_phys;
            cmd.src1 = p_idx;
            cmd.src2 = s_phys;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &gd, sizeof(gd));
        }

        case GGML_OP_CPY:
        case GGML_OP_CONT: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            uint32_t stype = 0, dtype = 0;
            if (!p_src0 || !p_dst ||
                !ggml_type_to_c7x_dtype(src0->type, &stype) ||
                !ggml_type_to_c7x_dtype(node->type, &dtype)) {
                return false;
            }
            struct c7x_cpy_desc cd {};
            for (int i = 0; i < 4; i++) {
                cd.ne0[i]    = (uint32_t) src0->ne[i];
                cd.ne_dst[i] = (uint32_t) node->ne[i];
                cd.nb0[i]    = (uint64_t) src0->nb[i];
                cd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            cmd.op   = C7X_CMD_CPY;
            cmd.p0   = stype;
            cmd.p1   = dtype;
            cmd.src0 = p_src0;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &cd, sizeof(cd));
        }

        case GGML_OP_SET_ROWS: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * idx  = node->src[1];
            const uint64_t p_src0 = phys(src0), p_idx = phys(idx), p_dst = phys(node);
            uint32_t dtype = 0;
            if (!p_src0 || !p_idx || !p_dst || !ggml_type_to_c7x_dtype(node->type, &dtype)) {
                return false;
            }
            struct c7x_set_rows_desc sd {};
            for (int i = 0; i < 4; i++) {
                sd.ne0[i]    = (uint32_t) src0->ne[i];
                sd.ne1[i]    = (uint32_t) idx->ne[i];
                sd.ne_dst[i] = (uint32_t) node->ne[i];
                sd.nb0[i]    = (uint64_t) src0->nb[i];
                sd.nb1[i]    = (uint64_t) idx->nb[i];
                sd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            cmd.op    = C7X_CMD_SET_ROWS;
            cmd.flags = (idx->type == GGML_TYPE_I64) ? C7X_CMD_FLAG_IDX_I64 : 0u;
            cmd.p0    = dtype;
            cmd.src0  = p_src0;
            cmd.src1  = p_idx;
            cmd.dst   = p_dst;
            return push_cmd(ctx, node, cmd, &sd, sizeof(sd));
        }

        case GGML_OP_NORM: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            if (!p_src0 || !p_dst) return false;
            float eps = 1e-5f;
            memcpy(&eps, node->op_params, sizeof(float));
            struct c7x_norm_desc nd {};
            for (int i = 0; i < 4; i++) {
                nd.ne0[i]    = (uint32_t) src0->ne[i];
                nd.nb0[i]    = (uint64_t) src0->nb[i];
                nd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            nd.eps       = eps;
            nd.has_gamma = 0u;
            nd.has_beta  = 0u;
            cmd.op   = C7X_CMD_NORM_F32;
            cmd.src0 = p_src0;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &nd, sizeof(nd));
        }

        case GGML_OP_IM2COL: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * src1 = node->src[1];
            const uint64_t p_src1 = phys(src1), p_dst = phys(node);
            uint32_t stype = 0, dtype = 0;
            if (!src0 || !p_src1 || !p_dst ||
                !ggml_type_to_c7x_dtype(src1->type, &stype) ||
                !ggml_type_to_c7x_dtype(node->type, &dtype)) {
                return false;
            }
            const int32_t * op_p = (const int32_t *) node->op_params;
            const bool is_2d = (op_p[6] == 1);
            struct c7x_im2col_desc id {};
            id.N        = (uint32_t) (is_2d ? src1->ne[3] : src1->ne[2]);
            id.IC       = (uint32_t) (is_2d ? src1->ne[2] : src1->ne[1]);
            id.IH       = (uint32_t) (is_2d ? src1->ne[1] : 1);
            id.IW       = (uint32_t) src1->ne[0];
            id.KH       = (uint32_t) (is_2d ? src0->ne[1] : 1);
            id.KW       = (uint32_t) src0->ne[0];
            id.OH       = (uint32_t) (is_2d ? node->ne[2] : 1);
            id.OW       = (uint32_t) node->ne[1];
            id.s0       = op_p[0];
            id.s1       = op_p[1];
            id.p0       = op_p[2];
            id.p1       = op_p[3];
            id.d0       = op_p[4];
            id.d1       = op_p[5];
            id.is_2d    = is_2d ? 1u : 0u;
            id.src_nb1  = is_2d ? (uint64_t) src1->nb[1] : 0ull;
            id.src_nb_c = (uint64_t) (is_2d ? src1->nb[2] : src1->nb[1]);
            id.src_nb_n = (uint64_t) (is_2d ? src1->nb[3] : src1->nb[2]);
            cmd.op   = C7X_CMD_IM2COL;
            cmd.p0   = stype;
            cmd.p1   = dtype;
            cmd.src1 = p_src1;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &id, sizeof(id));
        }

        case GGML_OP_POOL_2D: {
            const struct ggml_tensor * src0 = node->src[0];
            const uint64_t p_src0 = phys(src0), p_dst = phys(node);
            if (!p_src0 || !p_dst) return false;
            const int32_t * opts = (const int32_t *) node->op_params;
            struct c7x_pool2d_desc pd {};
            pd.op       = (opts[0] == GGML_OP_POOL_MAX) ? C7X_POOL_MAX : C7X_POOL_AVG;
            pd.IW       = (uint32_t) src0->ne[0];
            pd.IH       = (uint32_t) src0->ne[1];
            pd.OW       = (uint32_t) node->ne[0];
            pd.OH       = (uint32_t) node->ne[1];
            pd.n_planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
            pd.k0       = opts[1];
            pd.k1       = opts[2];
            pd.s0       = opts[3];
            pd.s1       = opts[4];
            pd.p0       = opts[5];
            pd.p1       = opts[6];
            pd.src_nb1  = (uint64_t) src0->nb[1];
            pd.src_nb2  = (uint64_t) src0->nb[2];
            pd.dst_nb1  = (uint64_t) node->nb[1];
            pd.dst_nb2  = (uint64_t) node->nb[2];
            cmd.op    = C7X_CMD_POOL_2D_F32;
            cmd.flags = (src0->type == GGML_TYPE_F16) ? C7X_CMD_FLAG_SRC0_F16 : 0u;
            cmd.src0  = p_src0;
            cmd.dst   = p_dst;
            return push_cmd(ctx, node, cmd, &pd, sizeof(pd));
        }

        case GGML_OP_CONV_2D: {
            const struct ggml_tensor * src0 = node->src[0];
            const struct ggml_tensor * src1 = node->src[1];
            const uint64_t p_src0 = phys(src0), p_src1 = phys(src1), p_dst = phys(node);
            if (!p_src0 || !p_src1 || !p_dst) return false;
            const int32_t * op_p = (const int32_t *) node->op_params;
            struct c7x_conv2d_f32_desc cd {};
            cd.N        = (uint32_t) src1->ne[3];
            cd.IC       = (uint32_t) src1->ne[2];
            cd.IH       = (uint32_t) src1->ne[1];
            cd.IW       = (uint32_t) src1->ne[0];
            cd.OC       = (uint32_t) src0->ne[3];
            cd.KH       = (uint32_t) src0->ne[1];
            cd.KW       = (uint32_t) src0->ne[0];
            cd.OH       = (uint32_t) node->ne[1];
            cd.OW       = (uint32_t) node->ne[0];
            cd.has_bias = 0u;
            cd.s0       = op_p[0];
            cd.s1       = op_p[1];
            cd.p0       = op_p[2];
            cd.p1       = op_p[3];
            cd.d0       = op_p[4];
            cd.d1       = op_p[5];
            for (int i = 0; i < 4; i++) {
                cd.src_nb[i] = (uint64_t) src1->nb[i];
                cd.dst_nb[i] = (uint64_t) node->nb[i];
            }
            cmd.op    = C7X_CMD_CONV_2D_F32;
            cmd.flags = (src0->type == GGML_TYPE_F16) ? C7X_CMD_FLAG_SRC0_F16 : 0u;
            cmd.src0  = p_src0;
            cmd.src1  = p_src1;
            cmd.dst   = p_dst;
            return push_cmd(ctx, node, cmd, &cd, sizeof(cd));
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor * q     = node->src[0];
            const struct ggml_tensor * k     = node->src[1];
            const struct ggml_tensor * v     = node->src[2];
            const struct ggml_tensor * mask  = node->src[3];
            const struct ggml_tensor * sinks = node->src[4];
            const uint64_t p_q = phys(q), p_k = phys(k), p_v = phys(v), p_dst = phys(node);
            uint32_t k_dt = 0, v_dt = 0, m_dt = C7X_DTYPE_F16;
            if (!p_q || !p_k || !p_v || !p_dst ||
                !ggml_type_to_c7x_dtype(k->type, &k_dt) ||
                !ggml_type_to_c7x_dtype(v->type, &v_dt)) {
                return false;
            }
            if (mask && !ggml_type_to_c7x_dtype(mask->type, &m_dt)) {
                return false;
            }
            float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
            memcpy(&scale,         (const float *) node->op_params + 0, sizeof(float));
            memcpy(&max_bias,      (const float *) node->op_params + 1, sizeof(float));
            memcpy(&logit_softcap, (const float *) node->op_params + 2, sizeof(float));

            const uint32_t n_head      = (uint32_t) q->ne[2];
            const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((double) n_head));
            struct c7x_flash_attn_desc fd {};
            for (int i = 0; i < 4; i++) {
                fd.neq[i]    = (uint32_t) q->ne[i];
                fd.nek[i]    = (uint32_t) k->ne[i];
                fd.nev[i]    = (uint32_t) v->ne[i];
                fd.nem[i]    = mask ? (uint32_t) mask->ne[i] : 0u;
                fd.nbq[i]    = (uint64_t) q->nb[i];
                fd.nbk[i]    = (uint64_t) k->nb[i];
                fd.nbv[i]    = (uint64_t) v->nb[i];
                fd.nbm[i]    = mask ? (uint64_t) mask->nb[i] : 0ull;
                fd.nb_dst[i] = (uint64_t) node->nb[i];
            }
            fd.k_dtype       = k_dt;
            fd.v_dtype       = v_dt;
            fd.mask_dtype    = m_dt;
            fd.n_head_log2   = n_head_log2;
            fd.scale         = scale;
            fd.max_bias      = max_bias;
            fd.logit_softcap = logit_softcap;
            fd.m0            = (max_bias > 0.0f) ? std::pow(2.0f, -max_bias / (float) n_head_log2) : 1.0f;
            fd.m1            = (max_bias > 0.0f) ? std::pow(2.0f, -(max_bias * 0.5f) / (float) n_head_log2) : 1.0f;
            fd.mask_addr     = mask  ? phys(mask)  : 0ull;
            fd.sinks_addr    = sinks ? phys(sinks) : 0ull;
            if ((mask && !fd.mask_addr) || (sinks && !fd.sinks_addr)) {
                return false;
            }
            cmd.op   = C7X_CMD_FLASH_ATTN_EXT;
            cmd.src0 = p_q;
            cmd.src1 = p_k;
            cmd.src2 = p_v;
            cmd.dst  = p_dst;
            return push_cmd(ctx, node, cmd, &fd, sizeof(fd));
        }

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Backend stream interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_c7x_get_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_c7x_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_c7x_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_c7x_context *>(backend->context);
    delete ctx;
    delete backend;
}

// Try to fuse consecutive node pairs (RMS_NORM + MUL weight, SILU + MUL SwiGLU)
// when `node` has a single consumer (`next`) and is not exposed as a graph output/view.
static bool try_lower_fused_pair(ggml_backend_c7x_context * ctx,
                                 struct ggml_tensor * node,
                                 struct ggml_tensor * next,
                                 int node_uses,
                                 const char ** fused_name_out) {
    if (!node || !next || node_uses != 1 ||
        (next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(next)) {
        return false;
    }
    c7x_ctx_t * rt = ctx->dev_ctx->rt;
    auto phys = [rt](const struct ggml_tensor * t) -> uint64_t {
        return (t && t->data) ? c7x_host_to_phys(rt, t->data) : 0ull;
    };

    // Pattern 1: RMS_NORM(x) -> MUL(norm, weight[ne0, 1, 1, 1]) ==> weighted C7X_CMD_RMS_NORM_F32
    if (node->op == GGML_OP_RMS_NORM && next->op == GGML_OP_MUL &&
        next->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(node, next) && ggml_is_contiguous(next)) {
        const struct ggml_tensor * w = nullptr;
        if (next->src[0] == node && next->src[1] && next->src[1] != node) {
            w = next->src[1];
        } else if (next->src[1] == node && next->src[0] && next->src[0] != node) {
            w = next->src[0];
        }
        if (w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) &&
            w->ne[0] == node->ne[0] && w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1) {
            const struct ggml_tensor * x = node->src[0];
            const uint64_t p_x = phys(x), p_w = phys(w), p_dst = phys(next);
            if (p_x && p_w && p_dst && p_w != p_dst) {
                float eps = 0.0f;
                memcpy(&eps, node->op_params, sizeof(float));
                struct c7x_cmd cmd {};
                cmd.op   = C7X_CMD_RMS_NORM_F32;
                cmd.p0   = (uint32_t) x->ne[0];
                cmd.p1   = (uint32_t) ggml_nrows(x);
                cmd.p2   = (uint32_t) (x->nb[1] / sizeof(float));
                cmd.p3   = (uint32_t) (next->nb[1] / sizeof(float));
                cmd.f0   = eps;
                cmd.src0 = p_x;
                cmd.src1 = p_w;
                cmd.dst  = p_dst;
                if (push_cmd(ctx, next, cmd)) {
                    if (fused_name_out) *fused_name_out = "RMS_NORM_MUL";
                    return true;
                }
            }
        }
    }

    // Pattern 2: UNARY(SILU, gate) -> MUL(silu_gate, up) ==> C7X_CMD_GLU_F32 (SWIGLU)
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU &&
        next->op == GGML_OP_MUL && next->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(node, next) && ggml_is_contiguous(next)) {
        const struct ggml_tensor * up = nullptr;
        if (next->src[0] == node && next->src[1] && next->src[1] != node) {
            up = next->src[1];
        } else if (next->src[1] == node && next->src[0] && next->src[0] != node) {
            up = next->src[0];
        }
        const struct ggml_tensor * gate = node->src[0];
        if (gate && up && up->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(gate, up) && ggml_is_contiguous(gate) && ggml_is_contiguous(up)) {
            const uint64_t p_gate = phys(gate), p_up = phys(up), p_dst = phys(next);
            if (p_gate && p_up && p_dst) {
                struct c7x_cmd cmd {};
                cmd.op   = C7X_CMD_GLU_F32;
                cmd.p0   = (uint32_t) next->ne[0];
                cmd.p1   = (uint32_t) ggml_nrows(next);
                cmd.p2   = C7X_GLU_SWIGLU;
                cmd.p4   = (uint32_t) (gate->nb[1] / sizeof(float));
                cmd.p5   = (uint32_t) (up->nb[1] / sizeof(float));
                cmd.src0 = p_gate;
                cmd.src1 = p_up;
                cmd.dst  = p_dst;
                if (push_cmd(ctx, next, cmd)) {
                    if (fused_name_out) *fused_name_out = "SILU_MUL_SWIGLU";
                    return true;
                }
            }
        }
    }

    return false;
}

static enum ggml_status ggml_backend_c7x_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_c7x_context *>(backend->context);
    static const int profile = []() {
        const char * e = std::getenv("GGML_C7X_PROFILE");
        return e ? std::atoi(e) : 0;
    }();
    c7x_cmdbuf_reset(ctx->cmdbuf);
    ctx->cmd_nodes.clear();

    // Count references to every tensor in cgraph so peephole fusion never drops a live intermediate.
    std::unordered_map<const struct ggml_tensor *, int> use_cnt;
    use_cnt.reserve((size_t) cgraph->n_nodes * 2u);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * t = cgraph->nodes[i];
        if (!t) continue;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t->src[s]) use_cnt[t->src[s]]++;
        }
        if (t->view_src) use_cnt[t->view_src]++;
    }
    if (cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes - 1]) {
        // Protect final graph output from being elided as an intermediate.
        use_cnt[cgraph->nodes[cgraph->n_nodes - 1]]++;
    }

    if (profile >= 2) {
        std::unordered_map<std::string, std::pair<uint32_t, uint64_t>> by_key;
        uint64_t total_cyc = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if (!node || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(node)) continue;
            const char * fused_name = nullptr;
            bool fused = (i + 1 < cgraph->n_nodes) &&
                         try_lower_fused_pair(ctx, node, cgraph->nodes[i + 1], use_cnt[node], &fused_name);
            if (fused) {
                i++;
            } else if (!lower_node(ctx, backend->device, node)) {
                GGML_LOG_ERROR("ggml-c7x: failed to lower node %d (%s, op=%s)\n",
                               i, node->name, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
            if (c7x_cmdbuf_count(ctx->cmdbuf) > 0u) {
                struct c7x_exec_stats st {};
                if (c7x_cmdbuf_submit(ctx->dev_ctx->rt, ctx->cmdbuf, &st) != 0) {
                    GGML_LOG_ERROR("ggml-c7x: PROFILE=2 EXEC_LIST failed at node %d (%s, op=%s, done=%u)\n",
                                   i, node->name, ggml_op_name(node->op), st.cmds_done);
                    c7x_cmdbuf_reset(ctx->cmdbuf);
                    ctx->cmd_nodes.clear();
                    return GGML_STATUS_FAILED;
                }
                std::string key = fused ? std::string(fused_name) : std::string(ggml_op_name(node->op));
                if (!fused && node->op == GGML_OP_MUL_MAT && node->src[0]) {
                    key += ":";
                    key += ggml_type_name(node->src[0]->type);
                    if (node->src[0]->nb[0] != ggml_type_size(node->src[0]->type)) {
                        key += "(T)";
                    }
                    if (node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32) {
                        key += ":k" + std::to_string(node->src[0]->ne[0]) +
                               "n" + std::to_string(node->src[0]->ne[1]) +
                               "m" + std::to_string(node->src[1]->ne[1]);
                    } else {
                        key += ":m=" + std::to_string(node->src[1]->ne[1]);
                    }
                }
                auto & e = by_key[key];
                e.first++;
                e.second += st.cycles;
                total_cyc += st.cycles;
                c7x_cmdbuf_reset(ctx->cmdbuf);
                ctx->cmd_nodes.clear();
            }
        }
        fprintf(stderr, "=== ggml-c7x per-op DSP profile (total=%.2f ms) ===\n", (double) total_cyc / 1e6);
        for (const auto & kv : by_key) {
            fprintf(stderr, "  %-28s x%-4u : %7.2f ms\n",
                    kv.first.c_str(), kv.second.first, (double) kv.second.second / 1e6);
        }
        return GGML_STATUS_SUCCESS;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(node)) {
            continue;
        }
        if (i + 1 < cgraph->n_nodes &&
            try_lower_fused_pair(ctx, node, cgraph->nodes[i + 1], use_cnt[node], nullptr)) {
            i++;
            continue;
        }
        if (!lower_node(ctx, backend->device, node)) {
            GGML_LOG_ERROR("ggml-c7x: failed to lower node %d (%s, op=%s)\n",
                           i, node->name, ggml_op_name(node->op));
            c7x_cmdbuf_reset(ctx->cmdbuf);
            ctx->cmd_nodes.clear();
            return GGML_STATUS_FAILED;
        }
    }

    return flush_cmdbuf(ctx) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static const struct ggml_backend_i ggml_backend_c7x_iface = {
    /* .get_name           = */ ggml_backend_c7x_get_name,
    /* .free               = */ ggml_backend_c7x_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_c7x_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend device interface
// ---------------------------------------------------------------------------

static ggml_guid_t ggml_backend_c7x_guid(void) {
    static ggml_guid guid = {
        0xc7, 0x10, 0x72, 0x1e, 0xbe, 0xa9, 0x1e, 0x64,
        0x4d, 0x4d, 0x41, 0x01, 0x43, 0x37, 0x58, 0x00
    };
    return &guid;
}

static const char * ggml_backend_c7x_dev_get_name(ggml_backend_dev_t dev) {
    auto * dctx = static_cast<ggml_backend_c7x_device_context *>(dev->context);
    return dctx->name.c_str();
}

static const char * ggml_backend_c7x_dev_get_description(ggml_backend_dev_t dev) {
    auto * dctx = static_cast<ggml_backend_c7x_device_context *>(dev->context);
    return dctx->desc.c_str();
}

static void ggml_backend_c7x_dev_get_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    auto * dctx = static_cast<ggml_backend_c7x_device_context *>(dev->context);
    c7x_rt_info info {};
    if (dctx && dctx->rt && c7x_get_info(dctx->rt, &info) == 0) {
        *free_mem  = info.ddr_free;
        *total_mem = info.ddr_total;
    } else {
        *free_mem  = 0;
        *total_mem = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_c7x_dev_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_c7x_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_c7x_dev_get_name(dev);
    props->description = ggml_backend_c7x_dev_get_description(dev);
    props->type        = ggml_backend_c7x_dev_get_type(dev);
    ggml_backend_c7x_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_c7x_dev_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dctx = static_cast<ggml_backend_c7x_device_context *>(dev->context);
    if (!dctx || !dctx->rt) {
        return nullptr;
    }
    auto * ctx = new ggml_backend_c7x_context();
    ctx->dev_ctx = dctx;
    ctx->name    = dctx->name;
    ctx->cmdbuf  = c7x_cmdbuf_create(dctx->rt, 4096u, 256u * 1024u);
    if (!ctx->cmdbuf) {
        delete ctx;
        return nullptr;
    }
    auto * backend = new ggml_backend {
        /* .guid    = */ ggml_backend_c7x_guid(),
        /* .iface   = */ ggml_backend_c7x_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_c7x_dev_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dctx = static_cast<ggml_backend_c7x_device_context *>(dev->context);
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_c7x_buft_iface,
        /* .device  = */ dev,
        /* .context = */ dctx,
    };
    buft.device  = dev;
    buft.context = dctx;
    return &buft;
}

static bool ggml_backend_c7x_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (!op) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1 || op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
                return true;
            }
            if (c7x_tensor_can_repack(src0) &&
                ggml_is_contiguous(src1) && ggml_is_contiguous(op)) {
                // Only accept llama_model_loader's 0-byte probe buffer or actual WEIGHTS buffers
                // so supports_op never promises quantized MUL_MAT on unallocated/non-WEIGHTS tensors.
                if (src0->buffer != nullptr &&
                    (src0->buffer->size == 0 ||
                     src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                    return true;
                }
            }
            // Non-WEIGHTS Q8_0 (e.g. -ctk q8_0 -ctv q8_0 KV cache) via exact C7X_CMD_MATMUL_F32 + SRC0_Q8_0
            if (src0->type == GGML_TYPE_Q8_0 &&
                src0->nb[0] == sizeof(block_q8_0) &&
                (src0->ne[0] % 32) == 0 &&
                (src0->buffer == nullptr || src0->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                return true;
            }
            return false;
        }

        case GGML_OP_RMS_NORM:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return op->src[0] && op->src[1] &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32;

        case GGML_OP_SCALE:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);

        case GGML_OP_UNARY: {
            uint32_t subop = 0;
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                   ggml_unary_to_c7x(ggml_get_unary_op(op), &subop);
        }

        case GGML_OP_GLU: {
            uint32_t subop = 0;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (op->src[1] && (op->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(op->src[1]))) {
                return false;
            }
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                   ggml_glu_to_c7x(ggml_get_glu_op(op), &subop);
        }

        case GGML_OP_SOFT_MAX: {
            const struct ggml_tensor * src0  = op->src[0];
            const struct ggml_tensor * mask  = op->src[1];
            const struct ggml_tensor * sinks = op->src[2];
            if (!src0 || src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                return false;
            }
            if (mask && mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
                return false;
            }
            if (mask && mask->nb[0] != ggml_type_size(mask->type)) {
                return false;
            }
            if (sinks && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks))) {
                return false;
            }
            return true;
        }

        case GGML_OP_ROPE: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * pos  = op->src[1];
            const struct ggml_tensor * ff   = op->src[2];
            if (!src0 || !pos || src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
                pos->type != GGML_TYPE_I32) {
                return false;
            }
            if (src0->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                return false;
            }
            if (ff && (ff->type != GGML_TYPE_F32 || !ggml_is_contiguous(ff))) {
                return false;
            }
            const int32_t mode   = ((const int32_t *) op->op_params)[2];
            const int32_t n_offs = ((const int32_t *) op->op_params)[15];
            return (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) && n_offs == 0;
        }

        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * idx  = op->src[1];
            uint32_t dt = 0;
            if (!src0 || !idx || idx->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q4_1 ||
                src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_K ||
                src0->type == GGML_TYPE_Q6_K) {
                return c7x_lookup_repacked(src0, dev, nullptr);
            }
            return ggml_type_to_c7x_dtype(src0->type, &dt) &&
                   src0->type != GGML_TYPE_I32;
        }

        case GGML_OP_CPY:
        case GGML_OP_CONT: {
            uint32_t st = 0, dt = 0;
            if (!op->src[0] || !ggml_type_to_c7x_dtype(op->src[0]->type, &st) ||
                !ggml_type_to_c7x_dtype(op->type, &dt)) {
                return false;
            }
            if (op->src[0]->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_0) {
                return false;
            }
            // Reject unsupported I32 <-> float/quant casts (only I32 <-> I32)
            if ((op->src[0]->type == GGML_TYPE_I32) != (op->type == GGML_TYPE_I32)) {
                return false;
            }
            if ((op->src[0]->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q8_0) &&
                !ggml_are_same_shape(op->src[0], op)) {
                return false;
            }
            if (op->src[0]->type == GGML_TYPE_Q8_0 && op->type == GGML_TYPE_Q8_0) {
                return false;
            }
            return true;
        }

        case GGML_OP_SET_ROWS: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * idx  = op->src[1];
            if (!src0 || !idx || src0->type != GGML_TYPE_F32 ||
                (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32)) {
                return false;
            }
            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 ||
                   op->type == GGML_TYPE_Q8_0;
        }

        case GGML_OP_NORM:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   ggml_are_same_shape(op->src[0], op);

        case GGML_OP_IM2COL: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1 || !ggml_is_contiguous(op)) {
                return false;
            }
            if (op->type == GGML_TYPE_F32) {
                return src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float);
            }
            if (op->type == GGML_TYPE_F16) {
                return (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                       src1->nb[0] == ggml_type_size(src1->type);
            }
            return false;
        }

        case GGML_OP_POOL_2D: {
            const struct ggml_tensor * src0 = op->src[0];
            if (!src0 || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                return false;
            }
            if (src0->nb[0] != ggml_type_size(src0->type)) {
                return false;
            }
            if (src0->ne[3] > 1 && src0->nb[3] != (size_t) src0->ne[2] * src0->nb[2]) {
                return false;
            }
            const int32_t * opts = (const int32_t *) op->op_params;
            return (opts[0] == GGML_OP_POOL_AVG || opts[0] == GGML_OP_POOL_MAX) &&
                   opts[1] > 0 && opts[2] > 0;
        }

        case GGML_OP_CONV_2D: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!src0 || !src1 || op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                return false;
            }
            return ggml_is_contiguous(src0) && src0->ne[2] == src1->ne[2];
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor * q       = op->src[0];
            const struct ggml_tensor * k       = op->src[1];
            const struct ggml_tensor * v       = op->src[2];
            const struct ggml_tensor * mask    = op->src[3];
            const struct ggml_tensor * sinks   = op->src[4];
            const struct ggml_tensor * kv_rows = op->src[5];
            if (!q || !k || !v || kv_rows != nullptr) {
                return false;
            }
            if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32) {
                return false;
            }
            if ((k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
                (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
                return false;
            }
            if (q->nb[0] != sizeof(float) || op->nb[0] != sizeof(float) ||
                k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) {
                return false;
            }
            if ((k->type == GGML_TYPE_Q8_0 && (k->ne[0] % 32) != 0) ||
                (v->type == GGML_TYPE_Q8_0 && (v->ne[0] % 32) != 0)) {
                return false;
            }
            if (k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3] ||
                k->ne[2] <= 0 || k->ne[3] <= 0 ||
                (q->ne[2] % k->ne[2]) != 0 || (q->ne[3] % k->ne[3]) != 0) {
                return false;
            }
            if (mask && ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
                         mask->nb[0] != ggml_type_size(mask->type))) {
                return false;
            }
            if (sinks && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks))) {
                return false;
            }
            return true;
        }

        default:
            return false;
    }
}

static bool ggml_backend_c7x_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->device == dev;
}

static const struct ggml_backend_device_i ggml_backend_c7x_dev_iface = {
    /* .get_name             = */ ggml_backend_c7x_dev_get_name,
    /* .get_description      = */ ggml_backend_c7x_dev_get_description,
    /* .get_memory           = */ ggml_backend_c7x_dev_get_memory,
    /* .get_type             = */ ggml_backend_c7x_dev_get_type,
    /* .get_props            = */ ggml_backend_c7x_dev_get_props,
    /* .init_backend         = */ ggml_backend_c7x_dev_init,
    /* .get_buffer_type      = */ ggml_backend_c7x_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_c7x_dev_supports_op,
    /* .supports_buft        = */ ggml_backend_c7x_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend registry interface & dynamic plugin entry points
// ---------------------------------------------------------------------------

struct ggml_backend_c7x_reg_context {
    std::vector<ggml_backend_dev_t> devices;
    std::vector<ggml_backend_c7x_device_context *> dev_ctxs;

    ~ggml_backend_c7x_reg_context() {
        for (auto * d : devices) delete d;
        for (auto * dc : dev_ctxs) {
            if (dc->rt) c7x_close(dc->rt);
            delete dc;
        }
    }
};

static const char * ggml_backend_c7x_reg_get_name(ggml_backend_reg_t reg) {
    return "C7X";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_c7x_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * rctx = static_cast<ggml_backend_c7x_reg_context *>(reg->context);
    return rctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_c7x_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * rctx = static_cast<ggml_backend_c7x_reg_context *>(reg->context);
    GGML_ASSERT(index < rctx->devices.size());
    return rctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_c7x_reg_iface = {
    /* .get_name         = */ ggml_backend_c7x_reg_get_name,
    /* .get_device_count = */ ggml_backend_c7x_reg_get_device_count,
    /* .get_device       = */ ggml_backend_c7x_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_c7x_reg(void) {
    static std::mutex init_mtx;
    static struct ggml_backend_reg reg;
    static ggml_backend_c7x_reg_context * rctx = nullptr;
    static bool initialized = false;

    std::lock_guard<std::mutex> lk(init_mtx);
    if (!initialized) {
        rctx = new ggml_backend_c7x_reg_context();
        reg = {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_c7x_reg_iface,
            /* .context     = */ rctx,
        };

        struct c7x_open_opts opts {};
        opts.transport = C7X_TRANSPORT_AUTO;
        if (const char * env_mb = std::getenv("GGML_C7X_EMU_MB")) {
            long mb = std::atol(env_mb);
            if (mb > 0) opts.emu_ddr_bytes = (size_t) mb * 1024u * 1024u;
        }

        c7x_ctx_t * rt = c7x_open(&opts);
        if (rt) {
            auto * dctx = new ggml_backend_c7x_device_context();
            dctx->rt = rt;
            c7x_get_info(rt, &dctx->info);
            dctx->name = "C7X0";
            dctx->desc = dctx->info.is_hostemu
                ? "TI C71x + MMA (x86_64 C7000 Host Emulation)"
                : "TI TDA4VM C71x DSP + MMA v1 (BeagleBone AI-64)";
            if (const char * no_msmc = std::getenv("GGML_C7X_NO_MSMC")) {
                if (no_msmc[0] == '1') dctx->enable_msmc_pin = false;
            }
            auto * dev = new ggml_backend_device {
                /* .iface   = */ ggml_backend_c7x_dev_iface,
                /* .reg     = */ &reg,
                /* .context = */ dctx,
            };
            rctx->dev_ctxs.push_back(dctx);
            rctx->devices.push_back(dev);
        }
        initialized = true;
    }
    return &reg;
}

static int ggml_backend_c7x_score(void) {
    return 1;
}

GGML_BACKEND_DL_IMPL(ggml_backend_c7x_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_c7x_score)
