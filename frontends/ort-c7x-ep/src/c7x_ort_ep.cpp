/* SPDX-License-Identifier: MIT */
/*
 * C7x ONNX Runtime Execution Provider (ort-c7x-ep) over libc7x command buffers.
 */

#include "c7x_ort_ep.h"
#include "onnx_proto.hpp"

#include "c7x_cmd_proto.h"
#include "c7x_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static uint32_t align64_u32(uint32_t v) { return (v + 63u) & ~63u; }

static size_t dtype_size(int32_t dt) {
	switch (dt) {
		case C7X_ORT_FLOAT:   return 4u;
		case C7X_ORT_UINT8:   return 1u;
		case C7X_ORT_INT8:    return 1u;
		case C7X_ORT_INT32:   return 4u;
		case C7X_ORT_INT64:   return 8u;
		case C7X_ORT_FLOAT16: return 2u;
		default:              return 0u;
	}
}

static bool validate_tensor_bytes(const c7x_onnx::TensorProto &t, const std::vector<uint8_t> &bytes) {
	size_t esz = dtype_size(t.data_type);
	size_t ne  = t.num_elements();
	if (esz == 0u || ne == 0u) return false;
	return (bytes.size() % esz) == 0u && (bytes.size() / esz) == ne;
}

struct CompiledTensor {
	std::string          name;
	int32_t              dtype = C7X_ORT_UNDEFINED;
	std::vector<int64_t> dims;
	size_t               nbytes = 0;
	c7x_buf_t           *buf = nullptr;
	bool                 owns_buf = false;

	size_t num_elements() const {
		if (dims.empty()) return 1u;
		size_t n = 1u;
		for (int64_t d : dims) {
			if (d <= 0) return 0u;
			n *= (size_t)d;
		}
		return n;
	}
};

static bool read_scalar_f32(const c7x_onnx::GraphProto &g, const std::string &name,
			    float &out, bool require_positive = false) {
	auto it = g.initializers.find(name);
	if (it == g.initializers.end() || it->second.data_type != C7X_ORT_FLOAT) return false;
	auto bytes = it->second.as_bytes();
	if (!validate_tensor_bytes(it->second, bytes) || it->second.num_elements() != 1u) return false;
	std::memcpy(&out, bytes.data(), 4u);
	if (!std::isfinite(out)) return false;
	if (require_positive && out <= 0.0f) return false;
	return true;
}

static bool read_vec_f32(const c7x_onnx::GraphProto &g, const std::string &name,
			 std::vector<float> &out, bool require_positive = false) {
	auto it = g.initializers.find(name);
	if (it == g.initializers.end() || it->second.data_type != C7X_ORT_FLOAT) return false;
	auto bytes = it->second.as_bytes();
	if (!validate_tensor_bytes(it->second, bytes)) return false;
	out.resize(bytes.size() / 4u);
	std::memcpy(out.data(), bytes.data(), bytes.size());
	for (float v : out) {
		if (!std::isfinite(v) || (require_positive && v <= 0.0f)) return false;
	}
	return true;
}

/* Read scalar or uniform per-channel zero-point; rejects non-uniform per-channel zero-points. */
static bool read_uniform_zp(const c7x_onnx::GraphProto &g, const std::string &name,
			    int32_t &zp_out, int32_t &dtype_out, size_t expected_max_ch = 1u) {
	auto it = g.initializers.find(name);
	if (it == g.initializers.end()) return false;
	dtype_out = it->second.data_type;
	if (dtype_out != C7X_ORT_UINT8 && dtype_out != C7X_ORT_INT8) return false;
	auto bytes = it->second.as_bytes();
	if (!validate_tensor_bytes(it->second, bytes)) return false;
	if (bytes.size() != 1u && bytes.size() != expected_max_ch) return false;
	for (size_t i = 1u; i < bytes.size(); i++) {
		if (bytes[i] != bytes[0]) {
			fprintf(stderr, "ort-c7x-ep: non-uniform per-channel zero_point '%s' not supported\n",
				name.c_str());
			return false;
		}
	}
	zp_out = (dtype_out == C7X_ORT_UINT8) ? (int32_t)bytes[0] : (int32_t)(int8_t)bytes[0];
	return true;
}

static bool read_vec_i64(const c7x_onnx::GraphProto &g, const std::string &name,
			 std::vector<int64_t> &out) {
	auto it = g.initializers.find(name);
	if (it == g.initializers.end() || it->second.data_type != C7X_ORT_INT64) return false;
	auto bytes = it->second.as_bytes();
	if (!validate_tensor_bytes(it->second, bytes)) return false;
	out.resize(bytes.size() / 8u);
	std::memcpy(out.data(), bytes.data(), bytes.size());
	return true;
}

static bool compute_2d_pads_and_out(const std::string &auto_pad,
				    uint32_t IH, uint32_t IW, uint32_t KH, uint32_t KW,
				    int32_t s1, int32_t s0, int32_t d1, int32_t d0,
				    const std::vector<int64_t> &pads_attr,
				    int32_t &p1, int32_t &p0, uint32_t &OH, uint32_t &OW) {
	if (s0 <= 0 || s1 <= 0 || d0 <= 0 || d1 <= 0) return false;
	int32_t ekh = d1 * ((int32_t)KH - 1) + 1;
	int32_t ekw = d0 * ((int32_t)KW - 1) + 1;
	int32_t p1_end = 0, p0_end = 0;

	if (auto_pad.empty() || auto_pad == "NOTSET") {
		p1     = pads_attr.size() >= 2 ? (int32_t)pads_attr[0] : 0;
		p0     = pads_attr.size() >= 2 ? (int32_t)pads_attr[1] : 0;
		p1_end = pads_attr.size() >= 4 ? (int32_t)pads_attr[2] : p1;
		p0_end = pads_attr.size() >= 4 ? (int32_t)pads_attr[3] : p0;
		int32_t oh_s = ((int32_t)IH + p1 + p1_end - ekh) / s1 + 1;
		int32_t ow_s = ((int32_t)IW + p0 + p0_end - ekw) / s0 + 1;
		if (oh_s <= 0 || ow_s <= 0) return false;
		OH = (uint32_t)oh_s;
		OW = (uint32_t)ow_s;
		return true;
	}
	if (auto_pad == "VALID") {
		p1 = p0 = 0;
		int32_t oh_s = ((int32_t)IH - ekh) / s1 + 1;
		int32_t ow_s = ((int32_t)IW - ekw) / s0 + 1;
		if (oh_s <= 0 || ow_s <= 0) return false;
		OH = (uint32_t)oh_s;
		OW = (uint32_t)ow_s;
		return true;
	}
	if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
		OH = (IH + (uint32_t)s1 - 1u) / (uint32_t)s1;
		OW = (IW + (uint32_t)s0 - 1u) / (uint32_t)s0;
		int32_t pad_h = std::max<int32_t>(0, ((int32_t)OH - 1) * s1 + ekh - (int32_t)IH);
		int32_t pad_w = std::max<int32_t>(0, ((int32_t)OW - 1) * s0 + ekw - (int32_t)IW);
		if (auto_pad == "SAME_UPPER") {
			p1 = pad_h / 2;
			p0 = pad_w / 2;
		} else {
			p1 = (pad_h + 1) / 2;
			p0 = (pad_w + 1) / 2;
		}
		return true;
	}
	return false;
}

/* Pad a 1..4D ONNX NCHW shape into 4D [d0, d1, d2, d3] (right-aligned for GGML/C7x binop). */
static void to_c7x_4d_shape_strides(const std::vector<int64_t> &dims, size_t elem_sz,
				    uint32_t ne[4], uint64_t nb[4]) {
	int nd = (int)dims.size();
	uint64_t stride = elem_sz;
	for (int i = 0; i < 4; i++) {
		int src_idx = nd - 1 - i;
		uint32_t d = (src_idx >= 0 && dims[src_idx] > 0) ? (uint32_t)dims[src_idx] : 1u;
		ne[i] = d;
		nb[i] = stride;
		stride *= d;
	}
}

} /* anonymous namespace */

struct c7x_ort_session {
	c7x_ctx_t                                        *rt = nullptr;
	c7x_cmdbuf_t                                     *cb = nullptr;
	c7x_ort_ep_options                                opts {};
	std::unordered_map<std::string, CompiledTensor>   tensors;
	std::vector<c7x_buf_t *>                          aux_bufs;
	std::vector<std::string>                          input_names;
	std::vector<std::string>                          output_names;

	~c7x_ort_session() {
		if (cb) c7x_cmdbuf_destroy(cb);
		for (auto &kv : tensors) {
			if (kv.second.owns_buf && kv.second.buf && rt) {
				c7x_free(rt, kv.second.buf);
			}
		}
		for (c7x_buf_t *b : aux_bufs) {
			if (b && rt) c7x_free(rt, b);
		}
		if (rt) c7x_close(rt);
	}

	c7x_buf_t *alloc_buf(size_t bytes, bool prefer_msmc = false) {
		if (bytes == 0u) bytes = 64u;
		c7x_buf_t *b = nullptr;
		if (prefer_msmc && opts.use_msmc_weights) {
			b = c7x_alloc(rt, bytes, C7X_MEM_MSMC);
		}
		if (!b) {
			b = c7x_alloc(rt, bytes, C7X_MEM_DDR);
		}
		return b;
	}

	CompiledTensor *alloc_tensor(const std::string &name, int32_t dtype,
				     const std::vector<int64_t> &dims, bool prefer_msmc = false) {
		if (name.empty() || tensors.count(name)) {
			fprintf(stderr, "ort-c7x-ep: duplicate or empty tensor name '%s'\n", name.c_str());
			return nullptr;
		}
		CompiledTensor ct;
		ct.name   = name;
		ct.dtype  = dtype;
		ct.dims   = dims;
		size_t ne = ct.num_elements();
		size_t es = dtype_size(dtype);
		if (ne == 0u || es == 0u) return nullptr;
		ct.nbytes = ne * es;
		ct.buf    = alloc_buf(ct.nbytes, prefer_msmc);
		if (!ct.buf) return nullptr;
		ct.owns_buf = true;
		auto res = tensors.emplace(name, std::move(ct));
		return &res.first->second;
	}

	CompiledTensor *alias_tensor(const std::string &name, const CompiledTensor &src,
				     const std::vector<int64_t> &new_dims) {
		if (name.empty() || tensors.count(name)) {
			fprintf(stderr, "ort-c7x-ep: duplicate alias tensor name '%s'\n", name.c_str());
			return nullptr;
		}
		CompiledTensor ct;
		ct.name     = name;
		ct.dtype    = src.dtype;
		ct.dims     = new_dims;
		ct.nbytes   = src.nbytes;
		ct.buf      = src.buf;
		ct.owns_buf = false;
		auto res = tensors.emplace(name, std::move(ct));
		return &res.first->second;
	}

	c7x_buf_t *upload_aux(const void *data, size_t bytes, bool prefer_msmc = false) {
		c7x_buf_t *b = alloc_buf(bytes, prefer_msmc);
		if (!b) return nullptr;
		if (bytes > 0u) {
			c7x_write(rt, b, 0u, data, bytes);
		}
		aux_bufs.push_back(b);
		return b;
	}

	CompiledTensor *ensure_initializer_tensor(const c7x_onnx::GraphProto &g,
						  const std::string &name) {
		auto it = tensors.find(name);
		if (it != tensors.end()) return &it->second;
		auto iit = g.initializers.find(name);
		if (iit == g.initializers.end()) return nullptr;
		auto bytes = iit->second.as_bytes();
		if (!validate_tensor_bytes(iit->second, bytes)) {
			fprintf(stderr, "ort-c7x-ep: initializer '%s' byte size mismatch\n", name.c_str());
			return nullptr;
		}
		CompiledTensor *ct = alloc_tensor(name, iit->second.data_type, iit->second.dims, false);
		if (!ct) return nullptr;
		c7x_write(rt, ct->buf, 0u, bytes.data(), ct->nbytes);
		return ct;
	}
};

static bool compile_graph(c7x_ort_session *s, const c7x_onnx::GraphProto &g) {
	/* Count consumers of every tensor for safe Conv+Relu fusion. */
	std::unordered_map<std::string, int> use_count;
	for (const auto &nd : g.nodes) {
		for (const auto &in : nd.inputs) {
			if (!in.empty()) use_count[in]++;
		}
	}
	for (const auto &out : g.outputs) {
		use_count[out.name]++;
	}

	/* Allocate graph inputs (skipping initializers). */
	for (const auto &vi : g.inputs) {
		if (g.initializers.count(vi.name)) continue;
		if (!s->alloc_tensor(vi.name, vi.elem_type, vi.dims, false)) {
			fprintf(stderr, "ort-c7x-ep: failed to allocate input '%s'\n", vi.name.c_str());
			return false;
		}
		s->input_names.push_back(vi.name);
	}
	for (const auto &vo : g.outputs) {
		s->output_names.push_back(vo.name);
	}

	for (size_t ni = 0; ni < g.nodes.size(); ni++) {
		const auto &nd = g.nodes[ni];
		const std::string &op = nd.op_type;

		if (!c7x_ort_ep_op_supported(op.c_str())) {
			fprintf(stderr, "ort-c7x-ep: unsupported ONNX op '%s' (node '%s')\n",
				op.c_str(), nd.name.c_str());
			return false;
		}
		if (nd.inputs.empty() || nd.outputs.empty()) return false;

		/* ---------- Zero-copy shape ops ---------- */
		if (op == "Reshape" || op == "Flatten" || op == "Squeeze" || op == "Unsqueeze") {
			CompiledTensor *in = s->ensure_initializer_tensor(g, nd.inputs[0]);
			if (!in) return false;
			std::vector<int64_t> new_dims;
			if (op == "Flatten") {
				int64_t axis = nd.get_int("axis", 1);
				if (axis < 0) axis += (int64_t)in->dims.size();
				if (axis < 0 || axis > (int64_t)in->dims.size()) return false;
				int64_t d0 = 1, d1 = 1;
				for (int64_t i = 0; i < (int64_t)in->dims.size(); i++) {
					if (i < axis) d0 *= in->dims[i];
					else d1 *= in->dims[i];
				}
				new_dims = {d0, d1};
			} else if (op == "Reshape") {
				if (nd.inputs.size() < 2u) return false;
				std::vector<int64_t> sh;
				if (!read_vec_i64(g, nd.inputs[1], sh)) return false;
				int64_t total = (int64_t)in->num_elements();
				int64_t known = 1, neg_idx = -1;
				for (size_t i = 0; i < sh.size(); i++) {
					if (sh[i] == 0 && i < in->dims.size()) sh[i] = in->dims[i];
					if (sh[i] == -1) {
						if (neg_idx >= 0) return false;
						neg_idx = (int64_t)i;
					} else if (sh[i] > 0) {
						known *= sh[i];
					} else {
						return false;
					}
				}
				if (neg_idx >= 0) {
					if (known <= 0 || (total % known) != 0) return false;
					sh[neg_idx] = total / known;
				} else if (known != total) {
					return false;
				}
				new_dims = std::move(sh);
			} else {
				/* Squeeze / Unsqueeze: support both opset <13 attr and opset 13+ input[1] axes. */
				std::vector<int64_t> axes;
				if (nd.inputs.size() >= 2u && !nd.inputs[1].empty()) {
					if (!read_vec_i64(g, nd.inputs[1], axes)) return false;
				} else {
					axes = nd.get_ints("axes");
				}
				if (op == "Squeeze") {
					std::vector<int64_t> norm_axes;
					for (int64_t a : axes) {
						if (a < 0) a += (int64_t)in->dims.size();
						if (a < 0 || a >= (int64_t)in->dims.size() || in->dims[a] != 1) return false;
						norm_axes.push_back(a);
					}
					for (size_t i = 0; i < in->dims.size(); i++) {
						bool drop = norm_axes.empty()
							    ? (in->dims[i] == 1)
							    : (std::find(norm_axes.begin(), norm_axes.end(), (int64_t)i) != norm_axes.end());
						if (!drop) new_dims.push_back(in->dims[i]);
					}
					if (new_dims.empty()) new_dims.push_back(1);
				} else { /* Unsqueeze */
					if (axes.empty()) return false;
					size_t out_nd = in->dims.size() + axes.size();
					new_dims.assign(out_nd, 0);
					for (int64_t a : axes) {
						if (a < 0) a += (int64_t)out_nd;
						if (a < 0 || (size_t)a >= out_nd || new_dims[a] == 1) return false;
						new_dims[a] = 1;
					}
					size_t si = 0;
					for (size_t i = 0; i < out_nd; i++) {
						if (new_dims[i] == 0 && si < in->dims.size()) new_dims[i] = in->dims[si++];
					}
				}
			}
			if (!s->alias_tensor(nd.outputs[0], *in, new_dims)) return false;
			continue;
		}

		/* ---------- Standalone DequantizeLinear (INT8/UINT8 -> FP32, tensor or per-axis) ---------- */
		if (op == "DequantizeLinear") {
			if (nd.inputs.size() < 2u) return false;
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			if (!x || (x->dtype != C7X_ORT_INT8 && x->dtype != C7X_ORT_UINT8)) return false;

			std::vector<float> scales;
			if (!read_vec_f32(g, nd.inputs[1], scales, true)) return false;

			std::vector<uint8_t> zp_bytes;
			int32_t zp_dt = x->dtype;
			bool has_zp = (nd.inputs.size() >= 3u && !nd.inputs[2].empty());
			if (has_zp) {
				auto zit = g.initializers.find(nd.inputs[2]);
				if (zit == g.initializers.end() || zit->second.data_type != x->dtype) return false;
				zp_bytes = zit->second.as_bytes();
				if (!validate_tensor_bytes(zit->second, zp_bytes)) return false;
				if (zp_bytes.size() != scales.size()) return false;
			}

			struct c7x_dequant_desc dd {};
			c7x_buf_t *b_sc = nullptr;
			c7x_buf_t *b_zp = nullptr;

			if (scales.size() == 1u) {
				dd.outer        = 1u;
				dd.axis_dim     = 1u;
				dd.inner        = (uint32_t)x->num_elements();
				dd.has_zp       = has_zp ? 1u : 0u;
				dd.scalar_zp    = has_zp
						  ? ((zp_dt == C7X_ORT_UINT8) ? (int32_t)zp_bytes[0]
									      : (int32_t)(int8_t)zp_bytes[0])
						  : 0;
				dd.scalar_scale = scales[0];
			} else {
				int64_t axis = nd.get_int("axis", 1);
				if (axis < 0) axis += (int64_t)x->dims.size();
				if (axis < 0 || axis >= (int64_t)x->dims.size()) return false;
				if ((size_t)x->dims[axis] != scales.size()) return false;
				uint32_t outer = 1u, inner = 1u;
				for (int64_t i = 0; i < axis; i++) outer *= (uint32_t)x->dims[i];
				for (size_t i = (size_t)axis + 1u; i < x->dims.size(); i++) inner *= (uint32_t)x->dims[i];
				dd.outer    = outer;
				dd.axis_dim = (uint32_t)scales.size();
				dd.inner    = inner;
				dd.has_zp   = has_zp ? 1u : 0u;
				b_sc = s->upload_aux(scales.data(), scales.size() * sizeof(float), false);
				if (!b_sc) return false;
				if (has_zp) {
					b_zp = s->upload_aux(zp_bytes.data(), zp_bytes.size(), false);
					if (!b_zp) return false;
				}
			}

			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, x->dims, false);
			if (!y) return false;

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &dd, sizeof(dd));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op    = C7X_CMD_DEQUANTIZE_LIN;
			cmd.flags = (x->dtype == C7X_ORT_UINT8) ? C7X_CMD_FLAG_A_UINT8 : 0u;
			cmd.src0  = c7x_buf_phys(x->buf);
			cmd.src1  = b_sc ? c7x_buf_phys(b_sc) : 0ull;
			cmd.src2  = b_zp ? c7x_buf_phys(b_zp) : 0ull;
			cmd.src3  = dp;
			cmd.dst   = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- QLinearConv (strict ONNX semantics: always requantizes to y_dtype) ---------- */
		if (op == "QLinearConv") {
			if (nd.inputs.size() < 8u) return false;
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			auto wit = g.initializers.find(nd.inputs[3]);
			if (!x || wit == g.initializers.end() || x->dims.size() != 4u || wit->second.dims.size() != 4u) {
				fprintf(stderr, "ort-c7x-ep: QLinearConv '%s' invalid shapes\n", nd.name.c_str());
				return false;
			}
			if (x->dtype != C7X_ORT_INT8 && x->dtype != C7X_ORT_UINT8) return false;
			if (wit->second.data_type != C7X_ORT_INT8 && wit->second.data_type != C7X_ORT_UINT8) return false;

			auto w_bytes = wit->second.as_bytes();
			if (!validate_tensor_bytes(wit->second, w_bytes)) return false;

			const uint32_t N  = (uint32_t)x->dims[0];
			const uint32_t IC = (uint32_t)x->dims[1];
			const uint32_t IH = (uint32_t)x->dims[2];
			const uint32_t IW = (uint32_t)x->dims[3];
			const uint32_t OC = (uint32_t)wit->second.dims[0];
			const uint32_t WIC = (uint32_t)wit->second.dims[1];
			const uint32_t KH = (uint32_t)wit->second.dims[2];
			const uint32_t KW = (uint32_t)wit->second.dims[3];
			const uint32_t groups = (uint32_t)nd.get_int("group", 1);
			if (groups == 0u || (uint64_t)WIC * groups != IC) return false;

			float x_scale = 0.0f, y_scale = 0.0f;
			int32_t x_zp_raw = 0, x_zp_dt = 0, w_zp_raw = 0, w_zp_dt = 0, y_zp = 0, y_zp_dt = 0;
			std::vector<float> w_scales;
			if (!read_scalar_f32(g, nd.inputs[1], x_scale, true) ||
			    !read_uniform_zp(g, nd.inputs[2], x_zp_raw, x_zp_dt, 1u) ||
			    !read_vec_f32(g, nd.inputs[4], w_scales, true) ||
			    !read_uniform_zp(g, nd.inputs[5], w_zp_raw, w_zp_dt, OC) ||
			    !read_scalar_f32(g, nd.inputs[6], y_scale, true) ||
			    !read_uniform_zp(g, nd.inputs[7], y_zp, y_zp_dt, 1u)) {
				fprintf(stderr, "ort-c7x-ep: QLinearConv '%s' invalid quant params\n", nd.name.c_str());
				return false;
			}
			if (x_zp_dt != x->dtype || w_zp_dt != wit->second.data_type) return false;
			if (w_scales.size() != 1u && w_scales.size() != OC) return false;

			auto strides   = nd.get_ints("strides");
			auto pads      = nd.get_ints("pads");
			auto dilations = nd.get_ints("dilations");
			int32_t s1 = strides.size() >= 2 ? (int32_t)strides[0] : 1;
			int32_t s0 = strides.size() >= 2 ? (int32_t)strides[1] : 1;
			int32_t d1 = dilations.size() >= 2 ? (int32_t)dilations[0] : 1;
			int32_t d0 = dilations.size() >= 2 ? (int32_t)dilations[1] : 1;
			int32_t p1 = 0, p0 = 0;
			uint32_t OH = 0u, OW = 0u;
			if (!compute_2d_pads_and_out(nd.get_str("auto_pad"), IH, IW, KH, KW,
						     s1, s0, d1, d0, pads, p1, p0, OH, OW)) {
				return false;
			}

			uint32_t flags = 0u;
			if (x->dtype == C7X_ORT_UINT8) flags |= C7X_CMD_FLAG_A_UINT8;
			if (wit->second.data_type == C7X_ORT_UINT8) flags |= C7X_CMD_FLAG_B_UINT8;
			if (y_zp_dt == C7X_ORT_UINT8) flags |= C7X_CMD_FLAG_Y_UINT8;

			int w_is_u8 = (wit->second.data_type == C7X_ORT_UINT8) ? 1 : 0;
			c7x_buf_t *b_w = nullptr;
			uint64_t bsum_phys = 0ull;
			uint32_t k_pad = 0u, n_pad = 0u;

			if (groups == 1u) {
				uint32_t k_col = IC * KH * KW;
				k_pad = align64_u32(k_col);
				n_pad = align64_u32(OC);
				size_t bsum_off = 0u;
				size_t packed_sz = c7x_qlinear_packed_bytes(OC, k_col, &bsum_off);
				std::vector<uint8_t> packed(packed_sz, 0u);
				c7x_pack_qlinear_weights(w_bytes.data(), w_is_u8, OC, k_col,
							 reinterpret_cast<int8_t *>(packed.data()),
							 reinterpret_cast<int32_t *>(packed.data() + bsum_off));
				b_w = s->upload_aux(packed.data(), packed_sz, true);
				if (!b_w) return false;
				bsum_phys = c7x_buf_phys(b_w) + bsum_off;
			} else if (groups == IC && groups == OC) {
				uint32_t k_dw = KH * KW;
				std::vector<int8_t> w_dw((size_t)OC * k_dw);
				std::vector<int32_t> bsum(OC, 0);
				for (uint32_t oc = 0u; oc < OC; oc++) {
					int32_t sum = 0;
					for (uint32_t j = 0u; j < k_dw; j++) {
						int32_t v = w_is_u8
							    ? ((int32_t)w_bytes[(size_t)oc * k_dw + j] - 128)
							    : (int32_t)(int8_t)w_bytes[(size_t)oc * k_dw + j];
						w_dw[(size_t)oc * k_dw + j] = (int8_t)v;
						sum += v;
					}
					bsum[oc] = sum;
				}
				b_w = s->upload_aux(w_dw.data(), w_dw.size(), true);
				c7x_buf_t *b_bsum = s->upload_aux(bsum.data(), bsum.size() * sizeof(int32_t), false);
				if (!b_w || !b_bsum) return false;
				bsum_phys = c7x_buf_phys(b_bsum);
			} else {
				fprintf(stderr, "ort-c7x-ep: unsupported QLinearConv group=%u\n", groups);
				return false;
			}

			c7x_buf_t *b_ws = s->upload_aux(w_scales.data(), w_scales.size() * sizeof(float), false);
			if (!b_ws) return false;

			uint64_t bias_phys = 0ull;
			if (nd.inputs.size() >= 9u && !nd.inputs[8].empty()) {
				auto bit = g.initializers.find(nd.inputs[8]);
				if (bit == g.initializers.end()) return false;
				auto bb = bit->second.as_bytes();
				if (!validate_tensor_bytes(bit->second, bb) || bit->second.num_elements() != OC) return false;
				if (bit->second.data_type != C7X_ORT_INT32 && bit->second.data_type != C7X_ORT_FLOAT) return false;
				c7x_buf_t *b_bias = s->upload_aux(bb.data(), bb.size(), false);
				if (!b_bias) return false;
				bias_phys = c7x_buf_phys(b_bias);
				if (bit->second.data_type == C7X_ORT_INT32) {
					flags |= C7X_CMD_FLAG_BIAS_I32;
				}
			}

			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], y_zp_dt, {N, OC, OH, OW}, false);
			if (!y) return false;

			struct c7x_qlinear_conv2d_desc cd {};
			cd.N  = N;  cd.IC = IC; cd.IH = IH; cd.IW = IW;
			cd.OC = OC; cd.KH = KH; cd.KW = KW; cd.OH = OH; cd.OW = OW;
			cd.groups = groups;
			cd.s0 = s0; cd.s1 = s1; cd.p0 = p0; cd.p1 = p1; cd.d0 = d0; cd.d1 = d1;
			cd.k_pad = k_pad;
			cd.n_pad = n_pad;
			cd.a_zp  = (x->dtype == C7X_ORT_UINT8) ? (x_zp_raw - 128) : x_zp_raw;
			cd.b_zp  = w_is_u8 ? (w_zp_raw - 128) : w_zp_raw;
			cd.y_zp  = y_zp;
			cd.per_channel_b = (w_scales.size() > 1u) ? 1u : 0u;
			cd.a_scale       = x_scale;
			cd.y_scale       = y_scale;
			cd.b_sum_addr    = bsum_phys;
			cd.b_scales_addr = c7x_buf_phys(b_ws);

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &cd, sizeof(cd));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op    = C7X_CMD_QLINEAR_CONV2D;
			cmd.flags = flags;
			cmd.src0  = c7x_buf_phys(b_w);
			cmd.src1  = c7x_buf_phys(x->buf);
			cmd.src2  = bias_phys;
			cmd.src3  = dp;
			cmd.dst   = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- QLinearMatMul (strict ONNX semantics) ---------- */
		if (op == "QLinearMatMul") {
			if (nd.inputs.size() < 8u) return false;
			CompiledTensor *a = s->ensure_initializer_tensor(g, nd.inputs[0]);
			auto bit = g.initializers.find(nd.inputs[3]);
			if (!a || bit == g.initializers.end() || a->dims.size() < 2u || bit->second.dims.size() != 2u) {
				fprintf(stderr, "ort-c7x-ep: QLinearMatMul '%s' invalid shapes\n", nd.name.c_str());
				return false;
			}
			if (a->dtype != C7X_ORT_INT8 && a->dtype != C7X_ORT_UINT8) return false;
			if (bit->second.data_type != C7X_ORT_INT8 && bit->second.data_type != C7X_ORT_UINT8) return false;

			auto b_raw = bit->second.as_bytes();
			if (!validate_tensor_bytes(bit->second, b_raw)) return false;

			uint32_t K = (uint32_t)a->dims.back();
			uint32_t M = (uint32_t)(a->num_elements() / K);
			uint32_t BK = (uint32_t)bit->second.dims[0];
			uint32_t N  = (uint32_t)bit->second.dims[1];
			if (BK != K || K == 0u || M == 0u || N == 0u) return false;

			float a_scale = 0.0f, y_scale = 0.0f;
			int32_t a_zp_raw = 0, a_zp_dt = 0, b_zp_raw = 0, b_zp_dt = 0, y_zp = 0, y_zp_dt = 0;
			std::vector<float> b_scales;
			if (!read_scalar_f32(g, nd.inputs[1], a_scale, true) ||
			    !read_uniform_zp(g, nd.inputs[2], a_zp_raw, a_zp_dt, 1u) ||
			    !read_vec_f32(g, nd.inputs[4], b_scales, true) ||
			    !read_uniform_zp(g, nd.inputs[5], b_zp_raw, b_zp_dt, N) ||
			    !read_scalar_f32(g, nd.inputs[6], y_scale, true) ||
			    !read_uniform_zp(g, nd.inputs[7], y_zp, y_zp_dt, 1u)) {
				return false;
			}
			if (a_zp_dt != a->dtype || b_zp_dt != bit->second.data_type) return false;
			if (b_scales.size() != 1u && b_scales.size() != N) return false;

			int b_is_u8 = (bit->second.data_type == C7X_ORT_UINT8) ? 1 : 0;
			std::vector<uint8_t> b_trans((size_t)N * K);
			for (uint32_t k = 0u; k < K; k++) {
				for (uint32_t n = 0u; n < N; n++) {
					b_trans[(size_t)n * K + k] = b_raw[(size_t)k * N + n];
				}
			}

			size_t bsum_off = 0u;
			size_t packed_sz = c7x_qlinear_packed_bytes(N, K, &bsum_off);
			std::vector<uint8_t> packed(packed_sz, 0u);
			c7x_pack_qlinear_weights(b_trans.data(), b_is_u8, N, K,
						 reinterpret_cast<int8_t *>(packed.data()),
						 reinterpret_cast<int32_t *>(packed.data() + bsum_off));
			c7x_buf_t *b_w = s->upload_aux(packed.data(), packed_sz, true);
			c7x_buf_t *b_s = s->upload_aux(b_scales.data(), b_scales.size() * sizeof(float), false);
			if (!b_w || !b_s) return false;

			uint32_t flags = 0u;
			if (a->dtype == C7X_ORT_UINT8) flags |= C7X_CMD_FLAG_A_UINT8;
			if (b_is_u8) flags |= C7X_CMD_FLAG_B_UINT8;
			if (y_zp_dt == C7X_ORT_UINT8) flags |= C7X_CMD_FLAG_Y_UINT8;

			std::vector<int64_t> y_dims = a->dims;
			y_dims.back() = N;
			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], y_zp_dt, y_dims, false);
			if (!y) return false;

			struct c7x_qlinear_matmul_desc md {};
			md.M             = M;
			md.K             = K;
			md.N             = N;
			md.k_pad         = align64_u32(K);
			md.n_pad         = align64_u32(N);
			md.a_zp          = (a->dtype == C7X_ORT_UINT8) ? (a_zp_raw - 128) : a_zp_raw;
			md.b_zp          = b_is_u8 ? (b_zp_raw - 128) : b_zp_raw;
			md.y_zp          = y_zp;
			md.a_scale       = a_scale;
			md.y_scale       = y_scale;
			md.per_channel_b = (b_scales.size() > 1u) ? 1u : 0u;
			md.b_sum_addr    = c7x_buf_phys(b_w) + bsum_off;
			md.b_scales_addr = c7x_buf_phys(b_s);

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &md, sizeof(md));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op    = C7X_CMD_QLINEAR_MATMUL;
			cmd.flags = flags;
			cmd.src0  = c7x_buf_phys(b_w);
			cmd.src1  = c7x_buf_phys(a->buf);
			cmd.src3  = dp;
			cmd.dst   = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- FP32 Conv (with optional bias & exact fused Relu) ---------- */
		if (op == "Conv") {
			if (nd.inputs.size() < 2u) return false;
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			CompiledTensor *w = s->ensure_initializer_tensor(g, nd.inputs[1]);
			CompiledTensor *b = (nd.inputs.size() >= 3u && !nd.inputs[2].empty())
					    ? s->ensure_initializer_tensor(g, nd.inputs[2]) : nullptr;
			if (!x || !w || x->dtype != C7X_ORT_FLOAT || w->dtype != C7X_ORT_FLOAT ||
			    x->dims.size() != 4u || w->dims.size() != 4u) return false;
			if (nd.get_int("group", 1) != 1) return false;

			uint32_t N = (uint32_t)x->dims[0], IC = (uint32_t)x->dims[1];
			uint32_t IH = (uint32_t)x->dims[2], IW = (uint32_t)x->dims[3];
			uint32_t OC = (uint32_t)w->dims[0], WIC = (uint32_t)w->dims[1];
			uint32_t KH = (uint32_t)w->dims[2], KW = (uint32_t)w->dims[3];
			if (WIC != IC) return false;
			if (b && (b->dtype != C7X_ORT_FLOAT || b->num_elements() != OC)) return false;

			auto strides = nd.get_ints("strides"), pads = nd.get_ints("pads"), dil = nd.get_ints("dilations");
			int32_t s1 = strides.size() >= 2 ? (int32_t)strides[0] : 1;
			int32_t s0 = strides.size() >= 2 ? (int32_t)strides[1] : 1;
			int32_t d1 = dil.size() >= 2 ? (int32_t)dil[0] : 1;
			int32_t d0 = dil.size() >= 2 ? (int32_t)dil[1] : 1;
			int32_t p1 = 0, p0 = 0;
			uint32_t OH = 0u, OW = 0u;
			if (!compute_2d_pads_and_out(nd.get_str("auto_pad"), IH, IW, KH, KW,
						     s1, s0, d1, d0, pads, p1, p0, OH, OW)) {
				return false;
			}

			uint32_t flags = 0u;
			std::string out_name = nd.outputs[0];
			if (ni + 1 < g.nodes.size() && use_count[out_name] == 1 &&
			    g.nodes[ni + 1].op_type == "Relu" && g.nodes[ni + 1].inputs[0] == out_name) {
				flags |= C7X_CMD_FLAG_RELU;
				out_name = g.nodes[ni + 1].outputs[0];
				ni++;
			}

			CompiledTensor *y = s->alloc_tensor(out_name, C7X_ORT_FLOAT, {N, OC, OH, OW}, false);
			if (!y) return false;

			struct c7x_conv2d_f32_desc cd {};
			cd.N = N; cd.IC = IC; cd.IH = IH; cd.IW = IW;
			cd.OC = OC; cd.KH = KH; cd.KW = KW; cd.OH = OH; cd.OW = OW;
			cd.has_bias = b ? 1u : 0u;
			cd.s0 = s0; cd.s1 = s1; cd.p0 = p0; cd.p1 = p1; cd.d0 = d0; cd.d1 = d1;
			cd.src_nb[0] = sizeof(float);
			cd.src_nb[1] = (uint64_t)IW * sizeof(float);
			cd.src_nb[2] = (uint64_t)IH * IW * sizeof(float);
			cd.src_nb[3] = (uint64_t)IC * IH * IW * sizeof(float);
			cd.dst_nb[0] = sizeof(float);
			cd.dst_nb[1] = (uint64_t)OW * sizeof(float);
			cd.dst_nb[2] = (uint64_t)OH * OW * sizeof(float);
			cd.dst_nb[3] = (uint64_t)OC * OH * OW * sizeof(float);

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &cd, sizeof(cd));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op    = C7X_CMD_CONV_2D_F32;
			cmd.flags = flags;
			cmd.src0  = c7x_buf_phys(w->buf);
			cmd.src1  = c7x_buf_phys(x->buf);
			cmd.src2  = b ? c7x_buf_phys(b->buf) : 0ull;
			cmd.src3  = dp;
			cmd.dst   = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- FP32 Gemm / MatMul ---------- */
		if (op == "Gemm" || op == "MatMul") {
			if (nd.inputs.size() < 2u) return false;
			CompiledTensor *a = s->ensure_initializer_tensor(g, nd.inputs[0]);
			auto bit = g.initializers.find(nd.inputs[1]);
			if (!a || a->dtype != C7X_ORT_FLOAT || a->dims.size() < 2u) return false;
			if (bit == g.initializers.end() || bit->second.data_type != C7X_ORT_FLOAT ||
			    bit->second.dims.size() != 2u) return false;

			bool transB = false;
			if (op == "Gemm") {
				if (nd.get_int("transA", 0) != 0) return false;
				if (std::fabs(nd.get_float("alpha", 1.0f) - 1.0f) > 1e-6f) return false;
				if (std::fabs(nd.get_float("beta", 1.0f) - 1.0f) > 1e-6f) return false;
				transB = (nd.get_int("transB", 0) != 0);
			}

			auto bb = bit->second.as_bytes();
			if (!validate_tensor_bytes(bit->second, bb)) return false;

			uint32_t K = (uint32_t)a->dims.back();
			uint32_t M = (uint32_t)(a->num_elements() / K);
			uint32_t N = 0u;
			std::vector<float> bf(bb.size() / 4u);
			std::memcpy(bf.data(), bb.data(), bb.size());
			c7x_buf_t *b_rowmajor_nk = nullptr;

			if (transB) {
				N = (uint32_t)bit->second.dims[0];
				if ((uint32_t)bit->second.dims[1] != K) return false;
				b_rowmajor_nk = s->upload_aux(bf.data(), bf.size() * 4u, false);
			} else {
				if ((uint32_t)bit->second.dims[0] != K) return false;
				N = (uint32_t)bit->second.dims[1];
				std::vector<float> bt((size_t)N * K);
				for (uint32_t k = 0u; k < K; k++) {
					for (uint32_t n = 0u; n < N; n++) {
						bt[(size_t)n * K + k] = bf[(size_t)k * N + n];
					}
				}
				b_rowmajor_nk = s->upload_aux(bt.data(), bt.size() * 4u, false);
			}
			if (!b_rowmajor_nk) return false;

			std::vector<int64_t> y_dims = a->dims;
			y_dims.back() = N;
			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, y_dims, false);
			if (!y) return false;

			struct c7x_matmul_f32_desc md {};
			md.ne0[0] = K; md.ne0[1] = N; md.ne0[2] = 1u; md.ne0[3] = 1u;
			md.ne1[0] = K; md.ne1[1] = M; md.ne1[2] = 1u; md.ne1[3] = 1u;
			md.ne_dst[0] = N; md.ne_dst[1] = M; md.ne_dst[2] = 1u; md.ne_dst[3] = 1u;
			md.nb0[0] = 4u; md.nb0[1] = (uint64_t)K * 4u; md.nb0[2] = (uint64_t)N * K * 4u; md.nb0[3] = md.nb0[2];
			md.nb1[0] = 4u; md.nb1[1] = (uint64_t)K * 4u; md.nb1[2] = (uint64_t)M * K * 4u; md.nb1[3] = md.nb1[2];
			md.nb_dst[0] = 4u; md.nb_dst[1] = (uint64_t)N * 4u; md.nb_dst[2] = (uint64_t)M * N * 4u; md.nb_dst[3] = md.nb_dst[2];

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &md, sizeof(md));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op   = C7X_CMD_MATMUL_F32;
			cmd.src0 = c7x_buf_phys(b_rowmajor_nk);
			cmd.src1 = c7x_buf_phys(a->buf);
			cmd.src3 = dp;
			cmd.dst  = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;

			if (op == "Gemm" && nd.inputs.size() >= 3u && !nd.inputs[2].empty()) {
				CompiledTensor *c = s->ensure_initializer_tensor(g, nd.inputs[2]);
				if (!c || c->dtype != C7X_ORT_FLOAT) return false;
				struct c7x_4d_binop_desc bd {};
				to_c7x_4d_shape_strides(y->dims, 4u, bd.ne0, bd.nb0);
				to_c7x_4d_shape_strides(c->dims, 4u, bd.ne1, bd.nb1);
				to_c7x_4d_shape_strides(y->dims, 4u, bd.ne_dst, bd.nb_dst);
				for (int i = 0; i < 4; i++) {
					if (bd.ne1[i] != bd.ne0[i] && bd.ne1[i] != 1u) return false;
				}
				uint64_t bdp = c7x_cmdbuf_push_desc(s->cb, &bd, sizeof(bd));
				if (!bdp) return false;
				struct c7x_cmd acmd {};
				acmd.op   = C7X_CMD_ADD_F32;
				acmd.src0 = c7x_buf_phys(y->buf);
				acmd.src1 = c7x_buf_phys(c->buf);
				acmd.src3 = bdp;
				acmd.dst  = c7x_buf_phys(y->buf);
				if (c7x_cmdbuf_push(s->cb, &acmd) != 0) return false;
			}
			continue;
		}

		/* ---------- LayerNormalization ---------- */
		if (op == "LayerNormalization") {
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			CompiledTensor *gamma = (nd.inputs.size() >= 2u && !nd.inputs[1].empty())
						? s->ensure_initializer_tensor(g, nd.inputs[1]) : nullptr;
			CompiledTensor *beta  = (nd.inputs.size() >= 3u && !nd.inputs[2].empty())
						? s->ensure_initializer_tensor(g, nd.inputs[2]) : nullptr;
			if (!x || x->dtype != C7X_ORT_FLOAT || x->dims.empty()) return false;
			int64_t axis = nd.get_int("axis", -1);
			if (axis < 0) axis += (int64_t)x->dims.size();
			if (axis != (int64_t)x->dims.size() - 1) return false;
			uint32_t row_len = (uint32_t)x->dims.back();
			if (gamma && (gamma->dtype != C7X_ORT_FLOAT || gamma->num_elements() != row_len)) return false;
			if (beta  && (beta->dtype  != C7X_ORT_FLOAT || beta->num_elements()  != row_len)) return false;

			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, x->dims, false);
			if (!y) return false;

			struct c7x_norm_desc ndesc {};
			to_c7x_4d_shape_strides(x->dims, 4u, ndesc.ne0, ndesc.nb0);
			to_c7x_4d_shape_strides(y->dims, 4u, ndesc.ne0, ndesc.nb_dst);
			ndesc.eps       = nd.get_float("epsilon", 1e-5f);
			ndesc.has_gamma = gamma ? 1u : 0u;
			ndesc.has_beta  = beta ? 1u : 0u;

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &ndesc, sizeof(ndesc));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op   = C7X_CMD_NORM_F32;
			cmd.src0 = c7x_buf_phys(x->buf);
			cmd.src1 = gamma ? c7x_buf_phys(gamma->buf) : 0ull;
			cmd.src2 = beta  ? c7x_buf_phys(beta->buf)  : 0ull;
			cmd.src3 = dp;
			cmd.dst  = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- Pooling (MaxPool, AveragePool, GlobalAveragePool) ---------- */
		if (op == "MaxPool" || op == "AveragePool" || op == "GlobalAveragePool") {
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			if (!x || x->dtype != C7X_ORT_FLOAT || x->dims.size() != 4u) return false;
			uint32_t N = (uint32_t)x->dims[0], C = (uint32_t)x->dims[1];
			uint32_t IH = (uint32_t)x->dims[2], IW = (uint32_t)x->dims[3];
			int32_t k1 = (int32_t)IH, k0 = (int32_t)IW, s1 = 1, s0 = 1, p1 = 0, p0 = 0;
			uint32_t OH = 1u, OW = 1u;
			uint32_t pool_op = C7X_POOL_AVG;

			if (op != "GlobalAveragePool") {
				if (nd.get_int("ceil_mode", 0) != 0) return false;
				auto dil = nd.get_ints("dilations");
				for (int64_t dv : dil) {
					if (dv != 1) return false;
				}
				auto ks = nd.get_ints("kernel_shape");
				auto st = nd.get_ints("strides");
				auto pd = nd.get_ints("pads");
				if (ks.size() < 2u || ks[0] <= 0 || ks[1] <= 0) return false;
				k1 = (int32_t)ks[0];
				k0 = (int32_t)ks[1];
				s1 = st.size() >= 2 ? (int32_t)st[0] : 1;
				s0 = st.size() >= 2 ? (int32_t)st[1] : 1;
				if (!compute_2d_pads_and_out(nd.get_str("auto_pad"), IH, IW,
							     (uint32_t)k1, (uint32_t)k0, s1, s0, 1, 1,
							     pd, p1, p0, OH, OW)) {
					return false;
				}
				if (op == "MaxPool") {
					pool_op = C7X_POOL_MAX;
				} else {
					pool_op = nd.get_int("count_include_pad", 0)
						  ? C7X_POOL_AVG : C7X_POOL_AVG_EXCL_PAD;
				}
			}

			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, {N, C, OH, OW}, false);
			if (!y) return false;

			struct c7x_pool2d_desc pd {};
			pd.op       = pool_op;
			pd.IW = IW; pd.IH = IH; pd.OW = OW; pd.OH = OH;
			pd.n_planes = N * C;
			pd.k0 = k0; pd.k1 = k1; pd.s0 = s0; pd.s1 = s1; pd.p0 = p0; pd.p1 = p1;
			pd.src_nb1  = (uint64_t)IW * sizeof(float);
			pd.src_nb2  = (uint64_t)IH * IW * sizeof(float);
			pd.dst_nb1  = (uint64_t)OW * sizeof(float);
			pd.dst_nb2  = (uint64_t)OH * OW * sizeof(float);

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &pd, sizeof(pd));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op   = C7X_CMD_POOL_2D_F32;
			cmd.src0 = c7x_buf_phys(x->buf);
			cmd.src3 = dp;
			cmd.dst  = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- Elementwise Unary (Relu, Gelu, Sigmoid, Tanh) ---------- */
		if (op == "Relu" || op == "Gelu" || op == "Sigmoid" || op == "Tanh") {
			CompiledTensor *x = s->ensure_initializer_tensor(g, nd.inputs[0]);
			if (!x || x->dtype != C7X_ORT_FLOAT) return false;
			if (op == "Gelu") {
				/* C7X_UNARY_GELU implements tanh-GELU; require explicit approximate="tanh" so erf-GELU ("none") falls back cleanly. */
				std::string approx = nd.get_str("approximate", "none");
				if (approx != "tanh") return false;
			}
			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, x->dims, false);
			if (!y) return false;

			uint32_t subop = C7X_UNARY_RELU;
			if (op == "Gelu")         subop = C7X_UNARY_GELU;
			else if (op == "Sigmoid") subop = C7X_UNARY_SIGMOID;
			else if (op == "Tanh")    subop = C7X_UNARY_TANH;

			struct c7x_cmd cmd {};
			cmd.op   = C7X_CMD_UNARY_F32;
			cmd.p0   = (uint32_t)x->num_elements();
			cmd.p1   = subop;
			cmd.src0 = c7x_buf_phys(x->buf);
			cmd.dst  = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		/* ---------- Elementwise Binary (Add, Mul; handles commuted broadcast) ---------- */
		if (op == "Add" || op == "Mul") {
			if (nd.inputs.size() < 2u) return false;
			CompiledTensor *a = s->ensure_initializer_tensor(g, nd.inputs[0]);
			CompiledTensor *b = s->ensure_initializer_tensor(g, nd.inputs[1]);
			if (!a || !b || a->dtype != C7X_ORT_FLOAT || b->dtype != C7X_ORT_FLOAT) return false;

			size_t nd_out = std::max(a->dims.size(), b->dims.size());
			if (nd_out > 4u) return false;
			std::vector<int64_t> out_dims(nd_out, 1);
			bool a_bcast = false, b_bcast = false;
			for (size_t i = 0; i < nd_out; i++) {
				int64_t da = (i < a->dims.size()) ? a->dims[a->dims.size() - 1 - i] : 1;
				int64_t db = (i < b->dims.size()) ? b->dims[b->dims.size() - 1 - i] : 1;
				if (da != db && da != 1 && db != 1) return false;
				if (da < db) a_bcast = true;
				if (db < da) b_bcast = true;
				out_dims[nd_out - 1 - i] = std::max(da, db);
			}
			if (a_bcast && b_bcast) {
				fprintf(stderr, "ort-c7x-ep: bidirectional broadcast in %s '%s' not supported\n",
					op.c_str(), nd.name.c_str());
				return false;
			}
			/* C7X binop broadcasts src1 into src0; swap commutative Add/Mul operands if src0 is broadcast. */
			if (a_bcast) {
				std::swap(a, b);
			}

			CompiledTensor *y = s->alloc_tensor(nd.outputs[0], C7X_ORT_FLOAT, out_dims, false);
			if (!y) return false;

			struct c7x_4d_binop_desc bd {};
			to_c7x_4d_shape_strides(a->dims, 4u, bd.ne0, bd.nb0);
			to_c7x_4d_shape_strides(b->dims, 4u, bd.ne1, bd.nb1);
			to_c7x_4d_shape_strides(y->dims, 4u, bd.ne_dst, bd.nb_dst);

			uint64_t dp = c7x_cmdbuf_push_desc(s->cb, &bd, sizeof(bd));
			if (!dp) return false;
			struct c7x_cmd cmd {};
			cmd.op   = (op == "Mul") ? C7X_CMD_MUL_F32 : C7X_CMD_ADD_F32;
			cmd.src0 = c7x_buf_phys(a->buf);
			cmd.src1 = c7x_buf_phys(b->buf);
			cmd.src3 = dp;
			cmd.dst  = c7x_buf_phys(y->buf);
			if (c7x_cmdbuf_push(s->cb, &cmd) != 0) return false;
			continue;
		}

		fprintf(stderr, "ort-c7x-ep: unhandled op '%s' (node '%s')\n",
			op.c_str(), nd.name.c_str());
		return false;
	}
	return true;
}

extern "C" {

int c7x_ort_ep_op_supported(const char *op_type) {
	if (!op_type) return 0;
	static const char *const kOps[] = {
		"QLinearConv", "QLinearMatMul", "DequantizeLinear",
		"Conv", "MatMul", "Gemm", "LayerNormalization",
		"Relu", "Gelu", "Sigmoid", "Tanh",
		"Add", "Mul",
		"MaxPool", "AveragePool", "GlobalAveragePool",
		"Reshape", "Flatten", "Squeeze", "Unsqueeze",
	};
	for (const char *k : kOps) {
		if (std::strcmp(op_type, k) == 0) return 1;
	}
	return 0;
}

c7x_ort_session_t *c7x_ort_session_create_from_memory(const void *onnx_bytes, size_t onnx_len,
						      const struct c7x_ort_ep_options *opts) {
	if (!onnx_bytes || onnx_len == 0u) return nullptr;
	c7x_onnx::ModelProto model;
	if (!c7x_onnx::parse_model_proto(static_cast<const uint8_t *>(onnx_bytes), onnx_len, model)) {
		fprintf(stderr, "ort-c7x-ep: failed to parse ONNX protobuf (%zu bytes)\n", onnx_len);
		return nullptr;
	}

	auto sess = std::make_unique<c7x_ort_session>();
	if (opts) {
		sess->opts = *opts;
	} else {
		sess->opts.use_msmc_weights = 1;
		sess->opts.enable_profiling = 0;
		sess->opts.force_hostemu    = 0;
		sess->opts.emu_ddr_bytes    = 0u;
	}

	struct c7x_open_opts ro {};
	ro.transport     = sess->opts.force_hostemu ? C7X_TRANSPORT_HOSTEMU : C7X_TRANSPORT_AUTO;
	ro.emu_ddr_bytes = sess->opts.emu_ddr_bytes ? sess->opts.emu_ddr_bytes : (256ull * 1024ull * 1024ull);
	sess->rt = c7x_open(&ro);
	if (!sess->rt) {
		fprintf(stderr, "ort-c7x-ep: c7x_open failed\n");
		return nullptr;
	}
	sess->cb = c7x_cmdbuf_create(sess->rt, 1024u, 256u * 1024u);
	if (!sess->cb) {
		fprintf(stderr, "ort-c7x-ep: c7x_cmdbuf_create failed\n");
		return nullptr;
	}
	if (!compile_graph(sess.get(), model.graph)) {
		return nullptr;
	}
	return sess.release();
}

c7x_ort_session_t *c7x_ort_session_create_from_file(const char *onnx_path,
						    const struct c7x_ort_ep_options *opts) {
	if (!onnx_path) return nullptr;
	std::ifstream f(onnx_path, std::ios::binary | std::ios::ate);
	if (!f) {
		fprintf(stderr, "ort-c7x-ep: cannot open '%s'\n", onnx_path);
		return nullptr;
	}
	std::streamsize sz = f.tellg();
	if (sz <= 0) return nullptr;
	f.seekg(0, std::ios::beg);
	std::vector<uint8_t> buf((size_t)sz);
	if (!f.read(reinterpret_cast<char *>(buf.data()), sz)) return nullptr;
	return c7x_ort_session_create_from_memory(buf.data(), buf.size(), opts);
}

void c7x_ort_session_destroy(c7x_ort_session_t *sess) {
	delete sess;
}

uint32_t c7x_ort_session_num_cmds(const c7x_ort_session_t *sess) {
	return (sess && sess->cb) ? c7x_cmdbuf_count(sess->cb) : 0u;
}

static int fill_tensor_view(const c7x_ort_session_t *sess, const std::string &name,
			    struct c7x_ort_tensor_view *out) {
	auto it = sess->tensors.find(name);
	if (it == sess->tensors.end()) return -1;
	const CompiledTensor &ct = it->second;
	std::memset(out, 0, sizeof(*out));
	out->name   = ct.name.c_str();
	out->dtype  = (enum c7x_ort_dtype)ct.dtype;
	out->ndim   = (int)std::min<size_t>(ct.dims.size(), 4u);
	for (int i = 0; i < out->ndim; i++) out->dims[i] = ct.dims[i];
	out->nbytes = ct.nbytes;
	return 0;
}

int c7x_ort_session_input_info(const c7x_ort_session_t *sess, size_t idx,
			       struct c7x_ort_tensor_view *out) {
	if (!sess || !out || idx >= sess->input_names.size()) return -1;
	return fill_tensor_view(sess, sess->input_names[idx], out);
}

int c7x_ort_session_output_info(const c7x_ort_session_t *sess, size_t idx,
				struct c7x_ort_tensor_view *out) {
	if (!sess || !out || idx >= sess->output_names.size()) return -1;
	return fill_tensor_view(sess, sess->output_names[idx], out);
}

int c7x_ort_session_run(c7x_ort_session_t *sess,
			const struct c7x_ort_tensor_view *inputs, size_t n_inputs,
			struct c7x_ort_tensor_view *outputs, size_t n_outputs,
			struct c7x_ort_run_stats *stats_out) {
	if (!sess || !sess->rt || !sess->cb) return -1;

	for (size_t i = 0; i < n_inputs; i++) {
		std::string name = inputs[i].name ? inputs[i].name
				   : (i < sess->input_names.size() ? sess->input_names[i] : "");
		auto it = sess->tensors.find(name);
		if (it == sess->tensors.end() || !inputs[i].data || inputs[i].nbytes < it->second.nbytes) {
			fprintf(stderr, "ort-c7x-ep: bad input '%s'\n", name.c_str());
			return -2;
		}
		c7x_write(sess->rt, it->second.buf, 0u, inputs[i].data, it->second.nbytes);
	}

	struct c7x_exec_stats st {};
	if (c7x_cmdbuf_submit(sess->rt, sess->cb, &st) != 0) {
		fprintf(stderr, "ort-c7x-ep: C7X_OP_EXEC_LIST failed at cmd %u (done=%u)\n",
			st.fail_idx, st.cmds_done);
		return -4;
	}

	for (size_t i = 0; i < n_outputs; i++) {
		std::string name = outputs[i].name ? outputs[i].name
				   : (i < sess->output_names.size() ? sess->output_names[i] : "");
		auto it = sess->tensors.find(name);
		if (it == sess->tensors.end() || !outputs[i].data || outputs[i].nbytes < it->second.nbytes) {
			fprintf(stderr, "ort-c7x-ep: bad output '%s'\n", name.c_str());
			return -5;
		}
		c7x_read(sess->rt, outputs[i].data, it->second.buf, 0u, it->second.nbytes);
	}

	if (stats_out) {
		stats_out->cmds_executed    = st.cmds_done;
		stats_out->dsp_cycles       = st.cycles;
		stats_out->dsp_total_cycles = st.total_cycles;
		stats_out->wall_us          = st.wall_us;
	}
	if (sess->opts.enable_profiling) {
		fprintf(stderr, "ort-c7x-ep: cmds=%u dsp=%.3f ms total_dsp=%.3f ms wall=%.3f ms\n",
			st.cmds_done, (double)st.cycles / 1e6, (double)st.total_cycles / 1e6,
			st.wall_us / 1e3);
	}
	return 0;
}

} /* extern "C" */
