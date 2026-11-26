/* SPDX-License-Identifier: MIT */
/*
 * Minimal zero-dependency ONNX binary protobuf reader (onnx.proto3 subset).
 *
 * Parses ModelProto -> GraphProto -> NodeProto / TensorProto / AttributeProto /
 * ValueInfoProto using exact official ONNX protobuf wire field numbers.
 */
#ifndef ORT_C7X_ONNX_PROTO_HPP_
#define ORT_C7X_ONNX_PROTO_HPP_

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace c7x_onnx {

struct TensorProto {
	std::string          name;
	std::vector<int64_t> dims;
	int32_t              data_type = 0; /* 1=F32, 2=U8, 3=I8, 6=I32, 7=I64, 10=F16 */
	std::vector<uint8_t> raw_data;
	std::vector<float>   float_data;
	std::vector<int32_t> int32_data;
	std::vector<int64_t> int64_data;

	size_t num_elements() const {
		if (dims.empty()) return 1u;
		size_t n = 1u;
		for (int64_t d : dims) {
			if (d <= 0) return 0u;
			n *= (size_t)d;
		}
		return n;
	}

	/* Return contiguous typed bytes (normalizing float_data/int32_data/int64_data). */
	std::vector<uint8_t> as_bytes() const {
		if (!raw_data.empty()) {
			return raw_data;
		}
		std::vector<uint8_t> out;
		if (data_type == 1 && !float_data.empty()) {
			out.resize(float_data.size() * sizeof(float));
			std::memcpy(out.data(), float_data.data(), out.size());
		} else if (data_type == 6 && !int32_data.empty()) {
			out.resize(int32_data.size() * sizeof(int32_t));
			std::memcpy(out.data(), int32_data.data(), out.size());
		} else if (data_type == 7 && !int64_data.empty()) {
			out.resize(int64_data.size() * sizeof(int64_t));
			std::memcpy(out.data(), int64_data.data(), out.size());
		} else if ((data_type == 2 || data_type == 3) && !int32_data.empty()) {
			out.resize(int32_data.size());
			for (size_t i = 0; i < int32_data.size(); i++) {
				out[i] = (uint8_t)(int32_data[i] & 0xFF);
			}
		} else if (data_type == 10 && !int32_data.empty()) {
			out.resize(int32_data.size() * 2u);
			for (size_t i = 0; i < int32_data.size(); i++) {
				uint16_t h = (uint16_t)(int32_data[i] & 0xFFFF);
				std::memcpy(out.data() + i * 2u, &h, 2u);
			}
		}
		return out;
	}
};

struct AttributeProto {
	std::string          name;
	int32_t              type = 0; /* 1=FLOAT, 2=INT, 3=STRING, 6=FLOATS, 7=INTS */
	float                f = 0.0f;
	int64_t              i = 0;
	std::string          s;
	std::vector<float>   floats;
	std::vector<int64_t> ints;
};

struct NodeProto {
	std::string                                     name;
	std::string                                     op_type;
	std::vector<std::string>                        inputs;
	std::vector<std::string>                        outputs;
	std::unordered_map<std::string, AttributeProto> attrs;

	int64_t get_int(const char *key, int64_t def = 0) const {
		auto it = attrs.find(key);
		return (it != attrs.end()) ? it->second.i : def;
	}
	float get_float(const char *key, float def = 0.0f) const {
		auto it = attrs.find(key);
		return (it != attrs.end()) ? it->second.f : def;
	}
	std::vector<int64_t> get_ints(const char *key) const {
		auto it = attrs.find(key);
		return (it != attrs.end()) ? it->second.ints : std::vector<int64_t>{};
	}
	std::string get_str(const char *key, const char *def = "") const {
		auto it = attrs.find(key);
		return (it != attrs.end()) ? it->second.s : std::string(def);
	}
};

struct ValueInfoProto {
	std::string          name;
	int32_t              elem_type = 0;
	std::vector<int64_t> dims;
};

struct GraphProto {
	std::string                                  name;
	std::vector<NodeProto>                       nodes;
	std::unordered_map<std::string, TensorProto> initializers;
	std::vector<ValueInfoProto>                  inputs;
	std::vector<ValueInfoProto>                  outputs;
};

struct ModelProto {
	int64_t    ir_version = 0;
	GraphProto graph;
};

class PbReader {
public:
	PbReader(const uint8_t *p, size_t n) : cur_(p), end_(p + n) {}

	bool eof() const { return cur_ >= end_; }

	bool read_varint(uint64_t &out) {
		out = 0;
		unsigned shift = 0;
		while (cur_ < end_ && shift < 64u) {
			uint8_t b = *cur_++;
			out |= (uint64_t)(b & 0x7Fu) << shift;
			if ((b & 0x80u) == 0u) return true;
			shift += 7u;
		}
		return false;
	}

	bool read_fixed32(uint32_t &out) {
		if ((size_t)(end_ - cur_) < 4u) return false;
		std::memcpy(&out, cur_, 4u);
		cur_ += 4u;
		return true;
	}

	bool read_fixed64(uint64_t &out) {
		if ((size_t)(end_ - cur_) < 8u) return false;
		std::memcpy(&out, cur_, 8u);
		cur_ += 8u;
		return true;
	}

	bool read_bytes(const uint8_t *&ptr, size_t &len) {
		uint64_t l = 0;
		if (!read_varint(l) || l > (uint64_t)(end_ - cur_)) return false;
		ptr = cur_;
		len = (size_t)l;
		cur_ += len;
		return true;
	}

	bool read_string(std::string &out) {
		const uint8_t *p = nullptr;
		size_t n = 0;
		if (!read_bytes(p, n)) return false;
		out.assign(reinterpret_cast<const char *>(p), n);
		return true;
	}

	bool skip_field(uint32_t wire_type) {
		uint64_t v = 0;
		uint32_t v32 = 0;
		const uint8_t *p = nullptr;
		size_t n = 0;
		switch (wire_type) {
			case 0: return read_varint(v);
			case 1: return read_fixed64(v);
			case 2: return read_bytes(p, n);
			case 5: return read_fixed32(v32);
			default: return false;
		}
	}

private:
	const uint8_t *cur_;
	const uint8_t *end_;
};

inline bool parse_tensor_shape(const uint8_t *p, size_t n, std::vector<int64_t> &dims) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 2u) { /* dim (Dimension) */
			const uint8_t *dp = nullptr;
			size_t dn = 0;
			if (!r.read_bytes(dp, dn)) return false;
			PbReader dr(dp, dn);
			int64_t dval = 0;
			bool has_dim_value = false;
			while (!dr.eof()) {
				uint64_t dtag = 0;
				if (!dr.read_varint(dtag)) return false;
				uint32_t dfn = (uint32_t)(dtag >> 3), dwt = (uint32_t)(dtag & 7u);
				if (dfn == 1u && dwt == 0u) {
					uint64_t v = 0;
					if (!dr.read_varint(v)) return false;
					dval = (int64_t)v;
					has_dim_value = true;
				} else if (dfn == 2u && dwt == 2u) {
					/* Symbolic dim_param is not supported for static compilation. */
					return false;
				} else if (!dr.skip_field(dwt)) {
					return false;
				}
			}
			if (!has_dim_value || dval <= 0) return false;
			dims.push_back(dval);
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_value_info(const uint8_t *p, size_t n, ValueInfoProto &vi) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 2u) {
			if (!r.read_string(vi.name)) return false;
		} else if (fn == 2u && wt == 2u) { /* TypeProto */
			const uint8_t *tp = nullptr;
			size_t tn = 0;
			if (!r.read_bytes(tp, tn)) return false;
			PbReader tr(tp, tn);
			while (!tr.eof()) {
				uint64_t ttag = 0;
				if (!tr.read_varint(ttag)) return false;
				uint32_t tfn = (uint32_t)(ttag >> 3), twt = (uint32_t)(ttag & 7u);
				if (tfn == 1u && twt == 2u) { /* tensor_type */
					const uint8_t *ttp = nullptr;
					size_t ttn = 0;
					if (!tr.read_bytes(ttp, ttn)) return false;
					PbReader ttr(ttp, ttn);
					while (!ttr.eof()) {
						uint64_t ktag = 0;
						if (!ttr.read_varint(ktag)) return false;
						uint32_t kfn = (uint32_t)(ktag >> 3), kwt = (uint32_t)(ktag & 7u);
						if (kfn == 1u && kwt == 0u) {
							uint64_t v = 0;
							if (!ttr.read_varint(v)) return false;
							vi.elem_type = (int32_t)v;
						} else if (kfn == 2u && kwt == 2u) {
							const uint8_t *sp = nullptr;
							size_t sn = 0;
							if (!ttr.read_bytes(sp, sn)) return false;
							if (!parse_tensor_shape(sp, sn, vi.dims)) return false;
						} else if (!ttr.skip_field(kwt)) {
							return false;
						}
					}
				} else if (!tr.skip_field(twt)) {
					return false;
				}
			}
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_tensor_proto(const uint8_t *p, size_t n, TensorProto &t) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 0u) { /* dims unpacked */
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			t.dims.push_back((int64_t)v);
		} else if (fn == 1u && wt == 2u) { /* dims packed */
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			PbReader pr(bp, bn);
			while (!pr.eof()) {
				uint64_t v = 0;
				if (!pr.read_varint(v)) return false;
				t.dims.push_back((int64_t)v);
			}
		} else if (fn == 2u && wt == 0u) { /* data_type */
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			t.data_type = (int32_t)v;
		} else if (fn == 4u && wt == 5u) { /* float_data unpacked */
			uint32_t bits = 0; float f = 0.0f;
			if (!r.read_fixed32(bits)) return false;
			std::memcpy(&f, &bits, 4u);
			t.float_data.push_back(f);
		} else if (fn == 4u && wt == 2u) { /* float_data packed */
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			size_t cnt = bn / 4u;
			size_t old = t.float_data.size();
			t.float_data.resize(old + cnt);
			if (cnt > 0u) std::memcpy(t.float_data.data() + old, bp, cnt * 4u);
		} else if (fn == 5u && wt == 0u) { /* int32_data unpacked */
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			t.int32_data.push_back((int32_t)v);
		} else if (fn == 5u && wt == 2u) { /* int32_data packed */
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			PbReader pr(bp, bn);
			while (!pr.eof()) {
				uint64_t v = 0;
				if (!pr.read_varint(v)) return false;
				t.int32_data.push_back((int32_t)v);
			}
		} else if (fn == 7u && wt == 0u) { /* int64_data unpacked */
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			t.int64_data.push_back((int64_t)v);
		} else if (fn == 7u && wt == 2u) { /* int64_data packed */
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			PbReader pr(bp, bn);
			while (!pr.eof()) {
				uint64_t v = 0;
				if (!pr.read_varint(v)) return false;
				t.int64_data.push_back((int64_t)v);
			}
		} else if (fn == 8u && wt == 2u) { /* name */
			if (!r.read_string(t.name)) return false;
		} else if (fn == 9u && wt == 2u) { /* raw_data */
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			t.raw_data.assign(bp, bp + bn);
		} else if (fn == 13u && wt == 2u) { /* external_data (unsupported) */
			return false;
		} else if (fn == 14u && wt == 0u) { /* data_location (0=DEFAULT, 1=EXTERNAL) */
			uint64_t v = 0;
			if (!r.read_varint(v) || v != 0u) return false;
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_attr_proto(const uint8_t *p, size_t n, AttributeProto &a) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 2u) {
			if (!r.read_string(a.name)) return false;
		} else if (fn == 2u && wt == 5u) {
			uint32_t bits = 0;
			if (!r.read_fixed32(bits)) return false;
			std::memcpy(&a.f, &bits, 4u);
		} else if (fn == 3u && wt == 0u) {
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			a.i = (int64_t)v;
		} else if (fn == 4u && wt == 2u) {
			if (!r.read_string(a.s)) return false;
		} else if (fn == 7u && wt == 5u) {
			uint32_t bits = 0; float f = 0.0f;
			if (!r.read_fixed32(bits)) return false;
			std::memcpy(&f, &bits, 4u);
			a.floats.push_back(f);
		} else if (fn == 7u && wt == 2u) {
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			size_t cnt = bn / 4u, old = a.floats.size();
			a.floats.resize(old + cnt);
			if (cnt > 0u) std::memcpy(a.floats.data() + old, bp, cnt * 4u);
		} else if (fn == 8u && wt == 0u) {
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			a.ints.push_back((int64_t)v);
		} else if (fn == 8u && wt == 2u) {
			const uint8_t *bp = nullptr; size_t bn = 0;
			if (!r.read_bytes(bp, bn)) return false;
			PbReader pr(bp, bn);
			while (!pr.eof()) {
				uint64_t v = 0;
				if (!pr.read_varint(v)) return false;
				a.ints.push_back((int64_t)v);
			}
		} else if (fn == 20u && wt == 0u) {
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			a.type = (int32_t)v;
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_node_proto(const uint8_t *p, size_t n, NodeProto &nd) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 2u) {
			std::string s;
			if (!r.read_string(s)) return false;
			nd.inputs.push_back(std::move(s));
		} else if (fn == 2u && wt == 2u) {
			std::string s;
			if (!r.read_string(s)) return false;
			nd.outputs.push_back(std::move(s));
		} else if (fn == 3u && wt == 2u) {
			if (!r.read_string(nd.name)) return false;
		} else if (fn == 4u && wt == 2u) {
			if (!r.read_string(nd.op_type)) return false;
		} else if (fn == 5u && wt == 2u) {
			const uint8_t *ap = nullptr; size_t an = 0;
			if (!r.read_bytes(ap, an)) return false;
			AttributeProto a;
			if (!parse_attr_proto(ap, an, a)) return false;
			nd.attrs[a.name] = std::move(a);
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_graph_proto(const uint8_t *p, size_t n, GraphProto &g) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 2u) {
			const uint8_t *np = nullptr; size_t nn = 0;
			if (!r.read_bytes(np, nn)) return false;
			NodeProto nd;
			if (!parse_node_proto(np, nn, nd)) return false;
			g.nodes.push_back(std::move(nd));
		} else if (fn == 2u && wt == 2u) {
			if (!r.read_string(g.name)) return false;
		} else if (fn == 5u && wt == 2u) {
			const uint8_t *tp = nullptr; size_t tn = 0;
			if (!r.read_bytes(tp, tn)) return false;
			TensorProto t;
			if (!parse_tensor_proto(tp, tn, t)) return false;
			std::string key = t.name;
			g.initializers[key] = std::move(t);
		} else if (fn == 11u && wt == 2u) {
			const uint8_t *vp = nullptr; size_t vn = 0;
			if (!r.read_bytes(vp, vn)) return false;
			ValueInfoProto vi;
			if (!parse_value_info(vp, vn, vi)) return false;
			g.inputs.push_back(std::move(vi));
		} else if (fn == 12u && wt == 2u) {
			const uint8_t *vp = nullptr; size_t vn = 0;
			if (!r.read_bytes(vp, vn)) return false;
			ValueInfoProto vi;
			if (!parse_value_info(vp, vn, vi)) return false;
			g.outputs.push_back(std::move(vi));
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

inline bool parse_model_proto(const uint8_t *p, size_t n, ModelProto &m) {
	PbReader r(p, n);
	while (!r.eof()) {
		uint64_t tag = 0;
		if (!r.read_varint(tag)) return false;
		uint32_t fn = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7u);
		if (fn == 1u && wt == 0u) {
			uint64_t v = 0;
			if (!r.read_varint(v)) return false;
			m.ir_version = (int64_t)v;
		} else if (fn == 7u && wt == 2u) {
			const uint8_t *gp = nullptr; size_t gn = 0;
			if (!r.read_bytes(gp, gn)) return false;
			if (!parse_graph_proto(gp, gn, m.graph)) return false;
		} else if (!r.skip_field(wt)) {
			return false;
		}
	}
	return true;
}

} /* namespace c7x_onnx */

#endif /* ORT_C7X_ONNX_PROTO_HPP_ */
