/* SPDX-License-Identifier: MIT */
/*
 * Self-contained verification of the ONNX Runtime C API registration shim
 * (`ort_c_api_shim.cpp`): verifies `OrtSessionOptionsAppendExecutionProvider_C7x`,
 * custom-op domain registration (`com.beagleboard.c7x` / `C7xCompiledGraph`),
 * kernel creation, and end-to-end inference through the OrtCustomOp callbacks.
 */
#include "c7x_ort_ep.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct OrtStatus {
	OrtErrorCode code;
	std::string  msg;
};

struct OrtCustomOpDomain {
	std::string name;
	std::vector<const OrtCustomOp *> ops;
};

struct OrtSessionOptions {
	std::map<std::string, std::string> config;
	std::vector<OrtCustomOpDomain *>   domains;
};

struct OrtKernelInfo {
	std::map<std::string, std::string> str_attrs;
};

struct OrtValue {
	std::vector<uint8_t> data;
	std::vector<int64_t> dims;
};

struct OrtKernelContext {
	std::vector<const OrtValue *> inputs;
	std::vector<OrtValue *>       outputs;
};

static OrtStatus *ORT_API_CALL mock_create_status(OrtErrorCode code, const char *msg) {
	return new OrtStatus{code, msg ? msg : ""};
}
static void ORT_API_CALL mock_release_status(OrtStatus *st) { delete st; }
static OrtStatus *ORT_API_CALL mock_add_config(OrtSessionOptions *o, const char *k, const char *v) {
	o->config[k] = v;
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_create_domain(const char *d, OrtCustomOpDomain **out) {
	*out = new OrtCustomOpDomain{d ? d : "", {}};
	return nullptr;
}
static void ORT_API_CALL mock_release_domain(OrtCustomOpDomain *d) { delete d; }
static OrtStatus *ORT_API_CALL mock_domain_add(OrtCustomOpDomain *d, const OrtCustomOp *op) {
	d->ops.push_back(op);
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_add_domain(OrtSessionOptions *o, OrtCustomOpDomain *d) {
	o->domains.push_back(d);
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_get_str_attr(const OrtKernelInfo *info, const char *name,
						 char *out, size_t *size) {
	auto it = info->str_attrs.find(name ? name : "");
	if (it == info->str_attrs.end()) return mock_create_status(ORT_FAIL, "missing attr");
	size_t need = it->second.size() + 1u;
	if (!out) { *size = need; return nullptr; }
	if (*size < need) return mock_create_status(ORT_FAIL, "buffer too small");
	std::memcpy(out, it->second.c_str(), need);
	*size = need;
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_get_input(const OrtKernelContext *c, size_t i, const OrtValue **out) {
	if (i >= c->inputs.size()) return mock_create_status(ORT_FAIL, "bad input idx");
	*out = c->inputs[i];
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_get_output(OrtKernelContext *c, size_t i, const int64_t *dims,
					       size_t ndim, OrtValue **out) {
	if (i >= c->outputs.size()) return mock_create_status(ORT_FAIL, "bad output idx");
	size_t n = 1u;
	c->outputs[i]->dims.assign(dims, dims + ndim);
	for (size_t d = 0; d < ndim; d++) n *= (size_t)dims[d];
	c->outputs[i]->data.resize(n * sizeof(float));
	*out = c->outputs[i];
	return nullptr;
}
static OrtStatus *ORT_API_CALL mock_get_data(OrtValue *v, void **out) {
	*out = v->data.data();
	return nullptr;
}

static OrtApi g_mock_api = {
	mock_create_status,
	mock_release_status,
	mock_add_config,
	mock_create_domain,
	mock_release_domain,
	mock_domain_add,
	mock_add_domain,
	mock_get_str_attr,
	mock_get_input,
	mock_get_output,
	mock_get_data,
};

static const OrtApi *ORT_API_CALL mock_get_api(uint32_t ver) {
	return (ver == ORT_API_VERSION) ? &g_mock_api : nullptr;
}
static const char *ORT_API_CALL mock_get_ver(void) { return "mock-1.18"; }
static OrtApiBase g_mock_base = { mock_get_api, mock_get_ver };

extern "C" const OrtApiBase *ORT_API_CALL OrtGetApiBase(void) { return &g_mock_base; }

static bool read_file_bytes(const std::string &path, std::vector<uint8_t> &out) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) return false;
	std::streamsize sz = f.tellg();
	if (sz <= 0) return false;
	f.seekg(0, std::ios::beg);
	out.resize((size_t)sz);
	return (bool)f.read(reinterpret_cast<char *>(out.data()), sz);
}

int main(int argc, char **argv) {
	std::string model_dir = (argc > 1) ? argv[1] : "build-hostemu/models";

	/* 1. Null options must return non-null OrtStatus error. */
	OrtStatus *bad = OrtSessionOptionsAppendExecutionProvider_C7x(nullptr, nullptr);
	if (!bad || bad->code != ORT_INVALID_ARGUMENT) {
		fprintf(stderr, "[ort-c-api-shim] expected ORT_INVALID_ARGUMENT on null options\n");
		return 1;
	}
	mock_release_status(bad);

	/* 2. Valid registration registers config entries and com.beagleboard.c7x domain. */
	OrtSessionOptions opts {};
	struct c7x_ort_ep_options c7x_opts {};
	c7x_opts.use_msmc_weights   = 1;
	c7x_opts.allow_cpu_fallback = 1;
	c7x_opts.force_hostemu      = 1;

	OrtStatus *st = OrtSessionOptionsAppendExecutionProvider_C7x(&opts, &c7x_opts);
	if (st != nullptr) {
		fprintf(stderr, "[ort-c-api-shim] append EP failed: %s\n", st->msg.c_str());
		mock_release_status(st);
		return 1;
	}
	if (opts.config["ep.c7x.allow_cpu_fallback"] != "1" ||
	    opts.domains.size() != 1u ||
	    opts.domains[0]->name != "com.beagleboard.c7x" ||
	    opts.domains[0]->ops.size() != 1u) {
		fprintf(stderr, "[ort-c-api-shim] domain/config mismatch\n");
		return 1;
	}

	/* 3. Execute whisper_erf_hybrid.onnx end-to-end through the OrtCustomOp kernel callbacks! */
	const OrtCustomOp *op = opts.domains[0]->ops[0];
	OrtKernelInfo kinfo {};
	kinfo.str_attrs["onnx_path"] = model_dir + "/audio_enc_block.onnx";

	void *kernel = op->CreateKernel(op, &g_mock_api, &kinfo);
	if (!kernel) {
		fprintf(stderr, "[ort-c-api-shim] CreateKernel failed\n");
		return 1;
	}

	OrtValue in_val {}, out_val {};
	std::vector<uint8_t> ref_bytes;
	if (!read_file_bytes(model_dir + "/audio_input.bin", in_val.data) ||
	    !read_file_bytes(model_dir + "/audio_ref.bin", ref_bytes)) {
		fprintf(stderr, "[ort-c-api-shim] failed reading audio test files\n");
		op->KernelDestroy(kernel);
		return 1;
	}

	OrtKernelContext ctx {};
	ctx.inputs.push_back(&in_val);
	ctx.outputs.push_back(&out_val);
	op->KernelCompute(kernel, &ctx);
	op->KernelDestroy(kernel);

	float max_err = 0.0f;
	size_t n = ref_bytes.size() / sizeof(float);
	if (out_val.data.size() != ref_bytes.size()) {
		fprintf(stderr, "[ort-c-api-shim] output size mismatch (%zu vs %zu)\n",
			out_val.data.size(), ref_bytes.size());
		return 1;
	}
	const float *got = reinterpret_cast<const float *>(out_val.data.data());
	const float *ref = reinterpret_cast<const float *>(ref_bytes.data());
	for (size_t i = 0; i < n; i++) {
		float e = std::fabs(got[i] - ref[i]);
		if (e > max_err) max_err = e;
	}
	for (auto *d : opts.domains) mock_release_domain(d);

	bool ok = (max_err < 1e-4f);
	printf("[ort-c-api-shim] OrtSessionOptionsAppendExecutionProvider_C7x + C7xCompiledGraph max_err=%.6f: %s\n",
	       max_err, ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
