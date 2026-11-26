/* SPDX-License-Identifier: MIT */
/*
 * Upstream ONNX Runtime C API Execution Provider registration shim for ort-c7x-ep.
 *
 * Compile with `-DHAVE_ONNXRUNTIME_C_API -I/path/to/onnxruntime/include` against
 * either upstream ONNX Runtime's `onnxruntime_c_api.h` or the minimal ABI test
 * header (`tests/mock_ort_c_api.h`).
 *
 * Exposes:
 *   - `OrtSessionOptionsAppendExecutionProvider_C7x(options, c7x_opts)`:
 *       Records C7x EP config entries on `OrtSessionOptions` and registers the
 *       `com.beagleboard.c7x` custom-op domain containing `C7xCompiledGraph`,
 *       allowing ONNX Runtime sessions to dispatch C7x-compiled `.onnx` subgraphs
 *       while falling back to ORT's CPU EP for surrounding ops.
 */
#ifdef HAVE_ONNXRUNTIME_C_API

#include "c7x_ort_ep.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

static std::mutex g_ort_c7x_mtx;
static struct c7x_ort_ep_options g_default_opts = {
	1,  /* use_msmc_weights */
	0,  /* enable_profiling */
	0,  /* force_hostemu */
	0u, /* emu_ddr_bytes */
	1,  /* allow_cpu_fallback */
};

struct C7xOrtKernel {
	const OrtApi      *api = nullptr;
	c7x_ort_session_t *sess = nullptr;
};

static const char *ORT_API_CALL c7x_op_get_name(const OrtCustomOp *op) {
	(void)op;
	return "C7xCompiledGraph";
}

static const char *ORT_API_CALL c7x_op_get_provider(const OrtCustomOp *op) {
	(void)op;
	return "CPUExecutionProvider";
}

static size_t ORT_API_CALL c7x_op_get_input_count(const OrtCustomOp *op) {
	(void)op;
	return 1u;
}

static size_t ORT_API_CALL c7x_op_get_output_count(const OrtCustomOp *op) {
	(void)op;
	return 1u;
}

static ONNXTensorElementDataType ORT_API_CALL c7x_op_get_input_type(const OrtCustomOp *op, size_t idx) {
	(void)op;
	(void)idx;
	return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static ONNXTensorElementDataType ORT_API_CALL c7x_op_get_output_type(const OrtCustomOp *op, size_t idx) {
	(void)op;
	(void)idx;
	return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static void *ORT_API_CALL c7x_op_create_kernel(const OrtCustomOp *op, const OrtApi *api,
					       const OrtKernelInfo *info) {
	(void)op;
	if (!api || !info) return nullptr;

	size_t path_len = 0u;
	OrtStatus *st = api->KernelInfoGetAttribute_string(info, "onnx_path", nullptr, &path_len);
	if (st || path_len == 0u) {
		if (st) api->ReleaseStatus(st);
		return nullptr;
	}
	std::string path(path_len, '\0');
	st = api->KernelInfoGetAttribute_string(info, "onnx_path", &path[0], &path_len);
	if (st) {
		api->ReleaseStatus(st);
		return nullptr;
	}
	while (!path.empty() && path.back() == '\0') path.pop_back();

	struct c7x_ort_ep_options opts {};
	{
		std::lock_guard<std::mutex> lk(g_ort_c7x_mtx);
		opts = g_default_opts;
	}

	c7x_ort_session_t *sess = c7x_ort_session_create_from_file(path.c_str(), &opts);
	if (!sess) return nullptr;

	auto *k = new C7xOrtKernel();
	k->api  = api;
	k->sess = sess;
	return k;
}

static void ORT_API_CALL c7x_op_compute(void *op_kernel, OrtKernelContext *context) {
	auto *k = static_cast<C7xOrtKernel *>(op_kernel);
	if (!k || !k->api || !k->sess || !context) return;

	const OrtApi *api = k->api;
	const OrtValue *in_val = nullptr;
	OrtStatus *st = api->KernelContext_GetInput(context, 0u, &in_val);
	if (st || !in_val) {
		if (st) api->ReleaseStatus(st);
		return;
	}

	struct c7x_ort_tensor_view in_meta {}, out_meta {};
	if (c7x_ort_session_input_info(k->sess, 0u, &in_meta) != 0 ||
	    c7x_ort_session_output_info(k->sess, 0u, &out_meta) != 0) {
		return;
	}

	void *in_data = nullptr;
	st = api->GetTensorMutableData(const_cast<OrtValue *>(in_val), &in_data);
	if (st || !in_data) {
		if (st) api->ReleaseStatus(st);
		return;
	}

	OrtValue *out_val = nullptr;
	st = api->KernelContext_GetOutput(context, 0u, out_meta.dims, (size_t)out_meta.ndim, &out_val);
	if (st || !out_val) {
		if (st) api->ReleaseStatus(st);
		return;
	}
	void *out_data = nullptr;
	st = api->GetTensorMutableData(out_val, &out_data);
	if (st || !out_data) {
		if (st) api->ReleaseStatus(st);
		return;
	}

	in_meta.data  = in_data;
	out_meta.data = out_data;
	c7x_ort_session_run(k->sess, &in_meta, 1u, &out_meta, 1u, nullptr);
}

static void ORT_API_CALL c7x_op_destroy_kernel(void *op_kernel) {
	auto *k = static_cast<C7xOrtKernel *>(op_kernel);
	if (!k) return;
	if (k->sess) c7x_ort_session_destroy(k->sess);
	delete k;
}

static OrtCustomOp g_c7x_custom_op = {
	ORT_API_VERSION,
	c7x_op_create_kernel,
	c7x_op_get_name,
	c7x_op_get_provider,
	c7x_op_get_input_type,
	c7x_op_get_input_count,
	c7x_op_get_output_type,
	c7x_op_get_output_count,
	c7x_op_compute,
	c7x_op_destroy_kernel,
};

} /* anonymous namespace */

extern "C" OrtStatus *OrtSessionOptionsAppendExecutionProvider_C7x(
	OrtSessionOptions *options, const struct c7x_ort_ep_options *c7x_opts) {
	const OrtApiBase *base = OrtGetApiBase();
	const OrtApi *api = base ? base->GetApi(ORT_API_VERSION) : nullptr;
	if (!api) {
		return nullptr;
	}
	if (!options) {
		return api->CreateStatus(ORT_INVALID_ARGUMENT, "OrtSessionOptions must not be null");
	}

	struct c7x_ort_ep_options eff {};
	if (c7x_opts) {
		eff = *c7x_opts;
	} else {
		eff.use_msmc_weights   = 1;
		eff.enable_profiling   = 0;
		eff.force_hostemu      = 0;
		eff.emu_ddr_bytes      = 0u;
		eff.allow_cpu_fallback = 1;
	}
	{
		std::lock_guard<std::mutex> lk(g_ort_c7x_mtx);
		g_default_opts = eff;
	}

	OrtStatus *st = api->AddSessionConfigEntry(
		options, "ep.c7x.use_msmc_weights", eff.use_msmc_weights ? "1" : "0");
	if (st) return st;
	st = api->AddSessionConfigEntry(
		options, "ep.c7x.allow_cpu_fallback", eff.allow_cpu_fallback ? "1" : "0");
	if (st) return st;
	st = api->AddSessionConfigEntry(
		options, "ep.c7x.enable_profiling", eff.enable_profiling ? "1" : "0");
	if (st) return st;

	OrtCustomOpDomain *domain = nullptr;
	st = api->CreateCustomOpDomain("com.beagleboard.c7x", &domain);
	if (st) return st;
	st = api->CustomOpDomain_Add(domain, &g_c7x_custom_op);
	if (st) {
		api->ReleaseCustomOpDomain(domain);
		return st;
	}
	st = api->AddCustomOpDomain(options, domain);
	if (st) {
		api->ReleaseCustomOpDomain(domain);
		return st;
	}
	return nullptr;
}

#endif /* HAVE_ONNXRUNTIME_C_API */
