/* SPDX-License-Identifier: MIT */
/*
 * C7x ONNX Runtime Execution Provider (ort-c7x-ep) over libc7x command buffers.
 *
 * Provides:
 *   1. Capability query (`c7x_ort_ep_op_supported`) and ORT-compatible EP
 *      configuration (`struct c7x_ort_ep_options`).
 *   2. Zero-dependency standard ONNX `.onnx` binary protobuf loader & hybrid
 *      graph compiler (`c7x_ort_session_t`) that packs INT8/UINT8/FP32 weights
 *      into the C7x DDR/MSMC arena, lowers contiguous C7x-supported subgraphs
 *      into single `C7X_OP_EXEC_LIST` RPCs via `libc7x`, and optionally partitions
 *      unsupported ops (QuantizeLinear, Softmax, erf-Gelu, Resize, Clip, etc.)
 *      onto built-in CPU fallback kernels (`allow_cpu_fallback = 1`).
 *   3. Upstream ONNX Runtime C API registration hook (`ort_c_api_shim.cpp`)
 *      when compiled with `-DHAVE_ONNXRUNTIME_C_API`.
 */
#ifndef ORT_C7X_EP_H_
#define ORT_C7X_EP_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum c7x_ort_dtype {
	C7X_ORT_UNDEFINED = 0,
	C7X_ORT_FLOAT     = 1,  /* ONNX TensorProto.FLOAT */
	C7X_ORT_UINT8     = 2,  /* ONNX TensorProto.UINT8 */
	C7X_ORT_INT8      = 3,  /* ONNX TensorProto.INT8 */
	C7X_ORT_INT32     = 6,  /* ONNX TensorProto.INT32 */
	C7X_ORT_INT64     = 7,  /* ONNX TensorProto.INT64 */
	C7X_ORT_FLOAT16   = 10, /* ONNX TensorProto.FLOAT16 */
};

struct c7x_ort_ep_options {
	int      use_msmc_weights;   /* Stage qualifying weights into on-chip MSMC SRAM */
	int      enable_profiling;   /* Print per-run DSP / wall-clock timing to stderr */
	int      force_hostemu;      /* 0 = AUTO (HW if /dev/mem + C7AI present, else hostemu) */
	size_t   emu_ddr_bytes;      /* Hostemu DDR arena size (0 = 256 MiB default) */
	int      allow_cpu_fallback; /* 0 = strict C7x-only (default); 1 = hybrid C7x + CPU partitioning */
};

struct c7x_ort_run_stats {
	uint32_t cmds_executed;
	uint64_t dsp_cycles;
	uint64_t dsp_total_cycles;
	double   wall_us;
	uint32_t c7x_subgraphs;
	uint32_t c7x_nodes;
	uint32_t cpu_fallback_nodes;
	double   cpu_fallback_us;
};

struct c7x_ort_tensor_view {
	const char        *name;
	enum c7x_ort_dtype dtype;
	int64_t            dims[4];
	int                ndim;
	void              *data;
	size_t             nbytes;
};

typedef struct c7x_ort_session c7x_ort_session_t;

/* Query whether an ONNX operator type string is natively accelerated on C7x. */
int c7x_ort_ep_op_supported(const char *op_type);

/* Query whether an ONNX operator has a built-in CPU fallback when allow_cpu_fallback=1. */
int c7x_ort_ep_op_cpu_fallback_supported(const char *op_type);

/* Create a compiled C7x EP session directly from a standard `.onnx` model file. */
c7x_ort_session_t *c7x_ort_session_create_from_file(const char *onnx_path,
						    const struct c7x_ort_ep_options *opts);

/* Create a compiled C7x EP session from an in-memory `.onnx` protobuf buffer. */
c7x_ort_session_t *c7x_ort_session_create_from_memory(const void *onnx_bytes, size_t onnx_len,
						      const struct c7x_ort_ep_options *opts);

void c7x_ort_session_destroy(c7x_ort_session_t *sess);

/* Introspect compiled graph stats, hybrid partition summary, and primary IO metadata. */
uint32_t c7x_ort_session_num_cmds(const c7x_ort_session_t *sess);
int      c7x_ort_session_partition_info(const c7x_ort_session_t *sess,
					uint32_t *c7x_subgraphs,
					uint32_t *c7x_nodes,
					uint32_t *cpu_nodes);
int      c7x_ort_session_input_info(const c7x_ort_session_t *sess, size_t idx,
				    struct c7x_ort_tensor_view *out);
int      c7x_ort_session_output_info(const c7x_ort_session_t *sess, size_t idx,
				     struct c7x_ort_tensor_view *out);

/*
 * Run one synchronous inference pass (pure C7x single-RPC or hybrid C7x + CPU):
 *   - Copies caller's `inputs[0..n_inputs-1]` into C7x DDR arena
 *   - Executes compiled C7x command buffers (and any partitioned CPU fallback stages)
 *   - Copies `outputs[0..n_outputs-1]` back to caller buffers
 */
int c7x_ort_session_run(c7x_ort_session_t *sess,
			const struct c7x_ort_tensor_view *inputs, size_t n_inputs,
			struct c7x_ort_tensor_view *outputs, size_t n_outputs,
			struct c7x_ort_run_stats *stats_out);

#ifdef HAVE_ONNXRUNTIME_C_API
#include <onnxruntime_c_api.h>
OrtStatus *OrtSessionOptionsAppendExecutionProvider_C7x(
	OrtSessionOptions *options, const struct c7x_ort_ep_options *c7x_opts);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ORT_C7X_EP_H_ */
