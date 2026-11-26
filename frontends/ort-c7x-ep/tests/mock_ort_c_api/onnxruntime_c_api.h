/* SPDX-License-Identifier: MIT */
/*
 * Minimal ONNX Runtime C API ABI subset for compiling and testing
 * `ort_c_api_shim.cpp` when upstream `onnxruntime_c_api.h` is not installed.
 * Matches upstream ONNX Runtime C API symbol & signature contracts.
 */
#ifndef ONNXRUNTIME_C_API_H_
#define ONNXRUNTIME_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ORT_API_VERSION
#define ORT_API_VERSION 18
#endif

#ifndef ORT_API_CALL
#define ORT_API_CALL
#endif

typedef enum OrtErrorCode {
	ORT_OK = 0,
	ORT_FAIL = 1,
	ORT_INVALID_ARGUMENT = 2,
} OrtErrorCode;

typedef enum ONNXTensorElementDataType {
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
} ONNXTensorElementDataType;

typedef struct OrtStatus OrtStatus;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtCustomOpDomain OrtCustomOpDomain;
typedef struct OrtKernelInfo OrtKernelInfo;
typedef struct OrtKernelContext OrtKernelContext;
typedef struct OrtValue OrtValue;
typedef struct OrtApi OrtApi;

typedef struct OrtCustomOp {
	uint32_t version;
	void *(ORT_API_CALL *CreateKernel)(const struct OrtCustomOp *op, const OrtApi *api,
					   const OrtKernelInfo *info);
	const char *(ORT_API_CALL *GetName)(const struct OrtCustomOp *op);
	const char *(ORT_API_CALL *GetExecutionProviderType)(const struct OrtCustomOp *op);
	ONNXTensorElementDataType(ORT_API_CALL *GetInputType)(const struct OrtCustomOp *op, size_t index);
	size_t(ORT_API_CALL *GetInputTypeCount)(const struct OrtCustomOp *op);
	ONNXTensorElementDataType(ORT_API_CALL *GetOutputType)(const struct OrtCustomOp *op, size_t index);
	size_t(ORT_API_CALL *GetOutputTypeCount)(const struct OrtCustomOp *op);
	void(ORT_API_CALL *KernelCompute)(void *op_kernel, OrtKernelContext *context);
	void(ORT_API_CALL *KernelDestroy)(void *op_kernel);
} OrtCustomOp;

struct OrtApi {
	OrtStatus *(ORT_API_CALL *CreateStatus)(OrtErrorCode code, const char *msg);
	void(ORT_API_CALL *ReleaseStatus)(OrtStatus *input);
	OrtStatus *(ORT_API_CALL *AddSessionConfigEntry)(OrtSessionOptions *options,
							 const char *config_key,
							 const char *config_value);
	OrtStatus *(ORT_API_CALL *CreateCustomOpDomain)(const char *domain,
							OrtCustomOpDomain **out);
	void(ORT_API_CALL *ReleaseCustomOpDomain)(OrtCustomOpDomain *ptr);
	OrtStatus *(ORT_API_CALL *CustomOpDomain_Add)(OrtCustomOpDomain *custom_op_domain,
						      const OrtCustomOp *op);
	OrtStatus *(ORT_API_CALL *AddCustomOpDomain)(OrtSessionOptions *options,
						     OrtCustomOpDomain *custom_op_domain);
	OrtStatus *(ORT_API_CALL *KernelInfoGetAttribute_string)(const OrtKernelInfo *info,
								 const char *name,
								 char *out, size_t *size);
	OrtStatus *(ORT_API_CALL *KernelContext_GetInput)(const OrtKernelContext *context,
							  size_t index, const OrtValue **out);
	OrtStatus *(ORT_API_CALL *KernelContext_GetOutput)(OrtKernelContext *context,
							   size_t index, const int64_t *dim_values,
							   size_t dim_count, OrtValue **out);
	OrtStatus *(ORT_API_CALL *GetTensorMutableData)(OrtValue *value, void **out);
};

typedef struct OrtApiBase {
	const OrtApi *(ORT_API_CALL *GetApi)(uint32_t version);
	const char *(ORT_API_CALL *GetVersionString)(void);
} OrtApiBase;

const OrtApiBase *ORT_API_CALL OrtGetApiBase(void);

#ifdef __cplusplus
}
#endif

#endif /* ONNXRUNTIME_C_API_H_ */
