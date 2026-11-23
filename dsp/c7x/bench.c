/* SPDX-License-Identifier: MIT */
/*
 * Benchmark & kernel command dispatcher for the C7x AI runtime.
 */

#include <stdint.h>
#include <string.h>
#include <c7x.h>

#include "c7x_ai_proto.h"
#include "c7x_llama_fmt.h"
#include "cmd_exec.h"
#include "dru.h"
#include "mma_kernels.h"
#include "mmu.h"
#include "transformer.h"
#include "vec_kernels.h"

#define DRU_CH_DEFAULT 0u

static struct c7x_dru_info g_dru;

static inline volatile struct c7x_ctrl *ctrl_ptr(void)
{
	return (volatile struct c7x_ctrl *)(uintptr_t)C7X_CTRL_ADDR;
}

static inline void mfence2(void)
{
	/* Double mfence per TI J721E C71x silicon errata (see PDK Cache_wait). */
	__memory_fence(0x1);
	__memory_fence(0x1);
}

static inline void cache_in_all(void)
{
	ai64_l1d_wbinv_all();
	mfence2();
	ai64_l2_wbinv_all();
	mfence2();
}

static inline void cache_out_all(void)
{
	ai64_l1d_wbinv_all();
	mfence2();
	ai64_l2_wbinv_all();
	mfence2();
}

void bench_init(void)
{
	volatile struct c7x_ctrl *c = ctrl_ptr();

	c->done_seq     = c->req_seq;
	c->version      = C7X_CTRL_VERSION;
	c->dnum         = (uint32_t)__DNUM;
	c->scr_lo       = (uint32_t)ai64_get_scr();
	c->l1dcfg       = (uint32_t)ai64_get_l1dcfg();
	c->l2cfg        = (uint32_t)ai64_get_l2cfg();

	c7x_dru_init(DRU_CH_DEFAULT, &g_dru);
	c->dru_pid      = g_dru.pid;
	c->dru_caps     = g_dru.caps;
	c->dru_ok       = g_dru.ok;
	c->cbank_max_ok = 0u;

	__memory_fence(0x1);
	c->magic        = C7X_CTRL_MAGIC;
	__memory_fence(0x1);
}

const struct c7x_dru_info *bench_dru_info(void)
{
	return &g_dru;
}

int bench_poll(void)
{
	volatile struct c7x_ctrl *c = ctrl_ptr();
	uint32_t seq = c->req_seq;
	uint32_t op, flags, iters, m, k, n, iter;
	uint64_t a_addr, b_addr, c_addr;
	uint64_t t_total0, t_kern0, t_kern1;
	uint32_t status = C7X_STATUS_OK;
	uint64_t aux0 = 0, aux1 = 0;

	if (seq == c->done_seq) {
		return 0;
	}
	__memory_fence(0x1);
	t_total0 = __TSC;

	op     = c->op;
	flags  = c->flags;
	iters  = c->iters ? c->iters : 1u;
	m      = c->m;
	k      = c->k;
	n      = c->n;
	a_addr = c->a_addr;
	b_addr = c->b_addr;
	c_addr = c->c_addr;

	cache_in_all();
	t_kern0 = __TSC;

	switch (op) {
	case C7X_OP_PING:
		break;

	case C7X_OP_CACHE_FLUSH:
		cache_out_all();
		break;

	case C7X_OP_PROBE_CBANK:
		if (k == 0u || k > 64u || a_addr == 0ull || b_addr == 0ull || c_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			c7x_mma_probe_cbank((const int8_t *)(uintptr_t)a_addr,
					    (const int8_t *)(uintptr_t)b_addr,
					    (int32_t *)(uintptr_t)c_addr, k);
		}
		break;

	case C7X_OP_MEM_BW_READ:
		if (m == 0u || (m % 64u) != 0u || a_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			aux0 ^= c7x_mem_bw_read((const void *)(uintptr_t)a_addr, m);
		}
		break;

	case C7X_OP_MEM_BW_COPY:
		if (m == 0u || (m % 64u) != 0u || a_addr == 0ull || b_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			c7x_mem_bw_copy((void *)(uintptr_t)b_addr,
					(const void *)(uintptr_t)a_addr, m);
		}
		break;

	case C7X_OP_DRU_COPY:
		if (!g_dru.ok) {
			status = C7X_STATUS_NO_DRU;
			break;
		}
		if (m == 0u || a_addr == 0ull || b_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			if (c7x_dru_copy_sync(DRU_CH_DEFAULT, b_addr, a_addr, m) != 0) {
				status = C7X_STATUS_DRU_ERR;
				break;
			}
		}
		break;

	case C7X_OP_GEMM_MMA_I8: {
		uint32_t b_trans = (flags & C7X_GEMM_B_TRANSPOSE) ? 1u : 0u;
		uint32_t block_m = (flags & C7X_GEMM_BLOCK_M_MASK) >> C7X_GEMM_BLOCK_M_SHIFT;

		if (m == 0u || k == 0u || n == 0u || (k % 64u) != 0u || (n % 64u) != 0u ||
		    a_addr == 0ull || b_addr == 0ull || c_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			c7x_mma_gemm_i8_i32((const int8_t *)(uintptr_t)a_addr,
					    (const int8_t *)(uintptr_t)b_addr,
					    (int32_t *)(uintptr_t)c_addr,
					    m, k, n, b_trans, block_m);
		}
		break;
	}

	case C7X_OP_GEMM_C_I8: {
		uint32_t b_trans = (flags & C7X_GEMM_B_TRANSPOSE) ? 1u : 0u;

		if (m == 0u || k == 0u || n == 0u ||
		    a_addr == 0ull || b_addr == 0ull || c_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			c7x_ref_gemm_i8_i32((const int8_t *)(uintptr_t)a_addr,
					    (const int8_t *)(uintptr_t)b_addr,
					    (int32_t *)(uintptr_t)c_addr,
					    m, k, n, b_trans);
		}
		break;
	}

	case C7X_OP_GEMV_W4A8:
		if (k == 0u || n == 0u || (k % 128u) != 0u || (n % 4u) != 0u ||
		    a_addr == 0ull || b_addr == 0ull || c_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		for (iter = 0; iter < iters; iter++) {
			c7x_mma_gemv_w4a8((const int8_t *)(uintptr_t)a_addr,
					  (const uint8_t *)(uintptr_t)b_addr,
					  (int32_t *)(uintptr_t)c_addr, k, n);
		}
		break;

	case C7X_OP_VEC_SELFTEST: {
		float max_err = 0.0f;
		int rc = c7x_vec_selftest(&max_err);
		uint32_t err_bits = 0u;

		memcpy(&err_bits, &max_err, sizeof(err_bits));
		aux0 = (uint64_t)err_bits;
		aux1 = (uint64_t)(int64_t)rc;
		if (rc != 0) {
			status = C7X_STATUS_MISMATCH;
		}
		break;
	}

	case C7X_OP_LLAMA_FORWARD: {
		const struct c7x_llama_req *req;
		uint32_t argmax = 0u;
		int rc;

		if (a_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		req = (const struct c7x_llama_req *)(uintptr_t)a_addr;
		rc = c7x_llama_forward_token(
			(const struct c7m_header *)(uintptr_t)req->model_addr,
			req->token, req->pos,
			(void *)(uintptr_t)req->k_cache_addr,
			(void *)(uintptr_t)req->v_cache_addr,
			req->k_scale_addr ? (float *)(uintptr_t)req->k_scale_addr : NULL,
			req->v_scale_addr ? (float *)(uintptr_t)req->v_scale_addr : NULL,
			req->flags,
			req->logits_addr ? (float *)(uintptr_t)req->logits_addr : NULL,
			&argmax,
			(void *)(uintptr_t)C7X_L2_SCRATCH_ADDR,
			C7X_L2_SCRATCH_SIZE,
			req->msmc_layers_addr ? (const void *)(uintptr_t)req->msmc_layers_addr : NULL,
			req->msmc_n_layers,
			req->prof_addr ? (struct c7x_llama_prof *)(uintptr_t)req->prof_addr : NULL,
			req->debug_x_addr ? (float *)(uintptr_t)req->debug_x_addr : NULL,
			req->debug_layer,
			req->stop_after_layer);
		aux0 = (uint64_t)(int64_t)rc;
		aux1 = (uint64_t)argmax;
		if (rc != 0) {
			status = C7X_STATUS_BAD_ARGS;
		}
		break;
	}

	case C7X_OP_LLAMA_PREFILL: {
		const struct c7x_llama_req *req;
		uint32_t argmax = 0u;
		int rc;

		if (a_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		req = (const struct c7x_llama_req *)(uintptr_t)a_addr;
		rc = c7x_llama_prefill_tokens(
			(const struct c7m_header *)(uintptr_t)req->model_addr,
			(const uint32_t *)(uintptr_t)req->tokens_addr,
			req->n_tokens, req->pos,
			(void *)(uintptr_t)req->k_cache_addr,
			(void *)(uintptr_t)req->v_cache_addr,
			req->k_scale_addr ? (float *)(uintptr_t)req->k_scale_addr : NULL,
			req->v_scale_addr ? (float *)(uintptr_t)req->v_scale_addr : NULL,
			req->flags,
			req->logits_addr ? (float *)(uintptr_t)req->logits_addr : NULL,
			&argmax,
			(void *)(uintptr_t)C7X_L2_SCRATCH_ADDR,
			C7X_L2_SCRATCH_SIZE,
			(void *)(uintptr_t)req->prefill_buf_addr,
			req->prefill_buf_bytes,
			req->msmc_layers_addr ? (const void *)(uintptr_t)req->msmc_layers_addr : NULL,
			req->msmc_n_layers,
			req->prof_addr ? (struct c7x_llama_prof *)(uintptr_t)req->prof_addr : NULL);
		aux0 = (uint64_t)(int64_t)rc;
		aux1 = (uint64_t)argmax;
		if (rc != 0) {
			status = C7X_STATUS_BAD_ARGS;
		}
		break;
	}

	case C7X_OP_EXEC_LIST: {
		uint32_t done = 0u, fail_idx = 0u;
		void *l2_buf;
		uint32_t l2_sz;
		int rc;

		if (k != 0u && k != C7X_CMD_ABI_VERSION) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		if ((b_addr != 0ull) != (n != 0u)) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		if (m > 0u && a_addr == 0ull) {
			status = C7X_STATUS_BAD_ARGS;
			break;
		}
		l2_buf = b_addr ? (void *)(uintptr_t)b_addr : (void *)(uintptr_t)C7X_L2_SCRATCH_ADDR;
		l2_sz  = n ? n : C7X_L2_SCRATCH_SIZE;
		rc = c7x_cmd_exec_list((const struct c7x_cmd *)(uintptr_t)a_addr,
				       m, l2_buf, l2_sz, &done, &fail_idx);
		aux0 = (uint64_t)done;
		aux1 = ((uint64_t)(uint32_t)rc << 32) | (uint64_t)fail_idx;
		if (rc != 0) {
			status = C7X_STATUS_CMD_ERR;
		}
		break;
	}

	default:
		status = C7X_STATUS_BAD_OP;
		break;
	}

	t_kern1 = __TSC;
	cache_out_all();

	c->cycles       = t_kern1 - t_kern0;
	c->total_cycles = __TSC - t_total0;
	c->aux0         = aux0;
	c->aux1         = aux1;
	c->status       = status;
	__memory_fence(0x1);
	c->done_seq     = seq;
	__memory_fence(0x1);
	return 1;
}
