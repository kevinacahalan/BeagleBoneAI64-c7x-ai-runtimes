/* SPDX-License-Identifier: MIT */
/*
 * C7x Command-Buffer Executor (C7X_OP_EXEC_LIST).
 *
 * Compiles identically for TI C7000 CGT (cl7x -mv7100 -O3) and x86_64 C7000
 * Host Emulation (g++ -D__C7X_HOSTEM__).
 */
#ifndef DSP_C7X_CMD_EXEC_H_
#define DSP_C7X_CMD_EXEC_H_

#include <stdint.h>

#include "c7x_cmd_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Execute `n_cmds` sequential commands from `cmds[0..n_cmds-1]` using
 * `l2_scratch` (size `l2_bytes`) for temporary quantized activations and
 * DRU0 L2 ping-pong buffering.
 *
 * Returns 0 on success, or a negative error code on the first failing command
 * (setting *fail_idx_out to the failing index and *done_out to commands completed).
 */
int c7x_cmd_exec_list(const struct c7x_cmd *cmds, uint32_t n_cmds,
		      void *l2_scratch, uint32_t l2_bytes,
		      uint32_t *done_out, uint32_t *fail_idx_out);

#ifdef __cplusplus
}
#endif

#endif /* DSP_C7X_CMD_EXEC_H_ */
