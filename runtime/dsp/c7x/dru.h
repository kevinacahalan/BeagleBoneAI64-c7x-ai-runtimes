/* SPDX-License-Identifier: MIT */
/*
 * Bare-metal C7x DRU (Data Routing Unit) driver for J721E / TDA4VM.
 *
 * Submits 64-byte Type-15 Transfer Requests atomically from the C7x via a
 * single 512-bit vector store to CHATOMIC[ch], with zero UDMA-C or RTOS overhead.
 */
#ifndef DSP_C7X_DRU_H_
#define DSP_C7X_DRU_H_

#include <stdint.h>

struct c7x_dru_info {
	uint64_t pid;
	uint64_t caps;
	uint32_t ok;
	uint32_t ch;
};

int c7x_dru_is_ok(void);

/*
 * Probe DRU0 (0x6d000000), configure queue 0, and enable direct-TR channel `ch`.
 * Returns 1 on success, 0 if DRU registers appear unreachable or misconfigured.
 */
int c7x_dru_init(uint32_t ch, struct c7x_dru_info *info);

/*
 * Synchronous 1D/2D block copy via DRU direct atomic TR9.
 * Handles lengths up to 64 MiB by splitting into 2D rows of 32 KiB if needed.
 * Returns 0 on success, non-zero on timeout or error.
 */
int c7x_dru_copy_sync(uint32_t ch, uint64_t dst, uint64_t src, uint32_t bytes);

/*
 * Asynchronous contiguous byte copy (1 or 2 buffers) with an ordered NC-L2
 * completion marker TR. Call c7x_dru_wait(ch) before consuming destinations.
 */
void c7x_dru_submit_bytes(uint32_t ch, uint64_t dst, uint64_t src, uint32_t bytes);
void c7x_dru_submit_pair_bytes(uint32_t ch,
			       uint64_t dst0, uint64_t src0, uint32_t bytes0,
			       uint64_t dst1, uint64_t src1, uint32_t bytes1);

/*
 * Asynchronous 2D tile copy (icnt0 bytes/row x icnt1 rows, src stride dim1,
 * dst stride ddim1). Call c7x_dru_wait(ch) before consuming destination data.
 */
void c7x_dru_submit_2d(uint32_t ch, uint64_t dst, uint64_t src,
		       uint16_t icnt0, uint16_t icnt1,
		       int32_t src_dim1, int32_t dst_dim1);

int c7x_dru_wait(uint32_t ch);

#endif /* DSP_C7X_DRU_H_ */
