/* SPDX-License-Identifier: MIT */
/*
 * Bare-metal C7x DRU (Data Routing Unit) driver for J721E / TDA4VM.
 *
 * Register offsets match the J721E TRM (COMPUTE_CLUSTER0_DRU @ 0x6d000000):
 *   0x00000: DRU_PID
 *   0x00008: DRU_CAPABILITIES
 *   0x08000 + q*8:       DRU_QUEUE_CFG[q]
 *   0x40000 + ch*0x100:  CHNRT_CFG[ch]
 *   0x40020 + ch*0x100:  CHNRT_CHOES0[ch]
 *   0x40060 + ch*0x100:  CHNRT_CHST_SCHED[ch]
 *   0x60000 + ch*0x100:  CHRT_CTL[ch]         (32-bit access)
 *   0x60010 + ch*0x100:  CHRT_STATUS_DET[ch]
 *   0x80000 + ch*0x100:  CHATOMIC[ch]         (64-byte atomic TR submit)
 */

#include <stdint.h>
#include <string.h>
#include <c7x.h>

#include "c7x_ai_proto.h"
#include "dru.h"
#include "mmu.h"

#define DRU_REG64(off)        (*(volatile uint64_t *)(uintptr_t)(C7X_DRU_BASE + (off)))
#define DRU_REG32(off)        (*(volatile uint32_t *)(uintptr_t)(C7X_DRU_BASE + (off)))

#define DRU_OFF_PID           0x00000u
#define DRU_OFF_CAPS          0x00008u
#define DRU_OFF_QUEUE_CFG(q)  (0x08000u + (q) * 8u)
#define DRU_OFF_CHNRT_CFG(c)  (0x40000u + (c) * 0x100u)
#define DRU_OFF_CHNRT_OES(c)  (0x40020u + (c) * 0x100u)
#define DRU_OFF_CHNRT_SCH(c)  (0x40060u + (c) * 0x100u)
#define DRU_OFF_CHRT_CTL(c)   (0x60000u + (c) * 0x100u)
#define DRU_OFF_CHRT_DET(c)   (0x60010u + (c) * 0x100u)
#define DRU_OFF_CHRT_CNT(c)   (0x60018u + (c) * 0x100u)
#define DRU_OFF_CHATOMIC(c)   (0x80000u + (c) * 0x100u)
#define DRU_OFF_CHCORE(c, co) (0xA0000u + (c) * 0x100u + (co) * 0x40u)

/* CHRT_CTL bits */
#define DRU_CHRT_CTL_ENABLE   (1u << 31)

#define DRU_DET_TR_CNT_SHIFT  16u
#define DRU_DET_TR_CNT_MASK   (0xFFull << DRU_DET_TR_CNT_SHIFT)
#define DRU_DET_TR_IN_Q_MASK  (0xFFull << 24)
#define DRU_DET_CH_ACTIVE     (1ull << 63)
#define DRU_DET_WR_ACTIVE     (1ull << 62)
#define DRU_DET_RD_ACTIVE     (1ull << 61)
#define DRU_DET_BUSY_MASK     (DRU_DET_CH_ACTIVE | DRU_DET_WR_ACTIVE | \
			       DRU_DET_RD_ACTIVE | DRU_DET_TR_IN_Q_MASK)

/* TR9 flags: Type 9 (4D block move with repacking, J721E DRU0 caps=0x0f20), EOP=1. */
#define TR9_FLAGS_4D_BLOCK    ((9u << 0) | (1u << 26) | (1u << 31))

/* Non-cacheable L2 SRAM completion-marker slots (top of 4 KiB control page). */
#define DRU_MARKER_SRC_PA(ch) (C7X_CTRL_ADDR + 0xE00u + ((uint64_t)((ch) & 3u) * 64ull))
#define DRU_MARKER_DST_PA(ch) (C7X_CTRL_ADDR + 0xF00u + ((uint64_t)((ch) & 3u) * 64ull))

static uint64_t g_dru_marker_seq[32];

/* 64-byte Type-15 Transfer Request matching TI UDMAP/DRU TR15 layout. */
struct __attribute__((__packed__, aligned(64))) dru_tr15 {
	uint32_t flags;
	uint16_t icnt0;
	uint16_t icnt1;
	uint64_t addr;
	int32_t  dim1;
	uint16_t icnt2;
	uint16_t icnt3;
	int32_t  dim2;
	int32_t  dim3;
	uint32_t fmtflags;
	int32_t  ddim1;
	uint64_t daddr;
	int32_t  ddim2;
	int32_t  ddim3;
	uint16_t dicnt0;
	uint16_t dicnt1;
	uint16_t dicnt2;
	uint16_t dicnt3;
};

static int g_dru_use_chcore = 0;
static int g_dru_ready = 0;

int c7x_dru_is_ok(void)
{
	return g_dru_ready;
}

static inline void dru_submit_tr15(uint32_t ch, const struct dru_tr15 *tr)
{
	if (g_dru_use_chcore) {
		const uint32_t *w = (const uint32_t *)tr;
		volatile uint32_t *dst = (volatile uint32_t *)(uintptr_t)(C7X_DRU_BASE + DRU_OFF_CHCORE(ch, 0u));
		uint32_t i;

		for (i = 1u; i < 16u; i++) {
			dst[i] = w[i];
		}
		__memory_fence(0x1);
		dst[0] = w[0];
		__memory_fence(0x1);
	} else {
		volatile __ulong8 *atomic_reg =
			(volatile __ulong8 *)(uintptr_t)(C7X_DRU_BASE + DRU_OFF_CHATOMIC(ch));
		*atomic_reg = *(const __ulong8 *)tr;
		__memory_fence(0x1);
	}
}

int c7x_dru_init(uint32_t ch, struct c7x_dru_info *info)
{
	uint64_t pid, caps;
	uint32_t ctl;
	int mode;

	pid = DRU_REG64(DRU_OFF_PID);
	caps = DRU_REG64(DRU_OFF_CAPS);
	if (info) {
		info->pid = pid;
		info->caps = caps;
		info->ok = 0;
	}
	if (pid == 0 || pid == ~0ull || ch >= 32u) {
		return 0;
	}

	/* Configure Queue 0 (priority queue): pri=0, qos=0, orderId=0, consecTrans=8, rearbWait=0. */
	DRU_REG64(DRU_OFF_QUEUE_CFG(0u)) = (8ull << 16);

	/* Disable channel first if already enabled from an earlier run. */
	DRU_REG32(DRU_OFF_CHRT_CTL(ch)) = 0u;
	__memory_fence(0x1);

	/* Configure Direct TR ownership (owner = 0, type = 0, pause_on_err = 0), queue 0. */
	DRU_REG64(DRU_OFF_CHNRT_CFG(ch)) = 0ull;
	DRU_REG64(DRU_OFF_CHNRT_OES(ch)) = 0xFFFFull;
	DRU_REG64(DRU_OFF_CHNRT_SCH(ch)) = 0ull;
	__memory_fence(0x1);

	/* Enable channel (32-bit write per TI CSL requirement). */
	DRU_REG32(DRU_OFF_CHRT_CTL(ch)) = DRU_CHRT_CTL_ENABLE;
	__memory_fence(0x1);

	ctl = DRU_REG32(DRU_OFF_CHRT_CTL(ch));
	if (info) {
		info->ch = ch;
	}
	if (!(ctl & DRU_CHRT_CTL_ENABLE)) {
		return 0;
	}

	/* Initialize NC-L2 completion marker slots for this channel. */
	g_dru_marker_seq[ch] = 0ull;
	*(volatile uint64_t *)(uintptr_t)DRU_MARKER_SRC_PA(ch) = 0ull;
	*(volatile uint64_t *)(uintptr_t)DRU_MARKER_DST_PA(ch) = 0ull;
	__memory_fence(0x1);

	/*
	 * Verify direct-TR submission: try CHATOMIC (0x80000) first (with retry for
	 * cold-boot channel settle), then CHCORE (0xA0000), using DDR arena (0xAC000000).
	 */
	for (mode = 0; mode < 3; mode++) {
		uint64_t src_pa = C7X_DDR_ARENA_BASE + C7X_DDR_ARENA_SIZE - 4096u;
		uint64_t dst_pa = src_pa + 64u;
		volatile uint32_t *s32 = (volatile uint32_t *)(uintptr_t)src_pa;
		volatile uint32_t *d32 = (volatile uint32_t *)(uintptr_t)dst_pa;
		int wrc;

		if (mode > 0) {
			/* Re-reset and re-enable channel cleanly before retry. */
			DRU_REG32(DRU_OFF_CHRT_CTL(ch)) = 0u;
			__memory_fence(0x1);
			DRU_REG64(DRU_OFF_CHNRT_CFG(ch)) = 0ull;
			DRU_REG64(DRU_OFF_CHNRT_OES(ch)) = 0xFFFFull;
			DRU_REG64(DRU_OFF_CHNRT_SCH(ch)) = 0ull;
			__memory_fence(0x1);
			DRU_REG32(DRU_OFF_CHRT_CTL(ch)) = DRU_CHRT_CTL_ENABLE;
			__memory_fence(0x1);
			g_dru_marker_seq[ch] = 0ull;
			*(volatile uint64_t *)(uintptr_t)DRU_MARKER_SRC_PA(ch) = 0ull;
			*(volatile uint64_t *)(uintptr_t)DRU_MARKER_DST_PA(ch) = 0ull;
			__memory_fence(0x1);
		}
		g_dru_use_chcore = (mode == 2) ? 1 : 0;
		s32[0] = 0xA55A1234u;
		d32[0] = 0xDEADBEEFu;
		__memory_fence(0x1);
		__memory_fence(0x1);
		ai64_l1d_wbinv_all();
		__memory_fence(0x1);
		__memory_fence(0x1);
		ai64_l2_wbinv_all();
		__memory_fence(0x1);
		__memory_fence(0x1);

		c7x_dru_submit_2d(ch, dst_pa, src_pa, 64u, 1u, 64, 64);
		wrc = c7x_dru_wait(ch);

		ai64_l1d_wbinv_all();
		__memory_fence(0x1);
		__memory_fence(0x1);
		ai64_l2_wbinv_all();
		__memory_fence(0x1);
		__memory_fence(0x1);
		if (wrc == 0 && d32[0] == 0xA55A1234u) {
			g_dru_ready = 1;
			if (info) {
				info->ok = 1;
			}
			return 1;
		}
	}
	/* Reset channel if neither submission path executed. */
	DRU_REG32(DRU_OFF_CHRT_CTL(ch)) = 0u;
	__memory_fence(0x1);
	return 0;
}

static void dru_queue_raw_2d(uint32_t ch, uint64_t dst, uint64_t src,
			     uint16_t icnt0, uint16_t icnt1,
			     int32_t src_dim1, int32_t dst_dim1)
{
	struct dru_tr15 tr;

	memset(&tr, 0, sizeof(tr));
	tr.flags  = TR9_FLAGS_4D_BLOCK;
	tr.icnt0  = icnt0;
	tr.icnt1  = icnt1;
	tr.icnt2  = 1u;
	tr.icnt3  = 1u;
	tr.addr   = src;
	tr.dim1   = src_dim1;
	tr.dim2   = (int32_t)((uint32_t)icnt0 * (uint32_t)icnt1);
	tr.dim3   = tr.dim2;
	tr.dicnt0 = icnt0;
	tr.dicnt1 = icnt1;
	tr.dicnt2 = 1u;
	tr.dicnt3 = 1u;
	tr.daddr  = dst;
	tr.ddim1  = dst_dim1;
	tr.ddim2  = (int32_t)((uint32_t)icnt0 * (uint32_t)icnt1);
	tr.ddim3  = tr.ddim2;
	dru_submit_tr15(ch, &tr);
}

static void dru_queue_raw_bytes(uint32_t ch, uint64_t dst, uint64_t src, uint32_t bytes)
{
	if (bytes == 0u) {
		return;
	}
	if (bytes <= 32768u) {
		dru_queue_raw_2d(ch, dst, src, (uint16_t)bytes, 1u,
				 (int32_t)bytes, (int32_t)bytes);
		return;
	}
	{
		uint32_t rows = bytes / 32768u;
		uint32_t rem  = bytes % 32768u;

		dru_queue_raw_2d(ch, dst, src, 32768u, (uint16_t)rows, 32768, 32768);
		if (rem != 0u) {
			uint64_t off = (uint64_t)rows * 32768ull;

			dru_queue_raw_2d(ch, dst + off, src + off,
					 (uint16_t)rem, 1u, (int32_t)rem, (int32_t)rem);
		}
	}
}

static void dru_queue_marker(uint32_t ch)
{
	uint64_t seq  = ++g_dru_marker_seq[ch];
	uint64_t msrc = DRU_MARKER_SRC_PA(ch);
	uint64_t mdst = DRU_MARKER_DST_PA(ch);

	*(volatile uint64_t *)(uintptr_t)msrc = seq;
	__memory_fence(0x1);
	dru_queue_raw_2d(ch, mdst, msrc, 8u, 1u, 8, 8);
}

void c7x_dru_submit_2d(uint32_t ch, uint64_t dst, uint64_t src,
		       uint16_t icnt0, uint16_t icnt1,
		       int32_t src_dim1, int32_t dst_dim1)
{
	dru_queue_raw_2d(ch, dst, src, icnt0, icnt1, src_dim1, dst_dim1);
	dru_queue_marker(ch);
}

void c7x_dru_submit_bytes(uint32_t ch, uint64_t dst, uint64_t src, uint32_t bytes)
{
	dru_queue_raw_bytes(ch, dst, src, bytes);
	dru_queue_marker(ch);
}

void c7x_dru_submit_pair_bytes(uint32_t ch,
			       uint64_t dst0, uint64_t src0, uint32_t bytes0,
			       uint64_t dst1, uint64_t src1, uint32_t bytes1)
{
	dru_queue_raw_bytes(ch, dst0, src0, bytes0);
	dru_queue_raw_bytes(ch, dst1, src1, bytes1);
	dru_queue_marker(ch);
}

int c7x_dru_wait(uint32_t ch)
{
	volatile uint64_t *mdst = (volatile uint64_t *)(uintptr_t)DRU_MARKER_DST_PA(ch);
	uint64_t want = g_dru_marker_seq[ch];
	uint32_t spin;

	for (spin = 0; spin < 5000000u; spin++) {
		if (*mdst == want) {
			__memory_fence(0x1);
			__memory_fence(0x1);
			return 0;
		}
	}
	return -1;
}

int c7x_dru_copy_sync(uint32_t ch, uint64_t dst, uint64_t src, uint32_t bytes)
{
	if (bytes == 0u) {
		return 0;
	}
	c7x_dru_submit_bytes(ch, dst, src, bytes);
	return c7x_dru_wait(ch);
}
