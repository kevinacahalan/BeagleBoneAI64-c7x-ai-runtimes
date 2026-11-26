/* SPDX-License-Identifier: MIT */
/*
 * C7x AI runtime firmware entry point.
 *
 * Boots from the C71x DDR carveout (0xa8200000):
 *   1. Enables MMU + L1D (32 KiB) + L2 (64 KiB cache / 448 KiB SRAM) + MSMC + DDR arena.
 *   2. Initializes remoteproc trace buffer in non-cacheable carveout (0xa8100000).
 *   3. Probes & initializes DRU0 channel 0 and publishes the L2 control block.
 *   4. Polls Linux benchmark/inference requests and mailbox shutdown (RP_MBOX_SHUTDOWN).
 */

#include <stdint.h>
#include <c7x.h>

#define AI64_TRACE_SIZE 4096
#define AI64_TRACE_NAME "trace:c7x-ai"
#include "ai64_dsp_trace.h"
#include "ai64_dsp_mbox.h"
#include "c7x_ai_proto.h"
#include "dru.h"
#include "mmu.h"

extern void bench_init(void);
extern int bench_poll(void);
extern const struct c7x_dru_info *bench_dru_info(void);

static void trace_hex64(uint64_t v)
{
	static const char hex[] = "0123456789abcdef";
	int shift;

	trace_puts("0x");
	for (shift = 60; shift >= 0; shift -= 4) {
		trace_putc(hex[(v >> shift) & 0xfu]);
	}
}

int main(void)
{
	uint32_t seconds = 0, jobs = 0;
	uint64_t last;
	const struct c7x_dru_info *dru;

	c7x_mmu_init();
	trace_init();
	trace_puts("c7x-ai boot: DNUM ");
	trace_u32((uint32_t)__DNUM);
	trace_puts((ai64_get_scr() & 1u) ? ", MMU on" : ", MMU off");
	trace_puts(", L1DCFG ");
	trace_u32((uint32_t)ai64_get_l1dcfg());
	trace_puts(", L2CFG ");
	trace_u32((uint32_t)ai64_get_l2cfg());
	trace_putc('\n');

	bench_init();
	dru = bench_dru_info();
	trace_puts("c7x-ai DRU0: ok=");
	trace_u32(dru->ok);
	trace_puts(" ch=");
	trace_u32(dru->ch);
	trace_puts(" pid=");
	trace_hex64(dru->pid);
	trace_puts(" caps=");
	trace_hex64(dru->caps);
	trace_putc('\n');
	trace_mark();

	last = __TSC;
	for (;;) {
		if (bench_poll() && ++jobs == 1u) {
			trace_puts("c7x-ai: first job done\n");
		}
		if (mbox_shutdown_requested((uint32_t)C7X_MBOX_BASE,
					    C7X_MBOX_LINUX_TX, C7X_MBOX_C7X_TX)) {
			trace_puts("c7x-ai: shutting down\n");
			ai64_l1d_wbinv_all();
			ai64_l2_wbinv_all();
			__memory_fence(0x1);
			for (;;) {
				__asm(" IDLE");
			}
		}
		if (__TSC - last >= C7X_CLOCK_HZ) {
			last += C7X_CLOCK_HZ;
			seconds++;
			if (seconds % 30u == 0u) {
				trace_puts("c7x-ai up ");
				trace_u32(seconds);
				trace_puts(" s, jobs=");
				trace_u32(jobs);
				trace_putc('\n');
			}
		}
	}
}
