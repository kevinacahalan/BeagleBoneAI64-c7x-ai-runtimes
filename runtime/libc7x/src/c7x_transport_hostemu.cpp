/* SPDX-License-Identifier: MIT */
/*
 * x86_64 C7000 Host Emulation transport for libc7x.
 *
 * Allocates host-RAM DDR and MSMC arenas + a 444 KiB L2 scratchpad and executes
 * the exact same C7x firmware command-list engine (`c7x_cmd_exec_list`)
 * compiled against TI's `libc70-host-emulation.a`.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c7x_internal.h"
#include "cmd_exec.h"

struct hostemu_priv {
	void  *ddr_mem;
	void  *msmc_mem;
	void  *l2_mem;
	size_t l2_bytes;
};

static void *alloc_aligned64(size_t bytes)
{
	void *p = NULL;
	size_t sz = (bytes + 63u) & ~((size_t)63u);

	if (sz == 0u) {
		sz = 64u;
	}
	if (posix_memalign(&p, 64u, sz) != 0) {
		return NULL;
	}
	memset(p, 0, sz);
	return p;
}

static int hostemu_ping(c7x_ctx_t *ctx, double *wall_us_out)
{
	double t0 = c7x_now_us();

	(void)ctx;
	if (wall_us_out) {
		*wall_us_out = c7x_now_us() - t0;
	}
	return 0;
}

static int hostemu_exec_list(c7x_ctx_t *ctx, uint64_t cmds_phys, uint32_t n_cmds,
			     struct c7x_exec_stats *stats)
{
	struct hostemu_priv *priv = (struct hostemu_priv *)ctx->transport_priv;
	const struct c7x_cmd *cmds = (const struct c7x_cmd *)(uintptr_t)cmds_phys;
	uint32_t done = 0u, fail_idx = 0u;
	double t0 = c7x_now_us();
	int rc;

	rc = c7x_cmd_exec_list(cmds, n_cmds, priv->l2_mem, (uint32_t)priv->l2_bytes,
			       &done, &fail_idx);
	if (stats) {
		stats->cycles       = 0ull;
		stats->total_cycles = 0ull;
		stats->wall_us      = c7x_now_us() - t0;
		stats->cmds_done    = done;
		stats->fail_idx     = fail_idx;
	}
	return rc;
}

static void hostemu_close(c7x_ctx_t *ctx)
{
	struct hostemu_priv *priv = (struct hostemu_priv *)ctx->transport_priv;

	if (priv) {
		free(priv->ddr_mem);
		free(priv->msmc_mem);
		free(priv->l2_mem);
		free(priv);
		ctx->transport_priv = NULL;
	}
}

extern "C" int c7x_transport_hostemu_init(c7x_ctx_t *ctx, const struct c7x_open_opts *opts)
{
	struct hostemu_priv *priv;
	size_t ddr_sz  = (opts && opts->emu_ddr_bytes)  ? opts->emu_ddr_bytes  : (256u * 1024u * 1024u);
	size_t msmc_sz = (opts && opts->emu_msmc_bytes) ? opts->emu_msmc_bytes : C7X_MSMC_SIZE;

	priv = (struct hostemu_priv *)calloc(1, sizeof(*priv));
	if (!priv) {
		return -1;
	}
	priv->l2_bytes = C7X_L2_SCRATCH_SIZE;
	priv->ddr_mem  = alloc_aligned64(ddr_sz);
	priv->msmc_mem = alloc_aligned64(msmc_sz);
	priv->l2_mem   = alloc_aligned64(priv->l2_bytes);
	if (!priv->ddr_mem || !priv->msmc_mem || !priv->l2_mem) {
		free(priv->ddr_mem);
		free(priv->msmc_mem);
		free(priv->l2_mem);
		free(priv);
		return -1;
	}

	/* In hostemu mode, physical address == host virtual pointer! */
	if (c7x_arena_init(&ctx->ddr, (uint64_t)(uintptr_t)priv->ddr_mem, priv->ddr_mem, ddr_sz) != 0 ||
	    c7x_arena_init(&ctx->msmc, (uint64_t)(uintptr_t)priv->msmc_mem, priv->msmc_mem, msmc_sz) != 0) {
		c7x_arena_destroy(&ctx->ddr);
		c7x_arena_destroy(&ctx->msmc);
		free(priv->ddr_mem);
		free(priv->msmc_mem);
		free(priv->l2_mem);
		free(priv);
		return -1;
	}

	ctx->transport_priv = priv;
	ctx->ops.ping       = hostemu_ping;
	ctx->ops.exec_list  = hostemu_exec_list;
	ctx->ops.close      = hostemu_close;
	ctx->dru_ok         = 0u;
	ctx->is_hostemu     = 1u;
	ctx->need_mmio      = 0;
	return 0;
}
