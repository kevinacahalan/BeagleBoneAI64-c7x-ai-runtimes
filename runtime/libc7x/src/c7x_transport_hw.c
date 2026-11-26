/* SPDX-License-Identifier: MIT */
/*
 * TDA4VM / BeagleBone AI-64 hardware transport for libc7x.
 *
 * Maps the non-cacheable C7x control block (0x64800000), on-chip MSMC SRAM
 * (0x70040000, 7.625 MiB), and the largest available reserved DDR arena
 * (1.5 GiB @ 0x880000000 -> 512 MiB @ 0xC0000000 -> 64 MiB @ 0xAC000000) via
 * /dev/mem (O_SYNC) and submits command lists via C7X_OP_EXEC_LIST.
 *
 * Acquires an exclusive non-blocking flock on /tmp/c7x-ai.lock so two host
 * processes cannot corrupt each other's shared arena allocations.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "c7x_internal.h"

#define C7X_HW_LOCK_PATH "/tmp/c7x-ai.lock"

struct hw_priv {
	int                       lock_fd;
	int                       dead;
	volatile struct c7x_ctrl *ctrl;
	void                     *ddr_map;
	size_t                    ddr_size;
	void                     *msmc_map;
	size_t                    msmc_size;
};

static void *hw_map_phys(uint64_t addr, size_t len)
{
	int fd = open("/dev/mem", O_RDWR | O_SYNC);
	void *p;

	if (fd < 0) {
		return NULL;
	}
	p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)addr);
	close(fd);
	if (p == MAP_FAILED) {
		return NULL;
	}
	return p;
}

static int dt_node_exists(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0;
}

static inline void hw_mem_barrier(void)
{
#if defined(__aarch64__)
	__asm__ volatile("dsb sy" ::: "memory");
#else
	__sync_synchronize();
#endif
}

static int hw_rpc(struct hw_priv *priv, uint32_t op,
		  uint32_t m, uint32_t k, uint32_t n,
		  uint64_t a_addr, uint64_t b_addr, uint64_t c_addr,
		  double timeout_us,
		  struct c7x_exec_stats *stats)
{
	volatile struct c7x_ctrl *ctrl = priv->ctrl;
	uint32_t next_seq;
	double t0;

	if (stats) {
		memset(stats, 0, sizeof(*stats));
		stats->fail_idx = 0xFFFFFFFFu;
	}
	if (priv->dead) {
		fprintf(stderr, "libc7x: context marked dead after prior DSP timeout\n");
		return -1;
	}

	next_seq = ctrl->req_seq + 1u;
	ctrl->op     = op;
	ctrl->flags  = 0u;
	ctrl->iters  = 1u;
	ctrl->m      = m;
	ctrl->k      = k;
	ctrl->n      = n;
	ctrl->a_addr = a_addr;
	ctrl->b_addr = b_addr;
	ctrl->c_addr = c_addr;
	hw_mem_barrier();

	t0 = c7x_now_us();
	ctrl->req_seq = next_seq;
	hw_mem_barrier();

	while (ctrl->done_seq != next_seq) {
		if (c7x_now_us() - t0 > timeout_us) {
			priv->dead = 1;
			if (stats) {
				stats->wall_us  = c7x_now_us() - t0;
				stats->fail_idx = 0xFFFFFFFFu;
			}
			fprintf(stderr, "libc7x: TIMEOUT waiting for C7x op=%u (context poisoned)\n", op);
			return -1;
		}
	}
	hw_mem_barrier();

	if (stats) {
		stats->cycles       = ctrl->cycles;
		stats->total_cycles = ctrl->total_cycles;
		stats->wall_us      = c7x_now_us() - t0;
		stats->cmds_done    = (uint32_t)ctrl->aux0;
		stats->fail_idx     = (uint32_t)(ctrl->aux1 & 0xFFFFFFFFull);
	}

	if (ctrl->status != C7X_STATUS_OK) {
		fprintf(stderr, "libc7x: C7x op=%u failed status=%u (aux0=0x%" PRIx64 ", aux1=0x%" PRIx64 ")\n",
			op, ctrl->status, ctrl->aux0, ctrl->aux1);
		return -1;
	}
	return 0;
}

static int hw_ping(c7x_ctx_t *ctx, double *wall_us_out)
{
	struct hw_priv *priv = (struct hw_priv *)ctx->transport_priv;
	struct c7x_exec_stats stats;
	int rc;

	rc = hw_rpc(priv, C7X_OP_PING, 0u, 0u, 0u, 0ull, 0ull, 0ull, 2000000.0, &stats);
	if (wall_us_out) {
		*wall_us_out = stats.wall_us;
	}
	return rc;
}

static int hw_exec_list(c7x_ctx_t *ctx, uint64_t cmds_phys, uint32_t n_cmds,
			struct c7x_exec_stats *stats)
{
	struct hw_priv *priv = (struct hw_priv *)ctx->transport_priv;

	return hw_rpc(priv, C7X_OP_EXEC_LIST, n_cmds, C7X_CMD_ABI_VERSION, 0u,
		      cmds_phys, 0ull, 0ull, 120000000.0, stats);
}

static void hw_priv_cleanup(struct hw_priv *priv)
{
	if (!priv) {
		return;
	}
	if (priv->ctrl) {
		munmap((void *)priv->ctrl, C7X_CTRL_SIZE);
	}
	if (priv->ddr_map) {
		munmap(priv->ddr_map, priv->ddr_size);
	}
	if (priv->msmc_map) {
		munmap(priv->msmc_map, priv->msmc_size);
	}
	if (priv->lock_fd >= 0) {
		flock(priv->lock_fd, LOCK_UN);
		close(priv->lock_fd);
	}
	free(priv);
}

static void hw_close(c7x_ctx_t *ctx)
{
	struct hw_priv *priv = (struct hw_priv *)ctx->transport_priv;

	hw_priv_cleanup(priv);
	ctx->transport_priv = NULL;
}

int c7x_transport_hw_init(c7x_ctx_t *ctx, const struct c7x_open_opts *opts)
{
	struct hw_priv *priv;
	struct {
		const char *dt_path;
		uint64_t    base;
		size_t      size;
	} candidates[3] = {
		{ "/proc/device-tree/reserved-memory/c7x-llm-arena@880000000", C7X_LLM_HI_ARENA_BASE, C7X_LLM_HI_ARENA_SIZE },
		{ "/proc/device-tree/reserved-memory/c7x-llm-arena@c0000000",  C7X_LLM_ARENA_BASE,    C7X_LLM_ARENA_SIZE },
		{ "/proc/device-tree/reserved-memory/c7x-ai-arena@ac000000",   C7X_DDR_ARENA_BASE,    C7X_DDR_ARENA_SIZE },
	};
	uint64_t ddr_base = 0ull;
	size_t   ddr_size = 0u;
	int i;

	(void)opts;
	priv = (struct hw_priv *)calloc(1, sizeof(*priv));
	if (!priv) {
		return -1;
	}
	priv->lock_fd = -1;

	/* Exclusive process lock so concurrent processes cannot clobber the shared arena */
	priv->lock_fd = open(C7X_HW_LOCK_PATH, O_CREAT | O_RDWR, 0660);
	if (priv->lock_fd < 0) {
		fprintf(stderr, "libc7x: warning: could not open lock file %s (%s); continuing unlocked\n",
			C7X_HW_LOCK_PATH, strerror(errno));
	} else if (flock(priv->lock_fd, LOCK_EX | LOCK_NB) != 0) {
		fprintf(stderr, "libc7x: C7x DSP is already locked by another process (%s)\n",
			C7X_HW_LOCK_PATH);
		close(priv->lock_fd);
		priv->lock_fd = -1;
		free(priv);
		return -1;
	}

	priv->ctrl = (volatile struct c7x_ctrl *)hw_map_phys(C7X_CTRL_ADDR, C7X_CTRL_SIZE);
	if (!priv->ctrl) {
		hw_priv_cleanup(priv);
		return -1;
	}
	if (priv->ctrl->magic != C7X_CTRL_MAGIC || priv->ctrl->version != C7X_CTRL_VERSION) {
		hw_priv_cleanup(priv);
		return -1;
	}

	/* Verify firmware supports C7X_OP_EXEC_LIST + matching ABI version up front (2 s timeout) */
	if (hw_rpc(priv, C7X_OP_EXEC_LIST, 0u, C7X_CMD_ABI_VERSION, 0u,
		   0ull, 0ull, 0ull, 2000000.0, NULL) != 0) {
		fprintf(stderr, "libc7x: C7x firmware does not support C7X_OP_EXEC_LIST ABI v%u\n",
			C7X_CMD_ABI_VERSION);
		hw_priv_cleanup(priv);
		return -1;
	}

	/* Try reserved DDR arenas from largest to smallest with mmap fallback */
	for (i = 0; i < 3; i++) {
		if (dt_node_exists(candidates[i].dt_path)) {
			void *m = hw_map_phys(candidates[i].base, candidates[i].size);
			if (m) {
				priv->ddr_map  = m;
				priv->ddr_size = candidates[i].size;
				ddr_base = candidates[i].base;
				ddr_size = candidates[i].size;
				break;
			}
		}
	}
	if (!priv->ddr_map) {
		/* Last-resort fallback to 64 MiB primary arena */
		priv->ddr_map = hw_map_phys(C7X_DDR_ARENA_BASE, C7X_DDR_ARENA_SIZE);
		if (!priv->ddr_map) {
			hw_priv_cleanup(priv);
			return -1;
		}
		priv->ddr_size = C7X_DDR_ARENA_SIZE;
		ddr_base = C7X_DDR_ARENA_BASE;
		ddr_size = C7X_DDR_ARENA_SIZE;
	}

	/* Map 7.625 MiB MSMC SRAM (optional) */
	priv->msmc_map = hw_map_phys(C7X_MSMC_BASE, C7X_MSMC_SIZE);
	if (priv->msmc_map) {
		priv->msmc_size = C7X_MSMC_SIZE;
	}

	if (c7x_arena_init(&ctx->ddr, ddr_base, priv->ddr_map, ddr_size) != 0) {
		hw_priv_cleanup(priv);
		return -1;
	}
	if (priv->msmc_map) {
		if (c7x_arena_init(&ctx->msmc, C7X_MSMC_BASE, priv->msmc_map, priv->msmc_size) != 0) {
			c7x_arena_destroy(&ctx->ddr);
			hw_priv_cleanup(priv);
			return -1;
		}
	}

	ctx->transport_priv = priv;
	ctx->ops.ping       = hw_ping;
	ctx->ops.exec_list  = hw_exec_list;
	ctx->ops.close      = hw_close;
	ctx->dru_ok         = priv->ctrl->dru_ok;
	ctx->is_hostemu     = 0u;
	ctx->need_mmio      = 1;
	return 0;
}
