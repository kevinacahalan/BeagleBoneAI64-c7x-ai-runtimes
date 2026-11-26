/* SPDX-License-Identifier: MIT */
/*
 * Internal definitions shared across libc7x translation units.
 */
#ifndef LIBC7X_INTERNAL_H_
#define LIBC7X_INTERNAL_H_

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "c7x_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

struct c7x_arena_block {
	size_t offset;
	size_t size;
	int    used;
	struct c7x_arena_block *prev;
	struct c7x_arena_block *next;
};

struct c7x_arena {
	uint64_t phys_base;
	uint8_t *host_base;
	size_t   total_size;
	size_t   free_size;
	struct c7x_arena_block *head;
};

struct c7x_buf {
	c7x_ctx_t              *ctx;
	enum c7x_mem_tier       tier;
	struct c7x_arena_block *blk;
	uint64_t                phys;
	uint8_t                *host;
	size_t                  size;
};

struct c7x_transport_ops {
	int  (*ping)(c7x_ctx_t *ctx, double *wall_us_out);
	int  (*exec_list)(c7x_ctx_t *ctx, uint64_t cmds_phys, uint32_t n_cmds,
			  struct c7x_exec_stats *stats);
	void (*close)(c7x_ctx_t *ctx);
};

struct c7x_ctx {
	pthread_mutex_t          lock;
	struct c7x_arena         ddr;
	struct c7x_arena         msmc;
	struct c7x_transport_ops ops;
	void                    *transport_priv;
	uint32_t                 dru_ok;
	uint32_t                 is_hostemu;
	int                      need_mmio;
};

/* Shared arena helpers */
int   c7x_arena_init(struct c7x_arena *a, uint64_t phys_base, void *host_base, size_t size);
void  c7x_arena_destroy(struct c7x_arena *a);
struct c7x_arena_block *c7x_arena_alloc(struct c7x_arena *a, size_t bytes);
void  c7x_arena_free(struct c7x_arena *a, struct c7x_arena_block *blk);

/* Transport constructors */
int c7x_transport_hw_init(c7x_ctx_t *ctx, const struct c7x_open_opts *opts);
int c7x_transport_hostemu_init(c7x_ctx_t *ctx, const struct c7x_open_opts *opts);

double c7x_now_us(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBC7X_INTERNAL_H_ */
