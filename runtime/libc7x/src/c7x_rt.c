/* SPDX-License-Identifier: MIT */
/*
 * Core libc7x runtime: coalescing arena allocator, safe O_SYNC transfer
 * helpers, command-buffer builder, and lifecycle API.
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "c7x_internal.h"

static inline size_t align64(size_t n)
{
	return (n + 63u) & ~((size_t)63u);
}

double c7x_now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* -------------------------------------------------------------------------- */
/* Safe O_SYNC /dev/mem helpers (no ARM64 DC ZVA SIGBUS)                      */
/* -------------------------------------------------------------------------- */

static void mmio_zero(void *dst, size_t n)
{
	volatile uint64_t *p64 = (volatile uint64_t *)dst;
	size_t i = 0;

	if ((((uintptr_t)dst) & 7u) == 0u) {
		for (; i + 8u <= n; i += 8u) {
			p64[i / 8u] = 0ull;
		}
	}
	for (; i < n; i++) {
		((volatile uint8_t *)dst)[i] = 0u;
	}
}

static void mmio_write(void *dst, const void *src, size_t n)
{
	volatile uint8_t *d8 = (volatile uint8_t *)dst;
	const uint8_t *s8 = (const uint8_t *)src;
	size_t i = 0;

	/* Align destination up to 8 bytes first */
	while (i < n && (((uintptr_t)(d8 + i)) & 7u) != 0u) {
		d8[i] = s8[i];
		i++;
	}
	/* Stream 8-byte stores (using memcpy to avoid UB/unaligned src penalty) */
	for (; i + 8u <= n; i += 8u) {
		uint64_t v;
		memcpy(&v, s8 + i, sizeof(v));
		*(volatile uint64_t *)(d8 + i) = v;
	}
	for (; i < n; i++) {
		d8[i] = s8[i];
	}
}

static void mmio_read(void *dst, const void *src, size_t n)
{
	uint8_t *d8 = (uint8_t *)dst;
	const volatile uint8_t *s8 = (const volatile uint8_t *)src;
	size_t i = 0;

	while (i < n && (((uintptr_t)(s8 + i)) & 7u) != 0u) {
		d8[i] = s8[i];
		i++;
	}
	for (; i + 8u <= n; i += 8u) {
		uint64_t v = *(const volatile uint64_t *)(s8 + i);
		memcpy(d8 + i, &v, sizeof(v));
	}
	for (; i < n; i++) {
		d8[i] = s8[i];
	}
}

/* -------------------------------------------------------------------------- */
/* Coalescing free-list arena allocator (64-byte aligned)                     */
/* -------------------------------------------------------------------------- */

int c7x_arena_init(struct c7x_arena *a, uint64_t phys_base, void *host_base, size_t size)
{
	struct c7x_arena_block *blk;

	memset(a, 0, sizeof(*a));
	if (size == 0u || !host_base) {
		return 0;
	}
	blk = (struct c7x_arena_block *)calloc(1, sizeof(*blk));
	if (!blk) {
		return -1;
	}
	blk->offset = 0u;
	blk->size   = size & ~((size_t)63u);
	blk->used   = 0;

	a->phys_base  = phys_base;
	a->host_base  = (uint8_t *)host_base;
	a->total_size = blk->size;
	a->free_size  = blk->size;
	a->head       = blk;
	return 0;
}

void c7x_arena_destroy(struct c7x_arena *a)
{
	struct c7x_arena_block *cur = a->head;

	while (cur) {
		struct c7x_arena_block *next = cur->next;
		free(cur);
		cur = next;
	}
	memset(a, 0, sizeof(*a));
}

struct c7x_arena_block *c7x_arena_alloc(struct c7x_arena *a, size_t bytes)
{
	size_t need;
	struct c7x_arena_block *cur;
	struct c7x_arena_block *best = NULL;

	if (bytes > SIZE_MAX - 63u) {
		return NULL;
	}
	need = align64(bytes ? bytes : 64u);

	/* Best-fit search to minimize fragmentation on 1.5 GiB arena */
	for (cur = a->head; cur; cur = cur->next) {
		if (!cur->used && cur->size >= need) {
			if (!best || cur->size < best->size) {
				best = cur;
				if (best->size == need) {
					break;
				}
			}
		}
	}
	if (!best) {
		return NULL;
	}

	if (best->size > need) {
		struct c7x_arena_block *tail =
			(struct c7x_arena_block *)calloc(1, sizeof(*tail));
		if (!tail) {
			return NULL;
		}
		tail->offset = best->offset + need;
		tail->size   = best->size - need;
		tail->used   = 0;
		tail->prev   = best;
		tail->next   = best->next;
		if (best->next) {
			best->next->prev = tail;
		}
		best->next = tail;
		best->size = need;
	}

	best->used = 1;
	a->free_size -= best->size;
	return best;
}

void c7x_arena_free(struct c7x_arena *a, struct c7x_arena_block *blk)
{
	if (!blk || !blk->used) {
		return;
	}
	blk->used = 0;
	a->free_size += blk->size;

	/* Coalesce with next block */
	if (blk->next && !blk->next->used) {
		struct c7x_arena_block *n = blk->next;
		blk->size += n->size;
		blk->next = n->next;
		if (n->next) {
			n->next->prev = blk;
		}
		free(n);
	}
	/* Coalesce with previous block */
	if (blk->prev && !blk->prev->used) {
		struct c7x_arena_block *p = blk->prev;
		p->size += blk->size;
		p->next = blk->next;
		if (blk->next) {
			blk->next->prev = p;
		}
		free(blk);
	}
}

/* -------------------------------------------------------------------------- */
/* Context lifecycle & public API                                             */
/* -------------------------------------------------------------------------- */

c7x_ctx_t *c7x_open(const struct c7x_open_opts *opts)
{
	struct c7x_open_opts def_opts;
	c7x_ctx_t *ctx;
	int rc = -1;

	memset(&def_opts, 0, sizeof(def_opts));
	if (!opts) {
		opts = &def_opts;
	}

	ctx = (c7x_ctx_t *)calloc(1, sizeof(*ctx));
	if (!ctx) {
		return NULL;
	}
	pthread_mutex_init(&ctx->lock, NULL);

	if (opts->transport == C7X_TRANSPORT_HW) {
#ifdef LIBC7X_ENABLE_HW
		rc = c7x_transport_hw_init(ctx, opts);
#endif
	} else if (opts->transport == C7X_TRANSPORT_HOSTEMU) {
#ifdef LIBC7X_ENABLE_HOSTEMU
		rc = c7x_transport_hostemu_init(ctx, opts);
#endif
	} else {
		/* AUTO: try HW first if compiled in, else HOSTEMU */
#ifdef LIBC7X_ENABLE_HW
		rc = c7x_transport_hw_init(ctx, opts);
#endif
#ifdef LIBC7X_ENABLE_HOSTEMU
		if (rc != 0) {
			rc = c7x_transport_hostemu_init(ctx, opts);
		}
#endif
	}

	if (rc != 0) {
		pthread_mutex_destroy(&ctx->lock);
		free(ctx);
		return NULL;
	}
	return ctx;
}

void c7x_close(c7x_ctx_t *ctx)
{
	if (!ctx) {
		return;
	}
	if (ctx->ops.close) {
		ctx->ops.close(ctx);
	}
	c7x_arena_destroy(&ctx->ddr);
	c7x_arena_destroy(&ctx->msmc);
	pthread_mutex_destroy(&ctx->lock);
	free(ctx);
}

int c7x_get_info(const c7x_ctx_t *ctx, struct c7x_rt_info *info)
{
	if (!ctx || !info) {
		return -1;
	}
	memset(info, 0, sizeof(*info));
	info->ddr_base    = ctx->ddr.phys_base;
	info->ddr_total   = ctx->ddr.total_size;
	info->ddr_free    = ctx->ddr.free_size;
	info->msmc_base   = ctx->msmc.phys_base;
	info->msmc_total  = ctx->msmc.total_size;
	info->msmc_free   = ctx->msmc.free_size;
	info->dru_ok      = ctx->dru_ok;
	info->is_hostemu  = ctx->is_hostemu;
	info->max_block_m = ctx->is_hostemu ? 16u : 60u;
	return 0;
}

int c7x_ping(c7x_ctx_t *ctx, double *wall_us_out)
{
	int rc;

	if (!ctx || !ctx->ops.ping) {
		return -1;
	}
	pthread_mutex_lock(&ctx->lock);
	rc = ctx->ops.ping(ctx, wall_us_out);
	pthread_mutex_unlock(&ctx->lock);
	return rc;
}

c7x_buf_t *c7x_alloc(c7x_ctx_t *ctx, size_t bytes, enum c7x_mem_tier tier)
{
	struct c7x_arena *a;
	struct c7x_arena_block *blk;
	c7x_buf_t *buf;

	if (!ctx) {
		return NULL;
	}
	a = (tier == C7X_MEM_MSMC) ? &ctx->msmc : &ctx->ddr;

	pthread_mutex_lock(&ctx->lock);
	blk = c7x_arena_alloc(a, bytes);
	pthread_mutex_unlock(&ctx->lock);
	if (!blk) {
		return NULL;
	}

	buf = (c7x_buf_t *)calloc(1, sizeof(*buf));
	if (!buf) {
		pthread_mutex_lock(&ctx->lock);
		c7x_arena_free(a, blk);
		pthread_mutex_unlock(&ctx->lock);
		return NULL;
	}
	buf->ctx  = ctx;
	buf->tier = tier;
	buf->blk  = blk;
	buf->phys = a->phys_base + blk->offset;
	buf->host = a->host_base + blk->offset;
	buf->size = blk->size;
	return buf;
}

void c7x_free(c7x_ctx_t *ctx, c7x_buf_t *buf)
{
	struct c7x_arena *a;

	if (!ctx || !buf) {
		return;
	}
	a = (buf->tier == C7X_MEM_MSMC) ? &ctx->msmc : &ctx->ddr;
	pthread_mutex_lock(&ctx->lock);
	c7x_arena_free(a, buf->blk);
	pthread_mutex_unlock(&ctx->lock);
	free(buf);
}

uint64_t c7x_buf_phys(const c7x_buf_t *buf)
{
	return buf ? buf->phys : 0ull;
}

void *c7x_buf_host_ptr(const c7x_buf_t *buf)
{
	return buf ? (void *)buf->host : NULL;
}

size_t c7x_buf_size(const c7x_buf_t *buf)
{
	return buf ? buf->size : 0u;
}

enum c7x_mem_tier c7x_buf_tier(const c7x_buf_t *buf)
{
	return buf ? buf->tier : C7X_MEM_DDR;
}

void c7x_write_ptr(c7x_ctx_t *ctx, void *dst_arena_ptr, const void *src, size_t n)
{
	if (!dst_arena_ptr || !src || n == 0u) return;
	if (ctx && ctx->need_mmio) {
		mmio_write(dst_arena_ptr, src, n);
	} else {
		memcpy(dst_arena_ptr, src, n);
	}
}

void c7x_read_ptr(c7x_ctx_t *ctx, void *dst, const void *src_arena_ptr, size_t n)
{
	if (!dst || !src_arena_ptr || n == 0u) return;
	if (ctx && ctx->need_mmio) {
		mmio_read(dst, src_arena_ptr, n);
	} else {
		memcpy(dst, src_arena_ptr, n);
	}
}

void c7x_zero_ptr(c7x_ctx_t *ctx, void *dst_arena_ptr, size_t n)
{
	if (!dst_arena_ptr || n == 0u) return;
	if (ctx && ctx->need_mmio) {
		mmio_zero(dst_arena_ptr, n);
	} else {
		memset(dst_arena_ptr, 0, n);
	}
}

void c7x_write(c7x_ctx_t *ctx, c7x_buf_t *buf, size_t off, const void *src, size_t n)
{
	if (!buf || off > buf->size || n > buf->size - off) return;
	c7x_write_ptr(ctx, buf->host + off, src, n);
}

void c7x_read(c7x_ctx_t *ctx, void *dst, const c7x_buf_t *buf, size_t off, size_t n)
{
	if (!buf || off > buf->size || n > buf->size - off) return;
	c7x_read_ptr(ctx, dst, buf->host + off, n);
}

void c7x_zero(c7x_ctx_t *ctx, c7x_buf_t *buf, size_t off, size_t n)
{
	if (!buf || off > buf->size || n > buf->size - off) return;
	c7x_zero_ptr(ctx, buf->host + off, n);
}

uint64_t c7x_host_to_phys(const c7x_ctx_t *ctx, const void *host_ptr)
{
	uintptr_t p = (uintptr_t)host_ptr;

	if (!ctx || !host_ptr) {
		return 0ull;
	}
	if (ctx->ddr.host_base) {
		uintptr_t lo = (uintptr_t)ctx->ddr.host_base;
		uintptr_t hi = lo + ctx->ddr.total_size;
		if (p >= lo && p < hi) {
			return ctx->ddr.phys_base + (uint64_t)(p - lo);
		}
	}
	if (ctx->msmc.host_base) {
		uintptr_t lo = (uintptr_t)ctx->msmc.host_base;
		uintptr_t hi = lo + ctx->msmc.total_size;
		if (p >= lo && p < hi) {
			return ctx->msmc.phys_base + (uint64_t)(p - lo);
		}
	}
	return 0ull;
}

/* -------------------------------------------------------------------------- */
/* Command-buffer builder & submit                                            */
/* -------------------------------------------------------------------------- */

struct c7x_cmdbuf {
	c7x_ctx_t      *ctx;
	c7x_buf_t      *dev_buf;     /* Holds [cmds | descs] in DDR arena */
	struct c7x_cmd *host_cmds;   /* Cached host staging copy (avoids O_SYNC writes until submit) */
	uint8_t        *host_descs;
	uint32_t        max_cmds;
	uint32_t        n_cmds;
	size_t          cmds_bytes;
	size_t          desc_cap;
	size_t          desc_used;
};

c7x_cmdbuf_t *c7x_cmdbuf_create(c7x_ctx_t *ctx, uint32_t max_cmds, size_t desc_bytes)
{
	c7x_cmdbuf_t *cb;
	size_t cmds_sz, total_sz;

	if (!ctx || max_cmds == 0u) {
		return NULL;
	}
	cmds_sz  = align64((size_t)max_cmds * sizeof(struct c7x_cmd));
	desc_bytes = align64(desc_bytes ? desc_bytes : 65536u);
	total_sz = cmds_sz + desc_bytes;

	cb = (c7x_cmdbuf_t *)calloc(1, sizeof(*cb));
	if (!cb) {
		return NULL;
	}
	cb->ctx = ctx;
	cb->dev_buf = c7x_alloc(ctx, total_sz, C7X_MEM_DDR);
	cb->host_cmds = (struct c7x_cmd *)calloc(max_cmds, sizeof(struct c7x_cmd));
	cb->host_descs = (uint8_t *)calloc(1, desc_bytes);
	if (!cb->dev_buf || !cb->host_cmds || !cb->host_descs) {
		c7x_cmdbuf_destroy(cb);
		return NULL;
	}
	cb->max_cmds   = max_cmds;
	cb->cmds_bytes = cmds_sz;
	cb->desc_cap   = desc_bytes;
	return cb;
}

void c7x_cmdbuf_destroy(c7x_cmdbuf_t *cb)
{
	if (!cb) {
		return;
	}
	if (cb->dev_buf) {
		c7x_free(cb->ctx, cb->dev_buf);
	}
	free(cb->host_cmds);
	free(cb->host_descs);
	free(cb);
}

void c7x_cmdbuf_reset(c7x_cmdbuf_t *cb)
{
	if (!cb) return;
	cb->n_cmds    = 0u;
	cb->desc_used = 0u;
}

int c7x_cmdbuf_push(c7x_cmdbuf_t *cb, const struct c7x_cmd *cmd)
{
	if (!cb || !cmd || cb->n_cmds >= cb->max_cmds) {
		return -1;
	}
	cb->host_cmds[cb->n_cmds++] = *cmd;
	return 0;
}

uint64_t c7x_cmdbuf_push_desc(c7x_cmdbuf_t *cb, const void *desc, size_t bytes)
{
	size_t aligned = align64(bytes);
	size_t off;

	if (!cb || !desc || bytes == 0u || cb->desc_used + aligned > cb->desc_cap) {
		return 0ull;
	}
	off = cb->desc_used;
	memcpy(cb->host_descs + off, desc, bytes);
	cb->desc_used += aligned;
	return cb->dev_buf->phys + cb->cmds_bytes + off;
}

uint32_t c7x_cmdbuf_count(const c7x_cmdbuf_t *cb)
{
	return cb ? cb->n_cmds : 0u;
}

int c7x_cmdbuf_submit(c7x_ctx_t *ctx, c7x_cmdbuf_t *cb, struct c7x_exec_stats *stats)
{
	int rc;

	if (!ctx || !cb || !ctx->ops.exec_list) {
		return -1;
	}
	if (cb->n_cmds == 0u) {
		if (stats) memset(stats, 0, sizeof(*stats));
		return 0;
	}

	/* Flush staged commands and descriptors to DDR arena in one burst */
	c7x_write(ctx, cb->dev_buf, 0u, cb->host_cmds,
		  (size_t)cb->n_cmds * sizeof(struct c7x_cmd));
	if (cb->desc_used > 0u) {
		c7x_write(ctx, cb->dev_buf, cb->cmds_bytes, cb->host_descs, cb->desc_used);
	}

	pthread_mutex_lock(&ctx->lock);
	rc = ctx->ops.exec_list(ctx, cb->dev_buf->phys, cb->n_cmds, stats);
	pthread_mutex_unlock(&ctx->lock);
	return rc;
}
