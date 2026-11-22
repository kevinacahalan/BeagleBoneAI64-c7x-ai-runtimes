/* SPDX-License-Identifier: MIT */
/*
 * C7x MMU and cache initialization for the TDA4VM (J721E) AI runtime.
 *
 * Cache sizes: L1D = 32 KiB cache, L2 = 64 KiB cache + 448 KiB SRAM.
 *
 * Identity-mapped regions (4 KiB granule, 48-bit VA):
 *   0x00000000  1 GiB   device       mailboxes, GTC, SoC peripherals
 *   0x40000000  1 GiB   device       compute cluster (incl. DRU @ 0x6d000000)
 *     0x64800000  4 KiB   normal NC    control block (struct c7x_ctrl)
 *     0x64801000  444 KiB normal WB    L2 SRAM scratchpad
 *     0x70200000  4 MiB   normal WB    on-chip MSMC SRAM arena (clear of ATF/DMSC)
 *   0xa8000000  2 MiB   normal NC    DMA pool, resource table & trace buffer
 *   0xa8200000  14 MiB  normal WB    C7x code, data, stack, MMU tables
 *   0xac000000  64 MiB  normal WB    shared DDR tensor arena
 *   0xc0000000  512 MiB normal WB    large LLM DDR arena
 */

#include <stdint.h>
#include <c7x.h>

#include "c7x_ai_proto.h"
#include "mmu.h"

/* regs.asm */
extern void ai64_set_mair(uint64_t mair);
extern void ai64_set_tcr(uint64_t tcr);
extern void ai64_set_tbr(uint64_t table);
extern void ai64_scr_or(uint64_t bits);
extern void ai64_set_l1dcfg(uint64_t cfg);
extern void ai64_set_l2cfg(uint64_t cfg);

/* MAIR bytes 0..7: 0 = device nGnRnE, 4 = normal non-cacheable, 7 = normal write-back. */
#define MAIR_VALUE   0x7dbb4f440c080400ull
#define ATTR_DEVICE  0u
#define ATTR_NC      4u
#define ATTR_WB      7u

/* TCR: 4 KiB granule, write-back outer-shareable table walks, 48-bit VA (T0SZ = 16). */
#define TCR_VALUE    (0x8000ull | 0x0400ull | 0x2000ull | 0x0800ull | \
		      ((64ull - 48ull) << 1) | 1ull)

/* SCR: bit 0 enables MMU; bits 6, 7, 63 match TI secure startup. */
#define SCR_MMU_ON   0x80000000000000c1ull

/* L1DCFG = 32 KiB cache, L2CFG = 64 KiB cache (leaving 448 KiB L2 SRAM). */
#define L1D_32K      4u
#define L2_64K       2u

#define DESC_TABLE   3ull
#define DESC_BLOCK   1ull
#define DESC_PAGE    3ull
#define DESC_AF      (1ull << 10)
#define DESC_SH_OUT  (2ull << 8)
#define DESC_PXN     (1ull << 53)
#define DESC_UXN     (1ull << 54)
#define DESC_ATTR(i) ((uint64_t)(i) << 2)

#define SZ_4K        0x1000ull
#define SZ_2M        0x200000ull
#define SZ_1G        0x40000000ull

#define CARVEOUT_BASE 0xa8000000ull
#define CARVEOUT_SIZE 0x01000000ull

#pragma DATA_SECTION(mmu_tables, ".ai64_mmu")
#pragma DATA_ALIGN(mmu_tables, 4096)
static uint64_t mmu_tables[6][512];

#define L0      mmu_tables[0]
#define L1      mmu_tables[1]
#define L2_1G   mmu_tables[2] /* 0x40000000 .. 0x7FFFFFFF */
#define L2_2G   mmu_tables[3] /* 0x80000000 .. 0xBFFFFFFF */
#define L3_SRAM mmu_tables[4] /* 0x64800000 .. 0x649FFFFF */
#define L2_3G   mmu_tables[5] /* 0xC0000000 .. 0xFFFFFFFF */

static uint64_t leaf(uint64_t type, uint64_t addr, uint32_t attr)
{
	uint64_t desc = type | addr | DESC_ATTR(attr) | DESC_AF | DESC_SH_OUT | DESC_UXN;

	if (attr == ATTR_DEVICE) {
		desc |= DESC_PXN;
	}
	return desc;
}

static uint64_t table(const uint64_t *next)
{
	return DESC_TABLE | (uint64_t)(uintptr_t)next;
}

void c7x_mmu_init(void)
{
	uint64_t addr;
	uint32_t i;

	for (i = 0; i < 6u * 512u; i++) {
		mmu_tables[i / 512u][i % 512u] = 0;
	}

	L0[0] = table(L1);
	L1[0] = leaf(DESC_BLOCK, 0, ATTR_DEVICE);
	L1[1] = table(L2_1G);
	L1[2] = table(L2_2G);
	L1[3] = table(L2_3G);

	/* 0x40000000..0x7FFFFFFF: device by default (covers DRU at 0x6d000000). */
	for (i = 0; i < 512u; i++) {
		L2_1G[i] = leaf(DESC_BLOCK, SZ_1G + i * SZ_2M, ATTR_DEVICE);
	}

	/* C7x L2 SRAM (0x64800000): 4 KiB NC control header + 444 KiB WB scratchpad. */
	L2_1G[(C7X_L2_BASE - SZ_1G) / SZ_2M] = table(L3_SRAM);
	L3_SRAM[0] = leaf(DESC_PAGE, C7X_CTRL_ADDR, ATTR_NC);
	for (i = 1; i < (C7X_CTRL_SIZE + C7X_L2_SCRATCH_SIZE) / SZ_4K; i++) {
		L3_SRAM[i] = leaf(DESC_PAGE, C7X_L2_BASE + i * SZ_4K, ATTR_WB);
	}

	/* On-chip MSMC SRAM arena (0x70200000, 4 MiB, normal write-back). */
	for (addr = C7X_MSMC_BASE; addr < C7X_MSMC_BASE + C7X_MSMC_SIZE; addr += SZ_2M) {
		L2_1G[(addr - SZ_1G) / SZ_2M] = leaf(DESC_BLOCK, addr, ATTR_WB);
	}

	/* C7x DDR carveout: 2 MiB NC (DMA pool + trace/rsc_table), 14 MiB WB (code/data). */
	L2_2G[(CARVEOUT_BASE - 2u * SZ_1G) / SZ_2M] = leaf(DESC_BLOCK, CARVEOUT_BASE, ATTR_NC);
	for (addr = CARVEOUT_BASE + SZ_2M; addr < CARVEOUT_BASE + CARVEOUT_SIZE; addr += SZ_2M) {
		L2_2G[(addr - 2u * SZ_1G) / SZ_2M] = leaf(DESC_BLOCK, addr, ATTR_WB);
	}

	/* Shared DDR tensor arena (0xac000000, 64 MiB, normal write-back). */
	for (addr = C7X_DDR_ARENA_BASE; addr < C7X_DDR_ARENA_BASE + C7X_DDR_ARENA_SIZE; addr += SZ_2M) {
		L2_2G[(addr - 2u * SZ_1G) / SZ_2M] = leaf(DESC_BLOCK, addr, ATTR_WB);
	}

	/* Large LLM DDR tensor arena (0xc0000000, 512 MiB, normal write-back). */
	for (addr = C7X_LLM_ARENA_BASE; addr < C7X_LLM_ARENA_BASE + C7X_LLM_ARENA_SIZE; addr += SZ_2M) {
		L2_3G[(addr - 3u * SZ_1G) / SZ_2M] = leaf(DESC_BLOCK, addr, ATTR_WB);
	}

	ai64_set_mair(MAIR_VALUE);
	ai64_set_tcr(TCR_VALUE);
	ai64_set_tbr((uint64_t)(uintptr_t)L0);
	ai64_set_l2cfg(L2_64K);
	ai64_set_l1dcfg(L1D_32K);
	__memory_fence(0x1);
	ai64_scr_or(SCR_MMU_ON);
}
