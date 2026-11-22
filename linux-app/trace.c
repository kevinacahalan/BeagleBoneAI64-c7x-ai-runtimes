/* SPDX-License-Identifier: MIT */
/*
 * Read or follow the C7x remoteproc trace buffer via /dev/mem.
 */
#define _GNU_SOURCE

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "c7x_ai.h"

static int elf_trace(const char *file, uint32_t *da, uint32_t *len)
{
	FILE *f = fopen(file, "rb");
	unsigned char *img = NULL;
	long size;
	int ret = -1;
	uint64_t shoff;
	unsigned shentsize, shnum, shstrndx, i;

	if (!f) {
		perror(file);
		return -1;
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	img = malloc((size_t)size);
	if (!img || fread(img, 1, (size_t)size, f) != (size_t)size ||
	    size < 64 || memcmp(img, "\177ELF", 4) || img[4] != 2) {
		goto out;
	}
#define U16(o) ((uint32_t)img[o] | (uint32_t)img[(o) + 1] << 8)
#define U32(o) (U16(o) | U16((o) + 2) << 16)
#define U64(o) ((uint64_t)U32(o) | (uint64_t)U32((o) + 4) << 32)
	shoff     = U64(0x28);
	shentsize = U16(0x3a);
	shnum     = U16(0x3c);
	shstrndx  = U16(0x3e);
	if (shoff + (uint64_t)shnum * shentsize > (uint64_t)size || shstrndx >= shnum) {
		goto out;
	}
	{
		uint64_t s = shoff + (uint64_t)shstrndx * shentsize;
		uint64_t stroff = U64(s + 0x18);

		for (i = 0; i < shnum; i++) {
			uint64_t h = shoff + (uint64_t)i * shentsize;
			uint64_t off = U64(h + 0x18);
			uint64_t sz  = U64(h + 0x20);
			uint32_t num, k;

			if (stroff + U32(h) >= (uint64_t)size ||
			    strcmp((const char *)img + stroff + U32(h), ".resource_table") ||
			    off + sz > (uint64_t)size || sz < 16) {
				continue;
			}
			num = U32(off + 4);
			for (k = 0; k < num && 16 + 4 * (k + 1) <= sz; k++) {
				uint64_t e = off + U32(off + 16 + 4 * k);

				if (e + 16 <= off + sz && U32(e) == 2) { /* RSC_TRACE */
					*da  = U32(e + 4);
					*len = U32(e + 8);
					ret = 0;
					goto out;
				}
			}
		}
	}
#undef U64
#undef U32
#undef U16
out:
	if (ret) {
		fprintf(stderr, "%s has no trace entry in .resource_table\n", file);
	}
	free(img);
	fclose(f);
	return ret;
}

int cmd_trace(int argc, char **argv)
{
	int follow = (argc > 2 && strcmp(argv[2], "-f") == 0);
	char fw[128], path[256], state[32];
	uint32_t da, len, shown = 0;
	long page = sysconf(_SC_PAGESIZE);
	char *mapped;
	volatile char *buf;
	int fd;

	if (!rproc_running(C7X_RPROC_NAME, state, sizeof(state))) {
		fprintf(stderr, "%s is %s\n", C7X_RPROC_NAME, state);
		return 1;
	}
	if (rproc_attr(C7X_RPROC_NAME, "firmware", fw, sizeof(fw))) {
		fprintf(stderr, "cannot read firmware attr for %s\n", C7X_RPROC_NAME);
		return 1;
	}
	snprintf(path, sizeof(path), "/lib/firmware/%s", fw);
	if (elf_trace(path, &da, &len)) {
		return 1;
	}
	fd = open("/dev/mem", O_RDONLY | O_SYNC);
	if (fd < 0) {
		perror("/dev/mem");
		return 1;
	}
	mapped = mmap(NULL, len + (da % page), PROT_READ, MAP_SHARED, fd, (off_t)(da - da % page));
	close(fd);
	if (mapped == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	buf = (volatile char *)(mapped + da % page);
	do {
		uint32_t n = 0;

		while (n < len && buf[n]) {
			n++;
		}
		if (n < shown) {
			shown = 0;
		}
		for (; shown < n; shown++) {
			putchar(buf[shown]);
		}
		fflush(stdout);
		if (follow) {
			usleep(200000);
		}
	} while (follow);

	munmap(mapped, len + (da % page));
	return 0;
}
