/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "c7x_ai.h"

double now_us(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

void *map_phys(off_t addr, size_t len)
{
	int fd = open("/dev/mem", O_RDWR | O_SYNC);
	void *p;

	if (fd < 0) {
		perror("/dev/mem");
		return NULL;
	}
	p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr);
	close(fd);
	if (p == MAP_FAILED) {
		perror("mmap");
		return NULL;
	}
	return p;
}

void unmap_phys(void *p, size_t len)
{
	if (p && p != MAP_FAILED) {
		munmap(p, len);
	}
}

int read_sysfs(const char *path, char *buf, size_t len)
{
	FILE *f = fopen(path, "r");

	if (!f) {
		return -1;
	}
	if (!fgets(buf, (int)len, f)) {
		fclose(f);
		return -1;
	}
	fclose(f);
	buf[strcspn(buf, "\n")] = '\0';
	return 0;
}

int rproc_attr(const char *rproc, const char *attr, char *val, size_t len)
{
	char path[128], got[64];
	int i;

	for (i = 0; i < 64; i++) {
		snprintf(path, sizeof(path), "/sys/class/remoteproc/remoteproc%d/name", i);
		if (read_sysfs(path, got, sizeof(got)) || strcmp(got, rproc)) {
			continue;
		}
		snprintf(path, sizeof(path), "/sys/class/remoteproc/remoteproc%d/%s", i, attr);
		return read_sysfs(path, val, len);
	}
	return -1;
}

int rproc_write(const char *rproc, const char *attr, const char *val)
{
	char path[128], got[64];
	FILE *f;
	int i, ret;

	for (i = 0; i < 64; i++) {
		snprintf(path, sizeof(path), "/sys/class/remoteproc/remoteproc%d/name", i);
		if (read_sysfs(path, got, sizeof(got)) || strcmp(got, rproc)) {
			continue;
		}
		snprintf(path, sizeof(path), "/sys/class/remoteproc/remoteproc%d/%s", i, attr);
		f = fopen(path, "w");
		if (!f) {
			return -1;
		}
		ret = fputs(val, f) < 0;
		ret |= fclose(f) != 0;
		return ret ? -1 : 0;
	}
	return -1;
}

int rproc_running(const char *rproc, char *state, size_t len)
{
	char buf[32];

	if (!state) {
		state = buf;
		len = sizeof(buf);
	}
	if (rproc_attr(rproc, "state", state, len)) {
		snprintf(state, len, "missing");
		return 0;
	}
	return strcmp(state, "running") == 0;
}
