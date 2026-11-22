/* SPDX-License-Identifier: MIT */
#ifndef LINUX_APP_C7X_AI_H_
#define LINUX_APP_C7X_AI_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "c7x_ai_proto.h"

#define C7X_RPROC_NAME  "64800000.dsp"
#define C7X_DEFAULT_FW  "c7x-ai.elf"

double now_us(void);
void *map_phys(off_t addr, size_t len);
void unmap_phys(void *p, size_t len);

int read_sysfs(const char *path, char *buf, size_t len);
int rproc_attr(const char *rproc, const char *attr, char *val, size_t len);
int rproc_write(const char *rproc, const char *attr, const char *val);
int rproc_running(const char *rproc, char *state, size_t len);

int c7x_send_shutdown(void);
int c7x_stop(void);
int c7x_load(const char *fw_name);

int cmd_status(int argc, char **argv);
int cmd_load(int argc, char **argv);
int cmd_stop(int argc, char **argv);
int cmd_trace(int argc, char **argv);
int cmd_bench(int argc, char **argv);
int cmd_llama(int argc, char **argv);

#endif /* LINUX_APP_C7X_AI_H_ */
