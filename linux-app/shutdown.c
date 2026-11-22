/* SPDX-License-Identifier: MIT */
/*
 * Clean C7x lifecycle management: mailbox shutdown + remoteproc stop/load/start.
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "c7x_ai.h"

#define RP_MBOX_SHUTDOWN 0xFFFFFF14u

int c7x_send_shutdown(void)
{
	volatile uint32_t *mbox;
	int ms;

	mbox = (volatile uint32_t *)map_phys((off_t)C7X_MBOX_BASE, 4096);
	if (!mbox) {
		return -1;
	}
	mbox[(0x40 + 4 * C7X_MBOX_LINUX_TX) / 4] = RP_MBOX_SHUTDOWN;
	for (ms = 0; ms < 500; ms++) {
		if (!(mbox[(0xc0 + 4 * C7X_MBOX_LINUX_TX) / 4] & 0x3fu)) {
			usleep(10000);
			unmap_phys((void *)mbox, 4096);
			return 0;
		}
		usleep(1000);
	}
	unmap_phys((void *)mbox, 4096);
	return -1;
}

int c7x_stop(void)
{
	char state[32];

	if (!rproc_running(C7X_RPROC_NAME, state, sizeof(state))) {
		return 0;
	}
	/*
	 * Try remoteproc stop directly first (TI 6.18 sends RP_MBOX_SHUTDOWN itself);
	 * if that fails (e.g. BeagleBoard 6.18 kernel), send the mailbox shutdown
	 * directly and retry.
	 */
	if (rproc_write(C7X_RPROC_NAME, "state", "stop") == 0) {
		return 0;
	}
	(void)c7x_send_shutdown();
	if (rproc_write(C7X_RPROC_NAME, "state", "stop") != 0) {
		fprintf(stderr, "failed to stop %s (is running firmware handling RP_MBOX_SHUTDOWN?)\n",
			C7X_RPROC_NAME);
		return -1;
	}
	return 0;
}

int c7x_load(const char *fw_name)
{
	volatile struct c7x_ctrl *ctrl;
	int ms;

	if (c7x_stop() != 0) {
		return -1;
	}
	if (rproc_write(C7X_RPROC_NAME, "firmware", fw_name) != 0) {
		fprintf(stderr, "failed to set firmware '%s' on %s\n", fw_name, C7X_RPROC_NAME);
		return -1;
	}
	if (rproc_write(C7X_RPROC_NAME, "state", "start") != 0) {
		fprintf(stderr, "failed to start %s with '%s'\n", C7X_RPROC_NAME, fw_name);
		return -1;
	}
	ctrl = (volatile struct c7x_ctrl *)map_phys((off_t)C7X_CTRL_ADDR, C7X_CTRL_SIZE);
	if (!ctrl) {
		return -1;
	}
	for (ms = 0; ms < 2000; ms++) {
		if (ctrl->magic == C7X_CTRL_MAGIC) {
			unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
			return 0;
		}
		usleep(1000);
	}
	fprintf(stderr, "C7x started, but control magic 0x%08x not seen (got 0x%08x)\n",
		C7X_CTRL_MAGIC, ctrl->magic);
	unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
	return -1;
}
