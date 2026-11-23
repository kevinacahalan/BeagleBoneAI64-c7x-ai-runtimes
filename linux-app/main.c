/* SPDX-License-Identifier: MIT */
/*
 * c7x-ai: Linux A72 host CLI for the BeagleBone AI-64 C7x + MMA AI runtime.
 */
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "c7x_ai.h"

int cmd_status(int argc, char **argv)
{
	char state[32], fw[128];

	(void)argc;
	(void)argv;
	if (rproc_attr(C7X_RPROC_NAME, "state", state, sizeof(state))) {
		fprintf(stderr, "remoteproc %s not found\n", C7X_RPROC_NAME);
		return 1;
	}
	if (rproc_attr(C7X_RPROC_NAME, "firmware", fw, sizeof(fw))) {
		snprintf(fw, sizeof(fw), "(unknown)");
	}
	printf("remoteproc: %s\nstate:      %s\nfirmware:   %s\n", C7X_RPROC_NAME, state, fw);

	if (strcmp(state, "running") == 0) {
		volatile struct c7x_ctrl *ctrl =
			(volatile struct c7x_ctrl *)map_phys((off_t)C7X_CTRL_ADDR, C7X_CTRL_SIZE);

		if (ctrl) {
			if (ctrl->magic == C7X_CTRL_MAGIC) {
				printf("control:    magic=C7AI v%u, DNUM=%u, MMU=%u, L1DCFG=%u, L2CFG=%u\n",
				       ctrl->version, ctrl->dnum, ctrl->scr_lo & 1u,
				       ctrl->l1dcfg, ctrl->l2cfg);
				printf("DRU0:       ok=%u pid=0x%016" PRIx64 " caps=0x%016" PRIx64 "\n",
				       ctrl->dru_ok, ctrl->dru_pid, ctrl->dru_caps);
				printf("seq:        req=%u done=%u\n", ctrl->req_seq, ctrl->done_seq);
			} else {
				printf("control:    not c7x-ai (magic=0x%08x)\n", ctrl->magic);
			}
			unmap_phys((void *)ctrl, C7X_CTRL_SIZE);
		}
	}
	return 0;
}

int cmd_load(int argc, char **argv)
{
	const char *fw = argc > 2 ? argv[2] : C7X_DEFAULT_FW;

	if (c7x_load(fw) != 0) {
		return 1;
	}
	printf("loaded and started %s on %s\n", fw, C7X_RPROC_NAME);
	return cmd_status(argc, argv);
}

int cmd_stop(int argc, char **argv)
{
	(void)argc;
	(void)argv;
	if (c7x_stop() != 0) {
		return 1;
	}
	printf("stopped %s\n", C7X_RPROC_NAME);
	return 0;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"usage: %s <command> [args]\n"
		"  status                   Show C7x remoteproc and control block status\n"
		"  load [firmware]          Stop C7x cleanly, load firmware (default %s), start\n"
		"  stop                     Cleanly shut down and stop C7x\n"
		"  trace [-f]               Dump or follow C7x remoteproc trace buffer\n"
		"  bench [all|cbank|mem|dru|gemm|gemv]\n"
		"                           Run C7x MMA/DRU/MSMC/DDR microbenchmarks & verify\n"
		"  llama <info|verify|run|chat|spec> ...\n"
		"                           Inspect, verify, run, chat, or speculatively decode on C7x+MMA\n"
		"  serve <model.c7m> [--host 0.0.0.0] [--port 8080]\n"
		"                           OpenAI-compatible HTTP/1.1 + SSE inference server\n",
		prog, C7X_DEFAULT_FW);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 2;
	}
	if (strcmp(argv[1], "status") == 0) {
		return cmd_status(argc, argv);
	}
	if (strcmp(argv[1], "load") == 0) {
		return cmd_load(argc, argv);
	}
	if (strcmp(argv[1], "stop") == 0) {
		return cmd_stop(argc, argv);
	}
	if (strcmp(argv[1], "trace") == 0) {
		return cmd_trace(argc, argv);
	}
	if (strcmp(argv[1], "bench") == 0) {
		return cmd_bench(argc, argv);
	}
	if (strcmp(argv[1], "llama") == 0) {
		return cmd_llama(argc, argv);
	}
	if (strcmp(argv[1], "serve") == 0) {
		return cmd_serve(argc, argv);
	}
	usage(argv[0]);
	return 2;
}
