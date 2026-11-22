# Build the C7x AI runtime firmware with TI C7000 CGT, plus x86_64 host emulation tests.

BUILD := build
INC := include

.PHONY: all clean host-test host-llama-test

all: $(BUILD)/c7x-ai.elf

C7X_SRC := c7x/main.c c7x/mmu.c c7x/dru.c c7x/mma_kernels.c c7x/vec_kernels.c \
           c7x/transformer.c c7x/bench.c c7x/regs.asm
C7X_OBJ := $(patsubst c7x/%.asm,$(BUILD)/c7x/%.obj,$(patsubst c7x/%.c,$(BUILD)/c7x/%.obj,$(C7X_SRC)))

C7X_HDR := $(wildcard c7x/*.h) $(wildcard $(INC)/*.h)
C7X_FLAGS := -mv7100 -O3 --display_error_number --diag_suppress=70 \
             --include_path=$(C7X_CGT)/include --include_path=$(INC)

# Warning 70: resource table stores 64-bit trace buffer address in uint32_t (in low 4 GiB DDR).
$(BUILD)/c7x/%.obj: c7x/%.c $(C7X_HDR)
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD)/c7x
	$(C7X_CGT)/bin/cl7x $(C7X_FLAGS) --obj_directory=$(BUILD)/c7x -c $<

$(BUILD)/c7x/%.obj: c7x/%.asm
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD)/c7x
	$(C7X_CGT)/bin/cl7x $(C7X_FLAGS) --obj_directory=$(BUILD)/c7x -c $<

$(BUILD)/c7x-ai.elf: $(C7X_OBJ) c7x/c7x.cmd
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	$(C7X_CGT)/bin/cl7x -mv7100 -z -i$(C7X_CGT)/lib \
		--reread_libs --warn_sections -m $(BUILD)/c7x-ai.map \
		-o $@ $(C7X_OBJ) c7x/c7x.cmd --library=libc.a

# Host emulation unit tests (MMA + FP32 vector kernels).
host-test: $(BUILD)/host-test
	$(BUILD)/host-test

$(BUILD)/host-test: host_test/test_mma.cpp c7x/mma_kernels.c c7x/mma_kernels.h \
                    c7x/vec_kernels.c c7x/vec_kernels.h
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD)
	g++ -O2 -std=c++14 -D__C7X_HOSTEM__ -D_HOST_BUILD \
		-I$(C7X_CGT)/host_emulation/include -Ic7x -I$(INC) \
		-x c++ host_test/test_mma.cpp c7x/mma_kernels.c c7x/vec_kernels.c \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a \
		-lm -o $@

# Host emulation full transformer verification against .c7m + golden traces.
host-llama-test: $(BUILD)/host-llama-test

$(BUILD)/host-llama-test: host_test/test_llama.cpp c7x/mma_kernels.c c7x/vec_kernels.c \
                          c7x/transformer.c $(wildcard c7x/*.h) $(wildcard $(INC)/*.h)
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD)
	g++ -O2 -std=c++14 -D__C7X_HOSTEM__ -D_HOST_BUILD \
		-I$(C7X_CGT)/host_emulation/include -Ic7x -I$(INC) \
		-x c++ host_test/test_llama.cpp c7x/mma_kernels.c c7x/vec_kernels.c c7x/transformer.c \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a \
		-lm -o $@

clean:
	rm -rf $(BUILD)
