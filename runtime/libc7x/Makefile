# Build libc7x for either:
#   - aarch64 Linux hardware (default: `make hw`) -> build-aarch64/libc7x.a
#   - x86_64 C7000 Host Emulation (`make hostemu C7X_CGT=...`) -> build-hostemu/libc7x_hostemu.a + test

CC_AARCH64 ?= aarch64-linux-gnu-gcc
AR_AARCH64 ?= aarch64-linux-gnu-ar
CC_HOST    ?= gcc
CXX_HOST   ?= g++
AR_HOST    ?= ar

INC_FLAGS  := -Iinclude -I../dsp/include
WARN_FLAGS := -Wall -Wextra

BUILD_HW   := build-aarch64
BUILD_EMU  := build-hostemu

.PHONY: all hw hostemu hostemu-test clean

all: hw

# ----------------------------------------------------------------------------
# aarch64 hardware library (uses /dev/mem + c7x_ctrl)
# ----------------------------------------------------------------------------
HW_CFLAGS := -O2 -fPIC -std=c11 $(WARN_FLAGS) $(INC_FLAGS) -DLIBC7X_ENABLE_HW
HW_OBJS   := $(BUILD_HW)/c7x_rt.o $(BUILD_HW)/c7x_pack.o $(BUILD_HW)/c7x_transport_hw.o

hw: $(BUILD_HW)/libc7x.a $(BUILD_HW)/test_libc7x

$(BUILD_HW)/test_libc7x: tests/test_libc7x.c $(BUILD_HW)/libc7x.a
	mkdir -p $(BUILD_HW)
	$(CC_AARCH64) $(HW_CFLAGS) tests/test_libc7x.c $(BUILD_HW)/libc7x.a -lpthread -lm -o $@

$(BUILD_HW)/%.o: src/%.c include/c7x_rt.h src/c7x_internal.h ../dsp/include/c7x_ai_proto.h ../dsp/include/c7x_cmd_proto.h
	mkdir -p $(BUILD_HW)
	$(CC_AARCH64) $(HW_CFLAGS) -c -o $@ $<

$(BUILD_HW)/libc7x.a: $(HW_OBJS)
	$(AR_AARCH64) rcs $@ $(HW_OBJS)

# ----------------------------------------------------------------------------
# x86_64 C7000 Host Emulation library + unit test
# ----------------------------------------------------------------------------
EMU_CFLAGS   := -O2 -fPIC -std=c11 $(WARN_FLAGS) $(INC_FLAGS) -DLIBC7X_ENABLE_HOSTEMU
EMU_CXXFLAGS := -O2 -fPIC -std=c++14 $(WARN_FLAGS) $(INC_FLAGS) -I../dsp/c7x \
                -DLIBC7X_ENABLE_HOSTEMU -D__C7X_HOSTEM__ -D_HOST_BUILD \
                -I$(C7X_CGT)/host_emulation/include

EMU_C_OBJS   := $(BUILD_EMU)/c7x_rt.o $(BUILD_EMU)/c7x_pack.o
EMU_CXX_OBJS := $(BUILD_EMU)/c7x_transport_hostemu.o $(BUILD_EMU)/cmd_exec.o \
                $(BUILD_EMU)/mma_kernels.o $(BUILD_EMU)/vec_kernels.o

hostemu: $(BUILD_EMU)/libc7x_hostemu.a

$(BUILD_EMU)/c7x_rt.o: src/c7x_rt.c include/c7x_rt.h src/c7x_internal.h ../dsp/include/c7x_cmd_proto.h
	mkdir -p $(BUILD_EMU)
	$(CC_HOST) $(EMU_CFLAGS) -c -o $@ $<

$(BUILD_EMU)/c7x_pack.o: src/c7x_pack.c include/c7x_rt.h
	mkdir -p $(BUILD_EMU)
	$(CC_HOST) $(EMU_CFLAGS) -c -o $@ $<

$(BUILD_EMU)/c7x_transport_hostemu.o: src/c7x_transport_hostemu.cpp include/c7x_rt.h src/c7x_internal.h ../dsp/c7x/cmd_exec.h
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(EMU_CXXFLAGS) -c -o $@ $<

$(BUILD_EMU)/cmd_exec.o: ../dsp/c7x/cmd_exec.c ../dsp/c7x/cmd_exec.h ../dsp/include/c7x_cmd_proto.h
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(EMU_CXXFLAGS) -x c++ -c -o $@ $<

$(BUILD_EMU)/mma_kernels.o: ../dsp/c7x/mma_kernels.c ../dsp/c7x/mma_kernels.h
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(EMU_CXXFLAGS) -x c++ -c -o $@ $<

$(BUILD_EMU)/vec_kernels.o: ../dsp/c7x/vec_kernels.c ../dsp/c7x/vec_kernels.h
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(EMU_CXXFLAGS) -x c++ -c -o $@ $<

$(BUILD_EMU)/libc7x_hostemu.a: $(EMU_C_OBJS) $(EMU_CXX_OBJS)
	$(AR_HOST) rcs $@ $(EMU_C_OBJS) $(EMU_CXX_OBJS)

hostemu-test: $(BUILD_EMU)/test_libc7x
	$(BUILD_EMU)/test_libc7x

$(BUILD_EMU)/test_libc7x: tests/test_libc7x.c $(BUILD_EMU)/libc7x_hostemu.a
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	$(CXX_HOST) -O2 $(INC_FLAGS) tests/test_libc7x.c $(BUILD_EMU)/libc7x_hostemu.a \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a -lpthread -lm -o $@

clean:
	rm -rf $(BUILD_HW) $(BUILD_EMU)
