# Build ONNX Runtime C7x Execution Provider (ort-c7x-ep) for:
#   - aarch64 BeagleBone AI-64 hardware (`make hw`)
#   - x86_64 C7000 Host Emulation       (`make hostemu C7X_CGT=...`)

CXX_AARCH64 ?= aarch64-linux-gnu-g++
AR_AARCH64  ?= aarch64-linux-gnu-ar
CXX_HOST    ?= g++
AR_HOST     ?= ar

INC_FLAGS   := -Iinclude -Isrc -I../libc7x/include -I../dsp/include
COMMON_CXX  := -O2 -fPIC -std=c++17 -Wall -Wextra $(INC_FLAGS)

BUILD_HW    := build-aarch64
BUILD_EMU   := build-hostemu

.PHONY: all hw hostemu hostemu-test clean

all: hw

hw: $(BUILD_HW)/libort-c7x-ep.so $(BUILD_HW)/test_ort_c7x_ep

$(BUILD_HW)/c7x_ort_ep.o: src/c7x_ort_ep.cpp src/onnx_proto.hpp include/c7x_ort_ep.h ../libc7x/include/c7x_rt.h ../dsp/include/c7x_cmd_proto.h
	mkdir -p $(BUILD_HW)
	$(CXX_AARCH64) $(COMMON_CXX) -c -o $@ $<

$(BUILD_HW)/libort-c7x-ep.so: $(BUILD_HW)/c7x_ort_ep.o ../libc7x/build-aarch64/libc7x.a
	$(CXX_AARCH64) $(COMMON_CXX) -shared -o $@ $(BUILD_HW)/c7x_ort_ep.o ../libc7x/build-aarch64/libc7x.a -lpthread -lm

$(BUILD_HW)/test_ort_c7x_ep: tests/test_ort_c7x_ep.cpp $(BUILD_HW)/c7x_ort_ep.o ../libc7x/build-aarch64/libc7x.a
	$(CXX_AARCH64) $(COMMON_CXX) tests/test_ort_c7x_ep.cpp $(BUILD_HW)/c7x_ort_ep.o ../libc7x/build-aarch64/libc7x.a -lpthread -lm -o $@

hostemu: $(BUILD_EMU)/libort-c7x-ep.so $(BUILD_EMU)/test_ort_c7x_ep

$(BUILD_EMU)/c7x_ort_ep.o: src/c7x_ort_ep.cpp src/onnx_proto.hpp include/c7x_ort_ep.h ../libc7x/include/c7x_rt.h ../dsp/include/c7x_cmd_proto.h
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(COMMON_CXX) -c -o $@ $<

$(BUILD_EMU)/libort-c7x-ep.so: $(BUILD_EMU)/c7x_ort_ep.o ../libc7x/build-hostemu/libc7x_hostemu.a
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	$(CXX_HOST) $(COMMON_CXX) -shared -o $@ $(BUILD_EMU)/c7x_ort_ep.o \
		../libc7x/build-hostemu/libc7x_hostemu.a \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a -lpthread -lm

$(BUILD_EMU)/test_ort_c7x_ep: tests/test_ort_c7x_ep.cpp $(BUILD_EMU)/c7x_ort_ep.o ../libc7x/build-hostemu/libc7x_hostemu.a
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	$(CXX_HOST) $(COMMON_CXX) tests/test_ort_c7x_ep.cpp $(BUILD_EMU)/c7x_ort_ep.o \
		../libc7x/build-hostemu/libc7x_hostemu.a \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a -lpthread -lm -o $@

hostemu-test: $(BUILD_EMU)/test_ort_c7x_ep
	@if [ ! -f $(BUILD_EMU)/models/mobilenet_qlinear.onnx ]; then \
		python3 tests/gen_onnx_models.py $(BUILD_EMU)/models; \
	fi
	$(BUILD_EMU)/test_ort_c7x_ep $(BUILD_EMU)/models 1

clean:
	rm -rf $(BUILD_HW) $(BUILD_EMU)
