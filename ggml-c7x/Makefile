# Build out-of-tree dynamic GGML backend plugin (libggml-c7x.so) for:
#   - aarch64 BeagleBone AI-64 hardware (`make hw`)      -> build-aarch64/libggml-c7x.so
#   - x86_64 C7000 Host Emulation       (`make hostemu`) -> build-hostemu/libggml-c7x.so
#
# Load into any GGML_BACKEND_DL=ON build of llama.cpp via:
#   GGML_BACKEND_PATH=/path/to/libggml-c7x.so llama-cli ...

LLAMA_DIR  ?= ../models/llama.cpp
CXX_AARCH64 ?= aarch64-linux-gnu-g++
CXX_HOST    ?= g++

GGML_INC   := -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src
C7X_INC    := -I../libc7x/include -I../dsp/include
COMMON_CXX := -O2 -fPIC -std=c++17 -Wall -Wextra -Wno-unused-function \
              -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_BACKEND_BUILD \
              $(GGML_INC) $(C7X_INC)

BUILD_HW   := build-aarch64
BUILD_EMU  := build-hostemu

.PHONY: all hw hostemu clean

all: hw

hw: $(BUILD_HW)/libggml-c7x.so

$(BUILD_HW)/libggml-c7x.so: ggml-c7x.cpp ggml-c7x.h ../libc7x/build-aarch64/libc7x.a
	mkdir -p $(BUILD_HW)
	$(CXX_AARCH64) $(COMMON_CXX) -shared -o $@ ggml-c7x.cpp \
		../libc7x/build-aarch64/libc7x.a -lpthread -lm

hostemu: $(BUILD_EMU)/libggml-c7x.so

$(BUILD_EMU)/libggml-c7x.so: ggml-c7x.cpp ggml-c7x.h ../libc7x/build-hostemu/libc7x_hostemu.a
	@if [ -z "$(C7X_CGT)" ]; then echo "C7X_CGT is not set" >&2; exit 1; fi
	mkdir -p $(BUILD_EMU)
	$(CXX_HOST) $(COMMON_CXX) -shared -o $@ ggml-c7x.cpp \
		../libc7x/build-hostemu/libc7x_hostemu.a \
		$(C7X_CGT)/host_emulation/libc70-host-emulation.a \
		-lpthread -lm

clean:
	rm -rf $(BUILD_HW) $(BUILD_EMU)
