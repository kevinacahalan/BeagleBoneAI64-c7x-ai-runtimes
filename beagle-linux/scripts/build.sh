#!/bin/bash
# SPDX-License-Identifier: MIT
# Build C7x AI firmware, x86_64 C7000 host-emulation tests, Linux userspace CLI,
# and the BeagleBone AI-64 device-tree overlay inside containers.
#
#   containers/dsp-cgt/Containerfile  TI C7000 CGT 1.2.0.STS + g++ (host emu)
#   containers/linux/Containerfile    Debian 13 aarch64 cross-compiler + dtc
#
# Usage: ./beagle-linux/scripts/build.sh [all|dsp|host-test|linux|overlay|bundle]

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

IMAGE_LINUX=ai64-linux-userspace
IMAGE_DSP=c7x-ai-dsp-cgt

select_engine() {
	local engine=${AI64_CONTAINER_ENGINE:-}
	if [[ -z $engine ]]; then
		if command -v podman >/dev/null 2>&1; then
			engine=podman
		elif command -v docker >/dev/null 2>&1; then
			engine=docker
		else
			echo "Install podman or docker." >&2
			exit 1
		fi
	fi
	AI64_CONTAINER_ENGINE=$engine
}

engine_build() {
	if [[ $AI64_CONTAINER_ENGINE == podman ]]; then
		podman build "$@"
	else
		docker build "$@"
	fi
}

engine_run() {
	if [[ $AI64_CONTAINER_ENGINE == podman ]]; then
		podman run --rm --security-opt label=disable \
			--userns=keep-id --user "$(id -u):$(id -g)" -e HOME=/tmp "$@"
	else
		docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp "$@"
	fi
}

ensure_image() {
	local tag=$1 file=$2
	select_engine
	if ! "$AI64_CONTAINER_ENGINE" image inspect "$tag" >/dev/null 2>&1; then
		engine_build -t "$tag" -f "$file" "$(dirname "$file")"
	fi
}

build_host_test() {
	ensure_image "$IMAGE_DSP" "$ROOT/containers/dsp-cgt/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_DSP" \
		make -C dsp C7X_CGT=/opt/ti/ti-cgt-c7000_1.2.0.STS host-test
}

build_dsp() {
	ensure_image "$IMAGE_DSP" "$ROOT/containers/dsp-cgt/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_DSP" \
		make -C dsp C7X_CGT=/opt/ti/ti-cgt-c7000_1.2.0.STS all
}

build_linux() {
	ensure_image "$IMAGE_LINUX" "$ROOT/containers/linux/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_LINUX" \
		make -C linux-app clean all
}

build_libc7x() {
	ensure_image "$IMAGE_DSP" "$ROOT/containers/dsp-cgt/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_DSP" \
		make -C libc7x C7X_CGT=/opt/ti/ti-cgt-c7000_1.2.0.STS hostemu-test
	ensure_image "$IMAGE_LINUX" "$ROOT/containers/linux/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_LINUX" \
		make -C libc7x hw
}

build_ggml() {
	local llama_dir=${LLAMA_DIR:-$ROOT/models/llama.cpp}
	if [[ ! -f $llama_dir/ggml/include/ggml.h ]]; then
		echo "skipping ggml-c7x ($llama_dir/ggml/include/ggml.h not found; clone llama.cpp or set LLAMA_DIR)" >&2
		return 0
	fi
	ensure_image "$IMAGE_DSP" "$ROOT/containers/dsp-cgt/Containerfile"
	engine_run -v "$ROOT:/work" -v "$llama_dir:/llama:ro" -w /work "$IMAGE_DSP" \
		make -C ggml-c7x LLAMA_DIR=/llama C7X_CGT=/opt/ti/ti-cgt-c7000_1.2.0.STS hostemu
	ensure_image "$IMAGE_LINUX" "$ROOT/containers/linux/Containerfile"
	engine_run -v "$ROOT:/work" -v "$llama_dir:/llama:ro" -w /work "$IMAGE_LINUX" \
		make -C ggml-c7x LLAMA_DIR=/llama hw
}

build_overlay() {
	ensure_image "$IMAGE_LINUX" "$ROOT/containers/linux/Containerfile"
	engine_run -v "$ROOT:/work" -w /work "$IMAGE_LINUX" \
		make -C beagle-linux/overlays clean all
}

bundle() {
	local out=$ROOT/out/c7x-ai-beagle f
	local files=(
		linux-app/c7x-ai
		dsp/build/c7x-ai.elf
		beagle-linux/overlays/k3-j721e-beagleboneai64-c7x-ai.dtbo
		beagle-linux/scripts/install-beagle.sh
	)
	for f in "${files[@]}"; do
		if [[ ! -f $ROOT/$f ]]; then
			echo "missing $f. Run ./beagle-linux/scripts/build.sh first." >&2
			exit 1
		fi
	done
	rm -rf "$out"
	mkdir -p "$out"
	for f in "${files[@]}"; do
		cp -p "$ROOT/$f" "$out/"
	done
	if [[ -f $ROOT/ggml-c7x/build-aarch64/libggml-c7x.so ]]; then
		cp -p "$ROOT/ggml-c7x/build-aarch64/libggml-c7x.so" "$out/"
	fi
	echo "bundle in $out"
	echo "  scp -r out/c7x-ai-beagle USER@BOARD:"
	echo "  ssh -t USER@BOARD 'sudo ./c7x-ai-beagle/install-beagle.sh && sudo reboot'"
}

case ${1:-all} in
	host-test) build_host_test ;;
	dsp) build_dsp ;;
	libc7x) build_libc7x ;;
	ggml) build_libc7x && build_ggml ;;
	linux) build_linux ;;
	overlay) build_overlay ;;
	bundle) bundle ;;
	all)
		build_host_test
		build_dsp
		build_libc7x
		build_ggml
		build_linux
		build_overlay
		;;
	*)
		echo "usage: $0 [all|host-test|dsp|libc7x|ggml|linux|overlay|bundle]" >&2
		exit 2
		;;
esac
