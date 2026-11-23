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
	echo "bundle in $out"
	echo "  scp -r out/c7x-ai-beagle USER@BOARD:"
	echo "  ssh -t USER@BOARD 'sudo ./c7x-ai-beagle/install-beagle.sh && sudo reboot'"
}

case ${1:-all} in
	host-test) build_host_test ;;
	dsp) build_dsp ;;
	linux) build_linux ;;
	overlay) build_overlay ;;
	bundle) bundle ;;
	all)
		build_host_test
		build_dsp
		build_linux
		build_overlay
		;;
	*)
		echo "usage: $0 [all|host-test|dsp|linux|overlay|bundle]" >&2
		exit 2
		;;
esac
