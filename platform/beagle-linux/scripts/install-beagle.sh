#!/bin/bash
# SPDX-License-Identifier: MIT
# Install c7x-ai userspace CLI, C7x firmware, and DDR arena overlay on a
# BeagleBone AI-64 running Debian or Yocto.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SRC=${1:-}
if [[ -z $SRC && -f $HERE/c7x-ai ]]; then
	SRC=$HERE
fi

EXTLINUX=/boot/firmware/extlinux/extlinux.conf
OVERLAY_DIR=/boot/firmware/overlays
OVERLAY=k3-j721e-beagleboneai64-c7x-ai.dtbo

if [[ $(id -u) -ne 0 ]]; then
	exec sudo "$0" "$@"
fi

find_artifact() {
	local name=$1 repo=$2
	if [[ -n $SRC && -f $SRC/$name ]]; then
		echo "$SRC/$name"
	elif [[ -f $ROOT/$repo ]]; then
		echo "$ROOT/$repo"
	else
		echo "missing $name" >&2
		return 1
	fi
}

find_optional() {
	local name=$1 repo=$2
	if [[ -n $SRC && -e $SRC/$name ]]; then
		echo "$SRC/$name"
	elif [[ -e $ROOT/$repo ]]; then
		echo "$ROOT/$repo"
	else
		echo ""
	fi
}

install_file() {
	local mode=$1 src=$2 dst=$3
	mkdir -p "$(dirname "$dst")"
	cp -f "$src" "$dst"
	chmod "$mode" "$dst"
}

CLI=$(find_artifact c7x-ai frontends/c7x-ai/c7x-ai)
FW=$(find_artifact c7x-ai.elf runtime/dsp/build/c7x-ai.elf)
DTBO=$(find_artifact "$OVERLAY" "platform/beagle-linux/overlays/$OVERLAY")
GGML_SO=$(find_optional libggml-c7x.so frontends/ggml-c7x/build-aarch64/libggml-c7x.so)
ORT_SO=$(find_optional libort-c7x-ep.so frontends/ort-c7x-ep/build-aarch64/libort-c7x-ep.so)
ORT_TEST=$(find_optional test_ort_c7x_ep frontends/ort-c7x-ep/build-aarch64/test_ort_c7x_ep)
ORT_MODELS=$(find_optional models frontends/ort-c7x-ep/build-hostemu/models)

install_file 0755 "$CLI" /usr/bin/c7x-ai
install_file 0644 "$FW" /lib/firmware/c7x-ai.elf
if command -v setcap >/dev/null 2>&1 || [[ -x /usr/sbin/setcap ]]; then
	"$(command -v setcap || echo /usr/sbin/setcap)" cap_sys_rawio+ep /usr/bin/c7x-ai
fi

if [[ -n $GGML_SO || -n $ORT_SO || -n $ORT_TEST ]]; then
	mkdir -p /usr/lib/c7x-ai
fi
if [[ -n $GGML_SO ]]; then
	install_file 0755 "$GGML_SO" /usr/lib/c7x-ai/libggml-c7x.so
	echo "Installed llama.cpp backend plugin: /usr/lib/c7x-ai/libggml-c7x.so"
fi
if [[ -n $ORT_SO ]]; then
	install_file 0755 "$ORT_SO" /usr/lib/c7x-ai/libort-c7x-ep.so
	echo "Installed ONNX C7x EP library:      /usr/lib/c7x-ai/libort-c7x-ep.so"
fi
if [[ -n $ORT_TEST ]]; then
	install_file 0755 "$ORT_TEST" /usr/lib/c7x-ai/test_ort_c7x_ep
	echo "Installed ONNX C7x EP self-test:    /usr/lib/c7x-ai/test_ort_c7x_ep"
fi
if [[ -n $ORT_MODELS && -d $ORT_MODELS ]]; then
	mkdir -p /usr/share/c7x-ai/models
	cp -fp "$ORT_MODELS"/* /usr/share/c7x-ai/models/
	echo "Installed ONNX test models:         /usr/share/c7x-ai/models/"
fi

if [[ -d $OVERLAY_DIR && -f $EXTLINUX ]]; then
	install_file 0644 "$DTBO" "$OVERLAY_DIR/$OVERLAY"
	echo "Installed $OVERLAY into $OVERLAY_DIR."
	echo "Add '/overlays/$OVERLAY' to fdtoverlays in $EXTLINUX and reboot if the DDR arenas (0xac000000 / 0x880000000) are not already reserved."
fi

echo ""
echo "Done! Next steps:"
echo "  Shared setup: c7x-ai load c7x-ai.elf && c7x-ai bench all"
echo "  Path 1 (.c7m): c7x-ai llama chat <model.c7m>"
if [[ -n $GGML_SO ]]; then
	echo "  Path 2 (GGUF): GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-completion -m <model.gguf> -ngl 99"
fi
if [[ -n $ORT_TEST && -d /usr/share/c7x-ai/models ]]; then
	echo "  Path 3 (ONNX): /usr/lib/c7x-ai/test_ort_c7x_ep /usr/share/c7x-ai/models 20"
fi
