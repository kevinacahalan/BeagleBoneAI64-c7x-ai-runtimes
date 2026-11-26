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
BASE_DTB=/boot/firmware/ti/k3-j721e-beagleboneai64.dtb
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
	if command -v fdtoverlay >/dev/null 2>&1 && [[ -f $BASE_DTB ]]; then
		trial=$(mktemp)
		if ! fdtoverlay -i "$BASE_DTB" -o "$trial" "$DTBO"; then
			rm -f "$trial"
			echo "ERROR: $OVERLAY does not apply cleanly to $BASE_DTB." >&2
			exit 1
		fi
		rm -f "$trial"
	fi

	install_file 0644 "$DTBO" "$OVERLAY_DIR/$OVERLAY"
	echo "Installed overlay:                  $OVERLAY_DIR/$OVERLAY"

	WANT="/overlays/$OVERLAY"
	if [[ ! -f $EXTLINUX.c7x-ai-bak ]]; then
		cp -f "$EXTLINUX" "$EXTLINUX.c7x-ai-bak"
	fi
	tmp_ext=$(mktemp)
	if ! awk -v want="$WANT" '
		BEGIN { def = ""; in_def = 0; done = 0; seen_lbl = 0 }
		{
			sub(/\r$/, "", $0)
			lines[NR] = $0
		}
		/^default[[:space:]]+/ {
			d = $0
			sub(/^default[[:space:]]+/, "", d)
			sub(/[[:space:]]+$/, "", d)
			def = d
		}
		END {
			for (i = 1; i <= NR; i++) {
				line = lines[i]
				if (line ~ /^label[[:space:]]+/) {
					if (in_def && !done) {
						print "    fdtoverlays " want
						done = 1
					}
					lbl = line
					sub(/^label[[:space:]]+/, "", lbl)
					sub(/[[:space:]]+$/, "", lbl)
					seen_lbl++
					in_def = (def != "" ? (lbl == def) : (seen_lbl == 1))
				}
				if (in_def && !done && line ~ /^[[:space:]]*fdtoverlays[[:space:]]+/) {
					sub(/[[:space:]]+$/, "", line)
					if (index(" " line " ", " " want " ") == 0) {
						line = line " " want
					}
					done = 1
				}
				print line
			}
			if (in_def && !done) {
				print "    fdtoverlays " want
				done = 1
			}
			if (!done) {
				exit 42
			}
		}
	' "$EXTLINUX" > "$tmp_ext"; then
		rm -f "$tmp_ext"
		echo "ERROR: could not find a matching default or first label in $EXTLINUX; add '$WANT' to fdtoverlays manually." >&2
		exit 1
	fi
	cp -f "$tmp_ext" "$EXTLINUX"
	rm -f "$tmp_ext"
	sync
	echo "Enabled $WANT in $EXTLINUX (backup: $EXTLINUX.c7x-ai-bak)."
	echo "Reboot if the DDR arenas (0xac000000 / 0x880000000) are not already reserved."
fi

echo ""
echo "Done! Next steps (after rebooting if the overlay was newly added):"
echo "  Shared setup:  sudo c7x-ai load c7x-ai.elf && sudo c7x-ai bench all"
echo "  Path 1 (.c7m): sudo c7x-ai llama chat <model.c7m>"
if [[ -n $GGML_SO ]]; then
	echo "  Path 2 (GGUF): sudo GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so llama-completion -m <model.gguf> -ngl 99"
fi
if [[ -n $ORT_TEST && -d /usr/share/c7x-ai/models ]]; then
	echo "  Path 3 (ONNX): sudo /usr/lib/c7x-ai/test_ort_c7x_ep /usr/share/c7x-ai/models 20"
fi
