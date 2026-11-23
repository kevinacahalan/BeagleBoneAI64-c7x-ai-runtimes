#!/bin/bash
# SPDX-License-Identifier: MIT
# Install c7x-ai userspace CLI, C7x firmware, and DDR arena overlay on a
# BeagleBone AI-64 running Debian or Yocto.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
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

CLI=$(find_artifact c7x-ai linux-app/c7x-ai)
FW=$(find_artifact c7x-ai.elf dsp/build/c7x-ai.elf)
DTBO=$(find_artifact "$OVERLAY" "beagle-linux/overlays/$OVERLAY")

install -m 0755 "$CLI" /usr/bin/c7x-ai
install -m 0644 "$FW" /lib/firmware/c7x-ai.elf
if command -v setcap >/dev/null 2>&1 || [[ -x /usr/sbin/setcap ]]; then
	"$(command -v setcap || echo /usr/sbin/setcap)" cap_sys_rawio+ep /usr/bin/c7x-ai
fi

if [[ -d $OVERLAY_DIR && -f $EXTLINUX ]]; then
	install -m 0644 "$DTBO" "$OVERLAY_DIR/$OVERLAY"
	echo "Installed $OVERLAY into $OVERLAY_DIR."
	echo "Add '/overlays/$OVERLAY' to fdtoverlays in $EXTLINUX and reboot if the DDR arenas (0xac000000 / 0x880000000) are not already reserved."
fi

echo "Done. Run: c7x-ai load && c7x-ai bench all"
