SUMMARY = "BeagleBone AI-64 C7x AI runtime image"
DESCRIPTION = "Minimal systemd image for BeagleBone AI-64 with the C7x AI runtime, c7x-ai host CLI, and DDR tensor arena overlay."

require recipes-core/images/core-image-minimal.bb

IMAGE_INSTALL:append = " \
    ai64-demo \
    ai64-dsp-fw \
    ai64-boot \
    ai64-sysconf \
    bb-usb-gadgets \
    kernel-modules \
    systemd \
    systemd-networkd \
"

IMAGE_FEATURES:append = " ssh-server-dropbear"

BAD_RECOMMENDATIONS += "ti-eth-fw-j721e"

IMAGE_LINGUAS = "c"

WKS_FILE = "ai64-beagle.wks"

IMAGE_BOOT_FILES:append = " \
    Image \
    k3-j721e-beagleboneai64.dtb;ti/k3-j721e-beagleboneai64.dtb \
    extlinux.conf;extlinux/extlinux.conf \
    sysconf.txt;sysconf.txt \
    k3-j721e-beagleboneai64-c7x-ai.dtbo;overlays/k3-j721e-beagleboneai64-c7x-ai.dtbo \
"

do_image_wic[depends] += "ai64-demo:do_deploy ai64-boot:do_deploy"

ai64_fstab() {
    cat >> ${IMAGE_ROOTFS}${sysconfdir}/fstab <<'EOF'
/dev/mmcblk1p1  /boot/firmware  vfat  user,uid=0,gid=0,defaults  0  2
/dev/mmcblk1p3  none            swap  sw  0  0
EOF
    install -d ${IMAGE_ROOTFS}/boot/firmware
}

ROOTFS_POSTPROCESS_COMMAND:append = " ai64_fstab;"

COMPATIBLE_MACHINE = "beaglebone-ai64"
