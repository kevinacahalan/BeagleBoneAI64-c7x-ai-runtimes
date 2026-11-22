SUMMARY = "BeagleBone AI-64 C7x AI host CLI and overlay"
DESCRIPTION = "Cross-compiles c7x-ai and the DDR tensor arena device-tree overlay."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

DEPENDS = "dtc-native"

inherit deploy

FILESEXTRAPATHS:prepend := "${@os.path.abspath(os.path.join(os.path.dirname(d.getVar('FILE')), '..', '..', '..', '..', '..'))}/:"

SRC_URI = " \
    file://linux-app;subdir=src \
    file://dsp/include/c7x_ai_proto.h;subdir=src \
    file://dsp/include/c7x_llama_fmt.h;subdir=src \
    file://beagle-linux/overlays/Makefile;subdir=src \
    file://beagle-linux/overlays/k3-j721e-beagleboneai64-c7x-ai.dtso;subdir=src \
"

S = "${UNPACKDIR}/src"

COMPATIBLE_MACHINE = "beaglebone-ai64"

OVERLAYS = "k3-j721e-beagleboneai64-c7x-ai.dtbo"

do_configure[noexec] = "1"

do_compile () {
    rm -f ${S}/linux-app/c7x-ai ${S}/linux-app/*.o ${S}/beagle-linux/overlays/*.dtbo
    oe_runmake -C ${S}/linux-app CC="${CC}" CFLAGS="${CFLAGS} -I. -I../dsp/include" LDFLAGS="${LDFLAGS}"
    oe_runmake -C ${S}/beagle-linux/overlays CPP="${BUILD_CC} -E"
}

do_install () {
    install -d ${D}${bindir}
    install -d ${D}${nonarch_base_libdir}/firmware
    install -d ${D}/boot/firmware/overlays
    install -m 0755 ${S}/linux-app/c7x-ai ${D}${bindir}/c7x-ai
    for o in ${OVERLAYS}; do
        install -m 0644 ${S}/beagle-linux/overlays/$o ${D}${nonarch_base_libdir}/firmware/
        install -m 0644 ${S}/beagle-linux/overlays/$o ${D}/boot/firmware/overlays/
    done
}

do_deploy() {
    install -d ${DEPLOYDIR}
    for o in ${OVERLAYS}; do
        install -m 0644 ${S}/beagle-linux/overlays/$o ${DEPLOYDIR}/
    done
}
addtask deploy after do_compile before do_build

FILES:${PN} += " \
    ${nonarch_base_libdir}/firmware/*.dtbo \
    /boot/firmware/overlays/*.dtbo \
"
