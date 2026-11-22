SUMMARY = "BeagleBone AI-64 C7x AI firmware"
DESCRIPTION = "Compiles the bare-metal C7x AI runtime (c7x-ai.elf) with TI C7000 CGT."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

DEPENDS = "ti-cgt-dsp-native"

FILESEXTRAPATHS:prepend := "${@os.path.abspath(os.path.join(os.path.dirname(d.getVar('FILE')), '..', '..', '..', '..', '..'))}/:"

SRC_URI = " \
    file://dsp;subdir=src \
"

S = "${UNPACKDIR}/src"

COMPATIBLE_MACHINE = "beaglebone-ai64"
PACKAGE_ARCH = "${MACHINE_ARCH}"
INSANE_SKIP:${PN} += "arch buildpaths"

do_configure[noexec] = "1"

do_compile () {
    rm -rf ${S}/dsp/build
    oe_runmake -C ${S}/dsp \
        C7X_CGT="${STAGING_DATADIR_NATIVE}/ti-cgt-c7x" \
        all
    test -f ${S}/dsp/build/c7x-ai.elf
}

do_install () {
    install -d ${D}${nonarch_base_libdir}/firmware
    install -m 0644 ${S}/dsp/build/c7x-ai.elf ${D}${nonarch_base_libdir}/firmware/c7x-ai.elf
}

FILES:${PN} = "${nonarch_base_libdir}/firmware/c7x-ai.elf"
