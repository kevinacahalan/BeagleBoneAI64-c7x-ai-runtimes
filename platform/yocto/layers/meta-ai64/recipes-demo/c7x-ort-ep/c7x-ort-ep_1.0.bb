SUMMARY = "BeagleBone AI-64 libc7x and ONNX Runtime C7x Execution Provider (ort-c7x-ep)"
DESCRIPTION = "Cross-compiles runtime/libc7x and frontends/ort-c7x-ep (libort-c7x-ep.so, test_ort_c7x_ep, and ONNX verification models) for the BeagleBone AI-64 C7x + MMA."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

FILESEXTRAPATHS:prepend := "${@os.path.abspath(os.path.join(os.path.dirname(d.getVar('FILE')), '..', '..', '..', '..', '..', '..'))}/:"

SRC_URI = " \
    file://runtime/dsp/include/c7x_ai_proto.h;subdir=src \
    file://runtime/dsp/include/c7x_cmd_proto.h;subdir=src \
    file://runtime/libc7x/Makefile;subdir=src \
    file://runtime/libc7x/include;subdir=src \
    file://runtime/libc7x/src;subdir=src \
    file://runtime/libc7x/tests;subdir=src \
    file://frontends/ort-c7x-ep/Makefile;subdir=src \
    file://frontends/ort-c7x-ep/include;subdir=src \
    file://frontends/ort-c7x-ep/src;subdir=src \
    file://frontends/ort-c7x-ep/tests/gen_onnx_models.py;subdir=src \
    file://frontends/ort-c7x-ep/tests/test_ort_c7x_ep.cpp;subdir=src \
"

S = "${UNPACKDIR}/src"

COMPATIBLE_MACHINE = "beaglebone-ai64"

# libort-c7x-ep.so is a private runtime plugin installed under /usr/lib/c7x-ai/.
SOLIBS = ".so"
FILES_SOLIBSDEV = ""

do_configure[noexec] = "1"

do_compile () {
    rm -rf ${S}/runtime/libc7x/build-aarch64 ${S}/frontends/ort-c7x-ep/build-aarch64 ${B}/models
    oe_runmake -C ${S}/runtime/libc7x \
        CC_AARCH64="${CC}" \
        AR_AARCH64="${AR}" \
        EXTRA_CFLAGS="${DEBUG_PREFIX_MAP}" \
        LDFLAGS="${LDFLAGS}" \
        hw
    oe_runmake -C ${S}/frontends/ort-c7x-ep \
        CXX_AARCH64="${CXX}" \
        AR_AARCH64="${AR}" \
        EXTRA_CXXFLAGS="${DEBUG_PREFIX_MAP}" \
        LDFLAGS="${LDFLAGS}" \
        hw
    python3 ${S}/frontends/ort-c7x-ep/tests/gen_onnx_models.py ${B}/models
}

do_install () {
    install -d ${D}${libdir}/c7x-ai
    install -d ${D}${datadir}/c7x-ai/models
    install -m 0755 ${S}/runtime/libc7x/build-aarch64/test_libc7x ${D}${libdir}/c7x-ai/test_libc7x
    install -m 0755 ${S}/frontends/ort-c7x-ep/build-aarch64/libort-c7x-ep.so ${D}${libdir}/c7x-ai/libort-c7x-ep.so
    install -m 0755 ${S}/frontends/ort-c7x-ep/build-aarch64/test_ort_c7x_ep ${D}${libdir}/c7x-ai/test_ort_c7x_ep
    install -m 0644 ${B}/models/* ${D}${datadir}/c7x-ai/models/
}

FILES:${PN} = " \
    ${libdir}/c7x-ai/test_libc7x \
    ${libdir}/c7x-ai/libort-c7x-ep.so \
    ${libdir}/c7x-ai/test_ort_c7x_ep \
    ${datadir}/c7x-ai/models/* \
"
