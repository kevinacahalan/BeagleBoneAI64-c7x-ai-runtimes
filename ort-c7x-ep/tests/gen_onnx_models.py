#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Pure-Python standard ONNX binary protobuf generator (zero external dependencies).

Generates two realistic non-LLM `.onnx` models + deterministic inputs & golden
reference outputs for verifying and benchmarking `ort-c7x-ep` on hostemu and
live BeagleBone AI-64 silicon:

  1. mobilenet_qlinear.onnx  — MobileNetV2-style INT8/UINT8 vision classifier:
       QLinearConv 3x3 s=2 (3->34, 64x64->32x32, UINT8 asymmetric, w_zp=-3)
       -> QLinearConv DW 3x3 s=2 (34->34, 32x32->16x16, group=34)
       -> QLinearConv PW 1x1 (34->64, 16x16)
       -> QLinearConv 3x3 s=2 (64->128, 16x16->8x8)
       -> DequantizeLinear -> Relu -> GlobalAveragePool -> Flatten -> Gemm(128->18)

  2. audio_enc_block.onnx    — Whisper / Audio-Encoder-style hybrid block:
       Conv 3x3 s=2 (1->64, 16x64->8x32) -> Gelu
       -> Reshape([1, 256, 64]) -> LayerNormalization(64)
       -> MatMul(64->128) -> Gelu -> MatMul(128->64) -> Add (residual)
"""

import math
import os
import struct
import sys

# --- Minimal standard protobuf wire encoder ---
def encode_varint(v: int) -> bytes:
    if v < 0:
        v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def pb_varint(fn: int, v: int) -> bytes:
    return encode_varint((fn << 3) | 0) + encode_varint(v)

def pb_float(fn: int, f: float) -> bytes:
    return encode_varint((fn << 3) | 5) + struct.pack("<f", f)

def pb_bytes(fn: int, b: bytes) -> bytes:
    return encode_varint((fn << 3) | 2) + encode_varint(len(b)) + b

def pb_str(fn: int, s: str) -> bytes:
    return pb_bytes(fn, s.encode("utf-8"))

# --- ONNX TensorProto / ValueInfoProto / AttributeProto / NodeProto ---
DT_FLOAT = 1
DT_UINT8 = 2
DT_INT8  = 3
DT_INT32 = 6
DT_INT64 = 7

def make_tensor(name: str, dims, dtype: int, raw: bytes) -> bytes:
    b = bytearray()
    for d in dims:
        b += pb_varint(1, d)
    b += pb_varint(2, dtype)
    b += pb_str(8, name)
    b += pb_bytes(9, raw)
    return bytes(b)

def make_scalar_f32(name: str, val: float) -> bytes:
    return make_tensor(name, [], DT_FLOAT, struct.pack("<f", val))

def make_vec_f32(name: str, vals) -> bytes:
    return make_tensor(name, [len(vals)], DT_FLOAT, struct.pack(f"<{len(vals)}f", *vals))

def make_scalar_u8(name: str, val: int) -> bytes:
    return make_tensor(name, [], DT_UINT8, struct.pack("<B", val & 0xFF))

def make_scalar_i8(name: str, val: int) -> bytes:
    return make_tensor(name, [], DT_INT8, struct.pack("<b", val))

def make_vec_i32(name: str, vals) -> bytes:
    return make_tensor(name, [len(vals)], DT_INT32, struct.pack(f"<{len(vals)}i", *vals))

def make_vec_i64(name: str, vals) -> bytes:
    return make_tensor(name, [len(vals)], DT_INT64, struct.pack(f"<{len(vals)}q", *vals))

def make_value_info(name: str, dtype: int, dims) -> bytes:
    shape = b"".join(pb_bytes(1, pb_varint(1, d)) for d in dims)
    ttype = pb_varint(1, dtype) + pb_bytes(2, shape)
    tproto = pb_bytes(1, ttype)
    return pb_str(1, name) + pb_bytes(2, tproto)

def attr_int(name: str, v: int) -> bytes:
    return pb_str(1, name) + pb_varint(3, v) + pb_varint(20, 2)

def attr_float(name: str, f: float) -> bytes:
    return pb_str(1, name) + pb_float(2, f) + pb_varint(20, 1)

def attr_str(name: str, s: str) -> bytes:
    return pb_str(1, name) + pb_str(4, s) + pb_varint(20, 3)

def attr_ints(name: str, vals) -> bytes:
    b = bytearray()
    b += pb_str(1, name)
    for v in vals:
        b += pb_varint(8, v)
    b += pb_varint(20, 7)
    return bytes(b)

def make_node(op: str, inputs, outputs, name: str, attrs=()) -> bytes:
    b = bytearray()
    for i in inputs:
        b += pb_str(1, i)
    for o in outputs:
        b += pb_str(2, o)
    b += pb_str(3, name)
    b += pb_str(4, op)
    for a in attrs:
        b += pb_bytes(5, a)
    return bytes(b)

def make_model(graph_name: str, nodes, inits, inputs, outputs) -> bytes:
    g = bytearray()
    for n in nodes:
        g += pb_bytes(1, n)
    g += pb_str(2, graph_name)
    for t in inits:
        g += pb_bytes(5, t)
    for vi in inputs:
        g += pb_bytes(11, vi)
    for vo in outputs:
        g += pb_bytes(12, vo)
    opset = pb_varint(2, 17)
    return pb_varint(1, 8) + pb_bytes(7, bytes(g)) + pb_bytes(8, opset)

# --- Deterministic LCG RNG ---
class Lcg:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF
    def next_u32(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s
    def f32(self, scale=1.0) -> float:
        return (((self.next_u32() >> 8) & 0xFFFF) - 32768) / 32768.0 * scale
    def i8(self, lo=-40, hi=40) -> int:
        return lo + (self.next_u32() % (hi - lo + 1))
    def u8(self, lo=60, hi=195) -> int:
        return lo + (self.next_u32() % (hi - lo + 1))

def round_even(v: float) -> int:
    return int(round(v))

def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)

# --- Pure-Python reference kernels ---
def ref_qlinear_conv(x, N, IC, IH, IW, x_sc, x_zp, x_u8,
                     w, OC, KH, KW, w_sc, w_zp, w_u8,
                     b_i32, y_sc, y_zp, y_u8, y_f32, relu,
                     s0, s1, p0, p1, groups=1):
    OH = (IH + 2 * p1 - KH) // s1 + 1
    OW = (IW + 2 * p0 - KW) // s0 + 1
    out = [0.0 if y_f32 else 0] * (N * OC * OH * OW)
    ic_per_g = IC // groups
    for n in range(N):
        for oc in range(OC):
            g = oc if groups == OC else 0
            ws = w_sc[oc] if len(w_sc) > 1 else w_sc[0]
            bi = b_i32[oc] if b_i32 is not None else 0
            for oy in range(OH):
                for ox in range(OW):
                    acc = 0
                    for ic_l in range(ic_per_g):
                        ic = g * ic_per_g + ic_l
                        for ky in range(KH):
                            sy = oy * s1 + ky - p1
                            for kx in range(KW):
                                sx = ox * s0 + kx - p0
                                if 0 <= sy < IH and 0 <= sx < IW:
                                    xv = x[((n * IC + ic) * IH + sy) * IW + sx] - x_zp
                                else:
                                    xv = 0
                                wv = w[((oc * ic_per_g + ic_l) * KH + ky) * KW + kx] - w_zp
                                acc += xv * wv
                    val = (acc + bi) * (x_sc * ws)
                    if relu and val < 0.0:
                        val = 0.0
                    idx = ((n * OC + oc) * OH + oy) * OW + ox
                    if y_f32:
                        out[idx] = val
                    else:
                        q = round_even(val / y_sc + y_zp)
                        out[idx] = clamp(q, 0, 255) if y_u8 else clamp(q, -128, 127)
    return out, OH, OW

def build_mobilenet_qlinear(out_dir: str):
    rng = Lcg(0xC7010001)
    # Input: [1, 3, 64, 64] UINT8 (asymmetric zp=128)
    N, C0, H0, W0 = 1, 3, 64, 64
    x_in = [rng.u8(40, 215) for _ in range(N * C0 * H0 * W0)]

    inits = []
    nodes = []

    # Layer 1: QLinearConv 3x3 s=2 p=1 (3 -> 34 non-16-multiple!, w1_zp=-3, 64x64 -> 32x32), UINT8 -> INT8
    OC1 = 34
    w1_zp = -3
    w1 = [rng.i8(-50, 50) for _ in range(OC1 * C0 * 9)]
    ws1 = [0.008 + 0.0002 * i for i in range(OC1)]
    b1 = [rng.i8(-20, 20) for _ in range(OC1)]
    x_sc, x_zp = 0.015, 128
    y1_sc, y1_zp = 0.04, -5

    inits += [
        make_scalar_f32("x_sc", x_sc), make_scalar_u8("x_zp", x_zp),
        make_tensor("w1", [OC1, C0, 3, 3], DT_INT8, struct.pack(f"<{len(w1)}b", *w1)),
        make_vec_f32("ws1", ws1), make_scalar_i8("w1_zp", w1_zp),
        make_scalar_f32("y1_sc", y1_sc), make_scalar_i8("y1_zp", y1_zp),
        make_vec_i32("b1", b1),
    ]
    nodes.append(make_node(
        "QLinearConv",
        ["input", "x_sc", "x_zp", "w1", "ws1", "w1_zp", "y1_sc", "y1_zp", "b1"],
        ["c1"], "conv1_stem",
        [attr_ints("strides", [2, 2]), attr_ints("pads", [1, 1, 1, 1]), attr_int("group", 1)],
    ))

    # Layer 2: Depthwise QLinearConv 3x3 s=2 p=1 (34 -> 34, group=34, 32x32 -> 16x16), INT8 -> INT8
    w2 = [rng.i8(-40, 40) for _ in range(OC1 * 1 * 9)]
    ws2 = [0.01 + 0.0001 * i for i in range(OC1)]
    b2 = [rng.i8(-15, 15) for _ in range(OC1)]
    y2_sc, y2_zp = 0.05, 3
    inits += [
        make_tensor("w2", [OC1, 1, 3, 3], DT_INT8, struct.pack(f"<{len(w2)}b", *w2)),
        make_vec_f32("ws2", ws2), make_scalar_i8("w2_zp", 0),
        make_scalar_f32("y2_sc", y2_sc), make_scalar_i8("y2_zp", y2_zp),
        make_vec_i32("b2", b2),
    ]
    nodes.append(make_node(
        "QLinearConv",
        ["c1", "y1_sc", "y1_zp", "w2", "ws2", "w2_zp", "y2_sc", "y2_zp", "b2"],
        ["c2"], "conv2_dw",
        [attr_ints("strides", [2, 2]), attr_ints("pads", [1, 1, 1, 1]), attr_int("group", OC1)],
    ))

    # Layer 3: Pointwise QLinearConv 1x1 s=1 p=0 (32 -> 64, 16x16), INT8 -> INT8
    OC3 = 64
    w3 = [rng.i8(-35, 35) for _ in range(OC3 * OC1)]
    ws3 = [0.009 + 0.0001 * i for i in range(OC3)]
    b3 = [rng.i8(-10, 10) for _ in range(OC3)]
    y3_sc, y3_zp = 0.06, -2
    inits += [
        make_tensor("w3", [OC3, OC1, 1, 1], DT_INT8, struct.pack(f"<{len(w3)}b", *w3)),
        make_vec_f32("ws3", ws3), make_scalar_i8("w3_zp", 0),
        make_scalar_f32("y3_sc", y3_sc), make_scalar_i8("y3_zp", y3_zp),
        make_vec_i32("b3", b3),
    ]
    nodes.append(make_node(
        "QLinearConv",
        ["c2", "y2_sc", "y2_zp", "w3", "ws3", "w3_zp", "y3_sc", "y3_zp", "b3"],
        ["c3"], "conv3_pw",
        [attr_ints("strides", [1, 1]), attr_ints("pads", [0, 0, 0, 0]), attr_int("group", 1)],
    ))

    # Layer 4: QLinearConv 3x3 s=2 p=1 (64 -> 128, 16x16 -> 8x8) -> Dequant -> Relu
    OC4 = 128
    w4 = [rng.i8(-30, 30) for _ in range(OC4 * OC3 * 9)]
    ws4 = [0.007 + 0.00005 * i for i in range(OC4)]
    b4 = [rng.i8(-10, 10) for _ in range(OC4)]
    y4_sc, y4_zp = 0.05, 0
    inits += [
        make_tensor("w4", [OC4, OC3, 3, 3], DT_INT8, struct.pack(f"<{len(w4)}b", *w4)),
        make_vec_f32("ws4", ws4), make_scalar_i8("w4_zp", 0),
        make_scalar_f32("y4_sc", y4_sc), make_scalar_i8("y4_zp", y4_zp),
        make_vec_i32("b4", b4),
    ]
    nodes.append(make_node(
        "QLinearConv",
        ["c3", "y3_sc", "y3_zp", "w4", "ws4", "w4_zp", "y4_sc", "y4_zp", "b4"],
        ["c4_q"], "conv4_down",
        [attr_ints("strides", [2, 2]), attr_ints("pads", [1, 1, 1, 1]), attr_int("group", 1)],
    ))
    nodes.append(make_node("DequantizeLinear", ["c4_q", "y4_sc", "y4_zp"], ["c4_f"], "deq4"))
    nodes.append(make_node("Relu", ["c4_f"], ["c4_relu"], "relu4"))
    nodes.append(make_node("GlobalAveragePool", ["c4_relu"], ["gap"], "gap"))
    nodes.append(make_node("Flatten", ["gap"], ["flat"], "flat", [attr_int("axis", 1)]))

    # Classifier head: Gemm (128 -> 18 non-16-multiple!, transB=1)
    NCLS = 18
    w_fc = [rng.f32(0.25) for _ in range(NCLS * OC4)]
    b_fc = [rng.f32(0.1) for _ in range(NCLS)]
    inits += [
        make_tensor("w_fc", [NCLS, OC4], DT_FLOAT, struct.pack(f"<{len(w_fc)}f", *w_fc)),
        make_vec_f32("b_fc", b_fc),
    ]
    nodes.append(make_node("Gemm", ["flat", "w_fc", "b_fc"], ["logits"], "fc", [attr_int("transB", 1)]))

    model_bytes = make_model(
        "mobilenet_qlinear",
        nodes, inits,
        [make_value_info("input", DT_UINT8, [N, C0, H0, W0])],
        [make_value_info("logits", DT_FLOAT, [N, NCLS])],
    )

    # Strict ONNX golden reference (exact layer-by-layer incl. QLinearConv4 INT8 requant -> DequantizeLinear -> Relu)
    c1, h1, w1_ = ref_qlinear_conv(x_in, N, C0, H0, W0, x_sc, x_zp, True,
                                   w1, OC1, 3, 3, ws1, w1_zp, False, b1, y1_sc, y1_zp, False, False, False, 2, 2, 1, 1, 1)
    c2, h2, w2_ = ref_qlinear_conv(c1, N, OC1, h1, w1_, y1_sc, y1_zp, False,
                                   w2, OC1, 3, 3, ws2, 0, False, b2, y2_sc, y2_zp, False, False, False, 2, 2, 1, 1, OC1)
    c3, h3, w3_ = ref_qlinear_conv(c2, N, OC1, h2, w2_, y2_sc, y2_zp, False,
                                   w3, OC3, 1, 1, ws3, 0, False, b3, y3_sc, y3_zp, False, False, False, 1, 1, 0, 0, 1)
    c4_q, h4, w4_ = ref_qlinear_conv(c3, N, OC3, h3, w3_, y3_sc, y3_zp, False,
                                     w4, OC4, 3, 3, ws4, 0, False, b4, y4_sc, y4_zp, False, False, False, 2, 2, 1, 1, 1)
    c4 = [max(0.0, (q - y4_zp) * y4_sc) for q in c4_q]
    gap = []
    for c in range(OC4):
        s = sum(c4[c * (h4 * w4_) + k] for k in range(h4 * w4_))
        gap.append(s / (h4 * w4_))
    logits = []
    for o in range(NCLS):
        s = b_fc[o] + sum(gap[k] * w_fc[o * OC4 + k] for k in range(OC4))
        logits.append(s)

    with open(os.path.join(out_dir, "mobilenet_qlinear.onnx"), "wb") as f:
        f.write(model_bytes)
    with open(os.path.join(out_dir, "mobilenet_input.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(x_in)}B", *x_in))
    with open(os.path.join(out_dir, "mobilenet_ref.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(logits)}f", *logits))

def build_audio_enc_block(out_dir: str):
    """Whisper-style Conv + GELU + LayerNorm + QLinearMatMul FFN block."""
    rng = Lcg(0xC7020002)
    N, C0, H0, W0 = 1, 1, 16, 32  # Mel spectrogram tile [1, 1, 16, 32]
    x_in = [rng.f32(1.0) for _ in range(N * C0 * H0 * W0)]

    OC = 64
    w_c = [rng.f32(0.2) for _ in range(OC * C0 * 9)]
    b_c = [rng.f32(0.05) for _ in range(OC)]
    gamma = [1.0 + rng.f32(0.1) for _ in range(OC)]
    beta  = [rng.f32(0.05) for _ in range(OC)]
    w_ff1 = [rng.f32(0.15) for _ in range(OC * 128)]  # [64, 128]
    w_ff2 = [rng.f32(0.15) for _ in range(128 * OC)]  # [128, 64]

    inits = [
        make_tensor("w_c", [OC, C0, 3, 3], DT_FLOAT, struct.pack(f"<{len(w_c)}f", *w_c)),
        make_vec_f32("b_c", b_c),
        make_vec_i64("sh_seq", [1, 128, OC]),  # 8*16 = 128 tokens x 64
        make_vec_f32("ln_g", gamma),
        make_vec_f32("ln_b", beta),
        make_tensor("w_ff1", [OC, 128], DT_FLOAT, struct.pack(f"<{len(w_ff1)}f", *w_ff1)),
        make_tensor("w_ff2", [128, OC], DT_FLOAT, struct.pack(f"<{len(w_ff2)}f", *w_ff2)),
    ]
    nodes = [
        make_node("Conv", ["mel", "w_c", "b_c"], ["h0"], "mel_conv",
                  [attr_ints("strides", [2, 2]), attr_ints("pads", [1, 1, 1, 1])]),
        make_node("Gelu", ["h0"], ["h1"], "gelu0", [attr_str("approximate", "tanh")]),
        make_node("Reshape", ["h1", "sh_seq"], ["seq"], "to_seq"),
        make_node("LayerNormalization", ["seq", "ln_g", "ln_b"], ["ln"], "ln0",
                  [attr_float("epsilon", 1e-5)]),
        make_node("MatMul", ["ln", "w_ff1"], ["ff1"], "ff1"),
        make_node("Gelu", ["ff1"], ["ff1_g"], "gelu1", [attr_str("approximate", "tanh")]),
        make_node("MatMul", ["ff1_g", "w_ff2"], ["ff2"], "ff2"),
        make_node("Add", ["seq", "ff2"], ["out"], "res_add"),
    ]
    model_bytes = make_model(
        "audio_enc_block",
        nodes, inits,
        [make_value_info("mel", DT_FLOAT, [N, C0, H0, W0])],
        [make_value_info("out", DT_FLOAT, [1, 128, OC])],
    )

    # Golden reference
    def gelu(v):
        return 0.5 * v * (1.0 + math.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))

    OH, OW = 8, 16
    h1 = [0.0] * (OC * OH * OW)
    for oc in range(OC):
        for oy in range(OH):
            for ox in range(OW):
                acc = b_c[oc]
                for ky in range(3):
                    sy = oy * 2 + ky - 1
                    for kx in range(3):
                        sx = ox * 2 + kx - 1
                        if 0 <= sy < H0 and 0 <= sx < W0:
                            acc += x_in[sy * W0 + sx] * w_c[(oc * 3 + ky) * 3 + kx]
                h1[(oc * OH + oy) * OW + ox] = gelu(acc)
    # Reshape [1, 64, 8, 16] -> [1, 128, 64] (contiguous reinterpretation)
    seq = h1
    out = [0.0] * len(seq)
    for t in range(128):
        row = seq[t * OC : (t + 1) * OC]
        mean = sum(row) / OC
        var = sum((v - mean) ** 2 for v in row) / OC
        inv = 1.0 / math.sqrt(var + 1e-5)
        ln = [(row[c] - mean) * inv * gamma[c] + beta[c] for c in range(OC)]
        ff1 = [gelu(sum(ln[k] * w_ff1[k * 128 + j] for k in range(OC))) for j in range(128)]
        for c in range(OC):
            out[t * OC + c] = row[c] + sum(ff1[j] * w_ff2[j * OC + c] for j in range(128))

    with open(os.path.join(out_dir, "audio_enc_block.onnx"), "wb") as f:
        f.write(model_bytes)
    with open(os.path.join(out_dir, "audio_input.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(x_in)}f", *x_in))
    with open(os.path.join(out_dir, "audio_ref.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(out)}f", *out))

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    build_mobilenet_qlinear(out_dir)
    build_audio_enc_block(out_dir)
    print(f"[gen_onnx_models] generated mobilenet_qlinear.onnx and audio_enc_block.onnx in {out_dir}")

if __name__ == "__main__":
    main()
