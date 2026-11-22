#!/usr/bin/env python3
"""
Host Python/NumPy reference and INT8 packer for llama2.c models (stories15M).

Implements:
  1. Exact FP32 reference forward pass matching karpathy/llama2.c.
  2. Exact W8A8 (per-output-channel INT8 weights, dynamic per-vector INT8
     activations, exact INT32 matmuls, FP32 RMSNorm/RoPE/Attention/SwiGLU)
     reference forward pass matching the C7x + MMA runtime.
  3. Packing into 64-byte-aligned .c7m binary format for the BeagleBone AI-64
     C7x DDR tensor arena, plus per-layer golden activation dumps.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

C7M_MAGIC = 0x314D3743  # "C7M1"
C7M_VERSION = 1
C7M_FLAG_TILED_WEIGHTS = 1 << 0
C7M_FLAG_W4A8_LAYERS = 1 << 1
C7M_FLAG_W4A8_WCLS = 1 << 2
HEADER_SIZE = 256  # Fixed 256-byte (64B-aligned) header


def align64(n: int) -> int:
    return (n + 63) & ~63


def pack_tile_major_64(mat: np.ndarray) -> np.ndarray:
    """Reorder [N_pad, K_pad] int8 into [N_pad/64, K_pad/64, 64, 64] contiguous MMA tiles."""
    n, k = mat.shape
    assert n % 64 == 0 and k % 64 == 0
    return np.ascontiguousarray(mat.reshape(n // 64, 64, k // 64, 64).transpose(0, 2, 1, 3))


def quantize_weight_w4a8_g64(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric group-64 4-bit quantization ([-7, +7]) in MMA tile-major layout:
      w_packed: uint8[N_pad/64, K_pad/64, 32, 64]
                vector v (0..31), byte c (0..63): low = row 2v col c, high = row 2v+1 col c
      w_scales: float32[N_pad/64, K_pad/64, 64] (per-row group-64 scales for each 64x64 tile)
      w_deq:    float32[N_pad, K_pad] (dequantized reference matrix)
    """
    n, k = mat.shape
    n_pad = align64(n)
    k_pad = align64(k)
    nt = n_pad // 64
    kt = k_pad // 64
    padded = np.zeros((n_pad, k_pad), dtype=np.float32)
    padded[:n, :k] = mat.astype(np.float32)

    # Reshape into tiles [nt, 64, kt, 64] -> [nt, kt, 64, 64]
    tiles = padded.reshape(nt, 64, kt, 64).transpose(0, 2, 1, 3)
    max_abs = np.max(np.abs(tiles), axis=3)  # [nt, kt, 64]
    scales = (max_abs / np.float32(7.0)).astype(np.float32)
    inv = np.where(scales > 0.0, np.float32(1.0) / np.where(scales > 0.0, scales, np.float32(1.0)), np.float32(0.0))
    q = np.clip(np.rint(tiles * inv[:, :, :, None]), -7, 7).astype(np.int8)  # [nt, kt, 64, 64]

    w_deq = (q.astype(np.float32) * scales[:, :, :, None]).transpose(0, 2, 1, 3).reshape(n_pad, k_pad)

    # Pack even/odd rows within each 64x64 tile: [nt, kt, 32, 64]
    lo = (q[:, :, 0::2, :].astype(np.uint8)) & np.uint8(0x0F)
    hi = ((q[:, :, 1::2, :].astype(np.uint8)) & np.uint8(0x0F)) << np.uint8(4)
    w_packed = np.ascontiguousarray(lo | hi)
    w_scales = np.ascontiguousarray(scales)
    return w_packed, w_scales, w_deq


def vexpf_f32(x: np.ndarray) -> np.ndarray:
    """Exact FP32 Cody-Waite exp matching C7x vexpf_f16()."""
    xc = np.clip(x.astype(np.float32), np.float32(-87.33654475), np.float32(88.72283905))
    k = np.rint(xc * np.float32(1.4426950408889634)).astype(np.int32)
    kf = k.astype(np.float32)
    r = ((xc - kf * np.float32(0.693359375)) - kf * np.float32(-2.12194440e-4)).astype(np.float32)
    p = (np.float32(0.0013888889) * r + np.float32(0.0083333338)).astype(np.float32)
    p = (p * r + np.float32(0.0416666679)).astype(np.float32)
    p = (p * r + np.float32(0.1666666716)).astype(np.float32)
    p = (p * r + np.float32(0.5)).astype(np.float32)
    p = (p * r + np.float32(1.0)).astype(np.float32)
    p = (p * r + np.float32(1.0)).astype(np.float32)
    exp_bits = ((k + np.int32(127)) << np.int32(23)).astype(np.int32)
    pow2k = exp_bits.view(np.float32)
    return (p * pow2k).astype(np.float32)


@dataclass
class ModelConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_weights: bool

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return (self.dim * self.n_kv_heads) // self.n_heads

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads


class Llama2Tokenizer:
    """Minimal karpathy/llama2.c tokenizer.bin reader & BPE encoder/decoder."""

    def __init__(self, path: str, vocab_size: int):
        self.vocab: List[str] = []
        self.scores: List[float] = []
        self.vocab_to_id: Dict[str, int] = {}
        with open(path, "rb") as f:
            self.max_token_length = struct.unpack("<i", f.read(4))[0]
            for i in range(vocab_size):
                score, slen = struct.unpack("<fi", f.read(8))
                piece = f.read(slen).decode("utf-8", errors="replace")
                self.scores.append(score)
                self.vocab.append(piece)
                self.vocab_to_id[piece] = i

    def decode(self, prev_token: int, token: int) -> str:
        piece = self.vocab[token]
        if prev_token == 1 and piece.startswith(" "):
            piece = piece[1:]
        if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            return bytes([int(piece[3:5], 16)]).decode("latin1")
        return piece

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if bos:
            tokens.append(1)
        if text:
            dummy = self.vocab_to_id.get(" ")
            if dummy is not None:
                tokens.append(dummy)
        for ch in text:
            tid = self.vocab_to_id.get(ch)
            if tid is not None:
                tokens.append(tid)
            else:
                for b in ch.encode("utf-8"):
                    tokens.append(b + 3)
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                merged = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                tid = self.vocab_to_id.get(merged)
                if tid is not None and self.scores[tid] > best_score:
                    best_score = self.scores[tid]
                    best_id = tid
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            tokens.pop(best_idx + 1)
        if text and len(tokens) > 1 and tokens[1] == self.vocab_to_id.get(" "):
            # Only strip dummy prefix if it did not merge
            pass
        if eos:
            tokens.append(2)
        return tokens


def load_llama2c_fp32(path: str) -> Tuple[ModelConfig, Dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        hdr = struct.unpack("<7i", f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_raw, seq_len = hdr
        shared = vocab_raw > 0
        vocab_size = abs(vocab_raw)
        cfg = ModelConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            seq_len=seq_len,
            shared_weights=shared,
        )
        data = np.fromfile(f, dtype=np.float32)

    off = 0

    def take(shape: Tuple[int, ...]) -> np.ndarray:
        nonlocal off
        cnt = int(np.prod(shape))
        arr = data[off : off + cnt].reshape(shape).copy()
        off += cnt
        return arr

    w: Dict[str, np.ndarray] = {}
    w["token_embedding_table"] = take((vocab_size, dim))
    w["rms_att_weight"] = take((n_layers, dim))
    w["wq"] = take((n_layers, dim, dim))
    w["wk"] = take((n_layers, cfg.kv_dim, dim))
    w["wv"] = take((n_layers, cfg.kv_dim, dim))
    w["wo"] = take((n_layers, dim, dim))
    w["rms_ffn_weight"] = take((n_layers, dim))
    w["w1"] = take((n_layers, hidden_dim, dim))
    w["w2"] = take((n_layers, dim, hidden_dim))
    w["w3"] = take((n_layers, hidden_dim, dim))
    w["rms_final_weight"] = take((dim,))
    # Skip legacy freq_cis_real/imag in v0 checkpoints if present
    rem = data.size - off
    freq_sz = seq_len * (cfg.head_size // 2)
    if not shared:
        if rem >= 2 * freq_sz + vocab_size * dim:
            off += 2 * freq_sz
        w["wcls"] = take((vocab_size, dim))
    else:
        w["wcls"] = w["token_embedding_table"]

    return cfg, w


def hsum16_f32(x: np.ndarray) -> np.float32:
    """Exact 16-lane FP32 vector reduction matching C7x hsum_float16()."""
    x32 = x.astype(np.float32)
    n16 = (x32.size // 16) * 16
    if n16 > 0:
        acc = np.zeros(16, dtype=np.float32)
        for i in range(0, n16, 16):
            acc = (acc + x32[i : i + 16]).astype(np.float32)
        s0 = np.float32(np.float32(acc[0] + acc[1]) + np.float32(acc[2] + acc[3]))
        s1 = np.float32(np.float32(acc[4] + acc[5]) + np.float32(acc[6] + acc[7]))
        s2 = np.float32(np.float32(acc[8] + acc[9]) + np.float32(acc[10] + acc[11]))
        s3 = np.float32(np.float32(acc[12] + acc[13]) + np.float32(acc[14] + acc[15]))
        s = np.float32(np.float32(s0 + s1) + np.float32(s2 + s3))
    else:
        s = np.float32(0.0)
    for i in range(n16, x32.size):
        s = np.float32(s + x32[i])
    return s


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32)
    ss = hsum16_f32((x32 * x32).astype(np.float32))
    inv = np.float32(1.0) / np.sqrt(
        np.float32(ss / np.float32(x32.size) + np.float32(eps)), dtype=np.float32
    )
    return ((x32 * inv) * weight.astype(np.float32)).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    m = np.max(x32)
    e = vexpf_f32((x32 - m).astype(np.float32))
    s = np.float32(0.0)
    for v in e:
        s = np.float32(s + v)
    inv = np.float32(1.0) / s
    return (e * inv).astype(np.float32)


def build_rope_tables(seq_len: int, head_size: int) -> Tuple[np.ndarray, np.ndarray]:
    half = head_size // 2
    cos_t = np.zeros((seq_len, half), dtype=np.float32)
    sin_t = np.zeros((seq_len, half), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(half):
            freq = 1.0 / (10000.0 ** ((2.0 * i) / float(head_size)))
            val = pos * freq
            cos_t[pos, i] = math.cos(val)
            sin_t[pos, i] = math.sin(val)
    return cos_t, sin_t


def apply_rope(
    vec: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray, n_heads: int, head_size: int
) -> np.ndarray:
    out = vec.astype(np.float32).copy()
    half = head_size // 2
    for h in range(n_heads):
        base = h * head_size
        for i in range(half):
            v0 = out[base + 2 * i]
            v1 = out[base + 2 * i + 1]
            c = cos_row[i]
            s = sin_row[i]
            out[base + 2 * i] = np.float32(v0 * c - v1 * s)
            out[base + 2 * i + 1] = np.float32(v0 * s + v1 * c)
    return out


def quantize_weight_per_row_pad64(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8 quantization with 64-element zero padding on N and K."""
    n, k = mat.shape
    n_pad = align64(n)
    k_pad = align64(k)
    q = np.zeros((n_pad, k_pad), dtype=np.int8)
    scales = np.zeros((n_pad,), dtype=np.float32)
    mat32 = mat.astype(np.float32)
    max_abs = np.max(np.abs(mat32), axis=1)
    s = (max_abs / np.float32(127.0)).astype(np.float32)
    scales[:n] = s
    inv = np.where(s > 0.0, np.float32(1.0) / np.where(s > 0.0, s, np.float32(1.0)), np.float32(0.0))
    scaled = np.rint(mat32 * inv[:, None])
    q[:n, :k] = np.clip(scaled, -127, 127).astype(np.int8)
    return q, scales


def quantize_act_i8_pad64(x: np.ndarray) -> Tuple[np.ndarray, np.float32]:
    """Dynamic per-vector symmetric INT8 quantization with 64-element zero padding."""
    k = x.size
    k_pad = align64(k)
    x32 = x.astype(np.float32)
    max_abs = float(np.max(np.abs(x32)))
    scale = np.float32(max_abs / 127.0)
    xq = np.zeros((k_pad,), dtype=np.int8)
    if scale > np.float32(0.0):
        inv = np.float32(1.0) / scale
        xq[:k] = np.clip(np.rint(x32 * inv), -127, 127).astype(np.int8)
    return xq, scale


def linear_w8a8(x: np.ndarray, w_q: np.ndarray, w_s: np.ndarray, out_dim: int) -> np.ndarray:
    xq, xs = quantize_act_i8_pad64(x)
    # Exact int32 dot product matching MMA C-bank accumulation
    y32 = w_q.astype(np.int32) @ xq.astype(np.int32)
    return (y32[:out_dim].astype(np.float32) * (w_s[:out_dim] * xs)).astype(np.float32)


class QuantizedModel:
    def __init__(self, cfg: ModelConfig, fp32_w: Dict[str, np.ndarray], quant_mode: str = "w8a8"):
        self.cfg = cfg
        self.quant_mode = quant_mode
        self.w4_layers = quant_mode == "w4a8"
        self.w4_wcls = quant_mode in ("w4a8", "hybrid")
        self.dim_pad = align64(cfg.dim)
        self.hidden_pad = align64(cfg.hidden_dim)
        self.vocab_pad = align64(cfg.vocab_size)
        self.rope_cos, self.rope_sin = build_rope_tables(cfg.seq_len, cfg.head_size)

        self.rms_att = fp32_w["rms_att_weight"].astype(np.float32).copy()
        self.rms_ffn = fp32_w["rms_ffn_weight"].astype(np.float32).copy()
        self.rms_final = fp32_w["rms_final_weight"].astype(np.float32).copy()

        self.layers: List[Dict[str, np.ndarray]] = []
        for l in range(cfg.n_layers):
            ld: Dict[str, np.ndarray] = {}
            for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                if self.w4_layers:
                    q, s, deq = quantize_weight_w4a8_g64(fp32_w[name][l])
                    ld[name + "_q"] = q
                    ld[name + "_s"] = s
                    ld[name + "_deq"] = deq
                else:
                    q, s = quantize_weight_per_row_pad64(fp32_w[name][l])
                    ld[name + "_q"] = q
                    ld[name + "_s"] = s
            self.layers.append(ld)

        if self.w4_wcls:
            self.wcls_q, self.wcls_s, self.wcls_deq = quantize_weight_w4a8_g64(fp32_w["wcls"])
            self.emb_deq = self.wcls_deq
        else:
            self.wcls_q, self.wcls_s = quantize_weight_per_row_pad64(fp32_w["wcls"])
            self.emb_q = self.wcls_q
            self.emb_s = self.wcls_s

    def _linear(self, x: np.ndarray, ld: Dict[str, np.ndarray], name: str, out_dim: int) -> np.ndarray:
        if self.w4_layers:
            xq, xs = quantize_act_i8_pad64(x)
            return ((ld[name + "_deq"][:out_dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
        return linear_w8a8(x, ld[name + "_q"], ld[name + "_s"], out_dim)

    def forward_token(
        self,
        token: int,
        pos: int,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        record_layers: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.cfg
        traces: Dict[str, np.ndarray] = {}

        # Token embedding lookup
        if self.w4_wcls:
            x = self.emb_deq[token, : cfg.dim].astype(np.float32).copy()
        else:
            x = (self.emb_q[token, : cfg.dim].astype(np.float32) * self.emb_s[token]).astype(np.float32)
        if record_layers:
            traces["emb"] = x.copy()

        cos_row = self.rope_cos[pos]
        sin_row = self.rope_sin[pos]
        inv_sqrt_hd = np.float32(1.0) / np.sqrt(np.float32(cfg.head_size), dtype=np.float32)

        for l in range(cfg.n_layers):
            ld = self.layers[l]
            xb = rmsnorm(x, self.rms_att[l])

            q = self._linear(xb, ld, "wq", cfg.dim)
            k = self._linear(xb, ld, "wk", cfg.kv_dim)
            v = self._linear(xb, ld, "wv", cfg.kv_dim)

            q = apply_rope(q, cos_row, sin_row, cfg.n_heads, cfg.head_size)
            k = apply_rope(k, cos_row, sin_row, cfg.n_kv_heads, cfg.head_size)

            k_cache[l, pos] = k
            v_cache[l, pos] = v

            # Multi-head causal attention (exact FP32 reduction order matching C7x)
            xb_att = np.zeros((cfg.dim,), dtype=np.float32)
            for h in range(cfg.n_heads):
                qh = q[h * cfg.head_size : (h + 1) * cfg.head_size]
                kvh = h // cfg.kv_mul
                kh = k_cache[l, : pos + 1, kvh * cfg.head_size : (kvh + 1) * cfg.head_size]
                vh = v_cache[l, : pos + 1, kvh * cfg.head_size : (kvh + 1) * cfg.head_size]
                att = np.empty((pos + 1,), dtype=np.float32)
                for t in range(pos + 1):
                    att[t] = np.float32(hsum16_f32((qh * kh[t]).astype(np.float32)) * inv_sqrt_hd)
                probs = softmax(att)
                oh = np.zeros((cfg.head_size,), dtype=np.float32)
                for t in range(pos + 1):
                    oh = (oh + probs[t] * vh[t]).astype(np.float32)
                xb_att[h * cfg.head_size : (h + 1) * cfg.head_size] = oh

            xb2 = self._linear(xb_att, ld, "wo", cfg.dim)
            x = (x + xb2).astype(np.float32)

            if record_layers:
                traces[f"l{l}_post_att"] = x.copy()

            # FFN SwiGLU
            xb = rmsnorm(x, self.rms_ffn[l])
            hb = self._linear(xb, ld, "w1", cfg.hidden_dim)
            hb2 = self._linear(xb, ld, "w3", cfg.hidden_dim)
            # SiLU(hb) * hb2 (exact vexpf matching C7x)
            silu = (hb / (np.float32(1.0) + vexpf_f32((-hb).astype(np.float32)))).astype(np.float32)
            hb = (silu * hb2).astype(np.float32)

            xb2 = self._linear(hb, ld, "w2", cfg.dim)
            x = (x + xb2).astype(np.float32)

            if record_layers:
                traces[f"l{l}_out"] = x.copy()

        x = rmsnorm(x, self.rms_final)
        if self.w4_wcls:
            xq, xs = quantize_act_i8_pad64(x)
            logits = ((self.wcls_deq[: cfg.vocab_size] @ xq.astype(np.float32)) * xs).astype(np.float32)
        else:
            logits = linear_w8a8(x, self.wcls_q, self.wcls_s, cfg.vocab_size)
        if record_layers:
            traces["final_norm"] = x.copy()
            traces["logits"] = logits.copy()
        return logits, traces

    def pack_c7m(self, out_path: str, tiled: bool = True) -> None:
        """Pack quantized model into 64-byte-aligned .c7m binary format."""
        cfg = self.cfg
        buf = bytearray(HEADER_SIZE)

        def append_aligned(arr: np.ndarray) -> int:
            rem = len(buf) % 64
            if rem:
                buf.extend(b"\x00" * (64 - rem))
            off = len(buf)
            buf.extend(np.ascontiguousarray(arr).tobytes())
            return off

        def maybe_tile_w8(arr: np.ndarray, is_w4: bool) -> np.ndarray:
            if is_w4:
                return arr
            return pack_tile_major_64(arr) if tiled else arr

        off_rope_cos = append_aligned(self.rope_cos)
        off_rope_sin = append_aligned(self.rope_sin)
        off_rms_final = append_aligned(self.rms_final)
        off_wcls_q = append_aligned(maybe_tile_w8(self.wcls_q, self.w4_wcls))
        off_wcls_s = append_aligned(self.wcls_s)

        # Layer table: 16 uint32 offsets per layer (64 bytes per layer entry)
        layer_table_entries = []
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        off_layer_table = len(buf)
        buf.extend(b"\x00" * (cfg.n_layers * 64))

        # Align start of contiguous layer blob (for MSMC staging)
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        off_layers_base = len(buf)
        layer_stride = 0

        for l in range(cfg.n_layers):
            rem = len(buf) % 64
            if rem:
                buf.extend(b"\x00" * (64 - rem))
            l_start = len(buf)
            ld = self.layers[l]
            offs = [
                append_aligned(self.rms_att[l]),
                append_aligned(self.rms_ffn[l]),
                append_aligned(maybe_tile_w8(ld["wq_q"], self.w4_layers)),
                append_aligned(ld["wq_s"]),
                append_aligned(maybe_tile_w8(ld["wk_q"], self.w4_layers)),
                append_aligned(ld["wk_s"]),
                append_aligned(maybe_tile_w8(ld["wv_q"], self.w4_layers)),
                append_aligned(ld["wv_s"]),
                append_aligned(maybe_tile_w8(ld["wo_q"], self.w4_layers)),
                append_aligned(ld["wo_s"]),
                append_aligned(maybe_tile_w8(ld["w1_q"], self.w4_layers)),
                append_aligned(ld["w1_s"]),
                append_aligned(maybe_tile_w8(ld["w2_q"], self.w4_layers)),
                append_aligned(ld["w2_s"]),
                append_aligned(maybe_tile_w8(ld["w3_q"], self.w4_layers)),
                append_aligned(ld["w3_s"]),
            ]
            rem = len(buf) % 64
            if rem:
                buf.extend(b"\x00" * (64 - rem))
            layer_stride = len(buf) - l_start
            layer_table_entries.append(offs)

        total_size = len(buf)

        # Fill layer table
        for l, offs in enumerate(layer_table_entries):
            struct.pack_into("<16I", buf, off_layer_table + l * 64, *offs)

        flags = 0
        if tiled and not (self.w4_layers and self.w4_wcls):
            flags |= C7M_FLAG_TILED_WEIGHTS
        if self.w4_layers:
            flags |= C7M_FLAG_W4A8_LAYERS
        if self.w4_wcls:
            flags |= C7M_FLAG_W4A8_WCLS

        # Fill header (matches struct c7m_header in dsp/include/c7x_llama_fmt.h)
        hdr = struct.pack(
            "<24If8I",
            C7M_MAGIC,
            C7M_VERSION,
            total_size,
            cfg.dim,
            self.dim_pad,
            cfg.hidden_dim,
            self.hidden_pad,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_size,
            cfg.kv_dim,
            cfg.vocab_size,
            self.vocab_pad,
            cfg.seq_len,
            off_rope_cos,
            off_rope_sin,
            off_rms_final,
            off_wcls_q,
            off_wcls_s,
            off_layer_table,
            flags,
            off_layers_base,
            layer_stride,
            1e-5,  # rms_norm_eps
            64 if (self.w4_layers or self.w4_wcls) else 0,  # w4_group_size
            0,  # off_emb_q
            0,  # off_emb_s
            0,  # off_bias_table
            0,  # off_tokenizer
            0,  # tokenizer_size
            1,  # bos_token_id
            2,  # eos_token_id
        )
        buf[0 : len(hdr)] = hdr

        with open(out_path, "wb") as f:
            f.write(buf)
        print(
            f"packed {out_path}: {total_size:,} bytes ({total_size / (1024*1024):.2f} MiB), "
            f"mode={self.quant_mode}, tiled={tiled}, layer_stride={layer_stride:,} B"
        )


def forward_fp32_token(
    cfg: ModelConfig,
    w: Dict[str, np.ndarray],
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    token: int,
    pos: int,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
) -> np.ndarray:
    x = w["token_embedding_table"][token].astype(np.float32).copy()
    cos_row = rope_cos[pos]
    sin_row = rope_sin[pos]
    inv_sqrt_hd = np.float32(1.0 / math.sqrt(float(cfg.head_size)))

    for l in range(cfg.n_layers):
        xb = rmsnorm(x, w["rms_att_weight"][l])
        q = (w["wq"][l] @ xb).astype(np.float32)
        k = (w["wk"][l] @ xb).astype(np.float32)
        v = (w["wv"][l] @ xb).astype(np.float32)
        q = apply_rope(q, cos_row, sin_row, cfg.n_heads, cfg.head_size)
        k = apply_rope(k, cos_row, sin_row, cfg.n_kv_heads, cfg.head_size)
        k_cache[l, pos] = k
        v_cache[l, pos] = v

        xb_att = np.zeros((cfg.dim,), dtype=np.float32)
        for h in range(cfg.n_heads):
            qh = q[h * cfg.head_size : (h + 1) * cfg.head_size]
            kvh = h // cfg.kv_mul
            kh = k_cache[l, : pos + 1, kvh * cfg.head_size : (kvh + 1) * cfg.head_size]
            vh = v_cache[l, : pos + 1, kvh * cfg.head_size : (kvh + 1) * cfg.head_size]
            att = (kh @ qh).astype(np.float32) * inv_sqrt_hd
            probs = softmax(att)
            xb_att[h * cfg.head_size : (h + 1) * cfg.head_size] = (probs @ vh).astype(np.float32)

        x = (x + (w["wo"][l] @ xb_att).astype(np.float32)).astype(np.float32)
        xb = rmsnorm(x, w["rms_ffn_weight"][l])
        hb = (w["w1"][l] @ xb).astype(np.float32)
        hb2 = (w["w3"][l] @ xb).astype(np.float32)
        hb = (hb / (np.float32(1.0) + np.exp(-hb).astype(np.float32)) * hb2).astype(np.float32)
        x = (x + (w["w2"][l] @ hb).astype(np.float32)).astype(np.float32)

    x = rmsnorm(x, w["rms_final_weight"])
    return (w["wcls"] @ x).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="llama2.c FP32 & W8A8 reference + .c7m packer")
    ap.add_argument("--model", required=True, help="Input llama2.c .bin checkpoint")
    ap.add_argument("--tokenizer", required=True, help="Input tokenizer.bin")
    ap.add_argument("--pack", help="Output .c7m quantized model path")
    ap.add_argument("--dump-golden", help="Output .npz golden trace path")
    ap.add_argument("--dump-golden-bin", help="Output flat binary golden trace for C/C++ tests")
    ap.add_argument("--quant", choices=["w8a8", "hybrid", "w4a8"], default="w8a8",
                    help="Quantization mode: w8a8 (all INT8), hybrid (INT8 layers + W4A8 wcls), w4a8 (all group-64 W4A8)")
    ap.add_argument("--steps", type=int, default=40, help="Greedy generation steps")
    ap.add_argument("--prompt", default="", help="Optional prompt string (empty = BOS only)")
    args = ap.parse_args()

    cfg, fp32_w = load_llama2c_fp32(args.model)
    tok = Llama2Tokenizer(args.tokenizer, cfg.vocab_size)
    print(
        f"model: dim={cfg.dim} (pad {align64(cfg.dim)}), hidden={cfg.hidden_dim} "
        f"(pad {align64(cfg.hidden_dim)}), layers={cfg.n_layers}, heads={cfg.n_heads}, "
        f"kv_heads={cfg.n_kv_heads}, vocab={cfg.vocab_size}, seq={cfg.seq_len}, "
        f"shared={cfg.shared_weights}, quant={args.quant}"
    )

    qmodel = QuantizedModel(cfg, fp32_w, quant_mode=args.quant)
    if args.pack:
        qmodel.pack_c7m(args.pack)

    prompt_tokens = tok.encode(args.prompt, bos=True, eos=False)
    print(f"prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    # Run FP32 greedy reference
    rope_cos, rope_sin = build_rope_tables(cfg.seq_len, cfg.head_size)
    kf = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
    vf = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
    fp32_tokens: List[int] = []
    token = prompt_tokens[0]
    fp32_text = []
    for pos in range(args.steps):
        logits = forward_fp32_token(cfg, fp32_w, rope_cos, rope_sin, token, pos, kf, vf)
        if pos + 1 < len(prompt_tokens):
            nxt = prompt_tokens[pos + 1]
        else:
            nxt = int(np.argmax(logits))
        fp32_tokens.append(nxt)
        fp32_text.append(tok.decode(token, nxt))
        token = nxt
    print("\n--- FP32 Reference ---")
    print("".join(fp32_text))

    # Run W8A8 greedy reference
    kq = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
    vq = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
    w8a8_tokens: List[int] = []
    token = prompt_tokens[0]
    w8a8_text = []
    first_traces: Dict[str, np.ndarray] = {}
    for pos in range(args.steps):
        logits, tr = qmodel.forward_token(token, pos, kq, vq, record_layers=(pos == 0))
        if pos == 0:
            first_traces = tr
            # Compare pos=0 logits against FP32
            kf0 = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
            vf0 = np.zeros((cfg.n_layers, cfg.seq_len, cfg.kv_dim), dtype=np.float32)
            logits_fp0 = forward_fp32_token(cfg, fp32_w, rope_cos, rope_sin, token, 0, kf0, vf0)
            cos_sim = float(
                np.dot(logits, logits_fp0)
                / (np.linalg.norm(logits) * np.linalg.norm(logits_fp0) + 1e-12)
            )
            max_err = float(np.max(np.abs(logits - logits_fp0)))
            print(
                f"\npos=0 W8A8 vs FP32 logits: cosine={cos_sim:.6f}, max_abs_err={max_err:.4f}, "
                f"top1_fp32={int(np.argmax(logits_fp0))}, top1_w8a8={int(np.argmax(logits))}"
            )
        if pos + 1 < len(prompt_tokens):
            nxt = prompt_tokens[pos + 1]
        else:
            nxt = int(np.argmax(logits))
        w8a8_tokens.append(nxt)
        w8a8_text.append(tok.decode(token, nxt))
        token = nxt

    print("\n--- W8A8 Quantized Reference ---")
    print("".join(w8a8_text))
    print(f"\nW8A8 token IDs ({len(w8a8_tokens)}): {w8a8_tokens}")
    matches = sum(1 for a, b in zip(fp32_tokens, w8a8_tokens) if a == b)
    print(f"FP32 vs W8A8 exact token match: {matches}/{len(w8a8_tokens)}")

    if args.dump_golden:
        save_dict = {
            "prompt_tokens": np.array(prompt_tokens, dtype=np.int32),
            "w8a8_tokens": np.array(w8a8_tokens, dtype=np.int32),
            "fp32_tokens": np.array(fp32_tokens, dtype=np.int32),
        }
        save_dict.update(first_traces)
        np.savez(args.dump_golden, **save_dict)
        print(f"saved golden traces to {args.dump_golden}")

    if args.dump_golden_bin:
        ref_tokens = fp32_tokens if args.quant == "w8a8" else w8a8_tokens
        with open(args.dump_golden_bin, "wb") as f:
            # Header: magic "GLD1", dim, vocab_size, n_steps, start_token
            f.write(
                struct.pack(
                    "<5I",
                    0x31444C47,
                    cfg.dim,
                    cfg.vocab_size,
                    len(ref_tokens),
                    prompt_tokens[0],
                )
            )
            f.write(np.array(ref_tokens, dtype=np.uint32).tobytes())
            f.write(first_traces["l0_out"].astype(np.float32).tobytes())
            f.write(first_traces[f"l{cfg.n_layers - 1}_out"].astype(np.float32).tobytes())
            f.write(first_traces["final_norm"].astype(np.float32).tobytes())
            f.write(first_traces["logits"].astype(np.float32).tobytes())
        print(f"saved binary golden trace to {args.dump_golden_bin}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
