#!/usr/bin/env python3
"""
Universal HuggingFace (safetensors + config.json + tokenizer.json) -> .c7m
quantized model converter & NumPy reference verifier for BeagleBone AI-64 C7x+MMA.

Supports any Llama / SmolLM / SmolLM2 / Qwen2 / Qwen2.5 / Mistral / TinyLlama
decoder-only causal LM checkpoint with ZERO PyTorch or safetensors pip dependencies
(pure Python + NumPy, including native BF16/F16/F32 decoding).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from llama_ref import (
    C7M_FLAG_TILED_WEIGHTS,
    C7M_FLAG_W4A8_LAYERS,
    C7M_FLAG_W4A8_WCLS,
    C7M_MAGIC,
    C7M_VERSION,
    HEADER_SIZE,
    align64,
    build_rope_tables,
    hsum16_f32,
    pack_tile_major_64,
    quantize_act_i8_pad64,
    quantize_weight_per_row_pad64,
    quantize_weight_w4a8_g64,
    rmsnorm,
    softmax,
    vexpf_f32,
)

C7M_FLAG_HAS_QKV_BIAS = 1 << 3
C7M_FLAG_UNTIED_EMB = 1 << 4
C7M_FLAG_ROPE_NEOX = 1 << 5

C7T_MAGIC = 0x31543743  # "C7T1"
C7T_FLAG_BYTE_LEVEL = 1 << 0
C7T_FLAG_ADD_BOS = 1 << 1


# ---------------------------------------------------------------------------
# Pure-Python Safetensors reader (supports F32, F16, BF16, single or sharded)
# ---------------------------------------------------------------------------

class SafetensorsReader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tensor_map: Dict[str, Tuple[str, dict, int]] = {}
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            files = sorted(set(idx["weight_map"].values()))
        else:
            files = ["model.safetensors"]

        for fn in files:
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len).decode("utf-8"))
                data_base = 8 + hdr_len
                for name, meta in hdr.items():
                    if name == "__metadata__":
                        continue
                    self.tensor_map[name] = (path, meta, data_base)

    def has(self, name: str) -> bool:
        return name in self.tensor_map

    def get_f32(self, name: str) -> np.ndarray:
        path, meta, data_base = self.tensor_map[name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(data_base + start)
            raw = f.read(end - start)
        if dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype} for {name}")
        return arr.reshape(shape)


# ---------------------------------------------------------------------------
# GPT-2 byte-level Unicode <-> raw byte mapping (SmolLM2, Qwen2.5, Llama-3)
# ---------------------------------------------------------------------------

def gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build_c7t_tokenizer_blob(
    tok_json_path: str, vocab_size: int
) -> Tuple[bytes, dict]:
    """
    Compile HuggingFace tokenizer.json into a compact binary C7T1 blob for C:
      Header (32 bytes):
        magic ("C7T1"), version (1), vocab_size, n_special, n_merges, flags,
        off_pieces_blob, pieces_blob_bytes
      byte_to_id[256]          (int32)
      piece_offs[vocab_size+1] (uint32)
      special_entries[n_special]: (id:u32, str_off:u32, str_len:u32)
      merge_entries[n_merges]:    (key:u64 = (left<<32)|right, rank:u32, merged_id:u32)
      pieces_blob              (raw UTF-8 decoded bytes for all tokens + specials)
    """
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    model = tj.get("model", {})
    vocab_dict: Dict[str, int] = model.get("vocab", {})
    merges_raw = model.get("merges", [])
    added_tokens = tj.get("added_tokens", [])

    # Detect byte-level BPE (SmolLM2, Qwen2.5, Llama-3, GPT-2)
    dec_str = json.dumps(tj.get("decoder", {})) + json.dumps(tj.get("pre_tokenizer", {}))
    is_byte_level = "ByteLevel" in dec_str
    b2u = gpt2_bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def decode_piece_to_bytes(piece: str, is_special: bool) -> bytes:
        if is_special:
            return piece.encode("utf-8")
        if is_byte_level:
            out = bytearray()
            for ch in piece:
                if ch in u2b:
                    out.append(u2b[ch])
                else:
                    out.extend(ch.encode("utf-8"))
            return bytes(out)
        # SentencePiece Metaspace fallback
        return piece.replace("\u2581", " ").encode("utf-8")

    id_to_piece: List[bytes] = [b""] * vocab_size
    special_ids = set()
    special_list: List[Tuple[int, bytes]] = []

    for at in added_tokens:
        tid = int(at["id"])
        content = at["content"]
        if 0 <= tid < vocab_size:
            raw = content.encode("utf-8")
            id_to_piece[tid] = raw
            special_ids.add(tid)
            special_list.append((tid, raw))

    for piece, tid in vocab_dict.items():
        tid = int(tid)
        if 0 <= tid < vocab_size and tid not in special_ids:
            id_to_piece[tid] = decode_piece_to_bytes(piece, False)

    # Single-byte token lookup table
    byte_to_id = [-1] * 256
    if is_byte_level:
        for b in range(256):
            ch = b2u[b]
            if ch in vocab_dict:
                byte_to_id[b] = int(vocab_dict[ch])
    else:
        for b in range(256):
            cand = bytes([b])
            for tid, pb in enumerate(id_to_piece):
                if pb == cand and tid not in special_ids:
                    byte_to_id[b] = tid
                    break

    # Compile merges into (key=(left<<32)|right, rank, merged_id) sorted by key
    merge_entries: List[Tuple[int, int, int]] = []
    for rank, m in enumerate(merges_raw):
        if isinstance(m, str):
            parts = m.split(" ")
            if len(parts) != 2:
                continue
            a, b = parts
        elif isinstance(m, list) and len(m) == 2:
            a, b = m[0], m[1]
        else:
            continue
        la = vocab_dict.get(a)
        rb = vocab_dict.get(b)
        mid = vocab_dict.get(a + b)
        if la is not None and rb is not None and mid is not None:
            key = (int(la) << 32) | int(rb)
            merge_entries.append((key, rank, int(mid)))

    # Deduplicate by key keeping lowest rank, then sort by key for C bsearch
    best_merge: Dict[int, Tuple[int, int]] = {}
    for key, rank, mid in merge_entries:
        if key not in best_merge or rank < best_merge[key][0]:
            best_merge[key] = (rank, mid)
    sorted_merges = sorted((k, v[0], v[1]) for k, v in best_merge.items())

    # Sort specials by descending byte length for greedy longest-match in C
    special_list.sort(key=lambda x: len(x[1]), reverse=True)

    # Pack pieces blob
    pieces_blob = bytearray()
    piece_offs: List[int] = []
    for tid in range(vocab_size):
        piece_offs.append(len(pieces_blob))
        pieces_blob.extend(id_to_piece[tid])
    piece_offs.append(len(pieces_blob))

    special_packed = bytearray()
    for tid, raw in special_list:
        so = len(pieces_blob)
        pieces_blob.extend(raw)
        special_packed.extend(struct.pack("<3I", tid, so, len(raw)))

    merges_packed = bytearray()
    for key, rank, mid in sorted_merges:
        merges_packed.extend(struct.pack("<Q2I", key, rank, mid))

    flags = C7T_FLAG_BYTE_LEVEL if is_byte_level else 0
    # Calculate offset to pieces_blob
    hdr_sz = 32
    b2id_sz = 256 * 4
    poff_sz = (vocab_size + 1) * 4
    spec_sz = len(special_packed)
    mrg_sz = len(merges_packed)
    off_pieces_blob = hdr_sz + b2id_sz + poff_sz + spec_sz + mrg_sz

    out = bytearray()
    out.extend(
        struct.pack(
            "<8I",
            C7T_MAGIC,
            1,
            vocab_size,
            len(special_list),
            len(sorted_merges),
            flags,
            off_pieces_blob,
            len(pieces_blob),
        )
    )
    out.extend(struct.pack("<256i", *byte_to_id))
    out.extend(struct.pack(f"<{vocab_size + 1}I", *piece_offs))
    out.extend(special_packed)
    out.extend(merges_packed)
    out.extend(pieces_blob)

    py_tok = {
        "id_to_piece": id_to_piece,
        "byte_to_id": byte_to_id,
        "special_list": special_list,
        "merge_map": best_merge,
        "is_byte_level": is_byte_level,
    }
    return bytes(out), py_tok


def c7t_encode_py(py_tok: dict, text: str) -> List[int]:
    """Exact Python mirror of the C7T1 C tokenizer encoder."""
    raw = text.encode("utf-8")
    specials = py_tok["special_list"]
    b2id = py_tok["byte_to_id"]
    merge_map = py_tok["merge_map"]

    def bpe_chunk(chunk: bytes) -> List[int]:
        if not chunk:
            return []
        ids = [b2id[b] for b in chunk if b2id[b] >= 0]
        while len(ids) >= 2:
            best_rank = 1 << 30
            best_pos = -1
            best_mid = -1
            for i in range(len(ids) - 1):
                key = (ids[i] << 32) | ids[i + 1]
                m = merge_map.get(key)
                if m is not None and m[0] < best_rank:
                    best_rank = m[0]
                    best_mid = m[1]
                    best_pos = i
            if best_pos < 0:
                break
            ids[best_pos] = best_mid
            ids.pop(best_pos + 1)
        return ids

    tokens: List[int] = []
    pos = 0
    cur = bytearray()
    while pos < len(raw):
        matched = False
        for tid, sbytes in specials:
            if sbytes and raw.startswith(sbytes, pos):
                if cur:
                    tokens.extend(bpe_chunk(bytes(cur)))
                    cur.clear()
                tokens.append(tid)
                pos += len(sbytes)
                matched = True
                break
        if not matched:
            cur.append(raw[pos])
            pos += 1
    if cur:
        tokens.extend(bpe_chunk(bytes(cur)))
    return tokens


def c7t_decode_py(py_tok: dict, tokens: List[int]) -> str:
    return b"".join(py_tok["id_to_piece"][t] for t in tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# RoPE (NeoX half-split) & NumPy Reference Forward Pass
# ---------------------------------------------------------------------------

def build_rope_tables_theta(seq_len: int, head_size: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    half = head_size // 2
    cos_t = np.zeros((seq_len, half), dtype=np.float32)
    sin_t = np.zeros((seq_len, half), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(half):
            freq = 1.0 / (theta ** ((2.0 * i) / float(head_size)))
            val = pos * freq
            cos_t[pos, i] = math.cos(val)
            sin_t[pos, i] = math.sin(val)
    return cos_t, sin_t


def apply_rope_neox(
    vec: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray, n_heads: int, head_size: int
) -> np.ndarray:
    out = vec.astype(np.float32).copy()
    half = head_size // 2
    for h in range(n_heads):
        base = h * head_size
        v0 = out[base : base + half].copy()
        v1 = out[base + half : base + head_size].copy()
        out[base : base + half] = (v0 * cos_row - v1 * sin_row).astype(np.float32)
        out[base + half : base + head_size] = (v0 * sin_row + v1 * cos_row).astype(np.float32)
    return out


def convert_and_optionally_verify(args: argparse.Namespace) -> int:
    with open(os.path.join(args.hf_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_size = int(cfg.get("head_dim", dim // n_heads))
    kv_dim = n_kv_heads * head_size
    vocab_size = int(cfg["vocab_size"])
    seq_len = min(int(cfg.get("max_position_embeddings", 2048)), args.seq_len)
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    bos_id = int(cfg.get("bos_token_id") or 1)
    eos_id = cfg.get("eos_token_id", 2)
    if isinstance(eos_id, list):
        eos_id = int(eos_id[0])
    else:
        eos_id = int(eos_id)

    dim_pad = align64(dim)
    hidden_pad = align64(hidden_dim)
    vocab_pad = align64(vocab_size)
    kv_dim_pad = align64(kv_dim)

    reader = SafetensorsReader(args.hf_dir)
    has_qkv_bias = reader.has("model.layers.0.self_attn.q_proj.bias")
    tied = bool(cfg.get("tie_word_embeddings", True)) or (not reader.has("lm_head.weight"))

    w4_layers = args.quant == "w4a8"
    w4_wcls = args.quant in ("w4a8", "hybrid")

    print(
        f"[convert_hf] {cfg.get('model_type')} | dim={dim} (pad {dim_pad}), "
        f"hidden={hidden_dim} (pad {hidden_pad}), layers={n_layers}, "
        f"heads={n_heads}/{n_kv_heads} (hd={head_size}), vocab={vocab_size}, "
        f"seq={seq_len}, eps={rms_eps}, theta={rope_theta}, "
        f"qkv_bias={has_qkv_bias}, tied={tied}, quant={args.quant}"
    )

    # Build embedded C7T1 tokenizer if tokenizer.json is present
    tok_json = os.path.join(args.hf_dir, "tokenizer.json")
    tok_blob = b""
    py_tok = None
    if os.path.exists(tok_json):
        tok_blob, py_tok = build_c7t_tokenizer_blob(tok_json, vocab_size)
        print(
            f"[convert_hf] compiled embedded C7T1 tokenizer: {len(tok_blob):,} bytes "
            f"(byte_level={py_tok['is_byte_level']}, specials={len(py_tok['special_list'])}, "
            f"merges={len(py_tok['merge_map'])})"
        )

    rope_cos, rope_sin = build_rope_tables_theta(seq_len, head_size, rope_theta)
    rms_final = reader.get_f32("model.norm.weight")

    # Embeddings & Classifier
    emb_fp32 = reader.get_f32("model.embed_tokens.weight")
    wcls_fp32 = emb_fp32 if tied else reader.get_f32("lm_head.weight")

    emb_q_sep = None
    emb_s_sep = None
    if not tied:
        emb_q_sep, emb_s_sep = quantize_weight_per_row_pad64(emb_fp32)
        emb_q_sep = emb_q_sep[:vocab_size]
        emb_s_sep = emb_s_sep[:vocab_size]

    if w4_wcls:
        wcls_q, wcls_s, wcls_deq = quantize_weight_w4a8_g64(wcls_fp32)
    else:
        wcls_q_raw, wcls_s = quantize_weight_per_row_pad64(wcls_fp32)
        wcls_q = pack_tile_major_64(wcls_q_raw)
        wcls_deq = (wcls_q_raw.astype(np.float32) * wcls_s[:, None])

    del wcls_fp32
    if tied:
        emb_deq_ref = wcls_deq
    else:
        emb_deq_ref = (emb_q_sep.astype(np.float32) * emb_s_sep[:, None])
    del emb_fp32

    # Pack binary buffer incrementally to keep host RAM low
    buf = bytearray(HEADER_SIZE)

    def append_aligned(arr: np.ndarray) -> int:
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        off = len(buf)
        buf.extend(np.ascontiguousarray(arr).tobytes())
        return off

    def append_bytes_aligned(raw: bytes) -> int:
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        off = len(buf)
        buf.extend(raw)
        return off

    off_rope_cos = append_aligned(rope_cos)
    off_rope_sin = append_aligned(rope_sin)
    off_rms_final = append_aligned(rms_final)
    off_wcls_q = append_aligned(wcls_q)
    off_wcls_s = append_aligned(wcls_s)

    off_emb_q = 0
    off_emb_s = 0
    if not tied and emb_q_sep is not None and emb_s_sep is not None:
        off_emb_q = append_aligned(emb_q_sep)
        off_emb_s = append_aligned(emb_s_sep)

    off_tok = append_bytes_aligned(tok_blob) if tok_blob else 0

    rem = len(buf) % 64
    if rem:
        buf.extend(b"\x00" * (64 - rem))
    off_layer_table = len(buf)
    buf.extend(b"\x00" * (n_layers * 64))

    off_bias_table = 0
    if has_qkv_bias:
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        off_bias_table = len(buf)
        buf.extend(b"\x00" * (n_layers * 16))

    rem = len(buf) % 64
    if rem:
        buf.extend(b"\x00" * (64 - rem))
    off_layers_base = len(buf)
    layer_stride = 0

    # Keep dequantized reference layers only if verifying
    ref_layers: List[dict] = []
    layer_table_entries: List[List[int]] = []
    bias_table_entries: List[List[int]] = []

    proj_map = [
        ("wq", "self_attn.q_proj.weight"),
        ("wk", "self_attn.k_proj.weight"),
        ("wv", "self_attn.v_proj.weight"),
        ("wo", "self_attn.o_proj.weight"),
        ("w1", "mlp.gate_proj.weight"),
        ("w2", "mlp.down_proj.weight"),
        ("w3", "mlp.up_proj.weight"),
    ]

    for l in range(n_layers):
        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        l_start = len(buf)

        pfx = f"model.layers.{l}."
        rms_att = reader.get_f32(pfx + "input_layernorm.weight")
        rms_ffn = reader.get_f32(pfx + "post_attention_layernorm.weight")

        offs = [append_aligned(rms_att), append_aligned(rms_ffn)]
        rld: dict = {"rms_att": rms_att, "rms_ffn": rms_ffn}

        for short_name, hf_suffix in proj_map:
            mat = reader.get_f32(pfx + hf_suffix)
            if w4_layers:
                q, s, deq = quantize_weight_w4a8_g64(mat)
                offs.append(append_aligned(q))
                offs.append(append_aligned(s))
                if args.verify_prompt is not None:
                    rld[short_name + "_deq"] = deq
            else:
                q_raw, s = quantize_weight_per_row_pad64(mat)
                offs.append(append_aligned(pack_tile_major_64(q_raw)))
                offs.append(append_aligned(s))
                if args.verify_prompt is not None:
                    rld[short_name + "_deq"] = (q_raw.astype(np.float32) * s[:, None])

        rem = len(buf) % 64
        if rem:
            buf.extend(b"\x00" * (64 - rem))
        cur_stride = len(buf) - l_start
        if l == 0:
            layer_stride = cur_stride
        else:
            assert cur_stride == layer_stride, f"layer stride mismatch {cur_stride} != {layer_stride}"
        layer_table_entries.append(offs)

        if has_qkv_bias:
            bq = reader.get_f32(pfx + "self_attn.q_proj.bias")
            bk = reader.get_f32(pfx + "self_attn.k_proj.bias")
            bv = reader.get_f32(pfx + "self_attn.v_proj.bias")
            if args.verify_prompt is not None:
                rld["bq"], rld["bk"], rld["bv"] = bq, bk, bv
            bias_table_entries.append([bq, bk, bv])  # append after contiguous layer blob!

        if args.verify_prompt is not None:
            ref_layers.append(rld)

        if (l + 1) % 6 == 0 or l == n_layers - 1:
            print(f"[convert_hf] packed layer {l + 1}/{n_layers} ({len(buf) / (1024*1024):.1f} MiB)")

    # Append biases after the contiguous layer blob so layer_stride stays uniform for MSMC staging!
    if has_qkv_bias:
        for l, (bq, bk, bv) in enumerate(bias_table_entries):
            obq = append_aligned(bq)
            obk = append_aligned(bk)
            obv = append_aligned(bv)
            struct.pack_into("<4I", buf, off_bias_table + l * 16, obq, obk, obv, 0)

    rem = len(buf) % 64
    if rem:
        buf.extend(b"\x00" * (64 - rem))
    total_size = len(buf)

    for l, offs in enumerate(layer_table_entries):
        struct.pack_into("<16I", buf, off_layer_table + l * 64, *offs)

    flags = C7M_FLAG_ROPE_NEOX
    if not (w4_layers and w4_wcls):
        flags |= C7M_FLAG_TILED_WEIGHTS
    if w4_layers:
        flags |= C7M_FLAG_W4A8_LAYERS
    if w4_wcls:
        flags |= C7M_FLAG_W4A8_WCLS
    if has_qkv_bias:
        flags |= C7M_FLAG_HAS_QKV_BIAS
    if not tied:
        flags |= C7M_FLAG_UNTIED_EMB

    hdr = struct.pack(
        "<24If8I",
        C7M_MAGIC,
        C7M_VERSION,
        total_size,
        dim,
        dim_pad,
        hidden_dim,
        hidden_pad,
        n_layers,
        n_heads,
        n_kv_heads,
        head_size,
        kv_dim,
        vocab_size,
        vocab_pad,
        seq_len,
        off_rope_cos,
        off_rope_sin,
        off_rms_final,
        off_wcls_q,
        off_wcls_s,
        off_layer_table,
        flags,
        off_layers_base,
        layer_stride,
        rms_eps,
        64 if (w4_layers or w4_wcls) else 0,
        off_emb_q,
        off_emb_s,
        off_bias_table,
        off_tok,
        len(tok_blob),
        bos_id,
        eos_id,
    )
    buf[0 : len(hdr)] = hdr

    with open(args.out, "wb") as f:
        f.write(buf)
    print(
        f"[convert_hf] wrote {args.out}: {total_size:,} bytes ({total_size / (1024*1024):.2f} MiB), "
        f"layer_stride={layer_stride:,} B, flags=0x{flags:02x}"
    )

    # Optional Python reference verification + golden trace export
    if args.verify_prompt is not None and py_tok is not None:
        ptoks = c7t_encode_py(py_tok, args.verify_prompt)
        print(f"[verify-py] prompt tokens ({len(ptoks)}): {ptoks}")

        kc = np.zeros((n_layers, seq_len, kv_dim), dtype=np.float32)
        vc = np.zeros((n_layers, seq_len, kv_dim), dtype=np.float32)
        inv_hd = np.float32(1.0) / np.sqrt(np.float32(head_size), dtype=np.float32)
        kv_mul = n_heads // n_kv_heads

        def forward_one(tok_id: int, pos: int, rec: bool = False):
            x = emb_deq_ref[tok_id, :dim].astype(np.float32).copy()
            cr, sr = rope_cos[pos], rope_sin[pos]
            tr = {}
            for l in range(n_layers):
                rld = ref_layers[l]
                xb = rmsnorm(x, rld["rms_att"], eps=rms_eps)
                xq, xs = quantize_act_i8_pad64(xb)
                q = ((rld["wq_deq"][:dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                k = ((rld["wk_deq"][:kv_dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                v = ((rld["wv_deq"][:kv_dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                if has_qkv_bias:
                    q = (q + rld["bq"]).astype(np.float32)
                    k = (k + rld["bk"]).astype(np.float32)
                    v = (v + rld["bv"]).astype(np.float32)
                q = apply_rope_neox(q, cr, sr, n_heads, head_size)
                k = apply_rope_neox(k, cr, sr, n_kv_heads, head_size)
                kc[l, pos] = k
                vc[l, pos] = v

                xb_att = np.zeros((dim,), dtype=np.float32)
                for h in range(n_heads):
                    qh = q[h * head_size : (h + 1) * head_size]
                    kvh = h // kv_mul
                    kh = kc[l, : pos + 1, kvh * head_size : (kvh + 1) * head_size]
                    vh = vc[l, : pos + 1, kvh * head_size : (kvh + 1) * head_size]
                    att = np.empty((pos + 1,), dtype=np.float32)
                    for t in range(pos + 1):
                        att[t] = np.float32(hsum16_f32((qh * kh[t]).astype(np.float32)) * inv_hd)
                    probs = softmax(att)
                    oh = np.zeros((head_size,), dtype=np.float32)
                    for t in range(pos + 1):
                        oh = (oh + probs[t] * vh[t]).astype(np.float32)
                    xb_att[h * head_size : (h + 1) * head_size] = oh

                xq, xs = quantize_act_i8_pad64(xb_att)
                x = (x + (rld["wo_deq"][:dim] @ xq.astype(np.float32)) * xs).astype(np.float32)

                xb = rmsnorm(x, rld["rms_ffn"], eps=rms_eps)
                xq, xs = quantize_act_i8_pad64(xb)
                hb = ((rld["w1_deq"][:hidden_dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                hb2 = ((rld["w3_deq"][:hidden_dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                silu = (hb / (np.float32(1.0) + vexpf_f32((-hb).astype(np.float32)))).astype(np.float32)
                hb = (silu * hb2).astype(np.float32)
                xq, xs = quantize_act_i8_pad64(hb)
                x = (x + (rld["w2_deq"][:dim] @ xq.astype(np.float32)) * xs).astype(np.float32)
                if rec and l == 0:
                    tr["l0_out"] = x.copy()
                if rec and l == n_layers - 1:
                    tr["last_out"] = x.copy()
            x = rmsnorm(x, rms_final, eps=rms_eps)
            xq, xs = quantize_act_i8_pad64(x)
            logits = ((wcls_deq[:vocab_size] @ xq.astype(np.float32)) * xs).astype(np.float32)
            if rec:
                tr["final_norm"] = x.copy()
                tr["logits"] = logits.copy()
            return logits, tr

        gen_toks: List[int] = []
        first_tr = {}
        tok = ptoks[0]
        total_steps = len(ptoks) - 1 + args.steps
        for pos in range(total_steps):
            logits, tr = forward_one(tok, pos, rec=(pos == 0))
            if pos == 0:
                first_tr = tr
            if pos + 1 < len(ptoks):
                nxt = ptoks[pos + 1]
            else:
                nxt = int(np.argmax(logits))
            gen_toks.append(nxt)
            tok = nxt

        print("[verify-py] generated text:\n" + c7t_decode_py(py_tok, [ptoks[0]] + gen_toks))
        print(f"[verify-py] token IDs: {gen_toks}")

        if args.dump_golden_bin:
            with open(args.dump_golden_bin, "wb") as f:
                if len(ptoks) > 1:
                    f.write(struct.pack("<6I", 0x32444C47, dim, vocab_size, len(gen_toks), ptoks[0], len(ptoks)))
                else:
                    f.write(struct.pack("<5I", 0x31444C47, dim, vocab_size, len(gen_toks), ptoks[0]))
                f.write(np.array(gen_toks, dtype=np.uint32).tobytes())
                f.write(first_tr["l0_out"].astype(np.float32).tobytes())
                f.write(first_tr["last_out"].astype(np.float32).tobytes())
                f.write(first_tr["final_norm"].astype(np.float32).tobytes())
                f.write(first_tr["logits"].astype(np.float32).tobytes())
            print(f"[verify-py] saved golden trace to {args.dump_golden_bin}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert any HF Llama/SmolLM/Qwen2/Mistral safetensors model to .c7m")
    ap.add_argument("--hf-dir", required=True, help="Directory with config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--out", required=True, help="Output .c7m model path")
    ap.add_argument("--quant", choices=["w4a8", "hybrid", "w8a8"], default="w4a8")
    ap.add_argument("--seq-len", type=int, default=1024, help="Max sequence length for RoPE table & KV cache")
    ap.add_argument("--verify-prompt", default=None, help="Optional prompt to run NumPy reference & dump golden")
    ap.add_argument("--steps", type=int, default=16, help="Reference greedy steps after prompt")
    ap.add_argument("--dump-golden-bin", default=None, help="Output golden.bin path (requires --verify-prompt)")
    args = ap.parse_args()
    return convert_and_optionally_verify(args)


if __name__ == "__main__":
    sys.exit(main())
