#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate reproducible dark-mode and light-mode SVG benchmark charts for docs/assets/
from measured live BeagleBone AI-64 (TDA4VM) silicon results in docs/RESULTS.md.

Zero third-party dependencies (Python stdlib only).

Usage:
    python3 tools/gen_charts.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

THEMES = {
    "dark": {
        "bg": "#0d1117",
        "card": "#161b22",
        "border": "#30363d",
        "track": "#21262d",
        "fg": "#e6edf3",
        "muted": "#8b949e",
        "cpu": "#6e7681",
        "ggml": "#388bfd",
        "mono": "#2ea043",
        "ort": "#a371f7",
        "accent": "#58a6ff",
    },
    "light": {
        "bg": "#ffffff",
        "card": "#f6f8fa",
        "border": "#d0d7de",
        "track": "#eaeef2",
        "fg": "#1f2328",
        "muted": "#656d76",
        "cpu": "#8c959f",
        "ggml": "#0969da",
        "mono": "#1a7f37",
        "ort": "#8250df",
        "accent": "#0969da",
    },
}

PREFILL_DATA = [
    ("SmolLM2-135M", 46.2, 175.1, 323.9, "54% of .c7m · 3.8x A72"),
    ("Qwen2.5-0.5B", 14.0, 118.3, 168.7, "70% of .c7m · 8.5x A72"),
    ("Qwen3-0.6B",   12.7,  68.1, 104.5, "65% of .c7m · 5.4x A72"),
    ("Qwen2.5-1.5B",  4.9,  63.0,  71.6, "88% of .c7m · 12.8x A72"),
]

DECODE_DATA = [
    ("SmolLM2-135M", 28.6, 47.7, 115.7, "41% of .c7m · 1.7x A72"),
    ("Qwen2.5-0.5B",  8.6, 14.3,  35.1, "41% of .c7m · 1.7x A72"),
    ("Qwen3-0.6B",    7.8, 13.1,  27.2, "48% of .c7m · 1.7x A72"),
    ("Qwen2.5-1.5B",  3.6,  8.3,  11.6, "72% of .c7m · 2.3x A72"),
]


def render_llm_tradeoff_svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    w, h = 880, 476
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="BeagleBone AI-64 Live Silicon LLM Performance Tradeoff">',
        "<defs>",
        '  <style>',
        '    .sans { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }',
        '    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }',
        "  </style>",
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{t["card"]}" stroke="{t["border"]}"/>',
        # Header
        f'<text x="24" y="34" class="sans" font-size="16" font-weight="700" fill="{t["fg"]}">Live BeagleBone AI-64 Silicon: Native .c7m vs. llama.cpp Plugin vs. Cortex-A72 CPU</text>',
        f'<text x="24" y="54" class="sans" font-size="12" fill="{t["muted"]}">Throughput in tok/s (higher is better) · Each model normalized to its fastest bar so the ratio is visible across 135M–1.5B</text>',
        # Legend
        f'<rect x="24" y="70" width="12" height="12" rx="2" fill="{t["cpu"]}"/>',
        f'<text x="42" y="80" class="sans" font-size="12" fill="{t["muted"]}">2T Cortex-A72 CPU (llama.cpp)</text>',
        f'<rect x="250" y="70" width="12" height="12" rx="2" fill="{t["ggml"]}"/>',
        f'<text x="268" y="80" class="sans" font-size="12" fill="{t["muted"]}">Path 2: Unmodified llama.cpp + libggml-c7x.so</text>',
        f'<rect x="572" y="70" width="12" height="12" rx="2" fill="{t["mono"]}"/>',
        f'<text x="590" y="80" class="sans" font-size="12" fill="{t["muted"]}">Path 1: Native .c7m (c7x-ai)</text>',
        # Divider
        f'<line x1="24" y1="96" x2="{w - 24}" y2="96" stroke="{t["border"]}" stroke-width="1"/>',
    ]

    panels = [
        ("Prompt Prefill (pp128 vs. pp146–160)", PREFILL_DATA, 24),
        ("Token Decode (tg32 vs. short-ctx)", DECODE_DATA, 456),
    ]

    bar_x_rel = 88
    bar_w = 248
    val_x_rel = bar_x_rel + bar_w + 56

    for title, data, x0 in panels:
        lines.append(
            f'<text x="{x0}" y="122" class="sans" font-size="13" font-weight="700" fill="{t["fg"]}">{escape(title)}</text>'
        )
        y = 148
        for model, cpu, ggml, mono, note in data:
            lines.append(
                f'<text x="{x0}" y="{y}" class="mono" font-size="12" font-weight="700" fill="{t["fg"]}">{escape(model)}</text>'
            )
            lines.append(
                f'<text x="{x0 + val_x_rel}" y="{y}" text-anchor="end" class="sans" font-size="11" fill="{t["muted"]}">{escape(note)}</text>'
            )
            rows = [
                ("A72 CPU", cpu, t["cpu"]),
                ("ggml-c7x", ggml, t["ggml"]),
                ("native .c7m", mono, t["mono"]),
            ]
            ry = y + 8
            for label, val, color in rows:
                fw = max(3.0, (val / mono) * bar_w)
                lines.append(
                    f'<text x="{x0 + bar_x_rel - 8}" y="{ry + 10}" text-anchor="end" class="sans" font-size="11" fill="{t["muted"]}">{label}</text>'
                )
                lines.append(
                    f'<rect x="{x0 + bar_x_rel}" y="{ry}" width="{bar_w}" height="12" rx="3" fill="{t["track"]}"/>'
                )
                lines.append(
                    f'<rect x="{x0 + bar_x_rel}" y="{ry}" width="{fw:.1f}" height="12" rx="3" fill="{color}"/>'
                )
                lines.append(
                    f'<text x="{x0 + val_x_rel}" y="{ry + 10}" text-anchor="end" class="mono" font-size="11" font-weight="600" fill="{t["fg"]}">{val:.1f}</text>'
                )
                ry += 17
            y += 74

    lines.append(
        f'<text x="24" y="{h - 14}" class="sans" font-size="11" fill="{t["muted"]}">Source: docs/RESULTS.md · BeagleBone AI-64 (TI TDA4VM C71x+MMA @ 1.0 GHz vs. dual Cortex-A72 @ 2.0 GHz)</text>'
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


def render_non_llm_svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    w, h = 880, 260
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="BeagleBone AI-64 Live Silicon Vision and Audio ONNX Results">',
        "<defs>",
        '  <style>',
        '    .sans { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }',
        '    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }',
        "  </style>",
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{t["card"]}" stroke="{t["border"]}"/>',
        f'<text x="24" y="34" class="sans" font-size="16" font-weight="700" fill="{t["fg"]}">Path 3 (ONNX Vision &amp; Audio) + ggml-c7x Non-LLM Ops on Live TDA4VM Silicon</text>',
        f'<text x="24" y="54" class="sans" font-size="12" fill="{t["muted"]}">Single C7X_OP_EXEC_LIST RPC per ONNX graph via ort-c7x-ep · Zero OpenVX / TIDL overhead</text>',
        f'<line x1="24" y1="68" x2="{w - 24}" y2="68" stroke="{t["border"]}" stroke-width="1"/>',
    ]

    cards = [
        (
            "mobilenet_qlinear.onnx",
            "81.6 FPS",
            "12.23 ms DSP · 9 cmds / 1 RPC",
            "4L INT8/UINT8 QLinearConv + GAP + Gemm",
            "cos = 1.000000",
            t["ort"],
        ),
        (
            "audio_enc_block.onnx",
            "882.6 FPS",
            "1.13 ms DSP · 7 cmds / 1 RPC",
            "Conv 3x3 s=2 + Gelu + LayerNorm + FFN",
            "max_err = 0.000000",
            t["ggml"],
        ),
        (
            "QLinearMatMul 60x256x256",
            "4.75x vs A72",
            "0.401 ms DSP vs 1.90 ms A72",
            "UINT8 x INT8 -> UINT8 · 19.6 GOPS",
            "max_qdiff = 0 (exact)",
            t["mono"],
        ),
    ]

    cw = 264
    gap = 20
    for idx, (title, big, sub1, sub2, badge, color) in enumerate(cards):
        cx = 24 + idx * (cw + gap)
        cy = 84
        ch = 136
        lines.append(
            f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="{t["bg"]}" stroke="{t["border"]}"/>'
        )
        lines.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="4" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{cx + 14}" y="{cy + 26}" class="mono" font-size="12" font-weight="700" fill="{t["fg"]}">{escape(title)}</text>'
        )
        lines.append(
            f'<text x="{cx + 14}" y="{cy + 58}" class="sans" font-size="24" font-weight="700" fill="{color}">{escape(big)}</text>'
        )
        lines.append(
            f'<text x="{cx + 14}" y="{cy + 79}" class="mono" font-size="11" fill="{t["fg"]}">{escape(sub1)}</text>'
        )
        lines.append(
            f'<text x="{cx + 14}" y="{cy + 98}" class="sans" font-size="10.5" fill="{t["muted"]}">{escape(sub2)}</text>'
        )
        lines.append(
            f'<text x="{cx + 14}" y="{cy + 120}" class="mono" font-size="11" font-weight="600" fill="{t["muted"]}">Accuracy: {escape(badge)}</text>'
        )

    lines.append(
        f'<text x="24" y="{h - 14}" class="sans" font-size="11" fill="{t["muted"]}">Plus ggml-c7x Vision/Audio ops on live silicon: test-backend-ops NORM|IM2COL|POOL_2D 710/710 PASS · CONV_2D 1,574/1,574 PASS (2,030/2,030 hostemu)</text>'
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for theme in ("dark", "light"):
        (ASSETS / f"llm-tradeoff-{theme}.svg").write_text(render_llm_tradeoff_svg(theme), encoding="utf-8")
        (ASSETS / f"non-llm-bench-{theme}.svg").write_text(render_non_llm_svg(theme), encoding="utf-8")
    print(f"[gen_charts] wrote 4 SVGs to {ASSETS}")


if __name__ == "__main__":
    main()
