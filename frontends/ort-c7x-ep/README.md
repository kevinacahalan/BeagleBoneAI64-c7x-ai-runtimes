# Path 3: `ort-c7x-ep` — ONNX Vision & Audio Execution Provider (`frontends/ort-c7x-ep/`)

Zero-dependency standard `.onnx` binary protobuf loader, graph compiler, and C7x execution provider (`libort-c7x-ep.so`) built on [`runtime/libc7x`](../../runtime/libc7x/). Compiles multi-layer quantized vision CNNs and audio encoder blocks into a **single `C7X_OP_EXEC_LIST` RPC** on the BeagleBone AI-64 C7x + MMA.

## Live Silicon Highlights

| ONNX Workload | Graph | Cmds / RPC | Accuracy | C7x DSP Latency | Throughput / Speedup |
| --- | --- | ---: | --- | ---: | --- |
| `mobilenet_qlinear.onnx` | 4-layer INT8/UINT8 `QLinearConv` (`3×3 s=2` → DW `3×3 s=2` → PW `1×1` → `3×3 s=2`) + `DequantizeLinear` + `Relu` + `GAP` + `Gemm` | `9` | `cos = 1.000000` | **`12.23 ms`** | **`81.6 FPS`** |
| `audio_enc_block.onnx` | Whisper-style `Conv 3×3 s=2` + `Gelu(tanh)` + `LayerNormalization` + FFN + residual `Add` | `7` | **`max_err = 0.000000`** | **`1.13 ms`** | **`882.6 FPS`** |
| `QLinearMatMul 60×256×256` | Asymmetric `UINT8 (zp=133) × INT8 (zp=-7) → UINT8 (zp=120)`, per-channel scales, `int32` bias | `1` | **`max_qdiff = 0`** | **`0.401 ms`** | **`19.6 GOPS` (`4.75×` vs A72)** |

## Supported ONNX Operators
- **Asymmetric Quantized INT8/UINT8:** `QLinearConv` (standard MMA `group=1` + depthwise `group=IC=OC`, per-tensor & per-channel scales, uniform zero-point), `QLinearMatMul`, `DequantizeLinear` (per-tensor & per-axis).
- **Float32 Vision/Audio/MLP:** `Conv` (2D), `MatMul`, `Gemm` (`transA=0, alpha=1, beta=1`), `LayerNormalization` (`axis=-1`), `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Add`, `Mul` (with broadcast validation), `Relu`, `Gelu` (`approximate="tanh"`), `Sigmoid`, `Tanh`.
- **Zero-copy shape ops:** `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze` (including opset 13+ input `axes`).
- **Strict validation:** Rejects external data tensors, symbolic `dim_param`, non-uniform per-channel zero-points, unsupported `ceil_mode`/`dilations`/`auto_pad`, and bidirectional broadcast cleanly instead of silently miscomputing.

## Build & Run

```bash
# Build & run host-emulation test suite (generates .onnx test models automatically)
./platform/beagle-linux/scripts/build.sh ort

# Outputs:
#   frontends/ort-c7x-ep/build-aarch64/libort-c7x-ep.so
#   frontends/ort-c7x-ep/build-aarch64/test_ort_c7x_ep
#   frontends/ort-c7x-ep/build-hostemu/models/{mobilenet_qlinear,audio_enc_block}.onnx
```

On the BeagleBone AI-64 (after `c7x-ai load c7x-ai.elf`):

```bash
./test_ort_c7x_ep ./models 20
```

See [`include/c7x_ort_ep.h`](include/c7x_ort_ep.h) for the C API (`c7x_ort_session_create`, `c7x_ort_session_run`, `c7x_ort_session_destroy`).
