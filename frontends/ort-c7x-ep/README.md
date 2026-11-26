# Path 3: `ort-c7x-ep` — ONNX Vision & Audio Execution Provider (`frontends/ort-c7x-ep/`)

Zero-dependency standard `.onnx` binary protobuf loader, graph partitioner/compiler, and C7x execution provider (`libort-c7x-ep.so`) built on [`runtime/libc7x`](../../runtime/libc7x/). Lowers maximal contiguous C7x subgraphs into single `C7X_OP_EXEC_LIST` RPCs on the BeagleBone AI-64 C7x + MMA, with optional automatic CPU fallback (`allow_cpu_fallback = 1`) and an ONNX Runtime C API shim (`OrtSessionOptionsAppendExecutionProvider_C7x`).

## Live Silicon Highlights

| ONNX Workload | Graph / Partitioning | Cmds / SGs | Accuracy | C7x DSP / Wall | Throughput / Speedup |
| --- | --- | ---: | --- | ---: | --- |
| `mobilenet_qlinear.onnx` | 4-layer INT8/UINT8 `QLinearConv` + `DequantizeLinear` + `Relu` + `GAP` + `Gemm` | `9` cmds / `1` SG | `cos = 1.000000` | **`12.22 ms`** / `12.30 ms` | **`81.6 FPS`** |
| `mobilenet_e2e_hybrid.onnx` | `QuantizeLinear` (**CPU**) → 4L `QLinearConv`+`GAP`+`Gemm` (**C7x**) → `Softmax` (**CPU**) | `9` cmds / `1` SG + `2` CPU | `cos = 1.000000` (`err=2.7e-5`) | `12.24 ms` / **`13.83 ms`** | **`72.3 FPS`** |
| `audio_enc_block.onnx` | Whisper-style `Conv 3×3 s=2` + `Gelu(tanh)` + `LayerNormalization` + FFN + residual `Add` | `7` cmds / `1` SG | **`max_err = 0.000000`** | **`1.13 ms`** / `1.75 ms` | **`882.6 FPS`** (DSP) |
| `whisper_erf_hybrid.onnx` | `Conv` (**C7x**) → `erf-Gelu` (**CPU**) → `LN+MatMul` (**C7x**) → `erf-Gelu` (**CPU**) → `MatMul+Add` (**C7x**) → `Softmax` (**CPU**) | `5` cmds / `3` SGs + `3` CPU | **`max_err = 0.000000`** | `1.06 ms` / **`5.25 ms`** | **`190.6 FPS`** |
| `QLinearMatMul 60×256×256` | Asymmetric `UINT8 (zp=133) × INT8 (zp=-7) → UINT8 (zp=120)`, per-channel scales, `int32` bias | `1` cmd | **`max_qdiff = 0`** | **`0.400 ms`** / `0.405 ms` | **`19.6 GOPS` (`4.75×` vs A72)** |

## Supported ONNX Operators
- **C7x DSP — Asymmetric Quantized INT8/UINT8:** `QLinearConv` (standard MMA `group=1` + depthwise `group=IC=OC`, per-tensor & per-channel scales, uniform zero-point), `QLinearMatMul`, `DequantizeLinear` (per-tensor & per-axis).
- **C7x DSP — Float32 Vision/Audio/MLP:** `Conv` (2D), `MatMul`, `Gemm` (`transA=0, alpha=1, beta=1`), `LayerNormalization` (`axis=-1`), `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Add`, `Mul`, `Relu`, `Gelu` (`approximate="tanh"`), `Sigmoid`, `Tanh`.
- **Zero-copy shape ops:** `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze` (including opset 13+ input `axes`).
- **Hybrid CPU Fallback (`opts.allow_cpu_fallback = 1`):** `QuantizeLinear`, `Softmax`, exact erf `Gelu` (`approximate="none"`), `Clip`, arbitrary-broadcast `Add`/`Sub`/`Mul`/`Div`, `Transpose`, `Concat`, `ReduceMean`, `Resize` (nearest/linear). Contiguous C7x ops automatically batch into single `EXEC_LIST` RPCs around CPU stages; default `allow_cpu_fallback = 0` preserves strict C7x-only rejection.
- **ONNX Runtime C API Shim (`src/ort_c_api_shim.cpp`):** Exposes `OrtSessionOptionsAppendExecutionProvider_C7x` (registers `com.beagleboard.c7x` custom op domain with `C7xCompiledGraph`). Build into `libort-c7x-ep.so` with `make hw ORT_INC=/path/to/onnxruntime/include`; verified in hostemu by `test_ort_c_api_shim`.

## Build & Run

```bash
# Build & run host-emulation tests (generates all 4 .onnx models + runs test_ort_c7x_ep & test_ort_c_api_shim)
./platform/beagle-linux/scripts/build.sh ort
```

On the BeagleBone AI-64 (installed by `install-beagle.sh`, after `c7x-ai load c7x-ai.elf`):

```bash
/usr/lib/c7x-ai/test_ort_c7x_ep /usr/share/c7x-ai/models 20
```

See [`include/c7x_ort_ep.h`](include/c7x_ort_ep.h) for the C API (`c7x_ort_session_create`, `c7x_ort_session_partition_info`, `c7x_ort_session_run`, `c7x_ort_session_destroy`).
