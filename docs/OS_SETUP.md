# Board OS Setup: BeagleBoard Debian 6.18 vs. Yocto Poky 6.18

This repository inherits **two distinct ways to boot Linux 6.18 on the BeagleBone AI-64** from [`BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo`](https://github.com/kevinacahalan/BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo):

1. **Option A — Standard BeagleBoard Debian 6.18 (`bbai64-debian-13.7-iot-v6.18-k3`)**: flash the official BeagleBoard.org Debian image, cross-compile the C7x runtimes on your PC in containers (`build.sh`), and deploy them with `install-beagle.sh`.
2. **Option B — Custom Yocto Poky 6.18 (`platform/yocto/`, `meta-ai64`)**: build a minimal, self-contained Poky Linux image with `kas-container` that bakes the C7x firmware, CLI, ONNX EP, verification models, and DDR arena overlay directly into a flashable `.wic.xz` SD image.

Both paths run the **exact same bare-metal C7x firmware (`c7x-ai.elf`), DDR arena overlay (`0xac000000` + `0x880000000`), and runtime performance** on live silicon.

---

## Side-by-Side Comparison

| | Option A: BeagleBoard Debian 6.18 *(Recommended for most users)* | Option B: Yocto Poky 6.18 (`meta-ai64`) |
| --- | --- | --- |
| **Best for** | Fast setup, interactive development, `apt` packages, Python tools, compiling `llama.cpp` on-board | Minimal reproducible embedded image, fast clean reflashing (~71 MiB `.wic.xz`), CI/appliance deployments |
| **Kernel & U-Boot** | BeagleBoard.org `6.18.x-arm64-k3` (e.g. `6.18.55-arm64-k3-r59`) | TI `linux-ti-staging` (`6.18.51-ti`) + TI U-Boot `2026.01` (`TI_PREFERRED_BSP = "ti-6_18"`) |
| **Out-of-tree kernel modules needed?** | **None.** Unlike the 19-core parent demo (which patches `pru_rproc`/`ti_k3_r5_remoteproc`), this repo only drives the C7x DSP and needs zero kernel modules. | **None** for C7x (`linux-ti-staging` already sends `RP_MBOX_SHUTDOWN` on C7x stop). |
| **How runtimes get installed** | Built in Podman/Docker via `./platform/beagle-linux/scripts/build.sh` and installed via `sudo ./c7x-ai-beagle/install-beagle.sh` + reboot | Baked directly into `ai64-demo-image` by BitBake (no `build.sh`, no `install-beagle.sh`, no post-install reboot needed) |
| **Pre-installed runtime paths** | `install-beagle.sh` installs **Path 1** (`c7x-ai`), **Path 2** (`libggml-c7x.so`, when `models/llama.cpp` is present), and **Path 3** (`libort-c7x-ep.so`, `test_ort_c7x_ep`, 4 `.onnx` models) | Image bakes **Path 1** (`c7x-ai`), shared `c7x-ai.elf` + DTBO, and **Path 3** (`libort-c7x-ep.so`, `test_ort_c7x_ep`, `test_libc7x`, 4 `.onnx` models) + `libstdc++`; copy cross-compiled `llama.cpp` + `libggml-c7x.so` for **Path 2** |
| **Device-tree overlay** | Enabled automatically in `/boot/firmware/extlinux/extlinux.conf` by `install-beagle.sh` (requires 1 reboot) | Baked into `/boot/firmware/extlinux/extlinux.conf` and `/boot/firmware/overlays/` at image build time |
| **Package manager / on-board compilers** | Full Debian 13 (`apt`, `python3`, optional `cmake`/`g++` on-board) | Minimal `core-image-minimal` + `systemd` + `dropbear` (no `apt`, no on-board compiler; cross-compile on PC) |
| **Login & `/dev/mem` permissions** | Non-root user (`sysconf.txt`); `/dev/mem` is `0640 root:kmem`, so run `c7x-ai`, `llama.cpp`, and `test_ort_c7x_ep` with **`sudo`** | Passwordless **`root`** development login (runs directly without `sudo`; keep on a trusted bench network) |
| **First-time PC build cost** | ~5–10 min, ~6 GB disk (Podman/Docker images) | ~3 hours, ~150 GB disk first build (minutes on rebuilds or with shared `DL_DIR`/`SSTATE_DIR`) |
| **SD image size** | ~1.0 GB `.img.xz` (10 GB uncompressed; auto-resizes over 3 boots) | ~70 MiB `.wic.xz` (boots straight to prompt in one cycle) |

---

## Option A: BeagleBoard Debian 6.18 Setup

### 1. Flash the Official BeagleBoard Debian 6.18 Image

Download [BeagleBone AI-64 Debian 13.7 IoT, v6.18 k3](https://www.beagleboard.org/distros/bbai64-debian-13-7-2026-10-09-iot-v6-18-x-k3) (`bbai64-debian-13.7-iot-v6.18-k3-arm64-2026-10-09-10gb.img.xz`).

Use **BeagleBoard Imager** to write the microSD card and configure your username, password, hostname, and SSH public key — or flash manually with `dd` and `sysconf.txt`:

<details>
<summary>Flashing manually with <code>dd</code> and <code>sysconf.txt</code></summary>

1. Verify your card device with `lsblk`, then write the image:
   ```bash
   xz -dc bbai64-debian-13.7-iot-v6.18-k3-arm64-2026-10-09-10gb.img.xz \
     | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress
   ```
2. Mount partition 1 (`BOOT`, vfat) and append your first-boot settings to `sysconf.txt`:
   ```ini
   user_name=debian
   user_password=choose-a-password
   user_authorized_key=ssh-ed25519 AAAA... you@pc
   hostname=bbai64
   ```
3. Unmount partition 1.

</details>

Insert the microSD card and power on (hold the `BOOT` button if your board boots from eMMC). A fresh Debian card **reboots automatically while applying `sysconf.txt` and resizing the root filesystem** (~2–3 minutes). Once it settles, verify SSH:

```bash
ssh debian@bbai64.local 'uname -r'
# Example: 6.18.55-arm64-k3-r59
```

### 2. Build the Unified Bundle on Your PC

All cross-compilers (TI C7000 CGT `1.2.0.STS` + Debian 13 `aarch64-linux-gnu-gcc`/`g++` + `dtc`) run inside Podman/Docker containers — no host toolchains needed:

```bash
# Optional: clone unmodified upstream llama.cpp so libggml-c7x.so is included in the bundle
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git models/llama.cpp

# Build & run all x86_64 host-emulation tests, cross-compile all aarch64 targets, and package bundle
./platform/beagle-linux/scripts/build.sh all
./platform/beagle-linux/scripts/build.sh bundle
```

This creates `out/c7x-ai-beagle/` containing `c7x-ai`, `c7x-ai.elf`, `k3-j721e-beagleboneai64-c7x-ai.dtbo`, `install-beagle.sh`, `libggml-c7x.so` (if `llama.cpp` was present), `libort-c7x-ep.so`, `test_ort_c7x_ep`, and `models/` (all 4 ONNX models).

### 3. Deploy, Install, and Reboot

```bash
scp -r out/c7x-ai-beagle USER@BOARD:
ssh -t USER@BOARD 'sudo ./c7x-ai-beagle/install-beagle.sh && sudo reboot'
```

`install-beagle.sh`:
- Installs `/usr/bin/c7x-ai`, `/lib/firmware/c7x-ai.elf`, `/usr/lib/c7x-ai/{libggml-c7x.so,libort-c7x-ep.so,test_ort_c7x_ep}`, and `/usr/share/c7x-ai/models/*`.
- Validates `k3-j721e-beagleboneai64-c7x-ai.dtbo` against `/boot/firmware/ti/k3-j721e-beagleboneai64.dtb` when both `fdtoverlay` and the base DTB are present.
- Backs up `/boot/firmware/extlinux/extlinux.conf` to `extlinux.conf.c7x-ai-bak` (once) and appends `/overlays/k3-j721e-beagleboneai64-c7x-ai.dtbo` to the default boot label's `fdtoverlays` line (or the first label if no `default` line is set).

### 4. Verify & Run on Debian 6.18

After the board reboots (~30 seconds), confirm the DDR tensor arenas are reserved and test the C7x runtimes (note `sudo` is required on Debian because `/dev/mem` is `0640 root:kmem`):

```bash
# Confirm both DDR arenas (64 MiB @ 0xac000000 and 1.5 GiB @ 0x880000000) are reserved
ls /proc/device-tree/reserved-memory/ | grep c7x

# Load shared C7x firmware and run bit-exact hardware benchmarks (Path 1 + shared base)
sudo c7x-ai load c7x-ai.elf
sudo c7x-ai bench all

# Verify Path 3 (ONNX Vision & Audio EP, including hybrid C7x+CPU models)
sudo /usr/lib/c7x-ai/test_ort_c7x_ep /usr/share/c7x-ai/models 20
```

For **Path 2 (`llama.cpp` + `libggml-c7x.so`)**, you can either cross-compile `llama.cpp` on your PC or compile it directly on the Debian board (`sudo apt update && sudo apt install -y cmake build-essential`):

```bash
# Build unmodified llama.cpp with dynamic backend loading:
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target llama-bench llama-completion

# Run with the installed C7x plugin:
sudo GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so \
  ./build/bin/llama-bench -m model-Q4_0.gguf -ngl 99 -fa 1 -p 128 -n 32
```

> **Iterating on Debian:** After rebuilding `c7x-ai`, `c7x-ai.elf`, `libggml-c7x.so`, or `libort-c7x-ep.so`, re-run `build.sh bundle`, `scp`, and `install-beagle.sh` — **no reboot is needed** unless you changed the `.dtso` overlay; just run `sudo c7x-ai load c7x-ai.elf`.

---

## Option B: Yocto Poky 6.18 Setup (`platform/yocto/`)

If you prefer a minimal, self-contained embedded Linux image (~70 MiB compressed `.wic.xz`) with the C7x runtimes and overlay pre-installed at build time, use the `kas` + `meta-ai64` configuration in [`platform/yocto/`](../platform/yocto/README.md).

### 1. Build the Image on Your PC

A first build from scratch takes ~3 hours on a 16-thread host and needs ~150 GB free disk. From the repository root:

```bash
wget -O kas-container https://raw.githubusercontent.com/siemens/kas/5.5/kas-container
chmod +x kas-container
KAS_BUILD_DIR=platform/yocto/build ./kas-container build platform/yocto/kas/ai64-poky.yml
```

- Uses `podman` or `docker` (override via `KAS_CONTAINER_ENGINE=podman` or `docker`).
- **Reusing existing Yocto caches:** `ti-poky-wrynose-config.yml` sets `DL_DIR ?=` and `SSTATE_DIR ?=`, so you can pass `DL_DIR=/path/to/downloads SSTATE_DIR=/path/to/sstate-cache` (for example from a checkout of `BeagleBoneAI64-linux-zephyr-pru-c7x-c66x-demo`) to cut rebuild time from ~3 hours to a few minutes.

### 2. Flash the `.wic.xz` Image

Verify `/dev/sdX` with `lsblk`, then flash the generated image:

```bash
xz -dc platform/yocto/build/deploy-ti/images/beaglebone-ai64/ai64-demo-image-beaglebone-ai64.rootfs.wic.xz \
  | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress
```

### 3. Boot & Run Immediately (No Install Script or Reboot Needed)

Boot the board and log in as `root` (no password; DHCP on Ethernet via Dropbear SSH, or serial console `ttyS2` 115200 8N1). Everything below is already baked into the image:

```bash
# Confirm kernel and reserved DDR arenas
uname -r                                            # 6.18.51-ti-...
ls /proc/device-tree/reserved-memory/ | grep c7x

# Path 1 + shared DSP firmware (pre-installed)
c7x-ai load c7x-ai.elf
c7x-ai bench all

# Path 3: ONNX Vision & Audio EP + libc7x self-test (pre-installed)
/usr/lib/c7x-ai/test_libc7x
/usr/lib/c7x-ai/test_ort_c7x_ep /usr/share/c7x-ai/models 20
```

For **Path 2 (`llama.cpp` + `libggml-c7x.so`)** on Yocto (which has no on-board compiler), clone unmodified `llama.cpp`, build `libggml-c7x.so` with `build.sh ggml`, cross-compile `llama.cpp` for `aarch64` (using your host `cmake` + `aarch64-linux-gnu-gcc`/`g++`, or a Debian 13 container with `cmake` + `g++-aarch64-linux-gnu`), and copy the outputs to the board (`libstdc++` is already included in `ai64-demo-image` via `c7x-ort-ep`):

```bash
# On PC (from repo root):
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git models/llama.cpp
./platform/beagle-linux/scripts/build.sh ggml

# Cross-compile unmodified llama.cpp with dynamic backend loading:
cmake -S models/llama.cpp -B out/llama-aarch64 \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build out/llama-aarch64 -j"$(nproc)" --target llama-bench llama-completion

# Copy plugin, llama.cpp binaries/shared libs, and a GGUF model to the Yocto board:
ssh root@BOARD 'mkdir -p /tmp/llama-c7x'
scp frontends/ggml-c7x/build-aarch64/libggml-c7x.so root@BOARD:/usr/lib/c7x-ai/
scp out/llama-aarch64/bin/llama-bench out/llama-aarch64/bin/llama-completion \
    out/llama-aarch64/bin/lib*.so* root@BOARD:/tmp/llama-c7x/
scp model-Q4_0.gguf root@BOARD:/tmp/

# On the Yocto board:
GGML_BACKEND_PATH=/usr/lib/c7x-ai/libggml-c7x.so LD_LIBRARY_PATH=/tmp/llama-c7x \
  /tmp/llama-c7x/llama-bench -m /tmp/model-Q4_0.gguf -ngl 99 -fa 1 -p 128 -n 32
```

*(You can also deploy the full `out/c7x-ai-beagle/` bundle with `./c7x-ai-beagle/install-beagle.sh` on Yocto to install `libggml-c7x.so` or update firmware/binaries in place without reflashing — `install-beagle.sh` is BusyBox-compatible and leaves the already-enabled overlay entry untouched.)*

> [!WARNING]
> `ai64-demo-image` is a development image with passwordless `root` SSH login. Set a root password (`passwd`) or keep the board on an isolated bench network.

---

## Under the Hood: Two 6.18 Kernels

Both options run Linux 6.18, but from different trees:

- **BeagleBoard Debian 6.18** runs BeagleBoard.org's `arm64-k3` kernel (`6.18.55-arm64-k3-r59`). Its `ti_k3_dsp_remoteproc` driver does **not** send `RP_MBOX_SHUTDOWN` before stopping the C7x core.
- **Yocto Poky 6.18** runs TI's `linux-ti-staging` (`6.18.51-ti`) with TI U-Boot `2026.01`. Its `ti_k3_dsp_remoteproc` driver **does** send `RP_MBOX_SHUTDOWN` and waits for the C7x to acknowledge and enter `IDLE`.

`c7x-ai stop` and `c7x-ai load` handle both transparently: they attempt `remoteproc stop` first (which completes via TI's mailbox handshake on Yocto) and, if the kernel rejects the stop before shutdown (as on BeagleBoard Debian 6.18), send `RP_MBOX_SHUTDOWN` (`0xFFFFFF14`) directly over the hardware mailbox and retry `remoteproc stop`. On Debian 6.18 you may see a single benign `k3-dsp-rproc 64800000.dsp: module-reset assert failed (-ENODEV)` line in `dmesg` right before `stopped remote processor 64800000.dsp` — the stop and restart still succeed cleanly with zero patched kernel modules required.

---

## Live Hardware Verification (Both OS Paths)

Both OS paths were freshly flashed and verified end-to-end on the same BeagleBone AI-64:

| Check | BeagleBoard Debian 13.7 (`6.18.55-arm64-k3-r59`) | Yocto Poky 6.0.4 (`6.18.51-ti`, `ai64-demo-image`) |
| --- | --- | --- |
| DDR arenas (`0xac000000`, `0x880000000`) | Active after `install-beagle.sh` + reboot | Active out-of-the-box on first boot |
| `c7x-ai load` + 3× `stop`/`load` restarts | **PASS** (zero patched kernel modules) | **PASS** (zero kernel warnings) |
| `c7x-ai bench all` (MMA/DRU/MSMC/DDR) | **PASS BIT-EXACT** (`ping=2.62 µs`, `DRU DDR→L2 10.92 GB/s`) | **PASS BIT-EXACT** (`ping=2.62 µs`, `DRU DDR→L2 10.92 GB/s`) |
| `test_ort_c7x_ep /usr/share/c7x-ai/models 20` | **PASS** (`mobilenet_qlinear` `12.22 ms`, `mobilenet_e2e_hyb` `13.61 ms`, `whisper_erf_hyb` `5.04 ms`, `QLinearMatMul` `19.6 GOPS`) | **PASS** (`mobilenet_qlinear` `12.23 ms`, `mobilenet_e2e_hyb` `13.62 ms`, `whisper_erf_hyb` `5.07 ms`, `QLinearMatMul` `19.7 GOPS`; plus `test_libc7x` PASS) |
| `test_ggml_c7x_aarch64` + `llama-bench` (`SmolLM2-135M Q4_0 -fa 1`) | **PASS** · **`185.81 t/s`** prefill / **`55.21 t/s`** decode | **PASS** · **`185.29 t/s`** prefill / **`55.39 t/s`** decode |
