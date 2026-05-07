# AGENTS.md

TL;DR for AI coding agents (Claude Code, Cursor, Copilot, …) picking up this repo.

## What this is

A single-process orchestrator (`morpheus_cam.py`) that runs a real-time, body-driven Stable Diffusion pipeline by spawning workers on every silicon device of an AMD Ryzen AI 300 + NVIDIA hybrid laptop:

1. Captures webcam frames in a thread (RGB + a zero-cost "fake" SD latent).
2. Spawns an **iGPU worker process** that runs Depth Anything v2 (FP16 DML) on the webcam and TAESD decoder on SD latents — both via `onnxruntime-directml` device id 1.
3. Runs an **NPU thread** in the main process that calls `StableDiffusionONNXPipelineAMD` (AMD Ryzen AI 1.7.1 SDK) with `vaiml_compile_v4` to do SD Turbo UNet inference at fp32 on the XDNA2 NPU. Depth from step 2 modulates the initial noise.
4. Spawns a **CUDA RIFE subprocess** in conda env `cuda` that flow-interpolates consecutive SD frames into 9-frame bursts (n_passes=3) for ~30 fps display.
5. Serves an **HTML control panel** (`control/server.py` + `control/panel.html`) on `127.0.0.1:54331` with WebSocket for live tweaks + a hardware sampler.
6. Outputs **three Spout senders** — `AMD_AI_SD`, `AMD_AI_SD_DEPTH`, `AMD_AI_RIFE`.

This is **not** a pip-installable library. It's a runner. Clone, install two conda envs, run.

## Read these first

1. `README.md` — full spec, hardware spotlight, the NPU strobe story, performance numbers.
2. `docs/ARCHITECTURE.md` — process model, SHM layout, render loop, frame format conventions.

## Run it

```bash
# env "npu" — main process (Ryzen AI 1.7.1 base)
conda activate npu
python morpheus_cam.py
python morpheus_cam.py --webcam 1 --http-port 54331
```

Open <http://127.0.0.1:54331/> in a browser for the control panel.

## Module map

| File | Role |
|---|---|
| `morpheus_cam.py` | Entry point — orchestrator + cam thread + NPU thread + display loop. Spawns iGPU process and RIFE subprocess. |
| `spout_sender.py` | `SpoutOutput`: thin wrapper around `UnveilStudio/SPOUT2ForPython`. Lazy-imports `spout`. RGBA buffer reuse, auto-resolution. |
| `download_models.py` | Fetches TAESD + Depth Anything v2 + RIFE weights from HuggingFace into `models/`. |
| `control/server.py` | `CamRifeControlState`, HTTP for `panel.html`, WebSocket `:54332`, hardware sampler thread (PowerShell + nvidia-smi + psutil). |
| `control/panel.html` | Browser UI — violet `#a855f7` accent, prompt textarea, sliders, hw bars, d_mean sparkline. |
| `rife/flow_worker.py` | RIFE CUDA worker subprocess (env `cuda`). Reads ctrl[2]=image_fid, writes ctrl[4]=display_fid, publishes 9 frames per pair into `amd_cam_ring`. |
| `rife/model.py` | `RIFEInterpolator`: PyTorch wrapper around `IFNet_HDv3` (Practical-RIFE, MIT). |
| `assets/build_banner.py` | Regenerates `banner.png` from `unveil_logo.png` + Segoe UI fonts. |

## Two conda envs (do NOT mix)

| Env | Activation | Use |
|---|---|---|
| `npu` | `conda run -n npu python ...` | Main process, NPU SD Turbo, iGPU TAESD/Depth (DML). Comes from Ryzen AI 1.7.1 `env.yaml`. **Golden env — do not modify.** |
| `cuda` | `conda run -n cuda --no-capture-output python ...` | Only for `rife/flow_worker.py`. PyTorch CUDA 12.x. |

`onnxruntime-vitisai` (NPU) and `onnxruntime-directml` (iGPU) ship with the Ryzen AI 1.7.1 SDK. Mixing pip wheels into the conda env breaks the VitisAI EP — DLL conflicts, silent CPU fallback. Don't.

## Common pitfalls

- **`VAIP_CONFIG_JSON` not set** — VitisAI EP silently falls back to CPU and you get garbage SD output at 1 fps. Always set:
  ```
  $env:VAIP_CONFIG_JSON = "C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"
  ```
- **NPU strobe bug** — INT8-quantised UNet inference flip-flops NaN every other call deterministically. We're already on fp32 + `vaiml_compile_v4` (zero-NaN verified). The `unet_step()` retry loop is a safety net — do not remove it.
- **DML device-id** — on hybrid laptops device 0 = NVIDIA, device 1 = AMD iGPU. The iGPU worker hard-codes `device_id: 1`. Don't "fix" this.
- **`models/sd/sd_turbo/` empty** — `download_models.py` does not fetch the SD Turbo NPU build. Compile via Ryzen AI SDK GenAI-SD quicktest, or pull the pre-built drop. See README "NPU model setup".
- **`models/cache/` stale** — clear it after updating any `.onnx` file under `models/`. The XDNA2 compiler caches by file path, not content hash, so an updated weight will reuse the old compiled binary.
- **`from src.StableDiffusionONNXPipelineTrigger import …` ImportError** — means the Ryzen AI SDK GenAI-SD path is missing from `sys.path`. The entry point inserts `C:\Program Files\RyzenAI\1.7.1\GenAI-SD` automatically; if your install path differs, edit `GENAI_SD_DIR` at the top of `morpheus_cam.py`.
- **Webcam already in use** — `cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)` fails if Zoom / Teams / Discord has the camera. Close them or use a different `--webcam` id.
- **3 conda processes lingering after Ctrl+C** — the orchestrator joins iGPU process + RIFE subprocess on shutdown, but if you SIGKILL the parent, the RIFE conda subprocess can leak. PowerShell:
  ```powershell
  Get-Process python | Where-Object { $_.MainWindowTitle -match 'morpheus|rife' } | Stop-Process
  ```
- **`Spout In TOP` shows nothing in TouchDesigner** — Sender has to be selected from the dropdown after the python side started. TD doesn't auto-discover senders that come up after it.

## Don't touch

- `morpheus_cam.py:unet_step()` retry loop — safety net for the NPU strobe bug. We're on the strobe-free codepath but the guard is free insurance.
- The `vaiml_compile_v4` configuration / sys.path GenAI-SD insertion order — the AMD SDK pipeline imports things in a specific order and re-ordering breaks it.
- `control/server.py:_HardwareSampler.PS_SCRIPT` — the long-lived PowerShell loop. Replacing it with per-tick `subprocess.run(["powershell", ...])` adds ~250 ms per sample (cold start of `powershell.exe`) and starves the WS pump.
- The `64-byte aligned` cam latent fake-encoding (`cam_thread()` in `morpheus_cam.py`) — `lat_rgba = np.concatenate([img_small, lum], axis=-1) * 2.0` is calibrated to roughly match real SD latent magnitudes. Changing the gain breaks the cam_mix slider.

## Tests

There's no formal test suite yet. Smoke check:

```bash
python -c "import ast, glob; [ast.parse(open(p, encoding='utf-8').read()) for p in glob.glob('**/*.py', recursive=True)]"
python morpheus_cam.py --help
```

Live verification:
1. Start the pipeline.
2. Open the panel — `connected` should appear within ~1 s.
3. After ~10 s the d_mean sparkline should be moving slightly (depth latent).
4. Within ~30 s `SD fps` should settle around `3.5 - 4.0` and `ms / step` around `230 - 270 ms`.
5. Wave a hand at the webcam — d_mean should pulse visibly on the sparkline.

If `SD fps` reads `0` for more than a minute: VAIP_CONFIG_JSON is unset, or the NPU build is missing. See README.
