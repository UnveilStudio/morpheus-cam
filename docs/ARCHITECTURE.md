# Architecture

Cross-checked against the source: `morpheus_cam.py`, `control/server.py`, `rife/flow_worker.py`, `rife/model.py`, `spout_sender.py`. Read this with the code open in another tab.

## Process model

morpheus-cam splits work across **three OS processes** plus daemon threads inside the main one. There is no shared GIL across the boundaries — DML, VitisAI and CUDA stacks are isolated as required by ONNX Runtime threading rules and conda env separation.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       MAIN PROCESS  (conda env: npu)                     │
│                                                                          │
│   thread  cam_thread     ─► amd_cam_webcam, amd_cam_camlat              │
│   thread  npu_thread     ─► amd_cam_latent              (SD Turbo on    │
│                                                          XDNA2 NPU,     │
│                                                          ~250 ms/step)  │
│   thread  http server    ─► serves panel.html                           │
│   thread  ws  server     ─► sends/receives panel state + metrics        │
│   thread  hw  sampler    ─► CPU/RAM (psutil) + iGPU/dGPU (PowerShell)   │
│                                                          + dGPU (smi)   │
│   main loop              ─► display ring read + Spout AMD_AI_RIFE       │
└──────────────────────────────────────────────────────────────────────────┘
        │ amd_cam_*                                       │ amd_cam_*
        ▼                                                  ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────┐
│ iGPU WORKER PROCESS (env: npu)       │    │ RIFE SUBPROCESS (env: cuda)  │
│ multiprocessing.Process              │    │ subprocess.Popen             │
│                                      │    │                              │
│ Depth Anything v2 small (DML id 1)   │    │ IFNet_HDv3 small (CUDA fp32) │
│   → amd_cam_dlat, amd_cam_depth      │    │   → amd_cam_ring (16 slots)  │
│ TAESD decoder (DML id 1)             │    │   ← amd_cam_image (2 slots)  │
│   → amd_cam_image                    │    │                              │
│ + Spout AMD_AI_SD                    │    │                              │
│ + Spout AMD_AI_SD_DEPTH              │    │                              │
└──────────────────────────────────────┘    └──────────────────────────────┘
```

The orchestrator (`main()` in `morpheus_cam.py`) creates eight named SHM blocks, spawns the iGPU process and RIFE subprocess, then enters the display loop. On `Ctrl+C` or the panel's Stop button, `ctrl[0] = 1` propagates shutdown to all workers; the orchestrator joins the iGPU process, terminates the RIFE subprocess, and unlinks SHM.

## SHM layout

All blocks are in the `amd_cam_*` namespace and Win32 named (single Windows session, `Global = OFF`). Counter fields are int64 indexes that producers atomically increment after writing the corresponding slot.

| Block | Shape | Bytes | Purpose |
|---|---|---|---|
| `amd_cam_ctrl` | `(6,)` int64 | 48 | Control / counters: `[shutdown, latent_fid, img_fid, depth_fid, display_fid, cam_fid]` |
| `amd_cam_latent` | `(2, 1, 4, 64, 64)` fp32 | 131 072 | NPU → iGPU: SD Turbo output latent, double-buffered |
| `amd_cam_image` | `(2, 512, 512, 3)` u8 | 1 572 864 | iGPU → main + RIFE: TAESD-decoded SD frame, double-buffered |
| `amd_cam_depth` | `(2, 512, 512, 3)` u8 | 1 572 864 | iGPU → main: depth visualisation (TURBO colormap), double-buffered |
| `amd_cam_ring` | `(16, 512, 512, 3)` u8 | 12 582 912 | RIFE → main: 16-slot ring of interpolated display frames |
| `amd_cam_dlat` | `(64, 64)` fp32 | 16 384 | iGPU → NPU: low-res depth latent, sigmoid-normalised |
| `amd_cam_webcam` | `(2, 512, 512, 3)` u8 | 1 572 864 | cam thread → iGPU: webcam RGB, double-buffered |
| `amd_cam_camlat` | `(2, 1, 4, 64, 64)` fp32 | 131 072 | cam thread → NPU: zero-cost fake SD latent, double-buffered |

Total ~17 MB. Every consumer reads `ctrl[<fid>]`, compares to a local `last_seen`, and copies the freshest slot. The double-buffer pattern + atomic counter is enough to avoid producer/consumer tearing without locks — readers check the counter again after reading and skip if it advanced inside their copy (only the ring uses an explicit "skip ahead on burst" rule because it has 16 slots and the producer can be arbitrarily ahead).

## CTRL layout (6 int64 slots)

```
[0] shutdown      0 = run, 1 = stop  (set by main on cleanup)
[1] latent_fid    NPU thread → iGPU worker  (SD latent ready)
[2] img_fid       iGPU worker → main + RIFE (TAESD-decoded RGB ready)
[3] depth_fid     iGPU worker → main        (depth viz ready)
[4] display_fid   RIFE worker → main        (ring slot count)
[5] cam_fid       cam thread → iGPU worker + NPU thread (webcam frame ready)
```

## Render loop (per stage)

### cam thread (main process)
```
loop:
    cap.read()                → BGR frame
    cv2.cvtColor + resize     → 512x512 RGB uint8
    write amd_cam_webcam[(cam_fid+1) % 2]
    img_f = (rgb/127.5) - 1   → fake latent magnitude
    avg_pool 8x to (64,64,3) + lum stack → (1,4,64,64) fp32
    write amd_cam_camlat[(cam_fid+1) % 2]
    ctrl[5] = cam_fid + 1
    sleep 20 ms
```

The "fake latent" is a zero-cost approximation of what the SD VAE encoder *would* produce — it's just the camera RGB downsampled to 64×64 with a luminance 4th channel and gain 2. It's good enough to drive the cam_mix slider as a style-mix in latent space, and saves ~10 ms per frame vs running the real TAESD encoder. The "real" depth path uses Depth Anything, not this latent.

### iGPU worker process

One polling loop, two consumers:

```
loop while ctrl[0] == 0:
    if ctrl[5] > last_cam:                           # webcam new
        cam_img = amd_cam_webcam[(cam_fid) % 2]
        d = DEPTH.run(...)                           # ImageNet-norm 518x518
        d_norm = sigmoid((d - mean)/std)
        write amd_cam_dlat = d_norm @ 64x64
        col = TURBO(d) @ 512x512 RGB
        write amd_cam_depth[(depth_fid+1) % 2]
        ctrl[3] = depth_fid + 1
        spout_depth.send(col)
    if ctrl[1] > last_lat:                           # SD latent new
        lat = amd_cam_latent[(latent_fid) % 2]
        img = TAESD.run(lat)                         # 1×3×512×512 fp32
        img_u8 = clip(img/2 + 0.5) * 255
        write amd_cam_image[(img_fid+1) % 2]
        ctrl[2] = img_fid + 1
        spout_sd.send(img_u8)
    if no work: sleep 3 ms
```

### NPU thread (main process)

```
loop while not stop_npu:
    cam_latent = amd_cam_camlat[ctrl[5] % 2]    # latest webcam fake latent
    d_snap = amd_cam_dlat                        # latest depth (64x64 fp32)
    seed = 1000 + cycle
    new_noise = randn((1,4,64,64), seed)
    if evolve and last_noise is not None:
        noise = a*new_noise + sqrt(1-a²)*last_noise   # std-preserving blend
    else:
        noise = new_noise
    last_noise = noise.clone()

    if apply_mod:                                   # alternance gate
        g = depth_g / 100
        mod = (1 - g/2) + g * d_snap                # range [1-g/2, 1+g/2]
        noise = noise * mod                         # depth-driven scale
    latents = noise * sched.init_noise_sigma
    for t in sched.timesteps:                       # STEPS=1 for SD Turbo
        sample = sched.scale_model_input(latents, t)
        noise_pred = UNET.run(sample, t, emb)       # NPU fp32, ~250 ms
        latents = sched.step(noise_pred, t, latents).prev_sample

    if cam_mix > 0:                                 # latent-space style mix
        latents = (1-cm)*latents + cm*cam_latent

    write amd_cam_latent[(latent_fid+1) % 2]
    ctrl[1] = latent_fid + 1
```

### RIFE subprocess (CUDA)

```
loop while ctrl[0] == 0:
    if ctrl[2] > last_consumed:
        curr = amd_cam_image[ctrl[2] % 2]
        if prev is not None:
            frames = rife.interpolate(prev, curr, n_passes=3)   # 9 frames
            for f in frames[1:]:                               # skip prev (already published)
                publish(f)                                      # ctrl[4]++
        else:
            publish(curr)
        prev = curr
    else: sleep 2 ms
```

### Main display loop

```
target_interval = 1/60
loop:
    if ctrl[2] > last_img_fid:    last_img    = amd_cam_image[ctrl[2] % 2]
    if ctrl[3] > last_depth_fid:  last_depth  = amd_cam_depth[ctrl[3] % 2]
    if ctrl[4] > last_display_fid:
        next = last_display_fid + 1
        if ctrl[4] - next > 14: next = ctrl[4] - 14   # skip-ahead on burst
        last_display = amd_cam_ring[next % 16]
        spout_rife.send(last_display)
        last_display_fid = next

    sync_panel_params()
    if cv2_preview_open: build canvas + imshow
    sleep until next_tick
```

The display loop runs at a virtual 60 Hz tick (one frame per 16.7 ms) but RIFE only delivers ~9 frames per 250 ms NPU step, so most ticks are no-ops on the display side. The Spout `AMD_AI_RIFE` sender is fed at the rate frames actually arrive — about 30 fps end to end.

## Cam-driven mode — body → noise

The pipeline is "body-driven" because the depth map of you in front of the camera modulates the SD initial noise. Mechanism:

1. `cam_thread` writes 30 webcam frames per second.
2. `iGPU worker` runs Depth Anything v2 on each frame, normalises the result, sigmoids it, and dumps it into `amd_cam_dlat` (64×64 fp32 in [0, 1]).
3. `NPU thread`, when starting an SD step, snapshots `amd_cam_dlat` and computes:
   ```
   g = depth_g / 100                        # 0..1.5
   mod = (1 - g/2) + g * depth_mask         # tensor (1,1,64,64)
   noise = randn(...) * mod
   ```
   `mod` is a per-pixel scalar that biases noise magnitude where you are vs where you aren't. With `depth_g = 60` (default), the noise gets scaled in `[0.7, 1.3]` per latent pixel. The SD Turbo UNet then "sees" a noise field with structure that follows your silhouette, and produces a frame that has structure in the same places.
4. Optional latent-space style-mix at the end:
   ```
   latents = (1 - cam_mix) * sd_latents + cam_mix * cam_latent
   ```
   With `cam_mix = 30 %` (default), 70 % of the SD output is preserved and 30 % is biased toward the webcam's fake latent — the result is a frame that *looks* like SD output but has subtle colour/composition cues from the camera.

## NPU pipeline — SD Turbo fp32 + vaiml_compile_v4

The AMD Ryzen AI 1.7.1 SDK ships `StableDiffusionONNXPipelineAMDTrigger` in `GenAI-SD/src/`. We import it at startup (sys.path injection) and instantiate with:

```python
StableDiffusionONNXPipelineAMDTrigger(
    model_id="stabilityai/sd-turbo",
    model_path=os.path.join(ROOT, "models/sd/sd_turbo"),
    custom_op_path=r"C:\Program Files\RyzenAI\1.7.1\deployment\onnx_custom_ops.dll",
    enable_compile=False, gpu=False,
)
```

This loads a fp32 UNet ONNX through the VitisAI EP. The first time it runs on a given machine the EP compiles the network for XDNA2 — ~60 s. The compile cache lives in `models/cache/` and subsequent runs are seconds. **Clear the cache after replacing any `.onnx` weight** — the compiler keys by file path, not content, and will reuse a stale binary.

### Why fp32 and not INT8

XDNA2 has a known silicon-level NaN flip-flop on INT8-quantised UNet inference: every other call returns NaN. We tested every software workaround we could think of (different quantizers, different calibration sets, retry loops, mixed-precision wrappers) — none of them work, the bug is in `dyn_bins.dll` and the silicon. The only path that doesn't strobe is **fp32 + the `vaiml_compile_v4` lowering**, where the NPU still runs the ops in BF16 internally but compiled through a different codegen.

The cost: roughly **3-5×** the latency of the INT8 path. We pay it for live performance — a single NaN frame on stage is unacceptable, and RIFE smooths out the ~250 ms keyframe cadence to 30 fps anyway.

When AMD ships the silicon revision that fixes this, swapping back to INT8 is a 1-line change.

## TAESD swap rationale (the math)

For a producer-consumer pipeline where the NPU is the producer and the iGPU decoder is the consumer, throughput is bounded by the slowest stage:

```
fps_keyframes = 1 / max( NPU_step_ms , iGPU_decode_ms )
```

The original choice would be to run the **full SD VAE decoder** on the iGPU. Measured: ~300 ms per frame on Radeon 880M at 512². So:

```
NPU = 250 ms,  full VAE = 300 ms   → fps = 1 / 0.30 = 3.33   (VAE-bound)
```

Swapping in **TAESD** (madebyollin) — a tiny autoencoder distilled from the SD VAE, trained to map SD latents to RGB:

```
NPU = 250 ms,  TAESD = 10 ms       → fps = 1 / 0.25 = 4.00   (NPU-bound)
```

Two effects:
1. **Throughput goes up by 20 %** end-to-end.
2. The bottleneck moves from a fungible iGPU stage to the dedicated NPU. That's the *correct* place for the bottleneck to live: the NPU is the most specialised silicon, the iGPU has slack we can use for other things (Depth Anything in our case, but it could be Real-ESRGAN, optical flow, anything).

This is why TAESD is "the cheat code". It doesn't make any single stage faster than a SOTA model — it makes the *pipeline* limited by the right stage.

Visual quality: noticeably degraded vs the full VAE on still detail, but at 30 fps with RIFE smoothing, the eye doesn't notice. For a live performance pipeline, the trade is correct. We don't claim TAESD-decoded SD looks like full-VAE SD; we claim it looks good *moving*.

## RIFE worker — flow interpolation on CUDA

Spawned in conda env `cuda` because PyTorch CUDA can't share a process with VitisAI EP DLLs without crashing. Communicates with the orchestrator only via SHM — no IPC, no pipes, no shared memory beyond the named blocks.

| Setting | Value | Why |
|---|---|---|
| `model_size` | `small` (`flownet_small.pkl`) | ~12 MB, ~50-60 ms per pair on RTX 5070 — full would be ~120 ms and we don't need the quality |
| `n_passes` | 3 | 2³+1 = 9 frames per pair → ~30 fps display from ~3.3 NPU keyframe-pairs / sec |
| `fp16` | False (default) | The flow predictions drift slightly between recursive passes in fp16 — fp32 is robust and fast enough |
| `ring_slots` | 16 | Bigger than `n_passes` → producer can stay ~1.5 pairs ahead without overwriting unread slots |

The recursive midpoint scheme is RIFE's standard trick: with one network call between two frames you get a midpoint; recurse on both halves and you get 3 mids; recurse again for 7; once more for 15. We stop at 3 passes (9 frames) — more passes give diminishing returns once the gap between input frames is < 100 ms.

All work stays in CUDA tensor space across recursive passes — no CPU↔GPU round-trips between sub-calls. Numpy conversion only on the way out, when publishing into `amd_cam_ring`.

## Web control panel architecture

`control/server.py` runs three things in daemon threads inside the main process:

- **HTTP server** (default 54331): serves `panel.html` and a JSON `/state` endpoint.
- **WebSocket server** (default 54332): bidirectional. Browser sends `{"name": "<control>", "value": <v>}` per slider drag; server pushes `{"type":"metrics","data":{...}}` once per second from the hardware sampler and `{"type":"update", ...}` on any state change.
- **Hardware sampler thread**: samples CPU/RAM via psutil, iGPU/dGPU via long-lived PowerShell `Get-Counter` loop, and dGPU temperature/power via per-tick `nvidia-smi`. NPU % is **derived from `ms_step`** because Windows perfmon doesn't expose `NPU 0` under the `\GPU Engine` path reliably — only iGPU + dGPU adapters show up there by LUID. The panel marks the NPU bar with `~` to be honest about the derivation.

The PowerShell process is spawned **once** and kept alive on stdin via an infinite `while($true){ Get-Counter ...; Start-Sleep 1000 }` loop. Cold-starting `powershell.exe` per tick costs ~250 ms, which would starve the WS pump.

LUID classification: the first time the sampler sees ≥ 2 active LUIDs in `Get-Counter`, the busier one is tagged `igpu` (TAESD + Depth always running in this pipeline) and the other `dgpu` (RIFE bursts). This is stable enough — the iGPU never goes below ~30 % active in normal operation.

## Three Spout outputs

| Sender | Format | Size | Rate | Source |
|---|---|---|---|---|
| `AMD_AI_SD` | RGB → RGBA8 (alpha = 255) | 512×512 | NPU rate, ~3-4 fps | TAESD-decoded SD frame |
| `AMD_AI_SD_DEPTH` | RGB → RGBA8 (alpha = 255) | 512×512 | webcam rate, ~30 fps | TURBO-colormapped Depth Anything output |
| `AMD_AI_RIFE` | RGB → RGBA8 (alpha = 255) | 512×512 | display rate, ~30 fps | RIFE-interpolated frame |

Internally `SpoutOutput` keeps a single RGBA buffer with `alpha = 255` and copies the producer's RGB into the first three channels. `SpoutLibrary.dll` then uploads to a hidden DX11 shared NT handle. CPU side cost is one `memcpy` per frame; GPU side is the GL→DX11 interop inherent to Spout (~2 ms at 512², much less than at 4K). See [`SPOUT2ForPython`](https://github.com/UnveilStudio/SPOUT2ForPython) for the underlying mechanism.

## Frame format conventions

| Surface | Format | Layout | Notes |
|---|---|---|---|
| Webcam capture | BGR uint8 (cv2) | `(480, 640, 3)` then resize → `(512, 512, 3)` | DSHOW backend |
| Webcam SHM (`amd_cam_webcam`) | RGB uint8 | `(2, 512, 512, 3)` double-buffered | matches Spout `AMD_AI_SD_DEPTH` orientation |
| SD latent (`amd_cam_latent`, `amd_cam_camlat`) | fp32 | `(2, 1, 4, 64, 64)` double-buffered | SD VAE latent space |
| Depth latent (`amd_cam_dlat`) | fp32 | `(64, 64)` single | sigmoid-normalised, mean ≈ 0.5 |
| Decoded RGB (`amd_cam_image`, `amd_cam_depth`) | RGB uint8 | `(2, 512, 512, 3)` double-buffered | sRGB-naive |
| RIFE display ring (`amd_cam_ring`) | RGB uint8 | `(16, 512, 512, 3)` single ring | producer atomically increments `ctrl[4]` |
| Spout payload | RGBA uint8 | `(H, W, 4)` contiguous | `alpha = 255` filled at sender init |
| cv2 preview | BGR uint8 | `(SIZE+90, 2*SIZE+20, 3)` | composited from depth + RIFE rows |

## Performance characteristics

Razer Blade 14 (2025) — Ryzen AI 9 365 + Radeon 880M + RTX 5070 Laptop, 32 GB LPDDR5X-8000, Win 11. Settings: 512², 1 SD step, n_passes=3, prompt encoded once.

### Per-stage breakdown

| Stage | Silicon | Cost | Notes |
|---|---|---|---|
| Webcam capture | CPU | ~6 ms | cv2 DSHOW decode + cvtColor + resize |
| Fake cam latent | CPU | ~1 ms | avg_pool + lum stack, vectorised numpy |
| Depth Anything v2 small (FP16) | iGPU DML | ~30 ms | 518×518 input |
| Depth post (sigmoid, colormap, resize) | CPU | ~2 ms | small tensors, well-cached |
| SD Turbo prompt encode | CPU (CLIP) | ~50 ms | once per prompt change, not per frame |
| **SD Turbo UNet (1 step, fp32, vaiml_compile_v4)** | **NPU** | **~250 ms** | **the throughput-defining stage** |
| Scheduler step | CPU | ~3 ms | DDIM-like, single timestep |
| TAESD decode | iGPU DML | ~10 ms | (1,4,64,64) → (1,3,512,512) |
| RIFE 9-frame interpolation | dGPU CUDA | ~50-60 ms | per pair, 3 recursive passes |
| Spout send (per stream) | CPU+GL | ~1 ms | RGBA memcpy + GL upload + DX11 interop |

### Steady-state utilisation

| Silicon | Util | What's running | Spare |
|---|---|---|---|
| **NPU XDNA2** | ~22 % | SD Turbo UNet | ≥ 70 % |
| **iGPU 880M** | ~41 % | Depth (30 ms × 30 fps) + TAESD (10 ms × 4 fps) | ~50 % |
| **dGPU RTX 5070** | ~9 % | RIFE small (50 ms × 4 pair/s) | ≥ 80 % |
| **CPU 8C/16T** | ~42 % | orchestration + cv2 + Spout | ~50 % |
| **RAM** | ~15 / 32 GB | SD weights + RIFE + ORT sessions + numpy scratch | ~50 % |
| **dGPU temp** | ~70 °C | sustained RIFE small | well within thermal envelope |
| **dGPU power** | ~22 W | RIFE small, fp32 | dGPU TGP up to 115 W |

> The pipeline draws ≈ 30 W combined for AI inference work. The Razer Blade 14 form factor handles this without throttling.

## Future / non-goals

- **Linux / macOS port** — non-goal. Ryzen AI Software (VitisAI EP, GenAI-SD pipeline, `vaip_config.json`, `onnx_custom_ops.dll`) is **Windows-only**. Without the SDK we don't have a path to the NPU at all. SHM is `multiprocessing.shared_memory` (cross-platform), so that part would port; everything NPU would not.
- **INT8 NPU re-enable** — when AMD ships the silicon revision that fixes the strobe, the swap is a 1-line change in `morpheus_cam.py` (replace the fp32 UNet path with the INT8 one in `StableDiffusionONNXPipelineAMDTrigger`). Expected speedup: 3-5×. Until then we stay on fp32.
- **dGPU 4× upscaler at 2K Spout out** — already proven possible. **Real-ESRGAN small** measured at ~33 ms for 512² → 2048² on the RTX 5070, with the dGPU at 9 % util in the current pipeline. Plugging it in is straightforward: add a CUDA stage in the RIFE worker (already in CUDA env) that upscales the interpolated frames before publishing to `amd_cam_ring`, and bump the ring slot resolution. Default config ships at 512² for legibility — the architecture is not the limit.
- **Multi-camera** — would need a second `cam_thread` and a `--webcam2` flag, plus a panel control to pick which feed drives the depth path. Not currently planned.
- **Audio reactive mode** — out of scope; use TouchDesigner downstream of the Spout outputs, that's exactly what the TD ecosystem is for.
