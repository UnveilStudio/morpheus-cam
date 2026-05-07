"""
morpheus_cam.py — Real-time body-driven Stable Diffusion pipeline.

Webcam → Depth Anything (iGPU) → modulates noise of SD Turbo (NPU) →
TAESD decode (iGPU) → RIFE flow interpolation (CUDA) → ~30 fps display
+ 3 Spout outputs + web control panel.

Architecture (all processes share-mem connected via amd_cam_*):

  Webcam thread (RGB capture + zero-cost fake latent)
    ─► amd_cam_webcam        ─┐
    ─► amd_cam_camlat         ├─► [NPU thread] SD Turbo img2img + temporal-smoothed noise
                              │       ─► amd_cam_latent
  iGPU worker process
    ─► Depth Anything (webcam → amd_cam_dlat + Spout AMD_AI_SD_DEPTH)
    ─► TAESD decode  (SD latent → amd_cam_image + Spout AMD_AI_SD)
  CUDA RIFE subprocess
    ─► RIFE flow interp (consecutive SD frames → amd_cam_ring + Spout AMD_AI_RIFE)

Web panel: http://127.0.0.1:54331/  (default).

Usage:
    conda run -n npu --no-capture-output python -u morpheus_cam.py
    conda run -n npu --no-capture-output python -u morpheus_cam.py --webcam 1 --port 54331
"""
from __future__ import annotations
import argparse
import copy
import os
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing import shared_memory

import cv2
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── pipeline constants ──────────────────────────────────────────────────────
SIZE         = 512
STEPS        = 1
DEPTH_SIZE   = 518
LATENT_SHAPE = (1, 4, SIZE // 8, SIZE // 8)
IMG_SHAPE    = (SIZE, SIZE, 3)
RING_SLOTS   = 16
N_PASSES     = 3                      # 2**3 + 1 = 9 frames per pair (~30 fps display)

SHM_CTRL    = "amd_cam_ctrl"
SHM_LATENT  = "amd_cam_latent"
SHM_IMG     = "amd_cam_image"
SHM_DEPTH   = "amd_cam_depth"
SHM_RING    = "amd_cam_ring"
SHM_DLATENT = "amd_cam_dlat"
SHM_CAM     = "amd_cam_webcam"
SHM_CAMLAT  = "amd_cam_camlat"

# CTRL layout (6 int64):
#   [0] shutdown
#   [1] latent_fid   (NPU thread → iGPU worker)
#   [2] img_fid      (iGPU worker → main display + RIFE)
#   [3] depth_fid    (iGPU worker → main display)
#   [4] display_fid  (RIFE → main display ring)
#   [5] cam_fid      (cam thread → iGPU worker + NPU thread)
CTRL_SLOTS = 6

TAESD_PATH       = os.path.join(ROOT, "models", "sd", "sd_turbo", "vae_decoder", "taesd_decoder.onnx")
TAESD_ENC_PATH   = os.path.join(ROOT, "models", "sd", "sd_turbo", "vae_encoder", "taesd_encoder_fp16.onnx")
DEPTH_ONNX       = os.path.join(ROOT, "models", "depth_anything_v2_small_fixed_fp16.onnx")
CUSTOM_OPS_DLL   = r"C:\Program Files\RyzenAI\1.7.1\deployment\onnx_custom_ops.dll"
GENAI_SD_DIR     = r"C:\Program Files\RyzenAI\1.7.1\GenAI-SD"

DEFAULT_PROMPT = (
    "intricate robotic creature, cyberpunk aesthetic, mechanical chassis, "
    "articulated limbs, glowing optic sensors, neon accents, sharp focus, "
    "photorealistic, 8k"
)


# ─── shared-memory views ─────────────────────────────────────────────────────
def view_ctrl(shm):    return np.ndarray((CTRL_SLOTS,), dtype=np.int64,   buffer=shm.buf)
def view_latent(shm):  return np.ndarray((2,) + LATENT_SHAPE, dtype=np.float32, buffer=shm.buf)
def view_img(shm):     return np.ndarray((2,) + IMG_SHAPE,    dtype=np.uint8,   buffer=shm.buf)
def view_ring(shm):    return np.ndarray((RING_SLOTS,) + IMG_SHAPE, dtype=np.uint8, buffer=shm.buf)
def view_dlat(shm):    return np.ndarray((64, 64), dtype=np.float32, buffer=shm.buf)
def view_cam(shm):     return np.ndarray((2,) + IMG_SHAPE,    dtype=np.uint8,   buffer=shm.buf)
def view_camlat(shm):  return np.ndarray((2,) + LATENT_SHAPE, dtype=np.float32, buffer=shm.buf)


# ─── iGPU worker (separate process: TAESD decode + Depth Anything + Spout) ──
def igpu_worker_proc(taesd_path: str, depth_onnx: str):
    """
    Runs in its own process so DML stays isolated from the NPU stack.
    Two consumer loops fused into one polling loop:
      A) webcam frame new (ctrl[5]) → Depth Anything + Spout AMD_AI_SD_DEPTH
      B) SD latent  new (ctrl[1])   → TAESD decode  + Spout AMD_AI_SD
    """
    import sys as _sys
    _sys.path.insert(0, ROOT)
    import numpy as _np
    import cv2 as _cv2
    import time as _time
    import onnxruntime as ort
    from multiprocessing import shared_memory as _shm

    print("[morpheus][iGPU] loading TAESD decoder + Depth Anything (DML device 1)...", flush=True)
    so_t = ort.SessionOptions()
    so_t.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    TAESD = ort.InferenceSession(
        taesd_path, sess_options=so_t,
        providers=[("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
    )

    so_d = ort.SessionOptions()
    so_d.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    DEPTH = ort.InferenceSession(
        depth_onnx, sess_options=so_d,
        providers=[("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
    )
    DIN = DEPTH.get_inputs()[0].name

    MEAN = _np.array([0.485, 0.456, 0.406], dtype=_np.float32).reshape(1, 1, 3)
    STD  = _np.array([0.229, 0.224, 0.225], dtype=_np.float32).reshape(1, 1, 3)

    ctrl_shm   = _shm.SharedMemory(name=SHM_CTRL)
    lat_shm    = _shm.SharedMemory(name=SHM_LATENT)
    img_shm    = _shm.SharedMemory(name=SHM_IMG)
    d_shm      = _shm.SharedMemory(name=SHM_DEPTH)
    dlat_shm   = _shm.SharedMemory(name=SHM_DLATENT)
    cam_shm    = _shm.SharedMemory(name=SHM_CAM)
    camlat_shm = _shm.SharedMemory(name=SHM_CAMLAT)
    ctrl    = view_ctrl(ctrl_shm)
    lats    = view_latent(lat_shm)
    imgs    = view_img(img_shm)
    dims    = view_img(d_shm)
    dlat    = view_dlat(dlat_shm)
    cams    = view_cam(cam_shm)

    from spout_sender import SpoutOutput
    spout_sd    = SpoutOutput("AMD_AI_SD")
    spout_depth = SpoutOutput("AMD_AI_SD_DEPTH")

    print("[morpheus][iGPU] ready", flush=True)
    last_lat = 0
    last_cam = 0
    try:
        while ctrl[0] == 0:
            did_work = False

            # A) webcam new → Depth Anything
            cam_fid = int(ctrl[5])
            if cam_fid > last_cam:
                cam_img = cams[cam_fid % 2].copy()
                last_cam = cam_fid
                did_work = True

                x = _cv2.resize(cam_img, (DEPTH_SIZE, DEPTH_SIZE)).astype(_np.float32) / 255.0
                x = (x - MEAN) / STD
                x = x.transpose(2, 0, 1)[None].astype(_np.float16)
                d = DEPTH.run(None, {DIN: x})[0][0].astype(_np.float32)
                dmin, dmax = float(d.min()), float(d.max())
                dn = (d - dmin) / (dmax - dmin + 1e-6)
                u8 = (dn * 255).astype(_np.uint8)
                col_bgr = _cv2.applyColorMap(u8, _cv2.COLORMAP_TURBO)
                col_bgr = _cv2.resize(col_bgr, (SIZE, SIZE))
                col_rgb = _cv2.cvtColor(col_bgr, _cv2.COLOR_BGR2RGB)
                d_small = _cv2.resize(d, (64, 64), interpolation=_cv2.INTER_AREA)
                mu, sd_ = float(d_small.mean()), float(d_small.std() + 1e-6)
                d_norm = (d_small - mu) / sd_
                d_sig = 1.0 / (1.0 + _np.exp(-1.5 * d_norm))
                _np.copyto(dlat, d_sig.astype(_np.float32))

                nd = int(ctrl[3]) + 1
                _np.copyto(dims[nd % 2], col_rgb)
                ctrl[3] = nd
                try: spout_depth.send(col_rgb)
                except Exception: pass

            # B) SD latent new → TAESD decode
            fid = int(ctrl[1])
            if fid > last_lat:
                lat = lats[fid % 2].copy()
                last_lat = fid
                did_work = True
                img = TAESD.run(None, {"latents": lat.astype(_np.float32)})[0]
                img = _np.clip(img[0] / 2 + 0.5, 0, 1)
                img_u8 = (img.transpose(1, 2, 0) * 255).astype(_np.uint8)
                nf = int(ctrl[2]) + 1
                _np.copyto(imgs[nf % 2], img_u8)
                ctrl[2] = nf
                try: spout_sd.send(img_u8)
                except Exception: pass

            if not did_work:
                _time.sleep(0.003)
    finally:
        print("[morpheus][iGPU] shutdown", flush=True)
        try: spout_sd.release()
        except Exception: pass
        try: spout_depth.release()
        except Exception: pass
        ctrl_shm.close(); lat_shm.close(); img_shm.close(); d_shm.close()
        dlat_shm.close(); cam_shm.close(); camlat_shm.close()


# ─── RIFE subprocess discovery + spawn ──────────────────────────────────────
def _find_cuda_python():
    env_var = os.environ.get("MORPHEUS_CUDA_PYTHON") or os.environ.get("AMD_AI_CUDA_PYTHON")
    if env_var and os.path.exists(env_var):
        return [env_var]
    candidates = [
        os.path.expanduser("~/miniconda3/envs/cuda/python.exe"),
        os.path.expanduser("~/anaconda3/envs/cuda/python.exe"),
        r"C:\ProgramData\miniconda3\envs\cuda\python.exe",
        r"C:\Miniconda3\envs\cuda\python.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return [c]
    return ["conda", "run", "-n", "cuda", "--no-capture-output", "python", "-u"]


def spawn_rife_worker():
    cmd = _find_cuda_python() + [
        os.path.join(ROOT, "rife", "flow_worker.py"),
        "--ctrl-name", SHM_CTRL,
        "--img-name",  SHM_IMG,
        "--ring-name", SHM_RING,
        "--size",       str(SIZE),
        "--n-passes",   str(N_PASSES),
        "--ring-slots", str(RING_SLOTS),
        "--model-size", "small",
    ]
    print(f"[morpheus][RIFE-spawn] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, universal_newlines=True,
    )

    def _relay():
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception:
            pass
    threading.Thread(target=_relay, daemon=True).start()
    return proc


# ─── main: NPU SD UNet + display loop + Spout RIFE ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="morpheus-cam — body-driven Stable Diffusion")
    ap.add_argument("--webcam",    type=int, default=0, help="webcam device id (default 0)")
    ap.add_argument("--http-port", type=int, default=54331, help="control panel HTTP port")
    ap.add_argument("--ws-port",   type=int, default=54332, help="control panel WebSocket port")
    ap.add_argument("--prompt",    type=str, default=DEFAULT_PROMPT, help="initial SD prompt")
    args = ap.parse_args()

    # SHM allocate
    shm_ctrl   = shared_memory.SharedMemory(create=True, size=CTRL_SLOTS * 8, name=SHM_CTRL)
    shm_lat    = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(LATENT_SHAPE)) * 4, name=SHM_LATENT)
    shm_img    = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(IMG_SHAPE)),  name=SHM_IMG)
    shm_depth  = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(IMG_SHAPE)),  name=SHM_DEPTH)
    shm_ring   = shared_memory.SharedMemory(create=True, size=RING_SLOTS * int(np.prod(IMG_SHAPE)), name=SHM_RING)
    shm_dlat   = shared_memory.SharedMemory(create=True, size=64 * 64 * 4, name=SHM_DLATENT)
    shm_cam    = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(IMG_SHAPE)), name=SHM_CAM)
    shm_camlat = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(LATENT_SHAPE)) * 4, name=SHM_CAMLAT)

    ctrl = view_ctrl(shm_ctrl); ctrl[:] = 0
    lats = view_latent(shm_lat)
    imgs = view_img(shm_img); dims = view_img(shm_depth)
    ring = view_ring(shm_ring)
    dlat = view_dlat(shm_dlat); dlat[:] = 0.5
    cams = view_cam(shm_cam)
    camlats = view_camlat(shm_camlat)

    # iGPU worker process
    igpu_p = mp.Process(target=igpu_worker_proc, args=(TAESD_PATH, DEPTH_ONNX), daemon=False)
    igpu_p.start()
    print(f"[morpheus][main] spawned iGPU worker pid={igpu_p.pid}", flush=True)

    # RIFE CUDA worker subprocess
    rife_proc = spawn_rife_worker()
    print(f"[morpheus][main] spawned RIFE worker pid={rife_proc.pid}", flush=True)

    # AMD Ryzen AI SDK GenAI-SD on sys.path so we can import the SD Turbo pipeline.
    # The 'src/' dir there ships StableDiffusionONNXPipelineTrigger.py — NOT redistributable.
    # User must install Ryzen AI 1.7.1 themselves; we just bolt onto its sys.path.
    sys.path.insert(0, ROOT)
    sys.path.insert(0, GENAI_SD_DIR)

    # Web control panel (HTTP + WS + hardware sampler in daemon threads)
    from control.server import CamRifeControlState, start_servers
    web_state = CamRifeControlState(default_prompt=args.prompt)
    start_servers(web_state, http_port=args.http_port, ws_port=args.ws_port)
    print(f"[morpheus][main] panel: http://127.0.0.1:{args.http_port}/", flush=True)

    os.environ["DISABLE_VAE_DD"] = "1"
    os.environ.setdefault(
        "VAIP_CONFIG_JSON",
        r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json",
    )

    print("[morpheus][main] loading SD Turbo pipeline (warmup ~20s)...", flush=True)
    from src.StableDiffusionONNXPipelineTrigger import StableDiffusionONNXPipelineAMDTrigger
    from src.pipeline_stable_diffusion_onnx_amd  import StableDiffusionONNXPipelineAMD
    import torch

    pipe_trigger = StableDiffusionONNXPipelineAMDTrigger(
        model_id="stabilityai/sd-turbo",
        model_path=os.path.join(ROOT, "models", "sd", "sd_turbo"),
        custom_op_path=CUSTOM_OPS_DLL,
        enable_compile=False, gpu=False,
    )
    pipe = StableDiffusionONNXPipelineAMD(
        vae=pipe_trigger.vae_decoder,
        text_encoder=pipe_trigger.text_encoder,
        tokenizer=pipe_trigger.tokenizer,
        unet=pipe_trigger.unet,
        scheduler=pipe_trigger.scheduler,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    PRISTINE = copy.deepcopy(pipe.scheduler)
    UNET = pipe.unet.model

    def encode_prompt(p):
        e, _ = pipe.encode_prompt(
            prompt=p, device=torch.device("cpu"),
            num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=None,
        )
        return e.detach().cpu().numpy().astype(np.float32)

    def unet_step(sample_np, ts_np, emb_np):
        # Strobe-bug retry: NPU XDNA2 returns NaN deterministically every other call
        # in the INT8 path. We're already on fp32 + vaiml_compile_v4 (no NaN), but we
        # keep the guard for free in case of upstream regressions.
        inputs = {"sample": sample_np, "timestep": ts_np, "encoder_hidden_states": emb_np}
        out = UNET.run(None, inputs)[0]
        for _ in range(4):
            if np.isfinite(out).all():
                return out
            out = UNET.run(None, inputs)[0]
        return out

    alt_flag   = [False]
    alt_parity = [0]
    params = {
        "cam_mix":   30,
        "noise_a":   30,
        "depth_g":   60,
    }
    evolve_flag = [True]
    last_noise  = [None]

    def generate_latent(emb_np, seed, d_snapshot, cam_latent_np):
        sched = copy.deepcopy(PRISTINE); sched.set_timesteps(STEPS)
        gen = torch.Generator("cpu").manual_seed(seed)
        new_noise = torch.randn((1, 4, SIZE // 8, SIZE // 8), generator=gen)

        # Temporal noise smoothing (correlated noise, std preserved):
        #   noise = a * new + sqrt(1 - a^2) * prev    (Gaussian-correct blend)
        if evolve_flag[0] and last_noise[0] is not None:
            a = float(params["noise_a"]) / 100.0
            a = max(0.02, min(1.0, a))
            b = float(np.sqrt(1.0 - a * a))
            noise = a * new_noise + b * last_noise[0]
        else:
            noise = new_noise
        last_noise[0] = noise.detach().clone()

        # Depth-driven noise modulation: webcam depth → per-pixel noise scale
        d_t = torch.from_numpy(d_snapshot).unsqueeze(0).unsqueeze(0)   # (1,1,64,64)
        apply_mod = (not alt_flag[0]) or (alt_parity[0] % 2 == 0)
        alt_parity[0] += 1
        if apply_mod:
            g = float(params["depth_g"]) / 100.0
            mod = (1.0 - g * 0.5) + g * d_t                            # range 1-g/2 .. 1+g/2
            noise_mod = noise * mod
        else:
            noise_mod = noise

        latents = noise_mod * sched.init_noise_sigma
        for t in sched.timesteps:
            lat_in = sched.scale_model_input(latents, t)
            sample = lat_in.numpy().astype(np.float32)
            ts = np.array([float(t.item())], dtype=np.float64)
            noise_pred = torch.from_numpy(unet_step(sample, ts, emb_np))
            latents = sched.step(noise_pred, t, latents).prev_sample

        # Style-transfer blend in latent space: SD output ↔ webcam fake latent.
        cm = float(params["cam_mix"]) / 100.0
        if cm > 0.001 and float(np.abs(cam_latent_np).mean()) > 1e-5:
            cam_lat_t = torch.from_numpy(cam_latent_np)
            latents = (1.0 - cm) * latents + cm * cam_lat_t
        return latents.numpy().astype(np.float32)

    print("[morpheus][main] warmup 2 cycles...", flush=True)
    current_prompt = [args.prompt]
    emb = encode_prompt(current_prompt[0])
    _d_warm = np.full((64, 64), 0.5, dtype=np.float32)
    _cam_warm = np.zeros(LATENT_SHAPE, dtype=np.float32)
    for i in range(2):
        t0 = time.perf_counter()
        _ = generate_latent(emb, seed=i, d_snapshot=_d_warm, cam_latent_np=_cam_warm)
        print(f"  [{i}] {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    # Spout RIFE display stream (interpolated frames, 30+ fps)
    from spout_sender import SpoutOutput
    spout_rife = SpoutOutput("AMD_AI_RIFE")
    print("[morpheus][main] Spout 'AMD_AI_RIFE' open (interpolated display stream)", flush=True)

    WIN = "morpheus-cam — preview"
    preview_open = [False]

    web_state.set_value("cam_mix", params["cam_mix"])
    web_state.set_value("noise_a", params["noise_a"])
    web_state.set_value("depth_g", params["depth_g"])

    fts_sd      = deque(maxlen=60)
    fts_display = deque(maxlen=120)
    last_img_fid = 0
    last_depth_fid = 0
    last_display_fid = 0
    last_img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    last_depth = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    last_display = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    print("\n=== LIVE === panel toggles cv2 preview, ESC quits when preview is open\n", flush=True)

    TARGET_INTERVAL = 1.0 / 60.0
    next_tick = time.perf_counter()

    # Webcam capture thread — RGB + zero-cost "fake" cam latent (avg-pool + lum stack).
    # Skips the full TAESD encoder (~10 ms saved per frame) — SD style-mix uses this
    # cheap latent and it visually matches.
    stop_cam = [False]
    def cam_thread():
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"[morpheus][cam] ERROR: webcam {args.webcam} not opened", flush=True)
            return
        print(f"[morpheus][cam] webcam {args.webcam} open (zero-cost fake-latent mode)", flush=True)
        while not stop_cam[0]:
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb512 = cv2.resize(rgb, (SIZE, SIZE))
            nf = int(ctrl[5]) + 1
            np.copyto(cams[nf % 2], rgb512)
            img_f = (rgb512.astype(np.float32) / 127.5) - 1.0
            img_small = cv2.resize(img_f, (64, 64), interpolation=cv2.INTER_AREA)
            lum = img_small.mean(axis=-1, keepdims=True)
            lat_rgba = np.concatenate([img_small, lum], axis=-1) * 2.0
            cam_lat = lat_rgba.transpose(2, 0, 1)[None].astype(np.float32)
            np.copyto(camlats[nf % 2], cam_lat)
            ctrl[5] = nf
            time.sleep(0.02)
        cap.release()
        print("[morpheus][cam] shutdown", flush=True)
    cam_t = threading.Thread(target=cam_thread, daemon=True)
    cam_t.start()

    # NPU SD UNet thread (the real keyframe generator)
    emb_holder = [emb]
    stop_npu = [False]
    def npu_thread():
        ncount = [0]
        prev_depth = np.full((64, 64), 0.5, dtype=np.float32)
        last_cam_seen = 0
        cam_latent_np = np.zeros(LATENT_SHAPE, dtype=np.float32)
        while not stop_npu[0]:
            cam_fid = int(ctrl[5])
            if cam_fid > 0 and cam_fid != last_cam_seen:
                cam_latent_np = camlats[cam_fid % 2].copy()
                last_cam_seen = cam_fid
            t0 = time.perf_counter()
            d_snap = prev_depth
            seed_now = 1000 + ncount[0]
            lat = generate_latent(emb_holder[0], seed=seed_now, d_snapshot=d_snap,
                                  cam_latent_np=cam_latent_np)
            ncount[0] += 1
            ms = (time.perf_counter() - t0) * 1000
            nf = int(ctrl[1]) + 1
            np.copyto(lats[nf % 2], lat)
            ctrl[1] = nf
            fts_sd.append(time.perf_counter())
            prev_depth = dlat.copy()
            sd_fps_now = (len(fts_sd) - 1) / max(fts_sd[-1] - fts_sd[0], 1e-3) if len(fts_sd) >= 2 else 0
            web_state.update_metrics(
                cycle=ncount[0], sd_fps=round(sd_fps_now, 2), ms_step=round(ms, 1),
                cam_fid=int(cam_fid), d_mean=round(float(prev_depth.mean()), 4),
            )
            if ncount[0] % 10 == 0:
                print(
                    f"  [morpheus][NPU] cycle {ncount[0]}  SD={sd_fps_now:.2f}fps  "
                    f"ms={ms:.0f}  cam_fid={cam_fid}  d_mean={float(prev_depth.mean()):.3f}",
                    flush=True,
                )
    npu_t = threading.Thread(target=npu_thread, daemon=True)
    npu_t.start()

    try:
        while True:
            if_ = int(ctrl[2])
            if if_ > last_img_fid:
                last_img = imgs[if_ % 2].copy()
                last_img_fid = if_
            df = int(ctrl[3])
            if df > last_depth_fid:
                last_depth = dims[df % 2].copy()
                last_depth_fid = df

            # RIFE display ring — advance one frame per tick, skip-ahead on burst.
            rfid = int(ctrl[4])
            if rfid > last_display_fid:
                next_to_show = last_display_fid + 1
                if rfid - next_to_show > RING_SLOTS - 2:
                    next_to_show = rfid - (RING_SLOTS - 2)
                slot = next_to_show % RING_SLOTS
                cand = ring[slot].copy()
                post_fid = int(ctrl[4])
                if (post_fid - next_to_show) < RING_SLOTS:
                    last_display = cand
                    fts_display.append(time.perf_counter())
                    try: spout_rife.send(last_display)
                    except Exception: pass
                last_display_fid = next_to_show

            # Sync live params from web panel
            params["cam_mix"] = web_state.get_value("cam_mix", params["cam_mix"])
            params["noise_a"] = web_state.get_value("noise_a", params["noise_a"])
            params["depth_g"] = web_state.get_value("depth_g", params["depth_g"])

            ws_pt = web_state.get_value("prompt_text", current_prompt[0])
            if ws_pt and ws_pt != current_prompt[0]:
                current_prompt[0] = ws_pt
                emb_holder[0] = encode_prompt(ws_pt)
                last_noise[0] = None
                print(f"  [morpheus][web] prompt -> {ws_pt[:80]}{'...' if len(ws_pt) > 80 else ''}  (noise reset)",
                      flush=True)
            ws_ev = web_state.get_value("evolve", evolve_flag[0])
            if ws_ev != evolve_flag[0]:
                evolve_flag[0] = ws_ev
                if not ws_ev:
                    last_noise[0] = None
                print(f"  [morpheus][web] EVOLVE -> {'ON' if ws_ev else 'OFF'}", flush=True)
            ws_alt = web_state.get_value("alternance", alt_flag[0])
            if ws_alt != alt_flag[0]:
                alt_flag[0] = ws_alt
                print(f"  [morpheus][web] ALTERNANCE -> {'ON' if alt_flag[0] else 'OFF'}", flush=True)

            want_preview = bool(web_state.get_value("cv2_preview", False))
            if want_preview and not preview_open[0]:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, SIZE * 2 + 20, SIZE + 90)
                preview_open[0] = True
            elif not want_preview and preview_open[0]:
                cv2.destroyWindow(WIN)
                preview_open[0] = False

            if preview_open[0]:
                dp_bgr = cv2.cvtColor(last_depth,   cv2.COLOR_RGB2BGR)
                rf_bgr = cv2.cvtColor(last_display, cv2.COLOR_RGB2BGR)
                gap = np.zeros((SIZE, 20, 3), dtype=np.uint8)
                row = np.hstack([rf_bgr, gap, dp_bgr])
                bar = np.zeros((90, SIZE * 2 + 20, 3), dtype=np.uint8)
                canvas = np.vstack([row, bar])

                sd_fps   = (len(fts_sd) - 1) / max(fts_sd[-1] - fts_sd[0], 1e-3) if len(fts_sd) >= 2 else 0
                disp_fps = (len(fts_display) - 1) / max(fts_display[-1] - fts_display[0], 1e-3) if len(fts_display) >= 2 else 0
                cv2.putText(
                    canvas, f"Disp={disp_fps:4.1f}  SD={sd_fps:.2f}  cam={params['cam_mix']}%  "
                            f"noise={params['noise_a']}%  depth={params['depth_g']}%",
                    (15, SIZE + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, f"img_fid={last_img_fid} depth_fid={last_depth_fid} disp_fid={last_display_fid}/{rfid}",
                    (15, SIZE + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, current_prompt[0][:80],
                    (15, SIZE + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
                )
                cv2.imshow(WIN, canvas)

            next_tick += TARGET_INTERVAL
            wait_ms = max(1, int((next_tick - time.perf_counter()) * 1000))
            if wait_ms > 33:
                wait_ms = 33

            if preview_open[0]:
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == 27:
                    web_state.set_value("running", False)
                elif key in (ord('a'), ord('A')):
                    alt_flag[0] = not alt_flag[0]
                    web_state.set_value("alternance", alt_flag[0])
                elif key in (ord('e'), ord('E')):
                    evolve_flag[0] = not evolve_flag[0]
                    if not evolve_flag[0]:
                        last_noise[0] = None
                    web_state.set_value("evolve", evolve_flag[0])
            else:
                time.sleep(wait_ms / 1000.0)

            if not web_state.get_value("running", True):
                break
    finally:
        cv2.destroyAllWindows()
        stop_npu[0] = True
        stop_cam[0] = True
        ctrl[0] = 1
        print("[morpheus][main] shutdown...", flush=True)
        igpu_p.join(timeout=5.0)
        if igpu_p.is_alive():
            igpu_p.terminate(); igpu_p.join(timeout=2.0)
        try: rife_proc.wait(timeout=5.0)
        except Exception: pass
        if rife_proc.poll() is None:
            rife_proc.terminate()
            try: rife_proc.wait(timeout=2.0)
            except Exception: rife_proc.kill()
        try: spout_rife.release()
        except Exception: pass
        for s in (shm_ctrl, shm_lat, shm_img, shm_depth, shm_ring, shm_dlat, shm_cam, shm_camlat):
            try: s.close()
            except Exception: pass
            try: s.unlink()
            except Exception: pass
        if len(fts_display) >= 2:
            final = (len(fts_display) - 1) / max(fts_display[-1] - fts_display[0], 1e-3)
            print(f"\n=== END  display fps_avg={final:.2f} ===", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
