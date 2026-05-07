"""
rife/flow_worker.py — RIFE CUDA worker (subprocess).

Reads (prev, curr) pairs from the SHM image double-buffer (uint8 H×W×3),
interpolates with RIFE IFNet_HDv3 on CUDA (RTX dGPU), publishes N
intermediate frames into the SHM ring and bumps ctrl[4] = display_fid.

Runs in conda env `cuda` (PyTorch CUDA + RIFE weights). Independent of
the NPU / DML stack.

Spawned by morpheus_cam.py:
    conda run -n cuda --no-capture-output python -u rife/flow_worker.py \\
        --ctrl-name amd_cam_ctrl --img-name amd_cam_image --ring-name amd_cam_ring \\
        --size 512 --n-passes 3 --ring-slots 16 --model-size small

CTRL layout (int64, shared with the main process):
  [0] shutdown       (1 = stop)
  [1] latent_fid     (NPU thread → iGPU worker)
  [2] image_fid      (iGPU worker → RIFE worker)  ← we read this
  [3] depth_fid      (iGPU worker → main display)
  [4] display_fid    (RIFE worker → main, ring buffer counter)  ← we write this
  [5] cam_fid        (cam thread → iGPU worker)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CTRL_SLOTS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-name", required=True)
    ap.add_argument("--img-name",  required=True)
    ap.add_argument("--ring-name", required=True)
    ap.add_argument("--size",        type=int, default=512)
    ap.add_argument("--n-passes",    type=int, default=3,
                    help="RIFE recursive passes: 2→5 frames/pair, 3→9 frames/pair")
    ap.add_argument("--ring-slots",  type=int, default=16)
    ap.add_argument("--model-size",  default="small", choices=["full", "small"])
    ap.add_argument("--fp16",        action="store_true", help="use fp16 (default fp32)")
    args = ap.parse_args()

    SIZE = args.size
    IMG_SHAPE = (SIZE, SIZE, 3)
    N_TOTAL = 2 ** args.n_passes + 1

    print(f"[morpheus][RIFE] init CUDA RIFE ({args.model_size}) n_passes={args.n_passes} "
          f"→ {N_TOTAL} frames/pair ({N_TOTAL - 1} new)", flush=True)

    from rife.model import RIFEInterpolator
    rife = RIFEInterpolator(device="cuda", fp16=args.fp16, model_size=args.model_size)

    ctrl_shm = shared_memory.SharedMemory(name=args.ctrl_name)
    img_shm  = shared_memory.SharedMemory(name=args.img_name)
    ring_shm = shared_memory.SharedMemory(name=args.ring_name)
    ctrl       = np.ndarray((CTRL_SLOTS,), dtype=np.int64, buffer=ctrl_shm.buf)
    img_slots  = np.ndarray((2,) + IMG_SHAPE, dtype=np.uint8, buffer=img_shm.buf)
    ring_slots = np.ndarray((args.ring_slots,) + IMG_SHAPE, dtype=np.uint8, buffer=ring_shm.buf)

    print("[morpheus][RIFE] ready, waiting for first keyframe...", flush=True)

    last_consumed = 0
    prev_img = None
    rife_times: list[float] = []

    def publish(frame: np.ndarray):
        fid = int(ctrl[4]) + 1
        np.copyto(ring_slots[fid % args.ring_slots], frame)
        ctrl[4] = fid

    try:
        while ctrl[0] == 0:
            img_fid = int(ctrl[2])
            if img_fid <= last_consumed:
                time.sleep(0.002)
                continue

            slot = img_fid % 2
            curr_img = img_slots[slot].copy()
            last_consumed = img_fid

            if prev_img is None:
                publish(curr_img)
                prev_img = curr_img
                continue

            t0 = time.perf_counter()
            frames = rife.interpolate(prev_img, curr_img, n_passes=args.n_passes)
            rife_times.append(time.perf_counter() - t0)
            if len(rife_times) > 30:
                rife_times.pop(0)

            # frames[0] is prev_img (already published) — skip it.
            for f in frames[1:]:
                publish(f)

            prev_img = curr_img

            if int(ctrl[4]) % 30 < N_TOTAL:
                avg = sum(rife_times) / len(rife_times) * 1000
                print(f"[morpheus][RIFE] pair {last_consumed}  interp_ms={avg:.1f}  "
                      f"disp_fid={int(ctrl[4])}", flush=True)
    finally:
        print("[morpheus][RIFE] shutdown", flush=True)
        try: ctrl_shm.close()
        except Exception: pass
        try: img_shm.close()
        except Exception: pass
        try: ring_shm.close()
        except Exception: pass


if __name__ == "__main__":
    main()
