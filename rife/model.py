"""
rife/model.py — RIFE Interpolator (PyTorch CUDA).

Wraps `IFNet_HDv3` from Practical_RIFE (MIT, by hzwer) for use on the dGPU.
Used by `rife/flow_worker.py` as a CUDA subprocess.

Requirements (conda env "cuda"):
    torch >= 2.4 with CUDA 12.x, numpy

Weights live in `models/rife/`:
    flownet.pkl         (~42 MB, full)
    flownet_small.pkl   (~12 MB, small — default for morpheus-cam)

See `download_models.py` for an automated fetch.

Usage:
    from rife.model import RIFEInterpolator
    rife = RIFEInterpolator(model_size="small")
    frames = rife.interpolate(frame_a, frame_b, n_passes=3)
    # → list of 9 np.ndarray H×W×3 uint8 (frame_a + 7 interp + frame_b)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIFE_DIR = os.path.join(ROOT, "models", "rife")


def _ensure_ifnet():
    """Import IFNet_HDv3 from this package (architecture is bundled, MIT)."""
    from rife.IFNet_HDv3 import IFNet
    return IFNet


class RIFEInterpolator:
    """
    High-level wrapper around IFNet_HDv3 (RIFE v4.6, PyTorch CUDA).

    Parameters
    ----------
    device     : str   "cuda" (default) or "cpu"
    fp16       : bool  use half precision on CUDA (default False)
    model_size : str   "full" (flownet.pkl) or "small" (flownet_small.pkl)
    """

    def __init__(self, device: str = "cuda", fp16: bool = False, model_size: str = "small"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[morpheus][RIFE] CUDA not available, falling back to CPU", flush=True)
            device = "cpu"
            fp16 = False

        self.device = device
        self.fp16 = fp16 and (device == "cuda")
        self._model = None

        fname = "flownet_small.pkl" if model_size == "small" else "flownet.pkl"
        weights_path = os.path.join(RIFE_DIR, fname)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"RIFE weights not found: {weights_path}\n"
                f"Place flownet.pkl (full) or flownet_small.pkl (small) in models/rife/.\n"
                f"See download_models.py for an automated fetch."
            )

        print(f"[morpheus][RIFE] loading IFNet_HDv3 on {device} (fp16={self.fp16})...", flush=True)
        IFNet = _ensure_ifnet()
        model = IFNet()

        state = torch.load(weights_path, map_location="cpu")
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        # block_tea is a teacher-only training branch — drop it.
        state = {k: v for k, v in state.items() if not k.startswith("block_tea")}
        # Drop unrelated heads (contextnet/unet) shipped in some checkpoints.
        model_keys = set(k for k, _ in model.named_parameters())
        model_keys |= set(k for k, _ in model.named_buffers())
        state = {k: v for k, v in state.items() if k in model_keys}
        model.load_state_dict(state, strict=True)

        model = model.to(device).eval()
        if self.fp16:
            model = model.half()
        self._model = model
        print("[morpheus][RIFE] model ready", flush=True)

    # ------------------------------------------------------------------ helpers
    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """np.ndarray H×W×3 uint8 → (1, 3, H, W) float tensor on device."""
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(self.device)
        if self.fp16:
            t = t.half()
        return t

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) float tensor → H×W×3 uint8."""
        t = t.squeeze(0).float().clamp(0, 1)
        return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _pad_to_multiple(self, t: torch.Tensor, multiple: int = 64):
        """Pad H and W to multiples of `multiple` for the network."""
        _, _, H, W = t.shape
        pH = (multiple - H % multiple) % multiple
        pW = (multiple - W % multiple) % multiple
        if pH or pW:
            t = F.pad(t, [0, pW, 0, pH])
        return t, H, W

    def _process_pair_tensor(self, ta: torch.Tensor, tb: torch.Tensor,
                             H: int, W: int) -> torch.Tensor:
        """
        Generate ONE midpoint frame from two padded device tensors.
        Crops back to (1, 3, H, W) and clamps to [0, 1] to prevent drift across passes.
        """
        x = torch.cat([ta, tb], 1)
        with torch.no_grad():
            merged, _, _ = self._model(x)
        return merged[:, :, :H, :W].clamp_(0, 1)

    # ------------------------------------------------------------------ public api
    def process_pair(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """Single midpoint between two H×W×3 uint8 RGB frames."""
        ta = self._to_tensor(frame_a)
        tb = self._to_tensor(frame_b)
        ta_p, H, W = self._pad_to_multiple(ta)
        tb_p, _, _ = self._pad_to_multiple(tb)
        mid = self._process_pair_tensor(ta_p, tb_p, H, W)
        return self._to_numpy(mid)

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        n_passes: int = 3,
    ) -> list[np.ndarray]:
        """
        Recursive midpoint interpolation between frame_a and frame_b.

        All work stays in CUDA tensor space (zero CPU↔GPU round-trips between
        passes). Numpy conversion happens only on the way out.

        n_passes=3 → 7 mid frames → list of 9 frames total:
            [frame_a, interp×7, frame_b]

        Total frames = 2**n_passes + 1.
        """
        ta = self._to_tensor(frame_a)
        tb = self._to_tensor(frame_b)
        ta_p, H, W = self._pad_to_multiple(ta)
        tb_p, _, _ = self._pad_to_multiple(tb)

        frames_t: list[torch.Tensor] = [ta_p, tb_p]

        for _ in range(n_passes):
            new_frames: list[torch.Tensor] = [frames_t[0]]
            for i in range(len(frames_t) - 1):
                mid = self._process_pair_tensor(frames_t[i], frames_t[i + 1], H, W)
                mid_p, _, _ = self._pad_to_multiple(mid)
                new_frames.append(mid_p)
                new_frames.append(frames_t[i + 1])
            frames_t = new_frames

        return [self._to_numpy(t[:, :, :H, :W]) for t in frames_t]
