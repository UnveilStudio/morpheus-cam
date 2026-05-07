"""
spout_sender.py — Single Spout sender, auto-resolution.

Wraps the `UnveilStudio/SPOUT2ForPython` package (BSD-2 SpoutLibrary.dll
bundled there) and adapts the resolution on the fly. RGBA buffer is
pre-allocated; on resolution change we reallocate once.

Install the underlying package:
    pip install git+https://github.com/UnveilStudio/SPOUT2ForPython
"""
from __future__ import annotations
import ctypes
import numpy as np

try:
    from spout import SpoutSender, GL_RGBA
except ImportError as e:
    raise ImportError(
        "SPOUT2ForPython is not installed.\n"
        "Install it with:\n"
        "    pip install git+https://github.com/UnveilStudio/SPOUT2ForPython"
    ) from e


class SpoutOutput:
    """Single Spout sender, auto-adapts resolution."""

    def __init__(self, name: str = "AMD_AI_SD"):
        self._sender = SpoutSender(name)
        self._rgba = np.empty((512, 512, 4), dtype=np.uint8)
        self._rgba[..., 3] = 255
        self._buf_h = 512
        self._buf_w = 512
        print(f"[morpheus][spout] sender '{name}' created", flush=True)

    def _ensure_buf(self, h: int, w: int):
        if h != self._buf_h or w != self._buf_w:
            self._rgba = np.empty((h, w, 4), dtype=np.uint8)
            self._rgba[..., 3] = 255
            self._buf_h = h
            self._buf_w = w

    def send(self, rgb: np.ndarray):
        """Send an RGB uint8 H×W×3 frame at any resolution."""
        h, w = rgb.shape[:2]
        self._ensure_buf(h, w)
        self._rgba[..., :3] = rgb
        ptr = self._rgba.ctypes.data_as(ctypes.c_char_p)
        self._sender.send_image(ptr, w, h, GL_RGBA)

    @property
    def fps(self) -> float:
        return self._sender.fps

    def release(self):
        try:
            self._sender.release()
        except Exception:
            pass
        print("[morpheus][spout] sender released", flush=True)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    spout = SpoutOutput()
    print("Sending 200 test frames (animated gradient)...")
    print("Open TouchDesigner > Spout In TOP > select 'AMD_AI_SD'")

    for i in range(200):
        t = i / 30.0
        y = np.linspace(0, 1, 512).reshape(-1, 1).astype(np.float32)
        x = np.linspace(0, 1, 512).reshape(1, -1).astype(np.float32)
        r = ((np.sin(2 * np.pi * (x + t)) * 0.5 + 0.5) * 255).astype(np.uint8)
        g = ((np.sin(2 * np.pi * (y + t * 1.3)) * 0.5 + 0.5) * 255).astype(np.uint8)
        b = ((np.cos(2 * np.pi * (x - t * 0.7)) * 0.5 + 0.5) * 255).astype(np.uint8)
        frame = np.stack(
            [np.broadcast_to(r, (512, 512)),
             np.broadcast_to(g, (512, 512)),
             np.broadcast_to(b, (512, 512))],
            axis=-1,
        )
        spout.send(frame)
        if i % 30 == 0:
            print(f"  frame {i}  fps={spout.fps:.1f}")
        time.sleep(1 / 30)

    spout.release()
    print("done.")
