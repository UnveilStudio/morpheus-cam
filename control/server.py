"""
control/server.py — HTTP + WebSocket control + telemetry server
for the morpheus-cam (body-driven SD) pipeline.

Thread-safe state, broadcast, daemon threads. Slim by design:
  - controls: prompt_text, cam_mix, noise_a, depth_g, evolve, alternance, cv2_preview, running
  - read-only metrics: cycle, sd_fps, ms_step, cam_fid, d_mean, hardware utilisation

A background sampler pushes hardware metrics (CPU, RAM, NPU, iGPU, dGPU,
dGPU temp/power) to all WS clients once per second. PowerShell is spawned
once and kept alive on stdin to avoid the ~250 ms per-call cost of cold-starting
powershell.exe.

Imported as:
    from control.server import CamRifeControlState, start_servers
"""
from __future__ import annotations
import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import psutil
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))


class CamRifeControlState:
    """Thread-safe shared state for the morpheus-cam pipeline."""

    CONTROL_DEFAULTS = {
        "prompt_text": "",
        "cam_mix":     30,
        "noise_a":     30,
        "depth_g":     60,
        "evolve":      True,
        "alternance":  False,
        "cv2_preview": False,
        "running":     True,
    }

    METRIC_DEFAULTS = {
        "cycle":       0,
        "sd_fps":      0.0,
        "ms_step":     0.0,
        "cam_fid":     0,
        "d_mean":      0.5,
        "cpu_pct":     0.0,
        "ram_pct":     0.0,
        "npu_pct":     0.0,
        "npu_derived": True,
        "igpu_pct":    0.0,
        "dgpu_pct":    0.0,
        "dgpu_temp":   0.0,
        "dgpu_power":  0.0,
    }

    def __init__(self, default_prompt: str):
        self._lock = threading.Lock()
        self._values = dict(self.CONTROL_DEFAULTS)
        self._values["prompt_text"] = default_prompt
        self._values.update(self.METRIC_DEFAULTS)
        self._callbacks: list = []
        self._t0 = time.time()

    def set_value(self, name: str, value):
        if name not in self.CONTROL_DEFAULTS:
            return
        if name == "prompt_text":
            value = str(value)[:1000].strip()
            if not value:
                return
        elif name == "cam_mix":
            value = max(0, min(100, int(value)))
        elif name == "noise_a":
            value = max(0, min(100, int(value)))
        elif name == "depth_g":
            value = max(0, min(150, int(value)))
        elif name in ("evolve", "alternance", "cv2_preview", "running"):
            value = bool(value)
        with self._lock:
            self._values[name] = value
        for cb in self._callbacks:
            try:
                cb(name, value)
            except Exception:
                pass

    def get_value(self, name: str, default=None):
        with self._lock:
            return self._values.get(name, default)

    def update_metrics(self, **kwargs):
        changed = []
        with self._lock:
            for k, v in kwargs.items():
                if k in self.METRIC_DEFAULTS and self._values.get(k) != v:
                    self._values[k] = v
                    changed.append((k, v))
        if changed:
            _push_metrics(dict(changed))

    def snapshot(self) -> dict:
        with self._lock:
            d = dict(self._values)
        d["uptime"] = time.time() - self._t0
        return d

    def on_change(self, callback):
        self._callbacks.append(callback)


# ─── Hardware sampler ─────────────────────────────────────────────
class _HardwareSampler:
    """
    Samples CPU / RAM / NPU / iGPU / dGPU utilisation every second.

    Strategy:
      - CPU/RAM:   psutil
      - iGPU/dGPU: long-lived powershell.exe process. We pipe a Get-Counter
                   loop on stdin (cheaper than spawning a PS host per tick).
      - dGPU temp/W: nvidia-smi (~30 ms; spawned once per tick is fine).
      - NPU%:      derived from ms_step  ( min(100, 100 * ms_step / 1000) )
                   because Windows perfmon does not expose the XDNA2 'NPU 0' engine
                   under the \\GPU Engine path reliably — Get-Counter only sees the
                   iGPU + dGPU adapters by LUID. Marked npu_derived=True in snapshot.
    """

    PS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
    try {
        $samples = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue -SampleInterval 1 -MaxSamples 1).CounterSamples
        $byLuid = @{}
        foreach ($s in $samples) {
            if ($s.CookedValue -le 0) { continue }
            if ($s.Path -match 'luid_0x[0-9A-Fa-f]+_0x([0-9A-Fa-f]+)') {
                $luid = $Matches[1]
                if (-not $byLuid.ContainsKey($luid)) { $byLuid[$luid] = 0.0 }
                $byLuid[$luid] += [double]$s.CookedValue
            }
        }
        $obj = @{}
        foreach ($k in $byLuid.Keys) { $obj[$k] = [math]::Min(100.0, $byLuid[$k]) }
        ConvertTo-Json -Compress -InputObject $obj
    } catch { Write-Output '{}' }
    Start-Sleep -Milliseconds 1000
}
"""

    def __init__(self, state: CamRifeControlState):
        self.state = state
        self._stop = threading.Event()
        self._ps = None
        self._luid_to_role: dict[str, str] = {}
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="hw-sampler").start()

    def stop(self):
        self._stop.set()
        if self._ps:
            try: self._ps.terminate()
            except Exception: pass

    def _spawn_ps(self):
        try:
            self._ps = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", self.PS_SCRIPT],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as e:
            print(f"[morpheus][hw] PowerShell spawn failed: {e}", flush=True)
            self._ps = None

    def _classify_luids(self, by_luid: dict[str, float]):
        # On the first tick that sees ≥ 2 active LUIDs, the busier one is the
        # iGPU (TAESD + Depth always running) and the other is the dGPU (RIFE).
        if len(self._luid_to_role) >= 2:
            return
        if not by_luid:
            return
        sorted_luids = sorted(by_luid.items(), key=lambda kv: -kv[1])
        if len(sorted_luids) >= 2:
            self._luid_to_role[sorted_luids[0][0]] = "igpu"
            self._luid_to_role[sorted_luids[1][0]] = "dgpu"
        elif len(sorted_luids) == 1 and not self._luid_to_role:
            self._luid_to_role[sorted_luids[0][0]] = "igpu"

    def _read_nvidia(self) -> tuple[float, float]:
        if not self._has_nvidia_smi:
            return 0.0, 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=2.0, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).strip().splitlines()[0]
            t, p = (s.strip() for s in out.split(","))
            return float(t), float(p)
        except Exception:
            return 0.0, 0.0

    def _run(self):
        self._spawn_ps()
        psutil.cpu_percent(interval=None)
        time.sleep(0.5)

        while not self._stop.is_set():
            tick_start = time.perf_counter()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            igpu_pct = 0.0
            dgpu_pct = 0.0
            if self._ps and self._ps.poll() is None:
                line = self._ps.stdout.readline().strip() if self._ps.stdout else ""
                if line:
                    try:
                        by_luid = json.loads(line)
                        if by_luid:
                            self._classify_luids({k: float(v) for k, v in by_luid.items()})
                            for luid, role in self._luid_to_role.items():
                                pct = float(by_luid.get(luid, 0.0))
                                if role == "igpu":   igpu_pct = pct
                                elif role == "dgpu": dgpu_pct = pct
                    except Exception:
                        pass
            else:
                self._spawn_ps()

            ms_step = float(self.state.get_value("ms_step", 0.0))
            npu_pct = min(100.0, 100.0 * ms_step / 1000.0) if ms_step > 0 else 0.0

            t_c, p_w = self._read_nvidia()

            self.state.update_metrics(
                cpu_pct=round(cpu, 1),
                ram_pct=round(ram, 1),
                igpu_pct=round(igpu_pct, 1),
                dgpu_pct=round(dgpu_pct, 1),
                npu_pct=round(npu_pct, 1),
                npu_derived=True,
                dgpu_temp=round(t_c, 1),
                dgpu_power=round(p_w, 1),
            )

            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, 1.0 - elapsed))


# ─── HTTP handler ────────────────────────────────────────────────
def _make_http_handler(state: CamRifeControlState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/panel.html"):
                p = os.path.join(HERE, "panel.html")
                with open(p, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
                return
            if path == "/state":
                self._send(200, json.dumps(state.snapshot()).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")
    return H


# ─── WebSocket plumbing ──────────────────────────────────────────
_ws_clients: set = set()
_ws_loop = None


async def _ws_handler(ws, state: CamRifeControlState):
    _ws_clients.add(ws)
    try:
        await ws.send(json.dumps({"type": "state", "data": state.snapshot()}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
                name = msg.get("name")
                value = msg.get("value")
                if name is not None:
                    state.set_value(name, value)
                    await _broadcast(
                        {"type": "update", "name": name, "value": state.get_value(name)},
                        exclude=ws,
                    )
            except Exception as e:
                print(f"[morpheus][ws] bad msg: {e}", flush=True)
    finally:
        _ws_clients.discard(ws)


async def _broadcast(msg: dict, exclude=None):
    if not _ws_clients:
        return
    raw = json.dumps(msg)
    for c in list(_ws_clients):
        if c is exclude:
            continue
        try:
            await c.send(raw)
        except Exception:
            _ws_clients.discard(c)


def _notify_ws(name, value):
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "update", "name": name, "value": value}),
            _ws_loop,
        )


def _push_metrics(changed: dict):
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "metrics", "data": changed}),
            _ws_loop,
        )


# ─── Lifecycle ───────────────────────────────────────────────────
def start_servers(state: CamRifeControlState,
                  http_port: int = 54331,
                  ws_port: int = 54332) -> None:
    global _ws_loop

    handler = _make_http_handler(state)
    httpd = HTTPServer(("127.0.0.1", http_port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http-morpheus").start()

    def _run_ws():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)

        async def main():
            async with websockets.serve(
                lambda w: _ws_handler(w, state),
                "127.0.0.1", ws_port,
                max_size=16384, compression=None,
            ):
                await asyncio.Future()

        _ws_loop.run_until_complete(main())

    threading.Thread(target=_run_ws, daemon=True, name="ws-morpheus").start()

    state.on_change(_notify_ws)

    sampler = _HardwareSampler(state)
    sampler.start()

    print(f"[morpheus][server] Panel: http://127.0.0.1:{http_port}/", flush=True)
    print(f"[morpheus][server] WS:    ws://127.0.0.1:{ws_port}/", flush=True)
