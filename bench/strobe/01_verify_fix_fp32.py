"""
01_verify_fix_fp32.py — does the fp32 + vaiml_compile_v4 path strobe?

Loads SD Turbo UNet exported as raw fp32 ONNX (no AMD-replaced custom ops)
on VitisAIExecutionProvider, then runs N back-to-back calls with bit-identical
fixed input. First run pays a one-time vaiml_compile_v4 compile cost (~15-25
min on a fresh cache, ~seconds on subsequent runs).

Pass criterion: 0 / N NaN frames. If we get that, the fp32 path is clean on
the same NPU silicon that strobes on the INT8 path — which is the whole
reason morpheus-cam ships fp32.

Run:
  conda run -n npu --no-capture-output python -u bench/strobe/01_verify_fix_fp32.py

Requires:
  - models/sd_turbo_unet_fp32.onnx  (exported via AMD GenAI-SD reference flow)
  - VAIP_CONFIG_JSON env var set
"""
from __future__ import annotations
import os, sys, time, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault(
    "VAIP_CONFIG_JSON",
    r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json",
)

UNET_PATH = os.path.join(ROOT, "models", "sd_turbo_unet_fp32.onnx")
VAIP_CFG  = os.environ["VAIP_CONFIG_JSON"]
N_CALL    = 20

print("=== SD Turbo UNet fp32 / VitisAI strobe verification ===\n")
if not os.path.exists(UNET_PATH):
    print(f"[error] {UNET_PATH} not found.")
    print("        Export it first via AMD's GenAI-SD reference flow.")
    sys.exit(1)

print(f"Model: {UNET_PATH}")
print(f"  size: {os.path.getsize(UNET_PATH) / 1e6:.1f} MB (file header only — heavy ops are external)\n")

import onnxruntime as ort

print("Load VitisAI...  (first run compiles vaiml_compile_v4, 15-25 min;")
print("                 subsequent runs use the cache at C:\\temp\\<user>\\vaip\\CACHE)")
t0 = time.perf_counter()
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(
    UNET_PATH,
    sess_options=so,
    providers=[
        ("VitisAIExecutionProvider", {"config_file": VAIP_CFG}),
        "CPUExecutionProvider",
    ],
)
print(f"Load OK in {time.perf_counter() - t0:.1f}s  provider={sess.get_providers()[0]}\n")

# Fixed bit-identical input
rng     = np.random.RandomState(42)
sample  = rng.randn(1, 4, 64, 64).astype(np.float32)
ts      = np.array([999.0], dtype=np.float32)
emb     = rng.randn(1, 77, 1024).astype(np.float32)
ts_inp  = next(i for i in sess.get_inputs() if "time" in i.name.lower())
if   "int64"  in ts_inp.type: ts = ts.astype(np.int64)
elif "double" in ts_inp.type: ts = ts.astype(np.float64)
else:                         ts = ts.astype(np.float32)

def h(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:12]

print(f"Input hashes (sanity): sample={h(sample)}  ts={h(ts)}  emb={h(emb)}\n")

feed = {"sample": sample, "timestep": ts, "encoder_hidden_states": emb}

print("Warmup (3 calls, ignored)...")
for _ in range(3):
    _ = sess.run(None, feed)

print(f"\nSweep {N_CALL} back-to-back calls with bit-identical input:")
print(f"{'#':>3} {'ms':>7} {'NaN':>5} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'pat':>3}")

nans, mss = [], []
for i in range(N_CALL):
    t0 = time.perf_counter()
    out = sess.run(None, feed)[0]
    ms = (time.perf_counter() - t0) * 1000
    finite = np.isfinite(out)
    nan_cnt = int((~finite).sum())
    f = out[finite] if finite.any() else np.array([0.0])
    print(f"{i:>3} {ms:7.1f} {nan_cnt:>5d} {f.min():+10.4f} {f.max():+10.4f} {f.mean():+10.4f} {f.std():10.4f} {'X' if nan_cnt else '.':>3}")
    nans.append(nan_cnt > 0)
    mss.append(ms)

total_nan = sum(nans)
pattern   = "".join("X" if n else "." for n in nans)
avg_ms    = sum(mss) / len(mss)

print("\n" + "=" * 70)
print("SUMMARY — SD Turbo UNet fp32 / VitisAI / vaiml_compile_v4")
print("=" * 70)
print(f"  NaN count   : {total_nan}/{N_CALL}")
print(f"  pattern     : {pattern}")
print(f"  avg latency : {avg_ms:.1f} ms  ({1000 / avg_ms:.2f} fps theoretical)")

print("\n=== VERDICT ===")
if total_nan == 0:
    print(f"OK  Zero NaN over {N_CALL} bit-identical calls.")
    print(f"    The fp32 + vaiml_compile_v4 path is clean on this hardware.")
    print(f"    Trade-off: ~{avg_ms:.0f} ms vs the INT8 path's ~110 ms (when it doesn't NaN).")
elif total_nan >= N_CALL // 3:
    print(f"FAIL  {total_nan}/{N_CALL} NaN — the symptom isn't isolated to the INT8 path on this machine.")
else:
    print(f"AMBIGUOUS  {total_nan}/{N_CALL} NaN — sporadic, not the deterministic strobe pattern we observed.")
