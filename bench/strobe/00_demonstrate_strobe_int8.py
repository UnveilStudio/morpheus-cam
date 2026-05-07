"""
00_demonstrate_strobe_int8.py — does the INT8 quantised path strobe?

Loads the INT8-quantised SD Turbo UNet (the one shipped by AMD's GenAI-SD
reference, with the AMD-replaced custom ops baked in), then runs back-to-back
inference with **bit-identical input** at varying sleep intervals. Counts NaN
outputs at each sleep level.

What we observed on Ryzen AI 1.7.1: ~50/50 NaN/clean alternation at every
sleep level (1 s down to 0 s). That rules out a timing race — the symptom
tracks **call parity**, not wall-clock interval. The fp32 + vaiml_compile_v4
path on the same hardware is clean (see 01_verify_fix_fp32.py).

We don't claim to know whether the root cause is silicon, driver, runtime,
or a specific lowering pass. We claim only the measurement.

Run:
  conda run -n npu --no-capture-output python -u bench/strobe/00_demonstrate_strobe_int8.py

Requires:
  - AMD Ryzen AI 1.7.1 SDK installed
  - GenAI-SD reference at C:\\Program Files\\RyzenAI\\1.7.1\\GenAI-SD
  - SD Turbo NPU build at models/sd/sd_turbo/ (compiled via the GenAI-SD quicktest)
  - VAIP_CONFIG_JSON env var set
"""
from __future__ import annotations
import os, sys, time, copy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, r"C:\Program Files\RyzenAI\1.7.1\GenAI-SD")
os.environ["DISABLE_VAE_DD"] = "1"

SIZE       = 512
STEPS      = 1
SLEEPS     = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0]
CYCLES_PER = 10
SEED       = 42

print("=== NPU strobe sweep — INT8 path, bit-identical input, varying sleep ===\n")

print("Loading the AMD pipeline (VitisAI warmup ~20 s)...")
from src.StableDiffusionONNXPipelineTrigger import StableDiffusionONNXPipelineAMDTrigger
from src.pipeline_stable_diffusion_onnx_amd  import StableDiffusionONNXPipelineAMD
import torch

pipe_trigger = StableDiffusionONNXPipelineAMDTrigger(
    model_id="stabilityai/sd-turbo",
    model_path=os.path.join(ROOT, "models", "sd", "sd_turbo"),
    custom_op_path=r"C:\Program Files\RyzenAI\1.7.1\deployment\onnx_custom_ops.dll",
    enable_compile=False,
    gpu=False,
)
pipe = StableDiffusionONNXPipelineAMD(
    vae=pipe_trigger.vae_decoder,
    text_encoder=pipe_trigger.text_encoder,
    tokenizer=pipe_trigger.tokenizer,
    unet=pipe_trigger.unet,
    scheduler=pipe_trigger.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)

embeds, _ = pipe.encode_prompt(
    prompt="a test prompt",
    device=torch.device("cpu"),
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
    negative_prompt=None,
)
emb_np = embeds.detach().cpu().numpy().astype(np.float32)

gen = torch.Generator().manual_seed(SEED)
lat = torch.randn((1, 4, SIZE // 8, SIZE // 8), generator=gen, dtype=torch.float32)
lat = lat * pipe.scheduler.init_noise_sigma

sched = copy.deepcopy(pipe.scheduler)
sched.set_timesteps(STEPS)
t = sched.timesteps[0]
lat_in = sched.scale_model_input(lat, t)
sample_np = lat_in.numpy().astype(np.float32)
ts_np = np.array([float(t.item())], dtype=np.float64)

unet_sess = pipe.unet.model

print(f"\nInput pinned: sample mean={sample_np.mean():.3f} std={sample_np.std():.3f}")
print(f"timestep={ts_np[0]:.1f}  embeds shape={emb_np.shape}\n")

print("Warmup (3 calls, ignored)...")
for _ in range(3):
    _ = unet_sess.run(
        None,
        {"sample": sample_np, "timestep": ts_np, "encoder_hidden_states": emb_np},
    )

print("\nsleep(s)   NaN/total  pattern                          notes")
print("--------   ---------  -------------------------------  ----------")

results = []
for sl in SLEEPS:
    pattern = []
    nan_count = 0
    for i in range(CYCLES_PER):
        time.sleep(sl)
        out = unet_sess.run(
            None,
            {"sample": sample_np, "timestep": ts_np, "encoder_hidden_states": emb_np},
        )[0]
        is_nan = not np.isfinite(out).all()
        pattern.append("X" if is_nan else ".")
        if is_nan:
            nan_count += 1
    ratio = nan_count / CYCLES_PER
    if   nan_count == 0:                        note = "clean"
    elif nan_count == CYCLES_PER:               note = "ALL NaN"
    elif abs(ratio - 0.5) < 0.15:               note = "strobe ~50/50"
    else:                                       note = f"{ratio * 100:.0f}% NaN"
    print(f"{sl:7.3f}    {nan_count:2d}/{CYCLES_PER:<2d}      {''.join(pattern):<32s} {note}")
    results.append((sl, nan_count))

print("\n=== END ===")
print("Reading the table:")
print("  - 'strobe ~50/50' at every sleep level → NaN tracks call parity, not wall time.")
print("  - 'clean' high sleep + 'strobe' low sleep → would suggest a timing race instead.")
print("  - any non-alternating pattern → something else again.")
