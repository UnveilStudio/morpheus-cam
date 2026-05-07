# NPU strobe reproduction

These scripts let you reproduce, on your own Ryzen AI 1.7.1 machine, the symptom that pushed `morpheus-cam` to ship the **fp32 + `vaiml_compile_v4`** path instead of INT8 quantised inference.

We do **not** claim to know whether the root cause is silicon, driver, runtime, or a specific lowering pass. We claim only what we measured. Run these on your hardware and form your own opinion.

## What you need

- AMD Ryzen AI 1.7.1 SDK installed at `C:\Program Files\RyzenAI\1.7.1\`
- `VAIP_CONFIG_JSON` env var set to `voe-4.0-win_amd64\vaip_config.json`
- The `npu` conda env (Ryzen AI SDK env)
- The SD Turbo NPU build (compiled via Ryzen AI GenAI-SD quicktest — see main README §4)
- For `00_demonstrate_strobe_int8.py`: the **INT8-quantised** UNet shipped by the GenAI-SD example
- For `01_verify_fix_fp32.py`: the **fp32** UNet exported via `_04_export_sd_turbo_unet_fp32.py` (also from the AMD reference)

## What each script does

### `00_demonstrate_strobe_int8.py` — sleep-sweep on the INT8 path

Loads the INT8 UNet that GenAI-SD ships, then runs **back-to-back inference with bit-identical input** at decreasing sleep intervals (1.0 s, 0.5 s, … 0 s), counting NaN outputs each time.

**Pass criterion**: clean output at all sleep levels.
**What we observed**: ~50/50 NaN/clean alternation at every sleep level — the symptom is not a timing race; it tracks call parity, not wall-clock interval.

### `01_verify_fix_fp32.py` — same input, fp32 + `vaiml_compile_v4` path

Loads the fp32 UNet on `VitisAIExecutionProvider` (which uses `vaiml_compile_v4` lowering), runs **20 back-to-back calls with the exact same fixed input**, counts NaNs.

**Pass criterion**: 0 / 20 NaN.
**What we observed**: 0 / 20 NaN, sustained, on the same hardware that strobes on INT8.

The first run pays a one-time `vaiml_compile_v4` compile cost (~15-25 minutes on a fresh cache, then under a second on subsequent runs).

## Reference results

A reference run on our test bench is committed at [`results_reference.json`](results_reference.json). That's *our* numbers on *our* machine — yours will differ in latency. What should match is the **NaN/total ratio**: ~50/50 on INT8, 0/N on fp32.

## Disclaimer

These scripts depend on AMD-provided SDK code and AMD-provided model bundles, neither of which we redistribute. They will not run on a clean checkout of `morpheus-cam` alone.
