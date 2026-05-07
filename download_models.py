"""
download_models.py — Fetch the open-weights bits of the morpheus-cam stack.

What this script does NOT download:
  - Stable Diffusion Turbo NPU build (`models/sd/sd_turbo/unet/...`).
    Compile it yourself via the Ryzen AI 1.7.1 SDK GenAI-SD quicktest, or pull
    a pre-built drop from HuggingFace — see README "NPU model setup".

What this script DOES download:
  - TAESD decoder + encoder (FP16) for SD latents     → models/sd/sd_turbo/{vae_decoder, vae_encoder}/
  - Depth Anything V2 small (FP16 ONNX, fixed shape)  → models/
  - RIFE v4.6 'small' weights guidance                 → models/rife/ (manual GDrive)

All sources are open weights with permissive licences (see README "Built on top of").
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SD_DIR = MODELS / "sd" / "sd_turbo"
RIFE_DIR = MODELS / "rife"

for d in (MODELS, SD_DIR, SD_DIR / "vae_decoder", SD_DIR / "vae_encoder", RIFE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _hf_dl(repo: str, filename: str, dest: Path):
    """Download a single file from a HuggingFace repo into `dest`."""
    from huggingface_hub import hf_hub_download
    print(f"  → {repo} :: {filename}", flush=True)
    p = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest),
                        local_dir_use_symlinks=False)
    print(f"     {p}", flush=True)
    return p


def main():
    print("[morpheus][download] TAESD decoder + encoder...")
    # madebyollin/taesd — Tiny Autoencoder for SD (MIT)
    # The repo ships .pth, .safetensors, .ckpt and ONNX exports. We want the ONNX ones.
    # NOTE: file paths inside the HF repo may vary — adjust if upstream renames.
    try:
        _hf_dl("madebyollin/taesd", "taesd_decoder.onnx", SD_DIR / "vae_decoder")
    except Exception as e:
        print(f"    [warn] decoder fetch failed: {e}")
        print("    Manual: https://huggingface.co/madebyollin/taesd  → taesd_decoder.onnx")
    try:
        _hf_dl("madebyollin/taesd", "taesd_encoder_fp16.onnx", SD_DIR / "vae_encoder")
    except Exception as e:
        print(f"    [warn] encoder fetch failed: {e}")
        print("    Manual: https://huggingface.co/madebyollin/taesd  → taesd_encoder_fp16.onnx")

    print("\n[morpheus][download] Depth Anything V2 small (FP16 ONNX)...")
    # depth-anything/Depth-Anything-V2-Small — Apache 2.0
    # The "fixed_fp16" file is a fixed-shape FP16 ONNX export; if upstream doesn't
    # ship it, you can re-export from the Depth-Anything-V2 repo with onnxsim.
    try:
        _hf_dl("depth-anything/Depth-Anything-V2-Small",
               "depth_anything_v2_small_fixed_fp16.onnx", MODELS)
    except Exception as e:
        print(f"    [warn] depth fetch failed: {e}")
        print("    Manual: re-export from https://github.com/DepthAnything/Depth-Anything-V2 "
              "to a fixed 518×518 FP16 ONNX, place at "
              "models/depth_anything_v2_small_fixed_fp16.onnx")

    print("\n[morpheus][download] RIFE v4.6 weights (small)...")
    # Practical-RIFE distributes weights via Google Drive (no official HF repo).
    # The IFNet architecture is bundled in rife/IFNet_HDv3.py (MIT, hzwer).
    # Only the .pkl checkpoint needs a manual fetch.
    flownet_small = RIFE_DIR / "flownet_small.pkl"
    if flownet_small.exists():
        print(f"    [skip] {flownet_small.name} already present")
    else:
        print("    Practical-RIFE ships weights via Google Drive — manual fetch:")
        print("      1. Open https://github.com/hzwer/Practical-RIFE")
        print("      2. Download the v4.6 'small' checkpoint package")
        print("      3. Place flownet_small.pkl into models/rife/")
        print("    (flownet.pkl, the ~42 MB full model, is optional — the small one")
        print("     is the default and sufficient for morpheus-cam.)")

    print("\n" + "=" * 70)
    print("[morpheus][download] DONE.")
    print("=" * 70)
    print("\nReminder: the SD Turbo NPU build is NOT downloaded by this script.")
    print("Compile it yourself via the Ryzen AI 1.7.1 SDK GenAI-SD quicktest —")
    print("see the README section 'NPU model setup' for the exact steps.\n")


if __name__ == "__main__":
    main()
