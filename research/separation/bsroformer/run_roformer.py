"""Run BS-RoFormer (audio-separator) on GPU and report wall time + peak VRAM.

audio-separator 0.48.0 does not list model_bs_roformer_ep_317_sdr_12.9755.ckpt in its
bundled models.json (it now ships Mel-Band models), so `Separator.download_model_files`
raises "not found in supported model files". We patch that one lookup in memory only
(no files outside this folder are modified) to point at the already-downloaded
checkpoint + its official config yaml from the model-configs release.

Usage:
  python run_roformer.py --input <wav> --outdir <dir> [--mode autocast|native_fp16|fp32]
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
YAML = "model_bs_roformer_ep_317_sdr_12.9755.yaml"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

from audio_separator.separator import Separator  # noqa: E402

_orig_download_model_files = Separator.download_model_files


def _patched_download_model_files(self, model_filename):
    if model_filename == MODEL:
        path = os.path.join(self.model_file_dir, MODEL)
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint missing: {path}")
        self.model_friendly_name = "BS-Roformer ep_317 (SDR 12.9755, 2-stem vocals)"
        self.model_is_uvr_vip = False
        return MODEL, "MDXC", self.model_friendly_name, path, YAML
    return _orig_download_model_files(self, model_filename)


Separator.download_model_files = _patched_download_model_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="autocast", choices=["autocast", "native_fp16", "fp32"])
    ap.add_argument("--chunk-duration", type=float, default=None,
                    help="optional audio-level chunk duration in seconds")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sep = Separator(
        log_level=logging.INFO,
        model_file_dir=MODEL_DIR,
        output_dir=args.outdir,
        output_format="WAV",
        # normalize() only scales when peak > max_peak; 1.0 + 0.0 == identity,
        # which keeps sum(stems) comparable to the original for the residual test.
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        use_autocast=(args.mode == "autocast"),
        use_native_fp16=(args.mode == "native_fp16"),
        chunk_duration=args.chunk_duration,
    )
    device = str(sep.torch_device)  # library already resolved cuda/cpu (torch.device)

    t0 = time.perf_counter()
    sep.load_model(model_filename=MODEL)
    t_load = time.perf_counter() - t0
    mem_after_load = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0

    t0 = time.perf_counter()
    outputs = sep.separate(args.input)
    torch.cuda.synchronize()
    t_sep = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2 if device == "cuda" else 0.0

    result = {
        "input": args.input,
        "mode": args.mode,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "load_seconds": round(t_load, 3),
        "separate_seconds": round(t_sep, 3),
        "peak_vram_allocated_mb": round(peak_alloc, 1),
        "peak_vram_reserved_mb": round(peak_reserved, 1),
        "vram_after_load_mb": round(mem_after_load, 1),
        "outputs": outputs,
        "total_seconds": round(t_load + t_sep, 3),
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
