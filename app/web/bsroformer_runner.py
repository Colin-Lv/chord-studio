"""Run the BS-RoFormer vocals checkpoint through audio-separator.

Called as a subprocess by the stems worker with the CUDA interpreter. Kept deliberately
close to the research script so the measured behaviour (peak VRAM 1.6 GB, 0.7x real time)
is the same one the product gets.

audio-separator 0.48 no longer lists this checkpoint in models.json, so the lookup is patched
in memory to point at the local file + yaml.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
YAML = "model_bs_roformer_ep_317_sdr_12.9755.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("STEMS_ROFORMER_DIR",
                                                             r"D:\ChordStudio\data\models\bsroformer"))
    parser.add_argument("--mode", default="autocast", choices=["autocast", "native_fp16", "fp32"])
    args = parser.parse_args()

    import torch
    from audio_separator.separator import Separator

    original = Separator.download_model_files

    def patched(self, model_filename):
        if model_filename == MODEL:
            path = os.path.join(self.model_file_dir, MODEL)
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint missing: {path}")
            self.model_friendly_name = "BS-RoFormer ep_317 (2-stem vocals)"
            self.model_is_uvr_vip = False
            return MODEL, "MDXC", self.model_friendly_name, path, YAML
        return original(self, model_filename)

    Separator.download_model_files = patched

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=args.model_dir,
        output_dir=args.outdir,
        output_format="WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        use_autocast=(args.mode == "autocast"),
        use_native_fp16=(args.mode == "native_fp16"),
    )
    started = time.perf_counter()
    separator.load_model(model_filename=MODEL)
    loaded = time.perf_counter()
    outputs = separator.separate(args.input)
    finished = time.perf_counter()
    print("RESULT " + json.dumps({
        "device": str(separator.torch_device),
        "load_seconds": round(loaded - started, 2),
        "separate_seconds": round(finished - loaded, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 0) if device == "cuda" else 0,
        "outputs": outputs,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
