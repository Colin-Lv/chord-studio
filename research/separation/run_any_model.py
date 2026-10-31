"""Run any audio-separator model on GPU and print the JSON result line."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default=r"D:\ChordStudio\data\models\audio_separator")
    parser.add_argument("--mode", default="autocast")
    args = parser.parse_args()

    import torch
    from audio_separator.separator import Separator

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)

    separator = Separator(
        log_level=logging.ERROR,
        model_file_dir=args.model_dir,
        output_dir=args.outdir,
        output_format="WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        use_autocast=(args.mode == "autocast"),
    )
    started = time.perf_counter()
    separator.load_model(model_filename=args.model)
    loaded = time.perf_counter()
    outputs = separator.separate(args.input)
    finished = time.perf_counter()
    print("RESULT " + json.dumps({
        "model": args.model,
        "device": str(separator.torch_device),
        "load_seconds": round(loaded - started, 2),
        "separate_seconds": round(finished - loaded, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 0)
        if torch.cuda.is_available() else 0,
        "outputs": outputs,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
