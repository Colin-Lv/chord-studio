"""Run MDX23C-DrumSep through audio-separator and print a JSON result line.

Called by app/web/drums_worker.py with the CUDA interpreter.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("STEMS_DRUMSEP_DIR",
                                                             r"D:\ChordStudio\data\models\drumsep"))
    args = parser.parse_args()

    import torch
    from audio_separator.separator import Separator

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    separator = Separator(
        log_level=logging.ERROR,
        model_file_dir=args.model_dir,
        output_dir=args.outdir,
        output_format="WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        mdxc_params={},
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
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 0),
        "outputs": outputs,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
