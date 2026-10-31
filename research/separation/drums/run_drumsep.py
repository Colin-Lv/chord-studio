"""Run MDX23C-DrumSep (aufr33-jarredou) via audio-separator on GPU.

This model IS listed in audio-separator 0.48.0's bundled models.json
(mdx23c_download_list -> "MDX23C Model: MDX23C DrumSep by aufr33-jarredou"),
so no models.json shim is needed: load_model() finds the registry entry and
only skips the download because the two files already exist in ./models.

Usage:
  python run_drumsep.py --input <wav> --outdir <dir> [--overlap N] [--batch N]
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
YAML = "config_drumsep_mdx23c.yaml"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

from audio_separator.separator import Separator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--normalize", type=float, default=1.0,
                    help="normalization_threshold; 1.0 + amplification 0 == identity")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    mdxc_params = {}
    if args.overlap is not None:
        mdxc_params["overlap"] = args.overlap
    if args.batch is not None:
        mdxc_params["batch_size"] = args.batch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sep = Separator(
        log_level=logging.INFO,
        model_file_dir=MODEL_DIR,
        output_dir=args.outdir,
        output_format="WAV",
        normalization_threshold=args.normalize,
        amplification_threshold=0.0,
        mdxc_params=mdxc_params,
    )
    device = str(sep.torch_device)

    t0 = time.perf_counter()
    sep.load_model(model_filename=MODEL)
    t_load = time.perf_counter() - t0
    torch.cuda.synchronize()
    mem_after_load = torch.cuda.max_memory_allocated() / 1024**2

    t0 = time.perf_counter()
    outputs = sep.separate(args.input)
    torch.cuda.synchronize()
    t_sep = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    result = {
        "input": args.input,
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "load_seconds": round(t_load, 3),
        "separate_seconds": round(t_sep, 3),
        "total_seconds": round(t_load + t_sep, 3),
        "peak_vram_allocated_mb": round(peak_alloc, 1),
        "peak_vram_reserved_mb": round(peak_reserved, 1),
        "vram_after_load_mb": round(mem_after_load, 1),
        "outputs": outputs,
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
