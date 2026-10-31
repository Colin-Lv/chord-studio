"""Which demucs dependencies already exist in the GPU runtime environment?"""
import importlib

import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"vram {free / 2**30:.1f}/{total / 2**30:.1f} GiB")

for name in ("demucs", "einops", "julius", "lameenc", "openunmix", "dora_search",
             "yaml", "tqdm", "soundfile", "torchaudio", "huggingface_hub"):
    try:
        module = importlib.import_module(name)
        print(f"  {name:16} {getattr(module, '__version__', '?')}")
    except Exception as exc:                 # noqa: BLE001
        print(f"  {name:16} MISSING ({type(exc).__name__})")
