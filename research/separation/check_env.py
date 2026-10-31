"""Which separation-relevant packages are installed in this interpreter?"""
import importlib
import sys

print("python", sys.version.split()[0])
print("exe", sys.executable)
for name in ("torch", "torchaudio", "demucs", "onnxruntime", "soundfile", "librosa",
             "numpy", "scipy", "transformers", "torchcrepe", "samplerate"):
    try:
        module = importlib.import_module(name)
        print(f"  {name:14} {getattr(module, '__version__', '?')}")
    except Exception as exc:                     # noqa: BLE001
        print(f"  {name:14} MISSING ({type(exc).__name__})")
try:
    import torch
    print("  cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  vram free {free / 2**30:.1f} GiB / {total / 2**30:.1f} GiB")
except Exception as exc:                         # noqa: BLE001
    print("  cuda check failed:", exc)
