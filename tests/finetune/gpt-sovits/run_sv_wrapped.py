"""Run prepare_datasets/2-get-sv.py with torchaudio.load monkey-patched.

The stock script calls torchaudio.load on each wav, which in torchaudio 2.11+
requires torchcodec — which fails to load without the right system ffmpeg libs.
Patch the load function to use soundfile (same trick the harness uses).

All env vars set by the caller are inherited.
"""
import runpy
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _torchaudio_load_via_soundfile

repo_root = Path("/workspace/VoiceReplication/generators/gpt-sovits")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "GPT_SoVITS"))

# Run the prepare_datasets script as __main__
runpy.run_path(
    str(repo_root / "GPT_SoVITS/prepare_datasets/2-get-sv.py"),
    run_name="__main__",
)
