"""Smoke test for Bark (Suno) TTS.

Loads Bark via `preload_models()` and synthesizes a fixed sentence with
`generate_audio()`. Saves to tests/outputs/smoke/bark.wav. Quality doesn't
matter — we just want to prove the model loads, generates on the 5090, and
produces a .wav.

Bark also supports inline non-verbal tags like [laughs] / [sighs], but the
contract for this smoke test is plain text only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Bark ships checkpoints with pickled numpy scalars (and friends). torch >= 2.6
# defaults `torch.load` to weights_only=True, which rejects them. Monkey-patch
# before importing bark — same trick we use for XTTS-v2 (see install-gotchas).
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from bark import SAMPLE_RATE, generate_audio, preload_models  # noqa: E402

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "bark.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  torch={torch.__version__}", flush=True)
if device != "cuda":
    print("CUDA unavailable — Bark smoke test requires the 5090. Aborting.", flush=True)
    sys.exit(1)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
preload_models()
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
audio_array = generate_audio(SMOKE_TEXT)
t_gen = time.time() - t0

# Bark returns a 1-D float32 numpy array at 24 kHz.
audio_array = np.asarray(audio_array, dtype=np.float32)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), audio_array, SAMPLE_RATE)

duration = audio_array.shape[-1] / SAMPLE_RATE
print(
    f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})",
    flush=True,
)
print(f"wrote {OUTPUT_PATH} (sr={SAMPLE_RATE})", flush=True)
sys.exit(0)
