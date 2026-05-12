"""Smoke test for Coqui XTTS-v2.

Loads the XTTS-v2 model and runs zero-shot inference using a short reference
wav bundled inside the Coqui TTS repo (an LJSpeech sample). Saves to
tests/outputs/smoke/xtts-v2.wav. Quality doesn't matter — we just want a wav.

Notes:
- XTTS-v2 requires the CPML license acceptance prompt to be bypassed via the
  COQUI_TOS_AGREED=1 environment variable (we set it programmatically below so
  this script Just Works without env wrangling).
- torchaudio 2.11 needs torchcodec for `ta.save`, so we save via soundfile.
- TTS.api.TTS.tts() returns a python list of float samples; we wrap with numpy.
"""
import os

# Must be set before importing TTS — the license prompt fires at model download.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# XTTS-v2 checkpoints contain XttsConfig / XttsAudioConfig pickled objects.
# torch >= 2.6 defaults torch.load() to weights_only=True which rejects them.
# We trust the upstream Coqui weights, so force-default weights_only=False here.
_orig_torch_load = torch.load
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_weights_only_false

from TTS.api import TTS

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "xtts-v2.wav"

# Reference speaker wav — XTTS-v2 needs one even for zero-shot synthesis.
# Use a bundled LJSpeech sample (~9.6s) from the Coqui TTS repo's test data.
REF_WAV = Path(__file__).resolve().parents[2] / "generators" / "xtts-v2" / "tests" / "data" / "ljspeech" / "wavs" / "LJ001-0001.wav"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke-test contract.", flush=True)
    sys.exit(1)

if not REF_WAV.exists():
    print(f"reference wav missing: {REF_WAV}", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
tts = TTS(model_name=MODEL_NAME, progress_bar=False).to(device)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
wav = tts.tts(
    text=SMOKE_TEXT,
    speaker_wav=str(REF_WAV),
    language="en",
)
t_gen = time.time() - t0

wav_np = np.asarray(wav, dtype=np.float32)
sr = tts.synthesizer.output_sample_rate

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav_np, sr)

duration = len(wav_np) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
