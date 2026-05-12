"""Smoke test for OmniVoice (k2-fsa, diffusion language model TTS).

Loads `k2-fsa/OmniVoice` from HuggingFace (cached after first run) and runs
zero-shot synthesis in **Auto Voice** mode — no reference audio, no instruct
attributes. This is the cleanest pipeline-runs check because it doesn't depend
on any bundled reference clip (the omnivoice repo's `examples/` folder ships
only training/finetune scripts, no audio).

Saves to tests/outputs/smoke/omnivoice.wav via soundfile.

Notes:
- Auto Voice mode: `model.generate(text=...)` returns a list of np.ndarray
  waveforms at `model.sampling_rate` (24 kHz per the README).
- We use `dtype=torch.float16` and `device_map="cuda:0"` per the README's
  NVIDIA example. Fails loudly if CUDA isn't present.
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from omnivoice import OmniVoice

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "omnivoice.wav"

if not torch.cuda.is_available():
    print("CUDA not available - failing loudly per smoke-test contract.", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
audio = model.generate(text=SMOKE_TEXT)
t_gen = time.time() - t0

# `generate` returns a list of np.ndarray waveforms; one entry for our single text.
wav = audio[0]
sr = model.sampling_rate

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav, sr)

duration = len(wav) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
