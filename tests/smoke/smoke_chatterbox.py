"""Smoke test for Chatterbox TTS.

Runs the base ChatterboxTTS model with its default voice on a fixed sentence,
saves to tests/outputs/smoke/chatterbox.wav. No reference voice yet — that
comes when we run the full bake-off against character data.
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "chatterbox.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  torch={torch.__version__}", flush=True)
if device == "cuda":
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
model = ChatterboxTTS.from_pretrained(device=device)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
wav = model.generate(SMOKE_TEXT)
t_gen = time.time() - t0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav.squeeze(0).cpu().numpy(), model.sr)

duration = wav.shape[-1] / model.sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={model.sr})", flush=True)
sys.exit(0)
