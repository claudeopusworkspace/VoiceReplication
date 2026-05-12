"""Smoke test for Tortoise-TTS.

Loads Tortoise's TextToSpeech, uses the built-in `tom` voice samples, generates
a fixed sentence with preset='ultra_fast' (Tortoise's default 'standard' takes
10+ minutes per sentence; quality at ultra_fast is poor but smoke only cares
about pipeline-runs). Saves to tests/outputs/smoke/tortoise-tts.wav.

Tortoise emits a 24 kHz waveform shaped [1, 1, samples] (k=1 default).
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "tortoise-tts.wav"
VOICE = "tom"
PRESET = "ultra_fast"
SR = 24000  # Tortoise's fixed output sample rate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  torch={torch.__version__}", flush=True)
if device != "cuda":
    print("FATAL: CUDA not available — Tortoise smoke requires the 5090.", flush=True)
    sys.exit(1)
print(f"gpu={torch.cuda.get_device_name(0)}  arch_list={torch.cuda.get_arch_list()}", flush=True)

t0 = time.time()
tts = TextToSpeech()
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

print(f"loading voice samples for '{VOICE}'", flush=True)
voice_samples, conditioning_latents = load_voice(VOICE)

t0 = time.time()
wav = tts.tts_with_preset(
    SMOKE_TEXT,
    voice_samples=voice_samples,
    conditioning_latents=conditioning_latents,
    preset=PRESET,
)
t_gen = time.time() - t0

# Tortoise returns either a single tensor [1, 1, T] (k=1) or a list of them.
if isinstance(wav, list):
    wav = wav[0]
audio = wav.squeeze().cpu().numpy()  # -> [T]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), audio, SR)

duration = audio.shape[-1] / SR
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={SR})  preset={PRESET}  voice={VOICE}", flush=True)
sys.exit(0)
