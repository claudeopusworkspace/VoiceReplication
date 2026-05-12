"""Smoke test for F5-TTS.

Loads the F5TTS_v1_Base model and runs zero-shot inference using the bundled
English reference audio + transcript shipped with the package. Saves to
tests/outputs/smoke/f5-tts.wav. Quality doesn't matter — we just want a wav.

Note: filename uses underscore (smoke_f5_tts.py) so it's a valid Python module;
output wav keeps the canonical f5-tts.wav name.
"""
import sys
import time
from importlib.resources import files
from pathlib import Path

import soundfile as sf
import torch

from f5_tts.api import F5TTS

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "f5-tts.wav"

# Bundled reference audio + transcript (zero-shot still needs a reference voice).
REF_AUDIO = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
REF_TEXT = "Some call me nature, others call me mother nature."

if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke-test contract.", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
model = F5TTS(model="F5TTS_v1_Base", device=device)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
wav, sr, _spec = model.infer(
    ref_file=REF_AUDIO,
    ref_text=REF_TEXT,
    gen_text=SMOKE_TEXT,
)
t_gen = time.time() - t0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav, sr)

duration = len(wav) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
