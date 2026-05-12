"""Smoke test for NeuTTS Air (Neuphonic).

Loads the full-precision (non-GGUF) NeuTTS Air backbone + NeuCodec onto CUDA
and runs zero-shot voice cloning against the bundled `samples/jo.wav`
reference shipped in the upstream repo. Saves to
tests/outputs/smoke/neutts-air.wav via soundfile.

Notes:
- NeuTTS requires *both* a reference wav AND its transcript — it's an
  LLM-backbone TTS that conditions on encoded ref audio plus text. The
  upstream `samples/jo.wav` + `samples/jo.txt` pair is the cleanest fixture.
- Constructor takes `backbone_device` and `codec_device` separately (no
  single `device=` kwarg). We set both to "cuda" per the smoke contract;
  if CUDA isn't available we fail loudly.
- We bypass the on-disk `.pt` cache in `samples/` and re-encode the
  reference each run — the smoke test should exercise the encoder path.
- Sample rate is fixed at 24 kHz (`tts.sample_rate`).
- Repo ships a stale `output.wav` in the repo root from prior testing;
  we write to tests/outputs/smoke/neutts-air.wav instead.
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from neuttsair.neutts import NeuTTSAir

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "neutts-air.wav"

NEUTTS_DIR = Path("/workspace/VoiceReplication/specialized/neutts-air")
REF_WAV = NEUTTS_DIR / "samples" / "jo.wav"
REF_TXT = NEUTTS_DIR / "samples" / "jo.txt"

if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke-test contract.", flush=True)
    sys.exit(1)

if not REF_WAV.exists() or not REF_TXT.exists():
    print(f"reference assets missing: {REF_WAV} / {REF_TXT}", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

ref_text = REF_TXT.read_text().strip()

t0 = time.time()
tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",
    backbone_device=device,
    codec_repo="neuphonic/neucodec",
    codec_device=device,
)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
ref_codes = tts.encode_reference(str(REF_WAV))
wav = tts.infer(SMOKE_TEXT, ref_codes, ref_text)
t_gen = time.time() - t0

sr = tts.sample_rate

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav, sr)

duration = len(wav) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
