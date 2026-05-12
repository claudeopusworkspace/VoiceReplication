"""Smoke test for VoxCPM (OpenBMB tokenizer-free TTS).

Loads VoxCPM2 from the HuggingFace Hub (cached after first run) and runs voice
cloning using the bundled `examples/reference_speaker.wav` shipped in the
generator repo. Saves to tests/outputs/smoke/voxcpm.wav via soundfile.

Notes:
- VoxCPM2's reference-only cloning path (`reference_wav_path=...`) does not
  require a transcript of the reference audio — VoxCPM2 treats the reference
  via its ref-audio tokens. Simplest path that exercises the full pipeline.
- We pass `load_denoiser=False` to skip the ZipEnhancer download (denoiser
  is only used when `denoise=True` at generate time, which we don't need for
  a pipeline-runs smoke test).
- VoxCPM's constructor warms up the model with a short generation when
  `optimize=True` (torch.compile path), so loading is intentionally slower
  but generation should be fast.
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from voxcpm import VoxCPM

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "voxcpm.wav"

VOXCPM_DIR = Path("/workspace/VoiceReplication/generators/voxcpm")
REF_WAV = VOXCPM_DIR / "examples" / "reference_speaker.wav"

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
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    device=device,
)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

t0 = time.time()
wav = model.generate(
    text=SMOKE_TEXT,
    reference_wav_path=str(REF_WAV),
)
t_gen = time.time() - t0

sr = model.tts_model.sample_rate

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav, sr)

duration = len(wav) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
