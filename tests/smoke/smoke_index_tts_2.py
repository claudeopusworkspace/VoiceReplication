"""Smoke test for IndexTTS-2.

Loads IndexTTS2 with default settings and synthesizes a fixed sentence using the
bundled `voice_01.wav` reference, then writes the result to
tests/outputs/smoke/index-tts-2.wav via soundfile (torchaudio 2.11 needs
torchcodec for ta.save, so we sidestep it).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# IndexTTS2 hard-codes relative paths (HF_HUB_CACHE, w2v-bert-2.0 download,
# bigvgan repo download). Run from the model dir so those land in the right
# place.
INDEXTTS_DIR = Path("/workspace/VoiceReplication/generators/index-tts-2")
os.chdir(INDEXTTS_DIR)
sys.path.insert(0, str(INDEXTTS_DIR))

from indextts.infer_v2 import IndexTTS2  # noqa: E402

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "index-tts-2.wav"
REF_AUDIO = INDEXTTS_DIR / "examples" / "voice_01.wav"

if not torch.cuda.is_available():
    print("CUDA not available — failing.", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

t0 = time.time()
tts = IndexTTS2(
    cfg_path=str(INDEXTTS_DIR / "checkpoints" / "config.yaml"),
    model_dir=str(INDEXTTS_DIR / "checkpoints"),
    device=device,
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False,
)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

t0 = time.time()
result = tts.infer(
    spk_audio_prompt=str(REF_AUDIO),
    text=SMOKE_TEXT,
    output_path=str(OUTPUT_PATH),
    verbose=False,
)
t_gen = time.time() - t0

# infer() with output_path writes the file itself. Read it back for sr/duration
# stats, and re-save via soundfile to match the rest of the bake-off.
wav, sr = sf.read(str(OUTPUT_PATH))
if wav.ndim > 1:
    wav = wav[:, 0]
sf.write(str(OUTPUT_PATH), wav.astype(np.float32), sr)

duration = len(wav) / sr
print(
    f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})",
    flush=True,
)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
