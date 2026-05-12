"""Smoke test for GPT-SoVITS (v2).

GPT-SoVITS is a zero-shot voice-cloning TTS — it needs a reference clip + its
transcript, not just the target text. We use the chatterbox smoke output as the
reference (it's a clear English clip of a known sentence, so the transcript is
known). Quality doesn't matter here; we only care that the pipeline loads,
runs end-to-end on CUDA, and produces a valid wav.

Gotchas this script works around:
  * GPT-SoVITS imports use bare 'GPT_SoVITS.*' paths and `tools.i18n` — must
    chdir into the repo root and add it to sys.path.
  * Pretrained weights aren't bundled; we expect them under
    GPT_SoVITS/pretrained_models/ (download per the README).
  * torchaudio in the gpt-sovits venv had to be pinned to 2.10.0+cu128 to
    match torch 2.10.0+cu128 (the default ships 2.11 which needs CUDA 13).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# --- Bootstrap: GPT-SoVITS must run from its own repo root ---
GPTSOVITS_DIR = Path("/workspace/VoiceReplication/generators/gpt-sovits")
os.chdir(GPTSOVITS_DIR)
sys.path.insert(0, str(GPTSOVITS_DIR))
sys.path.insert(0, str(GPTSOVITS_DIR / "GPT_SoVITS"))

# torchaudio 2.10+ delegates load/save to torchcodec, which isn't installed.
# Monkey-patch torchaudio.load to use soundfile + a torch tensor return, the
# shape/dtype TTS.py expects (channels, samples) float32 in [-1, 1].
import torchaudio  # noqa: E402

_sf_imported = sf  # alias to avoid shadowing in the patch closure


def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = _sf_imported.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns (samples, channels); torchaudio.load returns (channels, samples)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _torchaudio_load_via_soundfile

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "gpt-sovits.wav"

# Reference audio: reuse the chatterbox smoke output (same sentence, known transcript).
REF_AUDIO = Path(__file__).parent.parent / "outputs" / "smoke" / "chatterbox.wav"
REF_TEXT = SMOKE_TEXT  # chatterbox synthesized this exact text

if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke contract", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

if not REF_AUDIO.exists():
    print(f"Reference audio not found at {REF_AUDIO}; run smoke_chatterbox.py first.", flush=True)
    sys.exit(2)

# --- Build TTS pipeline ---
t0 = time.time()
config = TTS_Config({
    "device": device,
    "is_half": True,
    "version": "v2",
    "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
})
tts = TTS(config)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

# --- Run inference ---
req = {
    "text": SMOKE_TEXT,
    "text_lang": "en",
    "ref_audio_path": str(REF_AUDIO),
    "prompt_text": REF_TEXT,
    "prompt_lang": "en",
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "text_split_method": "cut5",
    "batch_size": 1,
    "speed_factor": 1.0,
    "seed": 42,
    "parallel_infer": True,
    "repetition_penalty": 1.35,
    "return_fragment": False,
    "streaming_mode": False,
}

t0 = time.time()
gen = tts.run(req)
sr, audio = next(gen)
# Drain any remaining fragments and concat (cut5 may yield multiple segments).
extra = []
for sr_i, audio_i in gen:
    extra.append(audio_i)
if extra:
    audio = np.concatenate([audio] + extra)
t_gen = time.time() - t0

# Audio comes back as int16 ndarray; soundfile.write handles that natively.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), audio, sr)

duration = audio.shape[-1] / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
