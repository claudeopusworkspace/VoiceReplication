"""Smoke test for CosyVoice 2 (0.5B).

CosyVoice2 is a two-stage zero-shot voice-cloning TTS — text -> semantic
tokens via a Qwen-based LM, then tokens -> waveform via flow-matching +
HiFi-GAN. It needs a reference clip + its transcript, not just the target
text. We use the bundled `asset/zero_shot_prompt.wav` (Chinese sample with a
known transcript from the upstream example.py). Quality doesn't matter here;
we only care that the pipeline loads, runs end-to-end on CUDA, and produces
a valid wav.

Gotchas this script works around:
  * Repo uses bare imports like `from cosyvoice.cli.cosyvoice import ...`
    and the bundled `example.py` appends `third_party/Matcha-TTS` to sys.path.
    We mirror that and chdir into the repo root.
  * Weights aren't bundled — we pull `FunAudioLLM/CosyVoice2-0.5B` from
    HuggingFace into `pretrained_models/CosyVoice2-0.5B/` on first run
    (~2 GB, several minutes).
  * torch was just upgraded to 2.11.0+cu128 to enable sm_120 (RTX 5090);
    the repo's pinned 2.3.1 in requirements.txt is stale.
"""
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# --- Bootstrap: CosyVoice must run from its own repo root ---
COSY_DIR = Path("/workspace/VoiceReplication/generators/cosyvoice")
os.chdir(COSY_DIR)
sys.path.insert(0, str(COSY_DIR))
sys.path.insert(0, str(COSY_DIR / "third_party" / "Matcha-TTS"))

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "cosyvoice.wav"

# Bundled reference clip + the matching transcript used in upstream example.py.
PROMPT_WAV = COSY_DIR / "asset" / "zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

MODEL_REPO = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_DIR = COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B"

if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke contract", flush=True)
    sys.exit(1)

device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

# --- Ensure weights are present ---
if not (MODEL_DIR / "cosyvoice2.yaml").exists():
    print(f"weights missing, downloading {MODEL_REPO} -> {MODEL_DIR} ...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODEL_DIR),
    )
    print("download complete", flush=True)

# --- Load model ---
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

t0 = time.time()
cosyvoice = CosyVoice2(model_dir=str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

sr = cosyvoice.sample_rate
print(f"sample_rate={sr}", flush=True)

# --- Run inference ---
t0 = time.time()
chunks = []
for out in cosyvoice.inference_zero_shot(
    SMOKE_TEXT,
    PROMPT_TEXT,
    str(PROMPT_WAV),
    stream=False,
):
    chunks.append(out["tts_speech"])
t_gen = time.time() - t0

if not chunks:
    print("no audio produced!", flush=True)
    sys.exit(3)

# Each chunk is a torch tensor [1, T]; concat along time.
audio = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), audio, sr)

duration = audio.shape[-1] / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
