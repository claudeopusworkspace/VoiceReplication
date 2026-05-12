"""Smoke test for Fish Speech (Fish Audio S2-Pro).

This repo is Fish Speech v2.0.0 — the inference entrypoint is the S2-Pro model
(`fishaudio/s2-pro`). Two-stage pipeline:
  1. DualARTransformer (4B-param LLM) emits semantic codes from text.
  2. DAC codec decoder turns those codes into a waveform.

Fish Speech supports reference-free generation (it falls back to a base
speaker), which is what we use here for a minimal smoke. This mirrors the
"dry run" the bundled `tools/run_webui.py` does at startup.

Gotchas this script works around:
  * Imports in `tools/run_webui.py` are bare (`fish_speech.*`, `tools.webui`),
    so we must chdir into the repo root before importing.
  * `s2-pro` checkpoint is ~11 GB — download once into
    `<repo>/checkpoints/s2-pro/`; subsequent runs reuse it.
  * Codec checkpoint lives inside the model dir at `codec.pth` (separate
    decoder model, distinct from the LLM safetensors).
"""
import os
import queue
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# --- Bootstrap: chdir into the fish-speech repo so its bare imports work ---
FISH_DIR = Path("/workspace/VoiceReplication/generators/fish-speech")
os.chdir(FISH_DIR)
sys.path.insert(0, str(FISH_DIR))

# Make einx happy (per run_webui.py)
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine  # noqa: E402
from fish_speech.models.dac.inference import load_model as load_decoder_model  # noqa: E402
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue  # noqa: E402
from fish_speech.utils.schema import ServeTTSRequest  # noqa: E402

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "fish-speech.wav"

LLAMA_CKPT = FISH_DIR / "checkpoints" / "s2-pro"
DECODER_CKPT = FISH_DIR / "checkpoints" / "s2-pro" / "codec.pth"
DECODER_CFG = "modded_dac_vq"

if not torch.cuda.is_available():
    print("CUDA not available - failing loudly per smoke contract", flush=True)
    sys.exit(1)

device = "cuda"
precision = torch.bfloat16
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

if not LLAMA_CKPT.exists() or not DECODER_CKPT.exists():
    print(
        f"Missing weights at {LLAMA_CKPT}. Run:\n"
        f"  hf download fishaudio/s2-pro --local-dir {LLAMA_CKPT}",
        flush=True,
    )
    sys.exit(2)

# --- Stage 1: load LLM (text -> semantic codes), runs on a worker thread ---
t0 = time.time()
llama_queue: queue.Queue = launch_thread_safe_queue(
    checkpoint_path=LLAMA_CKPT,
    device=device,
    precision=precision,
    compile=False,
)

# --- Stage 2: load DAC codec (semantic codes -> wav) ---
decoder_model = load_decoder_model(
    config_name=DECODER_CFG,
    checkpoint_path=DECODER_CKPT,
    device=device,
)
t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)

# Build the inference engine
engine = TTSInferenceEngine(
    llama_queue=llama_queue,
    decoder_model=decoder_model,
    precision=precision,
    compile=False,
)

# --- Inference: no reference -> use the base speaker distribution ---
req = ServeTTSRequest(
    text=SMOKE_TEXT,
    references=[],
    reference_id=None,
    max_new_tokens=1024,
    chunk_length=200,
    top_p=0.7,
    repetition_penalty=1.5,
    temperature=0.7,
    format="wav",
    streaming=False,
)

t0 = time.time()
audio = None
sr = None
for result in engine.inference(req):
    if result.code == "error":
        print(f"inference error: {result.error}", flush=True)
        sys.exit(3)
    if result.code == "final":
        sr, audio = result.audio
        break
t_gen = time.time() - t0

if audio is None or sr is None:
    print("no audio produced", flush=True)
    sys.exit(4)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), audio, sr)

duration = audio.shape[-1] / sr
print(
    f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})",
    flush=True,
)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
