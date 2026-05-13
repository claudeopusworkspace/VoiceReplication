"""Pre-generate source audio for the voice-conversion models.

For each test sentence, synthesize with Chatterbox's default voice (no
audio_prompt) so VC models have something to convert. Outputs land at
tests/outputs/harness/_source/<sentence_id>.wav.
"""
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parent.parent.parent
SENTENCES = json.loads((ROOT / "tests" / "harness" / "sentences.json").read_text())
OUT_DIR = ROOT / "tests" / "outputs" / "harness" / "_source"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, torch={torch.__version__}", flush=True)
assert device == "cuda", "need GPU"

t0 = time.time()
model = ChatterboxTTS.from_pretrained(device="cuda")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

for sid, text in SENTENCES.items():
    out = OUT_DIR / f"{sid}.wav"
    t0 = time.time()
    wav = model.generate(text)  # default voice, no audio_prompt
    gen_s = time.time() - t0
    sf.write(str(out), wav.squeeze(0).cpu().numpy(), model.sr)
    duration = wav.shape[-1] / model.sr
    print(f"  {sid}: {duration:.2f}s audio in {gen_s:.2f}s  → {out}", flush=True)
