"""Harness adapter for Tortoise-TTS.

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the model exactly once.

Notes (mirroring tests/smoke/smoke_tortoise_tts.py):
- Tortoise is reference-conditioned via `voice_samples` (list of tensors loaded
  at 22050 Hz). We pre-compute `conditioning_latents` per unique `ref_path`
  and reuse them across sentences — this avoids paying the autoregressive
  conditioning encoder cost on every cell.
- Preset is "ultra_fast" — quality is poor but the standard preset would take
  10+ minutes per cell × 15 cells which is unacceptable for a bake-off.
- Tortoise emits a 24 kHz waveform shaped [1, 1, T] (k=1 default). Squeeze
  to 1D before soundfile.write.
- Fail loudly if CUDA is unavailable; Tortoise on CPU is intractably slow.
"""
import argparse
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import soundfile as sf
import torch

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio

PRESET = "ultra_fast"
REF_SR = 22050  # Tortoise expects voice samples at 22050 Hz
OUT_SR = 24000  # Tortoise's fixed output sample rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to JSON-lines manifest from the orchestrator")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    rows = load_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    sw = stopwatch()
    tts = TextToSpeech()
    load_s = sw()

    # Pre-compute conditioning latents per unique ref_path.
    latents_cache: dict[str, tuple] = {}
    for row in rows:
        key = str(row.ref_path)
        if key in latents_cache:
            continue
        voice_sample = load_audio(str(row.ref_path), REF_SR)
        latents_cache[key] = tts.get_conditioning_latents([voice_sample])

    for i, row in enumerate(rows):
        try:
            conditioning_latents = latents_cache[str(row.ref_path)]

            t0 = time.time()
            wav = tts.tts_with_preset(
                row.text,
                voice_samples=None,
                conditioning_latents=conditioning_latents,
                preset=PRESET,
            )
            gen_s = time.time() - t0

            # Tortoise returns either a single tensor [1, 1, T] (k=1) or a list.
            if isinstance(wav, list):
                wav = wav[0]
            audio = wav.squeeze().cpu().numpy()

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), audio, OUT_SR)
            duration_s = audio.shape[-1] / OUT_SR

            emit(
                "tortoise-tts", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=OUT_SR,
                status="ok",
            )
        except Exception as e:
            emit("tortoise-tts", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
