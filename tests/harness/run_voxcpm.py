"""Harness adapter for VoxCPM2 (OpenBMB tokenizer-free TTS).

Reads a manifest from --manifest (one JSON object per line, see _common.Row) and
writes one wav per row. Loads the model exactly once.

VoxCPM quirks (preserved from smoke):
- `load_denoiser=False` to skip ZipEnhancer download (we don't pass denoise=True).
- VoxCPM's constructor runs a torch.compile warmup when optimize=True (default),
  so load_s includes ~2min of cold compile time — that's expected.
- Sample rate lives at `model.tts_model.sample_rate` (48000).
- Reference-only cloning path needs no ref transcript; we pass only
  `reference_wav_path` and let VoxCPM tokenize the reference audio.
- `model.generate(...)` returns a 1-D numpy ndarray, so no .cpu()/.squeeze() needed.
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
from voxcpm import VoxCPM


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
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        device="cuda",
    )
    load_s = sw()
    sr = model.tts_model.sample_rate

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav = model.generate(
                text=row.text,
                reference_wav_path=str(row.ref_path),
            )
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "voxcpm", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("voxcpm", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
