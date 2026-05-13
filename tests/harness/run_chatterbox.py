"""Harness adapter for Chatterbox TTS.

Reads a manifest from stdin (one JSON object per line, see _common.Row) and
writes one wav per row. Loads the model exactly once.
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
from chatterbox.tts import ChatterboxTTS


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
    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav = model.generate(row.text, audio_prompt_path=str(row.ref_path))
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav.squeeze(0).cpu().numpy(), model.sr)
            duration_s = wav.shape[-1] / model.sr

            emit(
                "chatterbox", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=model.sr,
                status="ok",
            )
        except Exception as e:
            emit("chatterbox", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
