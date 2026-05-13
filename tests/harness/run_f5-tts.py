"""Harness adapter for F5-TTS.

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the model exactly once.
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
from f5_tts.api import F5TTS


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
    model = F5TTS(model="F5TTS_v1_Base", device="cuda")
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav, sr, _spec = model.infer(
                ref_file=str(row.ref_path),
                ref_text=row.ref_text,
                gen_text=row.text,
            )
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "f5-tts", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("f5-tts", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
