"""Harness adapter for NeuTTS Air (Neuphonic).

Reads a manifest from stdin (one JSON object per line, see _common.Row) and
writes one wav per row. Loads the model exactly once.

NeuTTS quirks preserved from the smoke test:
- Constructor takes `backbone_device` and `codec_device` separately (no single
  `device=` kwarg). Both are set to "cuda".
- Voice cloning requires BOTH a reference wav AND its transcript — we use
  `row.ref_path` + `row.ref_text`. Rows with empty `ref_text` are failed.
- Inference is `encode_reference(ref_wav)` followed by
  `infer(text, ref_codes, ref_text)`. Output sample rate is `tts.sample_rate`.
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
from neuttsair.neutts import NeuTTSAir


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
    tts = NeuTTSAir(
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
    )
    load_s = sw()
    sr = tts.sample_rate

    # Cache encoded reference per ref_path — encoding is non-trivial and the
    # manifest typically pairs each ref with multiple sentences.
    ref_cache: dict[str, object] = {}

    for i, row in enumerate(rows):
        try:
            if not row.ref_text:
                raise ValueError("NeuTTS Air requires ref_text; got empty string")

            ref_key = str(row.ref_path)
            if ref_key not in ref_cache:
                ref_cache[ref_key] = tts.encode_reference(ref_key)
            ref_codes = ref_cache[ref_key]

            t0 = time.time()
            wav = tts.infer(row.text, ref_codes, row.ref_text)
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "neutts-air", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("neutts-air", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
