"""Harness adapter for NeuTTS-Air — fine-tuned on the Diana character.

Identical to run_neutts-air.py except the backbone path points at our local
fine-tuned weights (500 steps, ~8 epochs, batch=4, lr=4e-5) instead of the
pretrained `neuphonic/neutts-air` hub model.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import soundfile as sf
import torch
from neuttsair.neutts import NeuTTSAir


FT_DIR = "/workspace/VoiceReplication/tests/finetune/neutts-air/diana_ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
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
        backbone_repo=FT_DIR,
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda",
        language="en-us",  # required when backbone path isn't a neuphonic/* repo
    )
    load_s = sw()
    sr = tts.sample_rate

    ref_cache: dict[str, object] = {}

    for i, row in enumerate(rows):
        try:
            if not row.ref_text:
                raise ValueError("NeuTTS-Air requires ref_text")
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
                "neutts-air-ft", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("neutts-air-ft", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
