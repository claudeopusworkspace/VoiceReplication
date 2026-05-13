"""Chatterbox adapter — tuned variant (cfg_weight=0.3, exaggeration=0.7).

The default-tuned run (cfg=0.5, exag=0.5) produced a robotic/phasey artifact
on the character. Per Chatterbox's own README:
  > "If the reference speaker has a fast speaking style, lowering cfg_weight
  >  to around 0.3 can improve pacing."
  > "Expressive or Dramatic Speech: lower cfg_weight values (e.g. ~0.3) and
  >  increase exaggeration to around 0.7 or higher."

This adapter pins those values. Outputs go to tests/outputs/harness/chatterbox-tuned/
so the original chatterbox/ run remains for comparison.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS

CFG_WEIGHT = 0.3
EXAGGERATION = 0.7


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
    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav = model.generate(
                row.text,
                audio_prompt_path=str(row.ref_path),
                cfg_weight=CFG_WEIGHT,
                exaggeration=EXAGGERATION,
            )
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav.squeeze(0).cpu().numpy(), model.sr)
            duration_s = wav.shape[-1] / model.sr

            emit(
                "chatterbox-tuned", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=model.sr,
                status="ok",
            )
        except Exception as e:
            emit("chatterbox-tuned", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
