"""Harness adapter for Coqui XTTS-v2.

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the model exactly once.

Notes (mirroring tests/smoke/smoke_xtts-v2.py):
- COQUI_TOS_AGREED=1 must be set before importing TTS (license prompt fires at
  model download).
- XTTS-v2 checkpoints contain pickled XttsConfig/XttsAudioConfig objects.
  torch >= 2.6 defaults torch.load() to weights_only=True which rejects them;
  we monkey-patch torch.load to default weights_only=False.
- We save via soundfile to avoid torchaudio 2.11's torchcodec dependency.
- XTTS is reference-conditioned and doesn't need a reference transcript, just
  the wav. language="en".
"""
import os

# Must be set before importing TTS — the license prompt fires at model download.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import argparse
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import numpy as np
import soundfile as sf
import torch

# XTTS-v2 checkpoints contain XttsConfig / XttsAudioConfig pickled objects.
# torch >= 2.6 defaults torch.load() to weights_only=True which rejects them.
# We trust the upstream Coqui weights, so force-default weights_only=False here.
_orig_torch_load = torch.load
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_weights_only_false

from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


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
    tts = TTS(model_name=MODEL_NAME, progress_bar=False).to("cuda")
    load_s = sw()
    sr = tts.synthesizer.output_sample_rate

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav = tts.tts(
                text=row.text,
                speaker_wav=str(row.ref_path),
                language="en",
            )
            gen_s = time.time() - t0

            wav_np = np.asarray(wav, dtype=np.float32)
            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav_np, sr)
            duration_s = len(wav_np) / sr

            emit(
                "xtts-v2", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("xtts-v2", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
