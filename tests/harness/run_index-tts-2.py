"""Harness adapter for IndexTTS-2.

Reads a manifest from stdin (one JSON object per line, see _common.Row) and
writes one wav per row. Loads the model exactly once.

IndexTTS-2 quirks (preserved from tests/smoke/smoke_index_tts_2.py):
- Hard-codes relative paths (HF_HUB_CACHE, w2v-bert-2.0 download, bigvgan repo
  download); we chdir into the model dir so those land in the right place.
- `use_cuda_kernel=False` to avoid the BigVGAN custom CUDA kernel compile path.
- `infer(..., output_path=...)` writes the wav itself; we read it back and
  re-save via soundfile for consistent file format across the bake-off.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import numpy as np
import soundfile as sf
import torch

# IndexTTS2 hard-codes relative paths — chdir before importing.
INDEXTTS_DIR = Path("/workspace/VoiceReplication/generators/index-tts-2")
os.chdir(INDEXTTS_DIR)
sys.path.insert(0, str(INDEXTTS_DIR))

from indextts.infer_v2 import IndexTTS2  # noqa: E402


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
    tts = IndexTTS2(
        cfg_path=str(INDEXTTS_DIR / "checkpoints" / "config.yaml"),
        model_dir=str(INDEXTTS_DIR / "checkpoints"),
        device="cuda",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            row.output.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            tts.infer(
                spk_audio_prompt=str(row.ref_path),
                text=row.text,
                output_path=str(row.output),
                verbose=False,
            )
            gen_s = time.time() - t0

            # infer() wrote the file; read it back for stats and re-save via
            # soundfile to match the rest of the bake-off.
            wav, sr = sf.read(str(row.output))
            if wav.ndim > 1:
                wav = wav[:, 0]
            sf.write(str(row.output), wav.astype(np.float32), sr)
            duration_s = len(wav) / sr

            emit(
                "index-tts-2", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("index-tts-2", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
