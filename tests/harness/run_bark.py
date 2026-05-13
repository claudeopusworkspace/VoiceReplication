"""Harness adapter for Bark (Suno) TTS.

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads Bark via preload_models() exactly once.

IMPORTANT — Bark voice cloning caveat:
    Bark's "voice cloning" is not real reference-clip cloning. Its voice
    presets are pre-baked "history prompts" (semantic + coarse + fine
    tokens) shipped with the model; the public API does NOT accept an
    arbitrary reference wav. For this bake-off we accept the limitation
    and pin a single high-quality English preset (BARK_HISTORY_PROMPT
    below) for every row. The harness still produces N_refs * N_sentences
    cells, but every cell will sound like the chosen Bark preset rather
    than the character in row.ref_path. This is expected — it's part of
    what we're measuring (Bark is a baseline / sanity check, not a real
    voice-clone competitor).
"""
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

# Bark ships checkpoints with pickled numpy scalars (e.g.
# numpy.core.multiarray.scalar). torch >= 2.6 defaults torch.load to
# weights_only=True, which rejects them. Monkey-patch BEFORE importing
# bark — same trick as the smoke test and our XTTS-v2 adapter.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from bark import SAMPLE_RATE, generate_audio, preload_models  # noqa: E402

# Fixed English speaker preset. v2/en_speaker_6 is widely regarded as the
# most stable / natural-sounding of the official Bark presets.
BARK_HISTORY_PROMPT = "v2/en_speaker_6"


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
    preload_models()
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            audio_array = generate_audio(row.text, history_prompt=BARK_HISTORY_PROMPT)
            gen_s = time.time() - t0

            audio_array = np.asarray(audio_array, dtype=np.float32)

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), audio_array, SAMPLE_RATE)
            duration_s = audio_array.shape[-1] / SAMPLE_RATE

            emit(
                "bark", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=SAMPLE_RATE,
                status="ok",
            )
        except Exception as e:
            emit("bark", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
