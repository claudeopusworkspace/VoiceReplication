"""Harness adapter for OmniVoice (k2-fsa, diffusion language model TTS).

Reads a manifest from the orchestrator (one JSON object per line, see
_common.Row) and writes one wav per row. Loads OmniVoice exactly once.

Voice-cloning mode: ``model.generate(text=..., ref_audio=row.ref_path,
ref_text=row.ref_text)`` returns a list of np.ndarray waveforms at
``model.sampling_rate`` (24 kHz). We save the first element via soundfile.
``ref_text`` is forwarded only when the manifest supplies a non-empty
transcript; OmniVoice tolerates ``None`` and still clones from audio alone.
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
from omnivoice import OmniVoice


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
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    load_s = sw()
    sr = model.sampling_rate

    for i, row in enumerate(rows):
        try:
            ref_text = row.ref_text if row.ref_text else None

            t0 = time.time()
            audios = model.generate(
                text=row.text,
                ref_audio=str(row.ref_path),
                ref_text=ref_text,
            )
            gen_s = time.time() - t0

            # `generate` returns a list of np.ndarray waveforms; take the first.
            wav = audios[0]

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "omnivoice", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("omnivoice", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
