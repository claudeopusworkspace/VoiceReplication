"""Whisper-transcribe every clip ≥ min-duration in a character directory.

Output: <char_dir>.manifest.csv with columns
  filename, duration_s, sample_rate, rms, peak, transcript, lang_prob

Run via:
  tools/.venv/bin/python tools/transcribe.py reference_voice/<char_dir>
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char_dir", type=Path)
    ap.add_argument("--min-duration", type=float, default=2.0,
                    help="Skip clips shorter than this many seconds")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    ap.add_argument("--lang", default="en",
                    help="Language hint (set None for auto-detect)")
    args = ap.parse_args()

    if not args.char_dir.is_dir():
        sys.exit(f"not a directory: {args.char_dir}")

    out_csv = args.char_dir.parent / f"{args.char_dir.name}.manifest.csv"

    files = sorted(args.char_dir.glob("*.wav"))
    print(f"loading faster-whisper {args.model} on CUDA…", flush=True)
    t0 = time.time()
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"transcribing {len(files)} clips (min_duration={args.min_duration}s)…", flush=True)
    rows = []
    skipped = 0
    t0 = time.time()
    for i, f in enumerate(files):
        info = sf.info(str(f))
        if info.duration < args.min_duration:
            skipped += 1
            continue
        audio, sr = sf.read(str(f))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        segments, ti = model.transcribe(str(f), language=args.lang, beam_size=5, vad_filter=False)
        transcript = " ".join(s.text.strip() for s in segments).strip()
        rows.append({
            "filename": f.name,
            "duration_s": round(info.duration, 3),
            "sample_rate": sr,
            "rms": round(rms, 5),
            "peak": round(peak, 5),
            "transcript": transcript,
            "lang_prob": round(ti.language_probability, 3),
        })
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"  {i+1}/{len(files)} ({len(rows)} transcribed, {skipped} skipped)  elapsed={time.time()-t0:.0f}s", flush=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
