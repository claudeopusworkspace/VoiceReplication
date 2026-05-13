"""Rank candidate reference clips for zero-shot voice cloning.

Reads <char_dir>.manifest.csv produced by transcribe.py and scores each row by:
  - duration in [5s, 8s] (closer to mid = better)
  - peak headroom (penalize ≥ 0.99 — clipped)
  - RMS energy (higher = louder = better SNR usually)
  - word density (2.0-3.5 words/sec is natural; punish outliers)
  - language probability (must be ≥ 0.95 to compete)

Surfaces the top N candidates. Doesn't pick — Woj listens and chooses.
"""
import argparse
import csv
from pathlib import Path


def score_row(row):
    dur = float(row["duration_s"])
    rms = float(row["rms"])
    peak = float(row["peak"])
    lang_prob = float(row["lang_prob"])
    transcript = row["transcript"]
    n_words = len(transcript.split())

    if lang_prob < 0.95:
        return None  # disqualified
    if dur < 4 or dur > 10:
        return None  # outside usable range
    if peak >= 0.99:
        return None  # clipped

    # Duration: peak score at 6.5s, falls off outside [5, 8]
    if 5 <= dur <= 8:
        dur_score = 1.0 - abs(dur - 6.5) / 3.5
    else:
        dur_score = 0.5 - 0.1 * abs(dur - 6.5)
    dur_score = max(0, min(1, dur_score))

    # Word density (words per second)
    wps = n_words / dur if dur > 0 else 0
    if 2.0 <= wps <= 3.5:
        density_score = 1.0
    elif 1.5 <= wps < 2.0 or 3.5 < wps <= 4.0:
        density_score = 0.7
    else:
        density_score = 0.3

    # RMS energy — higher = better, normalize at 0.2 (typical for clean speech)
    rms_score = min(1.0, rms / 0.2)

    # Peak headroom — reward room below clipping
    peak_score = max(0, (0.99 - peak) / 0.5)
    peak_score = min(1.0, peak_score)

    composite = (
        0.35 * dur_score
        + 0.30 * density_score
        + 0.20 * rms_score
        + 0.15 * peak_score
    )
    return composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))

    scored = []
    for r in rows:
        s = score_row(r)
        if s is not None:
            scored.append((s, r))

    scored.sort(key=lambda x: -x[0])

    print(f"{'rank':>4}  {'score':>6}  {'dur':>5}  {'rms':>6}  {'peak':>5}  {'wps':>5}  {'lang':>5}  filename                 transcript")
    print("-" * 140)
    for i, (score, r) in enumerate(scored[: args.top], start=1):
        dur = float(r["duration_s"])
        wps = len(r["transcript"].split()) / dur
        print(f"{i:>4}  {score:>6.3f}  {dur:>5.2f}  {float(r['rms']):>6.3f}  {float(r['peak']):>5.2f}  {wps:>5.2f}  {float(r['lang_prob']):>5.2f}  {r['filename']:24}  {r['transcript'][:80]}")


if __name__ == "__main__":
    main()
