"""Convert reference_voice manifest CSV → GPT-SoVITS training list.

GPT-SoVITS's `prepare_datasets/1-get-text.py` reads a pipe-separated file
shaped `wav_name|spk_name|language|text`. We have a CSV with columns
`filename,duration_s,sample_rate,rms,peak,transcript,lang_prob`.

Output: tests/finetune/gpt-sovits/diana.list
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/workspace/VoiceReplication")
MANIFEST = ROOT / "reference_voice/ch05000_base_dialogue__en.manifest.csv"
WAV_DIR = ROOT / "reference_voice/ch05000_base_dialogue__en"
OUT = ROOT / "tests/finetune/gpt-sovits/diana.list"

SPK = "diana"
LANG = "en"


def main():
    rows = []
    with MANIFEST.open() as f:
        for r in csv.DictReader(f):
            wav = WAV_DIR / r["filename"]
            if not wav.exists():
                print(f"skip (missing wav): {r['filename']}")
                continue
            text = r["transcript"].strip().replace("|", " ")
            if not text:
                continue
            rows.append(f"{wav}|{SPK}|{LANG}|{text}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} lines to {OUT}")


if __name__ == "__main__":
    main()
