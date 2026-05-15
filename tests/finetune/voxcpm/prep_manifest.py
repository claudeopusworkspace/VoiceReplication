"""Build a VoxCPM training JSONL manifest from our reference_voice clips.

Each line: {"audio": "<abs path>", "text": "<transcript>"}
Output: tests/finetune/voxcpm/diana.jsonl
"""
import csv
import json
from pathlib import Path

ROOT = Path("/workspace/VoiceReplication")
MANIFEST = ROOT / "reference_voice/ch05000_base_dialogue__en.manifest.csv"
WAV_DIR = ROOT / "reference_voice/ch05000_base_dialogue__en"
OUT = ROOT / "tests/finetune/voxcpm/diana.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as fout, MANIFEST.open() as fin:
        for r in csv.DictReader(fin):
            wav = WAV_DIR / r["filename"]
            text = r["transcript"].strip()
            if not text or not wav.exists():
                continue
            fout.write(json.dumps({"audio": str(wav), "text": text, "duration": float(r["duration_s"])}) + "\n")
            n += 1
    print(f"wrote {n} entries to {OUT}")


if __name__ == "__main__":
    main()
