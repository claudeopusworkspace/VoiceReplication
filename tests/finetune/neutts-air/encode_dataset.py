"""Encode our 250-clip Diana dataset into a HF dataset compatible with
NeuTTS-Air's finetune.py. Each row gets `text` (transcript) and `codes`
(NeuCodec int sequence). Saved to disk via `Dataset.save_to_disk`.

Output: tests/finetune/neutts-air/diana_dataset/
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from librosa import load as load_wav
from neucodec import NeuCodec
from tqdm import tqdm

ROOT = Path("/workspace/VoiceReplication")
MANIFEST = ROOT / "reference_voice/ch05000_base_dialogue__en.manifest.csv"
WAV_DIR = ROOT / "reference_voice/ch05000_base_dialogue__en"
OUT = ROOT / "tests/finetune/neutts-air/diana_dataset"


def main():
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cuda")

    rows = []
    with MANIFEST.open() as fh:
        for r in csv.DictReader(fh):
            wav = WAV_DIR / r["filename"]
            text = r["transcript"].strip()
            if not text or not wav.exists():
                continue
            rows.append((wav, text))
    print(f"{len(rows)} clips to encode")

    out_rows = []
    for wav_path, text in tqdm(rows):
        audio, _ = load_wav(str(wav_path), sr=16000, mono=True)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            codes = codec.encode_code(audio_or_path=audio_t).squeeze().cpu().numpy().astype(np.int32)
        out_rows.append({
            "text": text,
            "codes": codes.tolist(),
            "__key__": wav_path.stem,
        })

    ds = Dataset.from_list(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT))
    print(f"saved {len(out_rows)} rows to {OUT}")
    print(f"sample row: text={out_rows[0]['text'][:60]!r}  codes_len={len(out_rows[0]['codes'])}")


if __name__ == "__main__":
    main()
