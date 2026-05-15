"""Sweep speed_factor on v2Pro FT, with em-dash → period fix in c_excited.

Loads the v2Pro fine-tuned weights once, then loops over (speed, ref, sentence).

Outputs land in tests/outputs/v2pro_speed_sweep/<speed>/<ref>__<sentence>.wav
and a listen gallery is auto-built at _listen/v2pro_speed_sweep/index.html.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

GPTSOVITS_DIR = Path("/workspace/VoiceReplication/generators/gpt-sovits")
ROOT = Path("/workspace/VoiceReplication")

os.chdir(GPTSOVITS_DIR)
sys.path.insert(0, str(GPTSOVITS_DIR))
sys.path.insert(0, str(GPTSOVITS_DIR / "GPT_SoVITS"))

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _torchaudio_load_via_soundfile

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


REFS_DIR = ROOT / "tests/harness/refs"
SENTENCES_PATH = ROOT / "tests/harness/sentences.json"
OUT_BASE = ROOT / "tests/outputs/v2pro_speed_sweep"
LISTEN_DIR = ROOT / "_listen/v2pro_speed_sweep"

SPEEDS = [1.0, 0.95, 0.90, 0.85]


def fix_dashes(text: str) -> str:
    """GPT-SoVITS's cut5 splitter doesn't recognize —. Convert to period for a
    real pause; replace surrounding double-space if it leaves one."""
    return text.replace(" — ", ". ").replace("—", ". ").replace("  ", " ")


def main():
    sentences = json.loads(SENTENCES_PATH.read_text())
    refs = []
    for wav in sorted(REFS_DIR.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        refs.append({
            "id": wav.stem,
            "wav": str(wav),
            "text": txt.read_text().strip() if txt.exists() else "",
        })

    config = TTS_Config({"custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v2Pro",
        "t2s_weights_path": "GPT_weights_v2Pro/diana_v2Pro-e15.ckpt",
        "vits_weights_path": "SoVITS_weights_v2Pro/diana_v2Pro_e8_s184.pth",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    }})
    print("--- loading TTS ---")
    t0 = time.time()
    tts = TTS(config)
    print(f"loaded in {time.time()-t0:.1f}s")

    n = 0
    for speed in SPEEDS:
        out_dir = OUT_BASE / f"speed_{speed:.2f}".replace(".", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        for ref in refs:
            for sid, stext in sentences.items():
                text = fix_dashes(stext)
                req = {
                    "text": text,
                    "text_lang": "en",
                    "ref_audio_path": ref["wav"],
                    "prompt_text": ref["text"],
                    "prompt_lang": "en",
                    "top_k": 15,
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "speed_factor": speed,
                    "seed": 42,
                    "parallel_infer": True,
                    "repetition_penalty": 1.35,
                    "return_fragment": False,
                    "streaming_mode": False,
                }
                t0 = time.time()
                gen = tts.run(req)
                sr, audio = next(gen)
                extra = []
                for _sr_i, ai in gen:
                    extra.append(ai)
                if extra:
                    audio = np.concatenate([audio] + extra)
                out_path = out_dir / f"{ref['id']}__{sid}.wav"
                sf.write(str(out_path), audio, sr)
                dur = audio.shape[-1] / sr
                n += 1
                print(f"  speed={speed:.2f}  {ref['id']}/{sid}  gen={time.time()-t0:.2f}s  out_dur={dur:.2f}s")

    print(f"--- {n} cells written under {OUT_BASE} ---")

    # Stage for listening
    if LISTEN_DIR.exists():
        shutil.rmtree(LISTEN_DIR)
    LISTEN_DIR.mkdir(parents=True)
    refs_listen = LISTEN_DIR / "refs"
    refs_listen.mkdir()
    for ref in refs:
        shutil.copy(ref["wav"], refs_listen / f"{ref['id']}.wav")
        (refs_listen / f"{ref['id']}.txt").write_text(ref["text"])
    for speed_dir in OUT_BASE.iterdir():
        shutil.copytree(speed_dir, LISTEN_DIR / speed_dir.name)
    build_gallery(refs, sentences)


def build_gallery(refs, sentences):
    speeds_ordered = [f"speed_{s:.2f}".replace(".", "") for s in SPEEDS]
    speed_labels = {f"speed_{s:.2f}".replace(".", ""): f"speed_factor = {s}" for s in SPEEDS}

    css = """body { font-family: -apple-system, system-ui, sans-serif; max-width: 1500px; margin: 1em auto; padding: 0 1em; color: #222; }
h1 { margin-bottom: 0.2em; }
h2 { border-bottom: 2px solid #444; padding-bottom: 0.3em; margin-top: 1.5em; }
h2 .meta { font-size: 0.7em; color: #666; font-weight: normal; }
.ref-block { margin: 1em 0; padding: 1em; background: #f7f7f7; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; margin-top: 0.5em; }
th, td { padding: 0.5em; text-align: left; vertical-align: top; border-top: 1px solid #ddd; }
th { background: #eee; }
.sid { font-family: ui-monospace, monospace; font-size: 0.9em; color: #444; white-space: nowrap; }
.stext { color: #666; font-style: italic; max-width: 320px; }
audio { width: 240px; height: 32px; }
"""
    parts = [f"<!doctype html><html lang=en><meta charset=utf-8><title>v2Pro speed sweep</title><style>{css}</style>"]
    parts.append("<h1>GPT-SoVITS v2Pro fine-tune — speed_factor sweep</h1>")
    parts.append(f"<p>{len(refs)} refs × {len(sentences)} sentences × {len(SPEEDS)} speeds. Em-dash in c_excited replaced with period (real pause).</p>")

    for ref in refs:
        parts.append(f'<h2>{ref["id"]} <span class="meta">— "{ref["text"][:80]}…"</span></h2>')
        parts.append('<div class="ref-block"><strong>Reference:</strong> ')
        parts.append(f'<audio controls src="refs/{ref["id"]}.wav"></audio></div>')
        parts.append("<table><thead><tr><th>Sentence</th>")
        for sd in speeds_ordered:
            parts.append(f"<th>{speed_labels[sd]}</th>")
        parts.append("</tr></thead><tbody>")
        for sid, stext in sentences.items():
            parts.append(f'<tr><td><div class="sid">{sid}</div><div class="stext">{stext}</div></td>')
            for sd in speeds_ordered:
                wav_rel = f"{sd}/{ref['id']}__{sid}.wav"
                parts.append(f'<td><audio controls src="{wav_rel}"></audio></td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</html>")
    (LISTEN_DIR / "index.html").write_text("\n".join(parts))
    print(f"--- gallery: {LISTEN_DIR}/index.html ---")


if __name__ == "__main__":
    main()
