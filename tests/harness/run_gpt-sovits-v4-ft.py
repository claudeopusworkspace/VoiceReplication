"""Harness adapter for GPT-SoVITS v4 — LoRA fine-tuned on the Diana character.

The LoRA checkpoint is auto-detected by TTS_Config — the base s2Gv4.pth gets
loaded first, then the LoRA weights from SoVITS_weights_v4/ are applied on top
(see TTS.py:495+ → get_sovits_version_from_path_fast).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

GPTSOVITS_DIR = Path("/workspace/VoiceReplication/generators/gpt-sovits")
os.chdir(GPTSOVITS_DIR)
sys.path.insert(0, str(GPTSOVITS_DIR))
sys.path.insert(0, str(GPTSOVITS_DIR / "GPT_SoVITS"))

import torchaudio  # noqa: E402

_sf_imported = sf


def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = _sf_imported.read(str(path), always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _torchaudio_load_via_soundfile

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    sw = stopwatch()
    # IMPORTANT: TTS_Config silently falls back to default weights unless the
    # config is wrapped in {"custom": {...}}. See TTS.py:318.
    config = TTS_Config({"custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v4",
        "t2s_weights_path": "GPT_weights_v4/diana_v4-e15.ckpt",
        "vits_weights_path": "SoVITS_weights_v4/diana_v4_e8_s184_l32.pth",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    }})
    tts = TTS(config)
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            req = {
                "text": row.text,
                "text_lang": "en",
                "ref_audio_path": str(row.ref_path),
                "prompt_text": row.ref_text,
                "prompt_lang": "en",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed_factor": 1.0,
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
            for _sr_i, audio_i in gen:
                extra.append(audio_i)
            if extra:
                audio = np.concatenate([audio] + extra)
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), audio, sr)
            duration_s = audio.shape[-1] / sr

            emit(
                "gpt-sovits-v4-ft", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("gpt-sovits-v4-ft", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
