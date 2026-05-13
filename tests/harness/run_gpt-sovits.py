"""Harness adapter for GPT-SoVITS (v2).

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the model exactly once.

GPT-SoVITS is the most quirk-laden generator in the bake-off; this adapter
faithfully reproduces every workaround proven out in tests/smoke/smoke_gpt-sovits.py:
  * Must run from the gpt-sovits repo root; its imports use bare 'GPT_SoVITS.*'
    and 'tools.i18n' paths, so we chdir and prepend two entries to sys.path.
  * torchaudio 2.10+ delegates load/save to torchcodec, which isn't installed
    in this venv. Monkey-patch torchaudio.load to use soundfile so TTS.py can
    load reference clips. The patch must be installed BEFORE importing TTS.
  * Pretrained weights (s1 / s2 / cn-hubert / cn-roberta) are expected under
    GPT_SoVITS/pretrained_models/ per the project README — not bundled.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd (we chdir below).
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

# --- Bootstrap: GPT-SoVITS must run from its own repo root ---
GPTSOVITS_DIR = Path("/workspace/VoiceReplication/generators/gpt-sovits")
os.chdir(GPTSOVITS_DIR)
sys.path.insert(0, str(GPTSOVITS_DIR))
sys.path.insert(0, str(GPTSOVITS_DIR / "GPT_SoVITS"))

# torchaudio 2.10+ delegates load/save to torchcodec, which isn't installed.
# Monkey-patch torchaudio.load to use soundfile + a torch tensor return, the
# shape/dtype TTS.py expects (channels, samples) float32 in [-1, 1].
import torchaudio  # noqa: E402

_sf_imported = sf  # alias to avoid shadowing in the patch closure


def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = _sf_imported.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns (samples, channels); torchaudio.load returns (channels, samples)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _torchaudio_load_via_soundfile

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


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
    config = TTS_Config({
        "device": "cuda",
        "is_half": True,
        "version": "v2",
        "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    })
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
            # Drain any remaining fragments and concat (cut5 may yield multiple segments).
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
                "gpt-sovits", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("gpt-sovits", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
