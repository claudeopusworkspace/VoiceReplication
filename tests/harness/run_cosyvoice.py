"""Harness adapter for CosyVoice 2 (0.5B).

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the model exactly once.

CosyVoice quirks mirrored from tests/smoke/smoke_cosyvoice.py:
  * The repo uses bare imports (`from cosyvoice.cli.cosyvoice import ...`)
    and bundles its deps under `third_party/Matcha-TTS`, so we chdir into
    the repo root and prepend both to sys.path before importing.
  * Inference is a generator that yields {"tts_speech": Tensor[1, T]} chunks;
    we concat along time to form the full waveform.
  * The zero-shot frontend's `_extract_*` helpers all call `load_wav(...)`
    on the prompt argument internally, so we still load+resample the ref
    once (to validate it + capture duration) but pass the path string to
    inference_zero_shot — matching upstream example.py / the smoke.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

# --- Bootstrap: CosyVoice must run from its own repo root ---
COSY_DIR = Path("/workspace/VoiceReplication/generators/cosyvoice")
os.chdir(COSY_DIR)
sys.path.insert(0, str(COSY_DIR))
sys.path.insert(0, str(COSY_DIR / "third_party" / "Matcha-TTS"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

MODEL_DIR = COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B"


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

    # Imports deferred until after sys.path/chdir bootstrap.
    from cosyvoice.cli.cosyvoice import CosyVoice2

    sw = stopwatch()
    model = CosyVoice2(model_dir=str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    load_s = sw()
    sr = model.sample_rate

    for i, row in enumerate(rows):
        try:
            # NB: CosyVoice2's zero-shot frontend re-loads the prompt audio
            # internally via torchaudio.load(path), so we pass the path
            # string rather than a pre-loaded tensor. (On torch 2.11+ the
            # torchcodec backend rejects float tensors handed to load().)
            t0 = time.time()
            chunks = []
            for out in model.inference_zero_shot(
                row.text,
                row.ref_text,
                str(row.ref_path),
                stream=False,
            ):
                chunks.append(out["tts_speech"])
            gen_s = time.time() - t0

            if not chunks:
                emit("cosyvoice", row, status="fail", error="no audio produced")
                continue

            audio = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), audio, sr)
            duration_s = audio.shape[-1] / sr

            emit(
                "cosyvoice", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("cosyvoice", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
