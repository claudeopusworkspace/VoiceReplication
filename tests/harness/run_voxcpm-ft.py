"""Harness adapter for VoxCPM2 — LoRA fine-tuned on the Diana character.

Loads the base openbmb/VoxCPM2 weights, then applies the LoRA adapter trained
by `tests/finetune/voxcpm/run_train.sh` (1000 steps, r=32, alpha=32, ~64
epochs through 250 clips). The constructor auto-detects the LoRA config from
the saved `lora_config.json`.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

import soundfile as sf
import torch
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig


LORA_PATH = "/workspace/VoiceReplication/tests/finetune/voxcpm/ckpt/step_0001000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    rows = load_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    sw = stopwatch()
    # Must pass the LoRAConfig matching our training (r=32, alpha=32) — otherwise
    # VoxCPM auto-creates a default r=8 config and the safetensors weights fail
    # to load with a shape mismatch at the rank dimension.
    lora_cfg = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False, r=32, alpha=32, dropout=0.0)
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        device="cuda",
        lora_config=lora_cfg,
        lora_weights_path=LORA_PATH,
    )
    load_s = sw()
    sr = model.tts_model.sample_rate

    for i, row in enumerate(rows):
        try:
            t0 = time.time()
            wav = model.generate(
                text=row.text,
                reference_wav_path=str(row.ref_path),
            )
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "voxcpm-ft", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("voxcpm-ft", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
