"""Harness adapter for DramaBox (Resemble AI / LTX-2.3 audio).

Uses TTSServer to keep weights warm across the manifest. Each test sentence is
wrapped with a tone-matched speaker description tuned to the target character
(young girl, modelled on Diana from Pragmata). Tone modifier maps to the
orchestrator's sentence_id:
  a_neutral  -> calmly
  b_wistful  -> wistfully
  c_excited  -> excitedly

This is a methodologically different treatment than the other models in the
bake-off (they received the bare sentence). Decision was deliberate: DramaBox
is designed for prompted input and the goal of this run is to confirm timbre
fidelity at the model's intended operating point.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch

DRAMABOX_DIR = Path("/workspace/VoiceReplication/generators/dramabox")
sys.path.insert(0, str(DRAMABOX_DIR / "ltx2"))
sys.path.insert(0, str(DRAMABOX_DIR / "src"))
os.chdir(DRAMABOX_DIR)

CACHE = Path(
    "/home/dev/.cache/dramabox/models--ResembleAI--Dramabox/snapshots/"
    "404f967f653fa1170dc15a9d1ddd3fdb9a0a842d"
)
DIT = CACHE / "dramabox-dit-v1.safetensors"
AUDIO_COMPONENTS = CACHE / "dramabox-audio-components.safetensors"
GEMMA = (
    "/home/dev/.cache/dramabox/models--unsloth--gemma-3-12b-it-bnb-4bit/"
    "snapshots/826e729dbaeea4ecb143738eed2bcf3539ebf7bf"
)

TONE = {
    "a_neutral": "calmly",
    "b_wistful": "wistfully",
    "c_excited": "excitedly",
}


def make_prompt(sentence_id: str, text: str) -> str:
    tone = TONE.get(sentence_id)
    if tone:
        return f'A young girl speaks {tone}, "{text}"'
    return f'A young girl speaks, "{text}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    rows = load_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    from inference_server import TTSServer
    import soundfile as sf

    sw = stopwatch()
    server = TTSServer(
        checkpoint=str(DIT),
        full_checkpoint=str(AUDIO_COMPONENTS),
        gemma_root=GEMMA,
        device="cuda",
        dtype="bf16",
        compile_model=False,
        bnb_4bit=True,
    )
    load_s = sw()

    for i, row in enumerate(rows):
        try:
            prompt = make_prompt(row.sentence_id, row.text)
            row.output.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            server.generate_to_file(
                prompt=prompt,
                output=str(row.output),
                voice_ref=str(row.ref_path),
                cfg_scale=2.5,
                stg_scale=1.5,
                watermark=True,
            )
            gen_s = time.time() - t0
            info = sf.info(str(row.output))
            emit(
                "dramabox", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=info.duration,
                sr=info.samplerate,
                status="ok",
            )
        except Exception as e:
            emit("dramabox", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
