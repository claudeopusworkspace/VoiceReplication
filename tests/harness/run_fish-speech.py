"""Harness adapter for Fish Speech (Fish Audio S2-Pro / v2).

Reads a manifest from --manifest (one JSON object per line, see _common.Row)
and writes one wav per row. Loads the two-stage model (DualARTransformer LLM
+ DAC codec decoder) exactly once -- it's heavy (~40s load, ~22 GB VRAM).

References are passed in-context via ServeReferenceAudio(audio=<wav bytes>,
text=<ref transcript>); Fish Speech does support and benefit from a reference
transcript, so we forward row.ref_text when present.
"""
import argparse
import os
import queue
import sys
import time
from pathlib import Path

# --- Bootstrap: chdir into the fish-speech repo so its bare imports work ---
FISH_DIR = Path("/workspace/VoiceReplication/generators/fish-speech")
os.chdir(FISH_DIR)
sys.path.insert(0, str(FISH_DIR))

# Make einx tracebacks readable (mirrors run_webui.py)
os.environ["EINX_FILTER_TRACEBACK"] = "false"

# Make _common.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch  # noqa: E402

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from fish_speech.inference_engine import TTSInferenceEngine  # noqa: E402
from fish_speech.models.dac.inference import load_model as load_decoder_model  # noqa: E402
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue  # noqa: E402
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest  # noqa: E402

LLAMA_CKPT = FISH_DIR / "checkpoints" / "s2-pro"
DECODER_CKPT = FISH_DIR / "checkpoints" / "s2-pro" / "codec.pth"
DECODER_CFG = "modded_dac_vq"


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

    if not LLAMA_CKPT.exists() or not DECODER_CKPT.exists():
        print(
            f'{{"status": "fail", "error": "missing weights at {LLAMA_CKPT}"}}',
            flush=True,
        )
        sys.exit(2)

    device = "cuda"
    precision = torch.bfloat16

    sw = stopwatch()
    # Stage 1: LLM (text -> semantic codes), runs on a worker thread.
    llama_queue: queue.Queue = launch_thread_safe_queue(
        checkpoint_path=LLAMA_CKPT,
        device=device,
        precision=precision,
        compile=False,
    )
    # Stage 2: DAC codec (semantic codes -> wav).
    decoder_model = load_decoder_model(
        config_name=DECODER_CFG,
        checkpoint_path=DECODER_CKPT,
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=False,
    )
    load_s = sw()

    # Cache reference bytes per ref_path to avoid re-reading the same file
    # for every sentence.
    ref_bytes_cache: dict[str, bytes] = {}

    for i, row in enumerate(rows):
        try:
            ref_key = str(row.ref_path)
            if ref_key not in ref_bytes_cache:
                with open(row.ref_path, "rb") as fh:
                    ref_bytes_cache[ref_key] = fh.read()
            references = [
                ServeReferenceAudio(
                    audio=ref_bytes_cache[ref_key],
                    text=row.ref_text or "",
                )
            ]

            req = ServeTTSRequest(
                text=row.text,
                references=references,
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
                streaming=False,
            )

            t0 = time.time()
            audio = None
            sr = None
            err = None
            for result in engine.inference(req):
                if result.code == "error":
                    err = str(result.error)
                    break
                if result.code == "final":
                    sr, audio = result.audio
                    break
            gen_s = time.time() - t0

            if err is not None:
                emit("fish-speech", row, status="fail", error=err)
                continue
            if audio is None or sr is None:
                emit("fish-speech", row, status="fail", error="no audio produced")
                continue

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), audio, sr)
            duration_s = audio.shape[-1] / sr

            emit(
                "fish-speech", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("fish-speech", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
