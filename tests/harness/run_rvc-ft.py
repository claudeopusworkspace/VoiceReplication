"""Harness adapter for RVC — fine-tuned on the Diana character.

Like the existing run_rvc.py adapter, but loads our trained `diana_rvc.pth`
(from 250 character clips × 100 epochs at 48 kHz v2) and uses the matching
FAISS retrieval index for stronger timbre adherence.

Unlike the prebuilt-JP-female zero-shot run, this DOES clone the character
(per-voice training is what RVC needs).

Input contract: same as run_rvc.py — manifest rows carry a `source` field
pointing at a pre-generated source wav (typically chatterbox output), one per
sentence id. The voice conversion overlays Diana's timbre onto that source.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, stopwatch, Row  # noqa: E402

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

# Bypass torch 2.6's weights_only=True default (fairseq HuBERT issue).
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe


RVC_DIR = Path(__file__).resolve().parent.parent.parent / "specialized" / "rvc"
MODEL_NAME = "diana_rvc.pth"  # written into assets/weights/ by training
INDEX_PATH = "logs/diana_rvc/added_IVF1280_Flat_nprobe_1_diana_rvc_v2.index"


def load_vc_manifest(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            row = Row(
                ref_id=obj["ref_id"],
                ref_path=Path(obj["ref_path"]),
                ref_text=obj.get("ref_text", ""),
                sentence_id=obj["sentence_id"],
                text=obj["text"],
                output=Path(obj["output"]),
            )
            out.append((row, obj.get("source", "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    pairs = load_vc_manifest(args.manifest)
    if not pairs:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    hubert_path = RVC_DIR / "assets" / "hubert" / "hubert_base.pt"
    weight_path = RVC_DIR / "assets" / "weights" / MODEL_NAME
    for p in (hubert_path, weight_path):
        if not p.exists():
            print(json.dumps({"status": "fail", "error": f"missing required file: {p}"}), flush=True)
            sys.exit(1)

    os.chdir(RVC_DIR)
    sys.path.insert(0, str(RVC_DIR))

    from dotenv import load_dotenv
    load_dotenv(RVC_DIR / ".env")
    sys.argv = sys.argv[:1]
    logging.basicConfig(level=logging.WARNING)

    sw = stopwatch()
    from configs.config import Config
    from infer.modules.vc.modules import VC

    config = Config()
    config.device = "cuda:0"
    vc = VC(config)
    vc.get_vc(MODEL_NAME)
    load_s = sw()

    for i, (row, source) in enumerate(pairs):
        try:
            if not source or not Path(source).exists():
                emit("rvc-ft", row, status="fail", error=f"missing source: {source}")
                continue

            t0 = time.time()
            info, result = vc.vc_single(
                0,                   # sid (single-speaker)
                source,              # input audio
                0,                   # f0_up_key (no transpose)
                None,                # f0_file
                "rmvpe",             # f0_method — RMVPE is the best of the bunch
                INDEX_PATH,          # file_index — use our trained retrieval index
                None,
                0.75,                # index_rate — strong retrieval to favor character timbre
                3,                   # filter_radius
                0,                   # resample_sr (keep model rate, 48k)
                0.25,                # rms_mix_rate
                0.33,                # protect
            )
            gen_s = time.time() - t0

            if result is None or result[1] is None:
                emit("rvc-ft", row, status="fail", error=f"vc_single returned no audio. info={info!r}")
                continue
            out_sr, wav_opt = result

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav_opt, out_sr)
            duration_s = len(wav_opt) / out_sr

            emit(
                "rvc-ft", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=out_sr,
                status="ok",
            )
        except Exception as e:
            emit("rvc-ft", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
