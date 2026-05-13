"""Harness adapter for RVC (Retrieval-based Voice Conversion).

RVC is voice-conversion, not TTS — it ingests source speech and re-renders it
in a target voice's timbre. The orchestrator builds rows with an extra
`source` field for VC models (the path to a pre-generated source wav, one
per sentence id). RVC ignores `row.text` and (importantly) cannot use
`row.ref_path` directly: it requires a trained `.pth` voice model for the
target voice, not just a reference clip.

KEY CAVEAT — DOES NOT CLONE THE CHARACTER:
We don't have a trained RVC model for our character's voice (training one
is a separate, multi-hour-of-audio process). So this adapter uses the same
prebuilt Japanese-female RVC v2 model (`V2-AISO-HOWATTO.pth`) that the
smoke test used, for ALL rows. Every output therefore sounds like that
single prebuilt voice — the 5 refs × 3 sentences = 15 cells WILL NOT vary
by character ref. The harness still produces 15 wavs so we can hear how
RVC transforms different source utterances, but cross-ref comparison is
not meaningful here. To actually clone the character with RVC you'd need
to train a per-voice .pth — out of scope for the bake-off.

DEPENDENCY ON PRE-GENERATED SOURCE AUDIO:
This adapter reads `row.source` (set by the orchestrator to
`tests/outputs/harness/_source/<sentence_id>.wav`) and DOES NOT generate
it. Those source files must be created out-of-band before this adapter
runs — typically by invoking Chatterbox with its default voice for each
sentence. The adapter fails per-row with `missing source` if a file is
absent, rather than aborting the whole run.

Contract:
  specialized/rvc/.venv/bin/python tests/harness/run_rvc.py --manifest <path>
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make _common.py importable regardless of cwd. We do this BEFORE chdir since
# os.chdir later moves us into the RVC repo root.
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch, Row

import soundfile as sf
import torch

# torch 2.6+ defaults torch.load to weights_only=True, which rejects fairseq's
# hubert checkpoint (contains pickled fairseq.data.dictionary.Dictionary). We
# trust these files (lj1995 + aa444rt). Monkey-patch back to the pre-2.6
# default. Same trick as smoke_rvc.py / XTTS-v2.
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe


RVC_DIR = Path(__file__).resolve().parent.parent.parent / "specialized" / "rvc"
MODEL_NAME = "V2-AISO-HOWATTO.pth"  # lives in assets/weights/ — PREBUILT, not the character


def load_vc_manifest(path: Path) -> list[tuple[Row, str]]:
    """Like _common.load_manifest, but also extracts the VC-only `source` field.

    Returns a list of (Row, source_path) tuples.
    """
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
            source = obj.get("source", "")
            out.append((row, source))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to JSON-lines manifest from the orchestrator")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    pairs = load_vc_manifest(args.manifest)
    if not pairs:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    # Verify required model files exist before importing the world.
    hubert_path = RVC_DIR / "assets" / "hubert" / "hubert_base.pt"
    weight_path = RVC_DIR / "assets" / "weights" / MODEL_NAME
    for p in (hubert_path, weight_path):
        if not p.exists():
            print(json.dumps({"status": "fail", "error": f"missing required file: {p}"}), flush=True)
            sys.exit(1)

    # RVC reads many paths via os.getenv (.env) and relative paths like
    # "assets/hubert/hubert_base.pt" and "configs/inuse/...". The repo only works
    # if cwd == repo root, so chdir before importing config/VC.
    os.chdir(RVC_DIR)
    sys.path.insert(0, str(RVC_DIR))

    from dotenv import load_dotenv
    load_dotenv(RVC_DIR / ".env")

    # Config() uses argparse on sys.argv — strip our own args first.
    sys.argv = sys.argv[:1]

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
            if not source:
                emit("rvc", row, status="fail", error="manifest row missing 'source' field")
                continue
            src_path = Path(source)
            if not src_path.exists():
                emit("rvc", row, status="fail",
                     error=f"missing source: {src_path} (generate it before running RVC)")
                continue

            t0 = time.time()
            info, result = vc.vc_single(
                0,                       # speaker id (ignored for single-speaker models)
                str(src_path),           # input audio path
                0,                       # f0_up_key: no transpose
                None,                    # f0_file
                "pm",                    # f0_method: pm = parselmouth (no rmvpe download)
                "",                      # file_index (no retrieval index)
                None,                    # file_index2
                0.0,                     # index_rate (irrelevant w/o index)
                3,                       # filter_radius
                0,                       # resample_sr (0 = keep model's sr)
                0.25,                    # rms_mix_rate
                0.33,                    # protect (consonants)
            )
            gen_s = time.time() - t0

            if result is None or result[1] is None:
                emit("rvc", row, status="fail",
                     error=f"vc_single returned no audio. info={info!r}")
                continue
            out_sr, wav_opt = result

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav_opt, out_sr)
            duration_s = len(wav_opt) / out_sr

            emit(
                "rvc", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=out_sr,
                status="ok",
            )
        except Exception as e:
            emit("rvc", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
