"""Orchestrate the bake-off: for each selected model, build a manifest of
(ref × sentence) combos and run the model's adapter.

Adapters live in tests/harness/run_<model>.py — each takes --manifest <path>
and reads JSON-lines rows.

Per-model venvs are at generators/<m>/.venv/ or specialized/<m>/.venv/.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # /workspace/VoiceReplication
HARNESS = ROOT / "tests" / "harness"
REFS_DIR = HARNESS / "refs"
SENTENCES_PATH = HARNESS / "sentences.json"
OUTPUTS_DIR = ROOT / "tests" / "outputs" / "harness"


# Model registry: name → (venv path, adapter path, kind)
# kind: "tts" takes text; "vc" takes a source wav (handled separately)
MODELS = {
    # Tier 1
    "chatterbox":   (ROOT / "generators/chatterbox/.venv/bin/python",     HARNESS / "run_chatterbox.py",   "tts"),
    "f5-tts":       (ROOT / "generators/f5-tts/.venv/bin/python",         HARNESS / "run_f5-tts.py",       "tts"),
    "xtts-v2":      (ROOT / "generators/xtts-v2/.venv/bin/python",        HARNESS / "run_xtts-v2.py",      "tts"),
    "index-tts-2":  (ROOT / "generators/index-tts-2/.venv/bin/python",    HARNESS / "run_index-tts-2.py",  "tts"),
    "gpt-sovits":   (ROOT / "generators/gpt-sovits/.venv/bin/python",     HARNESS / "run_gpt-sovits.py",   "tts"),
    # Tier 2
    "cosyvoice":    (ROOT / "generators/cosyvoice/.venv/bin/python",      HARNESS / "run_cosyvoice.py",    "tts"),
    "voxcpm":       (ROOT / "generators/voxcpm/.venv/bin/python",         HARNESS / "run_voxcpm.py",       "tts"),
    "fish-speech":  (ROOT / "generators/fish-speech/.venv/bin/python",    HARNESS / "run_fish-speech.py",  "tts"),
    "styletts2":    (ROOT / "generators/styletts2/.venv/bin/python",      HARNESS / "run_styletts2.py",    "tts"),
    "tortoise-tts": (ROOT / "generators/tortoise-tts/.venv/bin/python",   HARNESS / "run_tortoise-tts.py", "tts"),
    # Tier 3
    "bark":         (ROOT / "specialized/bark/.venv/bin/python",          HARNESS / "run_bark.py",         "tts"),
    "neutts-air":   (ROOT / "specialized/neutts-air/.venv/bin/python",    HARNESS / "run_neutts-air.py",   "tts"),
    "omnivoice":    (ROOT / "specialized/omnivoice/.venv/bin/python",     HARNESS / "run_omnivoice.py",    "tts"),
    "rvc":          (ROOT / "specialized/rvc/.venv/bin/python",           HARNESS / "run_rvc.py",          "vc"),
    "seed-vc":      (ROOT / "specialized/seed-vc/.venv/bin/python",       HARNESS / "run_seed-vc.py",      "vc"),
}


def collect_refs():
    refs = []
    for wav in sorted(REFS_DIR.glob("*.wav")):
        ref_id = wav.stem
        text_path = wav.with_suffix(".txt")
        ref_text = text_path.read_text().strip() if text_path.exists() else ""
        refs.append({"ref_id": ref_id, "ref_path": str(wav), "ref_text": ref_text})
    return refs


def build_manifest(model: str, kind: str, refs, sentences) -> list[dict]:
    rows = []
    model_dir = OUTPUTS_DIR / model
    if kind == "tts":
        for ref in refs:
            for sid, text in sentences.items():
                rows.append({
                    **ref,
                    "sentence_id": sid,
                    "text": text,
                    "output": str(model_dir / f"{ref['ref_id']}__{sid}.wav"),
                })
    elif kind == "vc":
        # VC adapters use the chatterbox-generated source audio for each sentence.
        source_dir = OUTPUTS_DIR / "_source"
        for ref in refs:
            for sid, text in sentences.items():
                rows.append({
                    **ref,
                    "sentence_id": sid,
                    "text": text,
                    "source": str(source_dir / f"{sid}.wav"),
                    "output": str(model_dir / f"{ref['ref_id']}__{sid}.wav"),
                })
    else:
        raise ValueError(f"unknown kind: {kind}")
    return rows


def write_manifest(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def run_model(model: str, kind: str, refs, sentences, timeout_s: int = 1800):
    venv_python, adapter, _kind = MODELS[model]
    if not venv_python.exists():
        return [{"model": model, "status": "fail", "error": f"venv missing: {venv_python}"}]
    if not adapter.exists():
        return [{"model": model, "status": "fail", "error": f"adapter missing: {adapter}"}]

    rows = build_manifest(model, kind, refs, sentences)
    manifest_path = OUTPUTS_DIR / "_manifests" / f"{model}.jsonl"
    write_manifest(rows, manifest_path)

    print(f"\n=== {model} ({kind}) — {len(rows)} cells, manifest at {manifest_path} ===", flush=True)
    cmd = [str(venv_python), str(adapter), "--manifest", str(manifest_path)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return [{"model": model, "status": "timeout", "error": f"exceeded {timeout_s}s"}]
    elapsed = time.time() - t0

    print(f"  exit={proc.returncode}  elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[-500:]}", flush=True)

    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="chatterbox",
                    help="Comma-separated model names, or 'all'")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-model timeout in seconds")
    args = ap.parse_args()

    if args.models == "all":
        selected = list(MODELS.keys())
    else:
        selected = [m.strip() for m in args.models.split(",")]
        for m in selected:
            if m not in MODELS:
                sys.exit(f"unknown model: {m}.  known: {sorted(MODELS)}")

    refs = collect_refs()
    if not refs:
        sys.exit(f"no refs found in {REFS_DIR}")
    sentences = json.loads(SENTENCES_PATH.read_text())

    print(f"refs: {[r['ref_id'] for r in refs]}", flush=True)
    print(f"sentences: {list(sentences)}", flush=True)
    print(f"models: {selected}", flush=True)
    print(f"cells per model: {len(refs)*len(sentences)}", flush=True)

    all_results = []
    overall_t0 = time.time()
    for m in selected:
        kind = MODELS[m][2]
        all_results.extend(run_model(m, kind, refs, sentences, args.timeout))

    # Write aggregate results
    results_path = OUTPUTS_DIR / f"_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(results_path, "w") as fh:
        for r in all_results:
            fh.write(json.dumps(r) + "\n")

    ok = sum(1 for r in all_results if r.get("status") == "ok")
    fail = sum(1 for r in all_results if r.get("status") != "ok")
    print(f"\nDONE in {time.time()-overall_t0:.0f}s — {ok} ok, {fail} fail")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
