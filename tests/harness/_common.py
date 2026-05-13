"""Shared helpers for harness adapters.

Each adapter loads its model once, then iterates over the (ref, sentence)
manifest from the orchestrator. This module provides the manifest reader and
the per-row JSON-line emitter so adapters stay terse.
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    ref_id: str       # e.g. "01_sad_slow_doctor"
    ref_path: Path    # absolute path to reference wav
    ref_text: str     # transcript of the reference wav (may be empty for VC models)
    sentence_id: str  # e.g. "a_neutral"
    text: str         # target text to synthesize
    output: Path      # absolute path to write the output wav


def load_manifest(path: Path) -> list[Row]:
    """Read the orchestrator's manifest CSV-like file (one row per output)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            rows.append(Row(
                ref_id=obj["ref_id"],
                ref_path=Path(obj["ref_path"]),
                ref_text=obj.get("ref_text", ""),
                sentence_id=obj["sentence_id"],
                text=obj["text"],
                output=Path(obj["output"]),
            ))
    return rows


def emit(model: str, row: Row, *, load_s: float | None = None, gen_s: float | None = None,
         duration_s: float | None = None, sr: int | None = None,
         status: str = "ok", error: str | None = None) -> None:
    """Print one JSON line summarizing the run of a single row."""
    rec = {
        "model": model,
        "ref_id": row.ref_id,
        "sentence_id": row.sentence_id,
        "output": str(row.output),
        "status": status,
    }
    if load_s is not None: rec["load_s"] = round(load_s, 3)
    if gen_s is not None: rec["gen_s"] = round(gen_s, 3)
    if duration_s is not None: rec["duration_s"] = round(duration_s, 3)
    if sr is not None: rec["sr"] = sr
    if error is not None: rec["error"] = error
    print(json.dumps(rec), flush=True)


def stopwatch():
    """Return a closure that yields elapsed seconds since the last call to it."""
    last = [time.time()]
    def split():
        now = time.time()
        delta = now - last[0]
        last[0] = now
        return delta
    return split
