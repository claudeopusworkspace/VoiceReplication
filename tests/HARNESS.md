# Bake-off harness — design

Goal: given a reference clip (+ its transcript) and a list of target sentences,
produce one synthesized output per (model × sentence × reference) so we can
listen side-by-side and decide which model clones the character best.

## Inputs (locked)

**Test sentences** — three lines covering the character's tonal range:

| id | text |
|---|---|
| `a_neutral` | My systems are running within normal parameters. There is nothing else to report. |
| `b_wistful` | Sometimes I wonder if remembering the people we've lost is what makes us truly alive. |
| `c_excited` | Look — I can see the city from up here! Everything sparkles when the lights come on. |

**Reference clip(s):** Woj's pick from `_listen/top5_references/`. Stored as
`tests/harness/refs/<ref_id>.wav` + `tests/harness/refs/<ref_id>.txt` (the
transcript from the Whisper manifest).

## Output layout

```
tests/outputs/harness/
├── <model>/
│   ├── <ref_id>__<sentence_id>.wav     # one file per cell of the grid
│   └── _meta.json                       # timings, status, errors
└── _index.html                          # auto-generated audio gallery
```

Example: `tests/outputs/harness/chatterbox/01_sad_slow_doctor__a_neutral.wav`

## Per-model adapter contract

Each generator/specialized model gets a Python script at
`tests/harness/run_<model>.py` with this CLI:

```bash
<venv>/bin/python tests/harness/run_<model>.py \
    --ref <path/to/ref.wav> \
    --ref-transcript "<text or empty>" \
    --text "<target sentence>" \
    --output <path/to/output.wav>
```

- Returns exit 0 on success, non-zero on failure.
- Prints one JSON line to stdout with `{model, load_s, gen_s, duration_s, sr, status}`.
- The adapter handles all model quirks (chdir, torch.load patches, transcript
  encoding, sample-rate matching) — the orchestrator stays clean.

The 15 existing smoke scripts already contain most of the loading logic;
adapters are mostly "smoke + argparse + parameterized inference."

## Voice-conversion models (RVC, Seed-VC)

These don't take *text* directly — they take *source audio* and convert its
timbre. Two-step strategy:

1. **Source generation pass:** run a fast TTS (Chatterbox at default voice)
   to synthesize each test sentence → `tests/outputs/harness/_source/<sentence_id>.wav`.
2. **Conversion pass:** run each VC model with source = `_source/<id>.wav`,
   target = the character reference clip.

The VC adapters take `--source` instead of `--text` and `--ref-transcript`.

## Orchestrator

`tests/harness/run_bakeoff.py` — single entry point. For each (model, ref,
sentence) triplet:

1. Resolve the model's venv python + adapter script.
2. Subprocess the adapter. Capture stdout JSON. Time out at 600s per call.
3. Aggregate results into `tests/outputs/harness/_results.csv`
   (model, ref, sentence, status, load_s, gen_s, output_path, error).
4. Generate `_index.html`: a simple sortable table with `<audio controls>` for
   every output, plus the original reference for comparison.

Run as:
```bash
tests/harness/.venv/bin/python tests/harness/run_bakeoff.py \
    --refs tests/harness/refs/*.wav \
    --models all
# or --models chatterbox,f5-tts,xtts-v2  for a partial sweep
```

Subprocess-per-model means a crash in one model doesn't poison the others, and
each adapter runs in its own venv (the only way given the dep conflicts).

## What we get out

- 1 ref × 3 sentences × 13 TTS models = 39 wav files
- 1 ref × 3 sentences × 2 VC models = 6 wav files
- Total: 45 outputs in `tests/outputs/harness/`, plus `_index.html` for side-by-side listening.

If we test multiple references (Woj may pick 2-3), multiply accordingly.

## Decision points still open

- **VoiceCraft:** deferred. Re-include only if we want speech editing
  (replacing words in existing audio) — different use case.
- **Fine-tune candidates:** GPT-SoVITS few-shot (1 min) and StyleTTS2 full
  retrain (~17 min may not be enough) are separate experiments after the
  zero-shot bake-off — not part of this first pass. Decide based on initial
  results.
- **Bark non-verbal augmentation:** Bark can do `[laughs]` / `[sighs]` —
  irrelevant for the head-to-head fidelity comparison but interesting for the
  winning model's pipeline. Defer.
