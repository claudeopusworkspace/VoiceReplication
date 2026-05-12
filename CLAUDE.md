# VoiceReplication — project conventions

## Goal
Compare current state-of-the-art open-source voice cloning models on a single
character voice. Both zero-shot (one clip) and few-shot/fine-tuned approaches.

## Hard constraints
- **Open source only.** No paid APIs (ElevenLabs, Resemble, PlayHT, etc.).
- Hardware: RTX 5090, 32 GB VRAM. Local inference only — no cloud GPU rental.

## Repository layout
- `generators/<model>/` and `specialized/<model>/` are upstream clones —
  **gitignored**. Treat as third-party code; do not commit changes from inside
  them to *this* repo.
- The catalog of cloned models lives in `MODELS.md` (URL + pinned commit + license).
  Update it when re-syncing or adding models.
- Per-model virtual envs (`generators/<model>/.venv/`) are also gitignored. Each
  model gets its own venv because their dependency pins clash badly.

## Reference voice data
- Lives in `reference_voice/` — gitignored by default. Add `.gitkeep` or a
  `README.md` if we want a tracked placeholder.
- Treat voice data as potentially sensitive (IP, likeness rights). Do not push
  it to remotes without explicit confirmation.

## Test harness
- Lives in `tests/`. Output audio goes to `tests/outputs/` (gitignored —
  regenerable). Aggregate result metadata goes to `tests/results/` (gitignored
  unless we decide to publish findings).
- Aim for: same target text + same reference clip → one output per model →
  side-by-side listening.
