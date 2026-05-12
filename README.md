# VoiceReplication

Experimenting with voice cloning AI to generate new lines from an existing
character's voice. The goal: an honest, ears-on comparison across the current
state-of-the-art open-source models — both zero-shot (one reference clip) and
few-shot/fine-tuned (with ~10-20 min of training data).

## Layout

```
VoiceReplication/
├── generators/        # Tier 1 + 2 — general-purpose text-to-speech voice cloners
├── specialized/       # Tier 3 — editing, conversion, non-verbal sounds, etc.
├── reference_voice/   # Character voice data (gitignored)
├── tests/             # Comparison harness + outputs (outputs gitignored)
├── notes/             # Research notes, observations
├── MODELS.md          # Catalog of every cloned model (URL + pinned commit + license)
└── CLAUDE.md          # Conventions for AI collaborator
```

The 16 model repos under `generators/` and `specialized/` are cloned locally
but **not committed** to this repo — see `MODELS.md` for the manifest and how to
re-clone them.

## Status

- [x] Research current open-source voice cloning landscape
- [x] Clone 16 candidate model repos (10 general-purpose, 6 specialized)
- [ ] Gather character voice reference data
- [ ] Stand up per-model virtual environments + verify each runs
- [ ] Design comparison test harness
- [ ] Run bake-off + capture results

## Hardware

NVIDIA RTX 5090 (32 GB VRAM). All 16 models are feasible to run locally.
