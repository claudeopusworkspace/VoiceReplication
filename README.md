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
- [x] Clone 16 candidate model repos (10 general-purpose, 6 specialized) — VoiceCraft deferred (torch 2.0/cu117 incompatible with sm_120)
- [x] Gather character voice reference data — 377 clips / ~17 min @ 48kHz mono
- [x] Stand up per-model virtual environments + verify each runs (15 / 16)
- [x] Design comparison test harness (`tests/HARNESS.md`)
- [x] Run zero-shot bake-off — **225 outputs across 15 models × 5 refs × 3 sentences**; see `tests/RESULTS.md`
- [x] Few-shot / fine-tune experiments — GPT-SoVITS (v2/v2Pro/v4), RVC, NeuTTS-Air, VoxCPM; see `tests/RESULTS.md` session 2
- [ ] ~~Pick winning approach and document a production recipe~~ — TTS timbre cloning is good, but text-to-prosody control is paradigm-limited; project parked in favor of a separate V2V experiment

## Hardware

NVIDIA RTX 5090 (32 GB VRAM, compute capability **sm_120**). All 15 installed models run locally. Note: torch wheels must be cu128 or cu130 — stock PyPI torch (max sm_90) will not work on this GPU.
