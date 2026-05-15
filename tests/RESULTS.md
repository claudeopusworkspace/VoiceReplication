# Bake-off results — session 1 (2026-05-13)

## Setup

- **Models:** 15 of 16 candidates (VoiceCraft deferred — torch 2.0/cu117 incompatible with RTX 5090's sm_120).
- **Reference clips:** 5 picks from the character's audio (5.6 – 6.9 s each, spanning emotional registers: sad, comforting, reflective, neutral, excited). Selected from the top of an auto-ranking over the 28 sweet-spot clips (5–10 s, ≥ 0.95 lang prob, no clipping).
- **Test sentences:** 3 lines covering tonal range — neutral exposition / wistful reflection / excited curiosity. See `tests/harness/sentences.json`.
- **Grid:** 15 models × 5 refs × 3 sentences = **225 outputs**. All cells succeeded.

## Listening result (Woj's ear)

Of the 15 zero-shot outputs, the models that produced *clearly or mostly clearly* recognizable clones of the character were:

1. **CosyVoice** (FunAudioLLM/CosyVoice2-0.5B)
2. **Fish Speech** (fishaudio/fish-speech-1.5, S2-Pro)
3. **IndexTTS-2** (IndexTeam/IndexTTS-2)
4. **VoxCPM** (openbmb/VoxCPM2)

Everything else was either too robotic, distorted, or not the right voice.

### Notable observations

- **Chatterbox** — current "open-source benchmark" per published evals (beats ElevenLabs in their blind tests) — did **not** land on this character. Possibly ref-audio sensitivity, possibly the dramatic / slow tone of the refs we picked. Worth a re-test with different refs before writing it off.
- **F5-TTS, XTTS-v2, GPT-SoVITS, StyleTTS2** — all credible in the wider TTS world; none made Woj's shortlist on this character.
- **Bark** does not cloud the character — its "voices" are pre-baked history-prompt presets. The bake-off adapter pinned `v2/en_speaker_6` and emitted 15 identical-voice cells. Real Bark cloning needs community forks (`bark-voice-cloning` etc.).
- **RVC** doesn't clone Diana either — we used a prebuilt JP-female `.pth` (no per-character training was performed). Proper RVC voice cloning needs a per-voice training run.

## Fish Speech pacing

Woj's read: "clean, but kind of slow and dramatic." Two findings while investigating:

1. **Fish Speech v2 has no `speed`/`pace` parameter.** Only `temperature`, `top_p`, `repetition_penalty`, `max_new_tokens`, `chunk_length`. (Pydantic schema in `fish_speech/utils/schema.py`.)
2. **The model inherits prosody from the reference clip.** Pace and dramatic-ness propagate from the ref's own pace and dramatic-ness. The 5 refs we picked all skew slow/emotional — Fish was being faithful, not slow on its own.

Temperature sweeps run (`tests/harness/fish_speech_sweep.py`, `fish_speech_sweep2.py`, bundles at `_listen/fish_sweep/` and `_listen/fish_sweep2/`): T010 (near-argmax) through T100 (max). Lower temperatures (~0.5) seemed to help slightly — they reduce dramatic flourishes but don't fundamentally speed things up. The real lever is the reference clip.

## What we did NOT do

- **Few-shot fine-tune** anything. The bake-off was pure zero-shot.
- **Try a per-character RVC training run.** Would replace the prebuilt-JP-voice hack with an actual Diana model. 17 min of data is comfortably above RVC's 5–10 min recommendation.
- **GPT-SoVITS fine-tune.** Their few-shot path wants only 1 min — we have 17×.
- **VoiceCraft.** Speech editing — would let us splice corrections into existing audio.

## Suggested next directions

Tier them roughly by likely value-per-effort on this dataset (17 min):

| Approach | Data fit | Risk | Why |
|---|---|---|---|
| **GPT-SoVITS few-shot** | comfortable (1 min needed) | low | Lightest fine-tune; if the zero-shot version was already in the credible-but-not-shortlist tier, fine-tuning might unlock it |
| **RVC training** | comfortable (5-10 min needed) | low | Currently doesn't clone Diana at all; a per-voice model would close that gap completely |
| **XTTS-v2 voice fine-tune** | borderline | medium | Coqui's per-voice fine-tune wants 1+ hour ideally; 17 min might work but quality TBD |
| **Tortoise voice customization** | borderline | medium | Has a lighter-weight voice-conditioning path; quality ceiling is high but speed is poor |
| **StyleTTS2 full fine-tune** | low (1+ hr recommended) | high | Likely insufficient data |
| **Bark voice clone (community fork)** | comfortable | medium-high | Would unlock Bark's non-verbal sounds; quality of the cloning forks is mixed |

A reasonable next-session plan is: GPT-SoVITS fine-tune + RVC training, run the same 3-sentence test harness, compare against the zero-shot winners and against the character ground truth.

## Repro

```bash
# Re-run the full bake-off (TTS only; VC adapters need _source/ pre-generated)
tests/harness/run_bakeoff.py --models all

# Listen
xdg-open _listen/bakeoff/index.html

# Re-run a single model (e.g. after adapter tweaks)
generators/<model>/.venv/bin/python tests/harness/run_bakeoff.py --models <model>
```

Outputs live at `tests/outputs/harness/<model>/<ref_id>__<sentence_id>.wav` (gitignored).

---

# Bake-off results — session 2 (2026-05-14 → 2026-05-15)

Fine-tune phase. All four planned experiments shipped on the 250-clip / ~17 min Diana dataset (curated subset of `reference_voice/ch05000_base_dialogue__en.manifest.csv`).

## What ran

| Model | Variant | Training config | Wall time | Where it lives |
|---|---|---|---|---|
| **GPT-SoVITS** | v2 (full FT) | 8 SoVITS ep + 15 GPT ep, batch 12/8 | ~3 min | `tests/finetune/gpt-sovits/`, weights at `generators/gpt-sovits/{SoVITS,GPT}_weights_v2/` |
| **GPT-SoVITS** | v2Pro (full FT + SV emb) | same | ~3 min | `..._v2Pro/` |
| **GPT-SoVITS** | v4 (LoRA r=32) | same | ~3 min | `..._v4/` |
| **RVC** | 48 kHz v2, with FAISS index | 100 ep, batch 8 | ~30 sec | `tests/finetune/rvc/`, weights at `specialized/rvc/assets/weights/diana_rvc.pth` + index at `specialized/rvc/logs/diana_rvc/` |
| **NeuTTS-Air** | full backbone FT | 500 steps, batch 4, lr 4e-5, ~8 epochs | ~86 min | `tests/finetune/neutts-air/diana_ckpt/` |
| **VoxCPM** | LoRA r=32 α=32 | 1000 steps, ~64 epochs | ~17 min | `tests/finetune/voxcpm/ckpt/step_0001000/` |

Inference adapters mirror the harness pattern: `tests/harness/run_<model>-ft.py`. Side-by-side gallery at `_listen/finetune/index.html` (10 columns: 4 zero-shot baselines + 6 FT variants).

## Listening verdict (Woj's ear)

> "The cloning is actually good — it's just the limitations of TTS at play when it comes to fine-tuned control of the speech."

**Timbre / identity** across the FT models was satisfactory. The wall was prosody:
- Em-dash (`—`) was not interpreted as a pause by *any* TTS variant. Mitigated mid-session by replacing `—` with `.` in `tests/harness/sentences.json` (`c_excited`).
- Exclamation-mark words (`here!`) didn't get an excited pitch rise in any GPT-SoVITS variant.
- Pacing inherited from the reference clip rather than the text emotion; `speed_factor` only post-hoc time-stretches.

GPT-SoVITS results converged at `top_3_acc ≈ 0.55` across v2/v2Pro/v4 — suggesting either capacity or step count was the bottleneck for the s1 (GPT) component, which is the prosody-controlling piece.

## What we learned about the toolchain

Patched upstream bugs (documented in `notes/install-gotchas.md`):
- **GPT-SoVITS `TTS_Config`** silently falls back to default v2 weights unless the config dict is wrapped in `{"custom": {...}}`. Affected all three FT adapters until detected by a listener finding every column bit-identical.
- **RVC** had four upstream bugs preventing training:
  - `"spec" in inp_path` substring bug in three f0-extraction scripts (false-matches when path contains `specialized/`).
  - `torch.load` `weights_only=True` rejecting the fairseq HuBERT checkpoint in feature extraction.
  - `matplotlib.figure.FigureCanvasAgg.tostring_rgb()` removed in 3.10+ — used in spectrogram TB logging.
  - `tools/infer/train-index-v2.py` hardcoded `./logs/anz/...` instead of taking the exp name as arg (replaced by `tests/finetune/rvc/build_index.py`).
- **NeuTTS-Air** inference venv missing several finetune-only deps: `loguru`, `fire`, `accelerate>=0.26`. Installed.
- **VoxCPM** `from_pretrained(lora_weights_path=...)` defaults to `LoRAConfig(r=8)` — must pass explicit `lora_config=LoRAConfig(r=32, alpha=32, ...)` matching training, or the safetensors load fails with a shape mismatch at the rank dimension.

## Project status

Parked. The TTS approaches we tested do clone the timbre well but can't drive prosody from text alone for emotive character work. Woj is opening a separate project to explore **Voice-to-Voice** — synthesize emotive source audio (user-read or another TTS), then convert with RVC / Seed-VC. The Diana RVC model trained this session is reusable there directly.
