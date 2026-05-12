# Model Catalog

The model repositories themselves live in `generators/` and `specialized/` but are
**not** committed to this project (see `.gitignore`). This file is the canonical
manifest — re-clone everything with `scripts/clone-models.sh` (TODO).

Pinned commits are the SHAs we cloned at on 2026-05-12, captured for
reproducibility. Update them whenever we re-sync from upstream.

## Tier 1 — Top contenders (`generators/`)

| Dir | Repo | Pinned | Ref audio | License | Notes |
|---|---|---|---|---|---|
| `chatterbox` | [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `3f35dfc` | ~5s | MIT | Current open-source benchmark; beats ElevenLabs in blind A/B |
| `f5-tts` | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | `6f91022` | ~10s | MIT (code) | Diffusion-based; very natural prosody |
| `xtts-v2` | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) | `dbf1a08` | ~6s | CPML (non-commercial) | Baseline; most-downloaded TTS on HuggingFace |
| `index-tts-2` | [index-tts/index-tts](https://github.com/index-tts/index-tts) | `830f6f8` | up to 15s | Apache 2.0 | **Emotion control** via vector/audio/text — best fit for character work |
| `gpt-sovits` | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | `08d627c` | 5s zero-shot, OR 1min fine-tune | MIT | Straddles zero-shot and few-shot fine-tuning |

## Tier 2 — Strong variety picks (`generators/`)

| Dir | Repo | Pinned | Ref audio | License | Notes |
|---|---|---|---|---|---|
| `cosyvoice` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `ace7c47` | 10-30s | Apache 2.0 | Top speaker similarity scores |
| `voxcpm` | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | `19b6bf7` | short | Apache 2.0 | Tokenizer-free, 30 languages, voice *design* by description |
| `fish-speech` | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | `a391467` | short | CC-BY-NC (weights) | Top TTS Arena ELO (1339); weights are non-commercial |
| `styletts2` | [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | `5cedc71` | ~1hr fine-tune | MIT | The "longer fine-tune" candidate; high quality |
| `tortoise-tts` | [neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts) | `8a2563e` | clips | Apache 2.0 | Older, slow, but historically top-tier — quality ceiling reference |

## Tier 3 — Specialized tools (`specialized/`)

Intended for **enhancing** or **augmenting** output from the Tier 1/2 models —
not standalone candidates in the main bake-off.

| Dir | Repo | Pinned | Role | License |
|---|---|---|---|---|
| `bark` | [suno-ai/bark](https://github.com/suno-ai/bark) | `f4f32d4` | Non-verbal sounds (laughs, sighs, grunts) | MIT |
| `voicecraft` | [jasonppy/VoiceCraft](https://github.com/jasonppy/VoiceCraft) | `a702dfd` | Speech *editing* — replace single words inside existing audio | CC-BY-NC-SA |
| `rvc` | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | `7ef1986` | Voice *conversion* — swap timbre of a source recording (5-10 min training data) | MIT |
| `seed-vc` | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | `51383ef` | Zero-shot voice conversion + singing voice conversion | GPL-3 |
| `omnivoice` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) | `d8b22ae` | 600+ languages, zero-shot | Apache 2.0 |
| `neutts-air` | [neuphonic/neutts-air](https://github.com/neuphonic/neutts-air) | `857bec0` | On-device, compact 0.5B LLM backbone | Apache 2.0 |
