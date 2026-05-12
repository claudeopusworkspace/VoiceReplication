# Install gotchas

Captured while standing up Tier 1 + 2 venvs on 2026-05-12. Each is a real
landmine; future-us should not re-discover them. Common tooling: `uv 0.11.14`
on Linux (WSL2), system Python 3.12, RTX 5090.

## Tooling decisions

- Each model gets its own `.venv/` because dependency pins clash badly. Tried
  to consolidate — don't. The disk cost (~70 GB across 10 envs) is cheaper than
  the human time spent unwedging version conflicts.
- `uv` is the right venv manager. It auto-downloads needed Python versions
  (e.g. 3.10 for IndexTTS-2, 3.11 for everything else). Faster than pip too.
- Models that ship a `uv.lock` use `uv sync` (IndexTTS-2, VoxCPM, Fish Speech).
  Others use `uv venv --python 3.11 .venv` then `uv pip install -e .` or
  `uv pip install -r requirements.txt`.

## Model-specific gotchas

### Coqui XTTS-v2 (`generators/xtts-v2`)
- `setup.py` declares `python >= 3.9, < 3.12`. System Python 3.12 won't work.
  Use `uv venv --python 3.11`. uv auto-fetches CPython 3.11.15.

### Tortoise-TTS (`generators/tortoise-tts`)
- `setup.py` has self-contradicting pins (`transformers==4.31.0` AND
  `tokenizers==0.14.0`, but `transformers==4.31.0` requires `tokenizers<0.14`).
  uv refuses to resolve. Workaround:
  ```bash
  uv pip install -r requirements.txt           # uses looser tokenizers pin
  uv pip install -e . --no-deps                # install tortoise itself
  ```

### Fish Speech (`generators/fish-speech`)
- `pyaudio` is a transitive dep that requires `portaudio.h` system-side.
  Install first: `sudo apt install portaudio19-dev`.
- `uv sync` is silent on success — empty output is fine, check `.venv` size.

### CosyVoice (`generators/cosyvoice`)
Multiple compounding issues:
1. `requirements.txt` pulls from an Azure DevOps index for some packages and
   `protobuf==4.25` only resolves there at a different version. Add
   `--index-strategy unsafe-best-match`.
2. Pins `openai-whisper==20231117`. That old version's `setup.py` imports
   `pkg_resources` at top level, and modern build isolation doesn't auto-inject
   it. The package isn't actually imported by CosyVoice's code (only
   architectural references). Workaround: strip the line:
   ```bash
   grep -v 'openai-whisper' requirements.txt > requirements.no-whisper.txt
   ```
   Install from the filtered file via vanilla pip (uv's resolver is too strict
   for the rest of this dep graph too):
   ```bash
   uv venv --python 3.11 --seed .venv         # --seed so pip is in the venv
   .venv/bin/pip install setuptools wheel
   .venv/bin/pip install -r requirements.no-whisper.txt
   ```
3. Pulls TensorRT — venv ends up at 12 GB. Heaviest of the lot.

### IndexTTS-2 (`generators/index-tts-2`)
- Requires `git-lfs` for weights (apt install git-lfs && git lfs install).
- We didn't run `git lfs pull` after the shallow clone — do that before the
  first smoke test if upstream stores weights in-repo.
- `uv sync` handles install via `uv.lock`. No extra steps.

### GPT-SoVITS (`generators/gpt-sovits`)
- Has a complex `install.sh` we bypassed — `uv pip install -r requirements.txt`
  was enough for inference. May need install.sh later for training/fine-tune.

## System packages needed (apt)
- `git-lfs` (for IndexTTS-2 weights and others)
- `portaudio19-dev` (for Fish Speech / pyaudio)
- `ffmpeg`, `libsndfile1` — already present on this container

## Lesson for future installs
**Don't pipe install commands through `tail` when running in background.**
`tail -N` buffers everything until the process exits, so you see nothing during
multi-minute installs. Use `2>&1 | tee` or write to a log file directly.
