# Install gotchas

Captured while standing up Tier 1 + 2 venvs on 2026-05-12. Each is a real
landmine; future-us should not re-discover them. Common tooling: `uv 0.11.14`
on Linux (WSL2), system Python 3.12, RTX 5090.

## CRITICAL: RTX 5090 and CUDA compute capability sm_120

The 5090 is compute capability **sm_120**. Stock PyTorch wheels from PyPI ship
with arch support up to sm_90. Trying to use the GPU produces:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 ... sm_90.
```

…followed at inference time by `RuntimeError: CUDA error: no kernel image is
available for execution on the device`.

**Fix:** install torch from the cu128 (or cu130) index:

```bash
uv pip install --python <venv>/bin/python --upgrade torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Most of our venvs got sm_120-capable torch *for free* — their pyproject pins
were loose enough that uv picked torch ≥ 2.7 / cu128 by default. Only repos
that hard-pin old torch (CosyVoice: `torch==2.3.1+cu121`; Seed-VC: `torch==2.4.0`
with no CUDA suffix → cu121) needed manual upgrade.

**Verify with:**
```python
import torch
print("sm_120" in torch.cuda.get_arch_list())  # must be True on the 5090
```

## torchaudio 2.11+: save/load now requires torchcodec

Starting in torchaudio 2.11, `torchaudio.save()` and `torchaudio.load()`
delegate to `torchcodec`, which isn't installed by default. Two workarounds:

1. **Sidestep torchaudio for I/O** — use `soundfile.write(path, ndarray, sr)`
   and `soundfile.read(path)`. This is the cleanest path when *we* control
   the I/O (smoke tests, harness).
2. **Install torchcodec** — required when *model internals* call
   `torchaudio.load/save` (e.g. XTTS-v2 loads the speaker reference via
   torchaudio.load). `uv pip install torchcodec` got us 0.11.1.

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
- Repo's `requirements.txt` says `transformers>=4.33.0` but XTTS's internal
  `TTS/tts/layers/xtts/stream_generator.py` imports `BeamSearchScorer`,
  `SampleOutput`, etc. which were removed around transformers 4.52. **Pin
  `transformers==4.40.2`** (pulls back `tokenizers==0.19.1`).
- torch ≥ 2.6 defaults `torch.load` to `weights_only=True`; this rejects the
  XTTS checkpoint because it contains pickled `XttsConfig`/`XttsAudioConfig`
  objects. Either monkey-patch `torch.load` to `weights_only=False` or call
  `torch.serialization.add_safe_globals([...])` with the XTTS config classes.
- XTTS internals call `torchaudio.load(speaker_wav)` → requires `torchcodec`
  (see global note above). Soundfile alone is *not* enough here.
- Set `COQUI_TOS_AGREED=1` programmatically to bypass the CPML click-through.

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
- **`git lfs pull` fails** — the `index-tts/index-tts` repo has exceeded its
  GitHub LFS budget. The bundled `examples/*.wav` files are 131-byte LFS
  pointers. Workaround: pull example wavs from the HuggingFace demo space
  `IndexTeam/IndexTTS-2-Demo` via
  `hf_hub_download(repo_type='space', filename='examples/voice_01.wav')`.
- Model weights are **not bundled**. Only `config.yaml` + `pinyin.vocab` ship
  in `checkpoints/`. Pull real weights from HF: `hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints`. ~5.5 GB.
- `infer_v2.py` hard-codes `os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'`
  at import time, so the script must `os.chdir(INDEXTTS_DIR)` before importing
  or auto-downloads land in the wrong place.
- Set `use_cuda_kernel=False` to avoid BigVGAN custom-kernel JIT compilation.
- `uv sync` handles base install via `uv.lock`.

### GPT-SoVITS (`generators/gpt-sovits`)
- Has a complex `install.sh` we bypassed — `uv pip install -r requirements.txt`
  was enough for inference. May need install.sh later for training/fine-tune.
- **No bundled weights.** `install.sh` downloads a 4.5 GB zip; minimum set for
  v2 EN inference (~1 GB total): `s2G2333k.pth`, `s1bert25hz-…ckpt`,
  `chinese-hubert-base/`, `chinese-roberta-wwm-ext-large/` — pull individually
  from HF `lj1995/GPT-SoVITS` into `GPT_SoVITS/pretrained_models/`.
- **torchaudio/torch ABI mismatch.** uv installed `torch 2.10.0+cu128` paired
  with `torchaudio 2.11.0` (built for CUDA 13 → `libcudart.so.13` missing at
  import). Pin matching: `torchaudio==2.10.0+cu128`.
- torchaudio 2.10+ delegates `load()` to torchcodec → monkey-patch
  `torchaudio.load` to a soundfile shim, or install torchcodec.
- NLTK data required for the English G2P frontend:
  `nltk.download('averaged_perceptron_tagger_eng' / 'punkt' / 'punkt_tab' / 'cmudict')`.
- Working-directory sensitive: the repo's TTS code uses bare paths like
  `tools.i18n` and `GPT_SoVITS/pretrained_models/...` — must `chdir` into the
  gpt-sovits repo root before importing.
- It's *zero-shot voice cloning*, not pure TTS: needs a reference WAV + the
  reference transcript + the target text.
- First `TTS_Config({...})` instantiation writes a `custom:` block back into
  `GPT_SoVITS/configs/tts_infer.yaml` — harmless but mutates the repo.

### Seed-VC (`specialized/seed-vc`)
- `requirements.txt` uses inline pip flags (`torch --pre --index-url ...`).
  `uv pip install` rejects those; use vanilla pip in a `--seed`ed venv:
  ```bash
  uv venv --python 3.10 --seed .venv
  .venv/bin/pip install -r requirements.txt
  ```
- Conda yaml specifies Python 3.10 — uv auto-fetches CPython 3.10.20.

### VoiceCraft (`specialized/voicecraft`) — SKIPPED
Not installed. Three blockers, mostly the third:
1. Requires Python 3.9.16 specifically.
2. Requires Montreal Forced Aligner from conda-forge (no PyPI equivalent).
3. Requires torch 2.0.1 with CUDA 11.7. The RTX 5090 is sm_120 and needs
   torch >= 2.6 for that compute capability. Even if install succeeded,
   inference would fail with `no kernel image is available for execution`.
Revisit only if we actually need speech editing. Recommended path is installing
miniconda and following their README literally — uv can't handle this one.

## System packages needed (apt)
- `git-lfs` (for IndexTTS-2 weights and others)
- `portaudio19-dev` (for Fish Speech / pyaudio)
- `ffmpeg`, `libsndfile1` — already present on this container

## Lesson for future installs
**Don't pipe install commands through `tail` when running in background.**
`tail -N` buffers everything until the process exits, so you see nothing during
multi-minute installs. Use `2>&1 | tee` or write to a log file directly.
