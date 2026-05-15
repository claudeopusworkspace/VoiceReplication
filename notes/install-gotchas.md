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
   it. **Correction to earlier note**: whisper IS used at runtime —
   `cli/frontend.py` calls `whisper.log_mel_spectrogram` on the prompt audio
   inside `_extract_speech_token`. Stripping it from install is fine, but you
   must install it back before inference. Use a newer release that builds
   cleanly:
   ```bash
   .venv/bin/pip install openai-whisper tiktoken --no-deps
   ```
   `--no-deps` avoids pulling a torch/numba downgrade that would clobber cu128.
3. The `third_party/Matcha-TTS` submodule is empty after a non-recursive clone.
   `cosyvoice/flow/flow_matching.py` does `from matcha.models.components...` so
   model load fails with `ModuleNotFoundError: No module named 'matcha'` if you
   miss this. Init the submodule:
   ```bash
   git submodule update --init --recursive
   ```
   The smoke test also appends `third_party/Matcha-TTS` to `sys.path` (mirror
   what upstream `example.py` does on line 2).
4. Pulls TensorRT — venv ends up at 12 GB. Heaviest of the lot.
5. **torch 2.11 upgrade for sm_120 works fine** — repo's pinned torch 2.3.1 was
   bumped to 2.11.0+cu128 (matched torchaudio 2.11.0+cu128). No ABI breakage in
   the CosyVoice2 inference path despite the 8-minor-version jump.
6. torchaudio 2.11 + torchcodec: `utils/file_utils.load_wav` calls
   `torchaudio.load(wav, backend='soundfile')`. The `backend=` kwarg is now
   ignored and the call routes through torchcodec. `pip install torchcodec`
   (got 0.11.1) fixes it. A soundfile-only monkey-patch would also work but
   requires touching repo code.
7. First inference triggers a modelscope download of the `pengzhendong/wetext`
   text-normalization FSTs (~10 MB) into `~/.cache/modelscope/`. Harmless,
   just expect the network hit on a cold run.
8. onnxruntime-gpu warns `libcudnn.so.8: cannot open shared object file` and
   falls back to CPU EP for the campplus/speech-tokenizer ONNX sessions.
   Doesn't fail the run — those models are tiny — but throughput is slightly
   below what the GPU EP would give.

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

### VoxCPM (`generators/voxcpm`)
- `uv sync` handles base install via `uv.lock`. Got `torch 2.10.0+cu128` for
  free — sm_120 compatible out of the box.
- Default `from_pretrained("openbmb/VoxCPM2")` pulls **two** model bundles
  before inference can run: (a) the VoxCPM2 snapshot (~6 GB across 9 files,
  ~3.5 min on this link), and (b) the ZipEnhancer denoiser via ModelScope when
  `load_denoiser=True` (the default). The smoke test passes `load_denoiser=False`
  to skip the denoiser fetch — only needed at generate time with `denoise=True`.
- `optimize=True` (the default) runs a torch.compile warm-up generation inside
  the constructor. First-time JIT cost was ~1:45 of the ~5:50 total cold load
  on the 5090. Subsequent loads stay slow if the inductor cache is cold; cache
  hits make this dramatically cheaper.
- Reference-only voice cloning (`reference_wav_path=...`) needs no transcript —
  VoxCPM2 routes the ref clip through ref-audio tokens. Cleanest entrypoint for
  a smoke test. The "ultimate cloning" mode (`prompt_wav_path` + `prompt_text`)
  is the one that needs a transcript.
- Sample rate is 48 kHz (`model.tts_model.sample_rate`), highest in the bake-off.
- Cold-start RTF ~2.3 on the first real generation (torch.compile still settling);
  subsequent generations should be much faster once kernels are cached.

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

### Bark (`specialized/bark`)
- `soundfile` is not in Bark's default install — `uv pip install soundfile`
  before running anything that uses it.
- Bark checkpoints contain pickled `numpy.core.multiarray.scalar` objects, so
  torch ≥ 2.6's `weights_only=True` default rejects them at load with
  `UnpicklingError`. Same class of issue as XTTS-v2. Fix: monkey-patch
  `torch.load` to force `weights_only=False` **before** importing `bark`.
- API is clean: `from bark import generate_audio, preload_models; preload_models(); audio = generate_audio(text)`. Returns numpy float32 at 24 kHz.

### NeuTTS Air (`specialized/neutts-air`)
- Constructor takes **separate device kwargs** — `backbone_device` and
  `codec_device`, not a unified `device=`. Set both to `"cuda"` for full GPU.
- Voice cloning requires **both** a reference wav AND a reference transcript
  (LLM is conditioned on text+audio). The repo bundles `samples/jo.wav` +
  `samples/jo.txt` — use those for a hands-off smoke test.
- The real class is `neutts.neutts.NeuTTS`; `neuttsair.neutts.NeuTTSAir` is a
  thin re-export. Either import works.
- A stale `output.wav` ships in the repo root from prior testing — write your
  output elsewhere (we use `tests/outputs/smoke/neutts-air.wav`).

### OmniVoice (`specialized/omnivoice`)
- `uv sync` gave us `torch 2.8.0+cu128` for free (sm_120 in arch list).
- API: `from omnivoice import OmniVoice; m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)`.
- `m.generate(text=...)` returns a **list of numpy arrays**, one per input —
  use `audio[0]` for a single-text call, not a tensor.
- Sample rate at `model.sampling_rate` (24 kHz), not the `tts_model.sample_rate`
  path some other models use.
- Auto Voice mode (no reference) works out of the box and is the easiest
  smoke-test entrypoint. The repo's `examples/` ships training shell scripts
  only — **no bundled reference wavs**, so voice-cloning mode needs audio from
  elsewhere if you want to test it.

### RVC (`specialized/rvc`)
- **No usable weights bundled.** The `assets/{hubert,pretrained,weights,indices,rmvpe}/`
  folders contain only tiny `*_inputs.pth` shape stubs (~30–170 KB). Real
  weights must be fetched:
  - HuBERT content encoder (`hubert_base.pt`, ~190 MB) from HF `lj1995/VoiceConversionWebUI`
  - A target-voice `.pth` from any community RVC voice repo (we used
    `aa444rt/RVC_V2_models_5_japanese_womens` for `V2-AISO-HOWATTO.pth`, ~55 MB)
  - `rmvpe.pt` (~180 MB) only if using f0-conditioned voice models; pick
    `f0_method='pm'` to avoid this download for non-f0 voice models.
- For **training** specifically, also fetch (per sample rate):
  - `pretrained_v2/f0G48k.pth` + `pretrained_v2/f0D48k.pth` (~70 + 130 MB) for
    48 kHz v2 training — the base generator/discriminator weights to fine-tune from.
- torch ≥ 2.6's `weights_only=True` rejects fairseq's HuBERT checkpoint (it
  pickles `fairseq.data.dictionary.Dictionary`). Same monkey-patch as Bark/XTTS.
  This also hits **`infer/modules/train/extract_feature_print.py`** during
  training pre-prep — wrap it with a runpy launcher that patches torch.load
  first (see `tests/finetune/rvc/run_feature_extract_wrapped.py`).
- **`"spec" in inp_path` substring-match bug in 3 f0 extraction scripts.** Any
  exp_dir whose absolute path contains the literal substring "spec" (e.g.
  living under `specialized/rvc/...`) will cause every file to be skipped and
  the script prints `no-f0-todo` with empty output. Affects:
  - `infer/modules/train/extract/extract_f0_rmvpe.py:118`
  - `infer/modules/train/extract/extract_f0_rmvpe_dml.py:118`
  - `infer/modules/train/extract/extract_f0_print.py:157`
  Patch: change `if "spec" in inp_path:` → `if "spec" in name:`. Already applied
  in our clone (gitignored upstream — re-apply on reclone).
- **CWD-sensitive.** Everything uses relative paths (`assets/hubert/hubert_base.pt`,
  `os.getenv("weight_root")`, etc.). `os.chdir(RVC_DIR)` + `load_dotenv(RVC_DIR/".env")`
  before any RVC import.
- `Config()` calls `argparse` on `sys.argv` directly — would crash on stray
  args from the harness. Clear with `sys.argv = sys.argv[:1]` before
  instantiating.
- The repo is a Gradio WebUI, not an SDK. Calling `VC` from `infer.modules.vc.modules`
  directly is cleaner than wrapping `tools/infer_cli.py`.

### Seed-VC (`specialized/seed-vc`)
- `requirements.txt` uses inline pip flags (`torch --pre --index-url ...`).
  `uv pip install` rejects those; use vanilla pip in a `--seed`ed venv:
  ```bash
  uv venv --python 3.10 --seed .venv
  .venv/bin/pip install -r requirements.txt
  ```
- Conda yaml specifies Python 3.10 — uv auto-fetches CPython 3.10.20.
- `inference.py` hard-codes `HF_HUB_CACHE = './checkpoints/hf_cache'` — must
  `os.chdir(SEED_VC_DIR)` before import.
- Bundled reference voices in `examples/reference/` (e.g. `teio_0.wav`,
  `azuma_0.wav`, `dingzhen_0.wav`, `trump_0.wav`) — pick any for smoke.
- Benign load warnings: "Skipped loading some keys: estimator.f0_embedder /
  input_pos" — those are V2/f0-variant weights not used by V1.

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
