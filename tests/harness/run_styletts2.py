"""Harness adapter for StyleTTS2 (LibriTTS multi-speaker checkpoint).

StyleTTS2 ships as a training repo, not a packaged inference SDK, so this
adapter ports Demo/Inference_LibriTTS.ipynb into a runnable script. We use the
LibriTTS checkpoint (not LJSpeech) because the LJSpeech model is single-speaker
and cannot clone voices — it would ignore the reference audio entirely. The
LibriTTS model performs zero-shot voice cloning by computing a 256-dim style
vector from the reference wav (concat of style_encoder and predictor_encoder
outputs) and conditioning the diffusion sampler on it.

Quirks preserved from the smoke test:
  - chdir into the repo so its bare imports (`from models import ...`) and the
    config.yml-relative paths (`Utils/ASR/...`) resolve.
  - sys.path prepend so the repo's modules are importable.
  - torch.load monkey-patch with weights_only=False — torch >= 2.6 defaults to
    True, which rejects the pickled training-state objects in all 4 checkpoints
    (ASR, F0/JDC, PLBERT, and the main epoch_2nd checkpoint).
  - NLTK punkt is needed by word_tokenize.
  - espeak-ng / phonemizer is needed for the EspeakBackend.
"""
import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Make _common.py importable regardless of cwd (we chdir below).
sys.path.insert(0, str(Path(__file__).parent))
from _common import emit, load_manifest, stopwatch  # noqa: E402

STYLETTS2_DIR = Path("/workspace/VoiceReplication/generators/styletts2")
MODEL_DIR = STYLETTS2_DIR / "Models" / "LibriTTS"
CONFIG_PATH = MODEL_DIR / "config.yml"
CHECKPOINT_PATH = MODEL_DIR / "epochs_2nd_00020.pth"  # note plural "epochs"

# chdir + sys.path so the repo's bare imports and config-relative Utils/... paths resolve.
os.chdir(STYLETTS2_DIR)
sys.path.insert(0, str(STYLETTS2_DIR))

# Reproducibility (mirrors the notebook).
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to JSON-lines manifest from the orchestrator")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)
    device = "cuda"

    rows = load_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    # NLTK punkt is required by word_tokenize.
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    import yaml
    import librosa
    import torchaudio
    import phonemizer
    from nltk.tokenize import word_tokenize

    # torch >= 2.6 flipped torch.load default to weights_only=True, which rejects
    # the pickled training-state objects in ASR/F0/PLBERT/main checkpoints.
    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat

    from models import build_model, load_ASR_models, load_F0_models  # noqa: E402
    from utils import recursive_munch  # noqa: E402
    from text_utils import TextCleaner  # noqa: E402
    from Utils.PLBERT.util import load_plbert  # noqa: E402
    from Modules.diffusion.sampler import (  # noqa: E402
        DiffusionSampler, ADPM2Sampler, KarrasSchedule,
    )

    textcleaner = TextCleaner()
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True,
    )

    # Mel transform for reference style extraction (24kHz, matches notebook).
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300,
    )
    mean, std = -4, 4

    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

    sw = stopwatch()

    # ----- Load model (once) -----
    config = yaml.safe_load(open(CONFIG_PATH))

    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    for key in model:
        model[key].eval()
        model[key].to(device)

    params_whole = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    params = params_whole["net"]
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                new_state_dict = OrderedDict()
                for k, v in params[key].items():
                    new_state_dict[k[7:]] = v  # strip leading `module.`
                model[key].load_state_dict(new_state_dict, strict=False)
    for key in model:
        model[key].eval()

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    sr = config["preprocess_params"]["sr"]  # 24000
    is_hifigan = model_params.decoder.type == "hifigan"

    def compute_style(path):
        """Extract the 256-dim reference style vector (timbre + prosody)."""
        wave, _sr = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        mel_tensor = preprocess(audio).to(device)
        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        """Synthesize `text` in the voice described by `ref_s` (LibriTTS API)."""
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = " ".join(word_tokenize(ps[0]))
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # LibriTTS sampler call: noise is (1, 1, 256) and features=ref_s.
            s_pred = sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            # Blend sampled style with reference (alpha=timbre, beta=prosody).
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
            if is_hifigan:
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
            if is_hifigan:
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        # Strip the weird pulse at the end (per the notebook's comment).
        return out.squeeze().cpu().numpy()[..., :-50]

    load_s = sw()

    # ----- Per-row synthesis -----
    # Cache style vectors keyed by ref_id so repeated refs aren't recomputed.
    style_cache: dict[str, torch.Tensor] = {}

    for i, row in enumerate(rows):
        try:
            if row.ref_id not in style_cache:
                style_cache[row.ref_id] = compute_style(str(row.ref_path))
            ref_s = style_cache[row.ref_id]

            t0 = time.time()
            wav = inference(
                row.text, ref_s,
                alpha=0.3, beta=0.7,
                diffusion_steps=5, embedding_scale=1,
            )
            gen_s = time.time() - t0

            row.output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(row.output), wav, sr)
            duration_s = len(wav) / sr

            emit(
                "styletts2", row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit("styletts2", row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
