"""Smoke test for StyleTTS2 (LJSpeech single-speaker checkpoint).

StyleTTS2 ships as a training repo, not a packaged inference SDK. This script
ports the canonical Demo/Inference_LJSpeech.ipynb walkthrough to a runnable
script: chdir into the repo, import its top-level `models`/`utils`/`text_utils`,
build the model from `Models/LJSpeech/config.yml`, load `epoch_2nd_00100.pth`,
run the diffusion sampler + decoder, and save the resulting numpy waveform via
soundfile (sidesteps the torchaudio 2.11+ → torchcodec dependency).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SMOKE_TEXT = "Hello! This is a test of voice synthesis. The output should sound clear and natural."
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "smoke" / "styletts2.wav"

STYLETTS2_DIR = Path("/workspace/VoiceReplication/generators/styletts2")
# HF download puts files at Models/LJSpeech/Models/LJSpeech/ (nested) — config_yml + epoch_2nd_00100.pth live there.
MODEL_DIR = STYLETTS2_DIR / "Models" / "LJSpeech" / "Models" / "LJSpeech"
CONFIG_PATH = MODEL_DIR / "config.yml"
CHECKPOINT_PATH = MODEL_DIR / "epoch_2nd_00100.pth"

# chdir + sys.path so the repo's bare imports (`from models import ...`,
# `Utils/ASR/...` relative paths in config.yml) resolve.
os.chdir(STYLETTS2_DIR)
sys.path.insert(0, str(STYLETTS2_DIR))

# Reproducibility (mirrors the notebook).
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random
random.seed(0)
np.random.seed(0)

# Hard-fail if CUDA is missing — this is the GPU bake-off, not a CPU fallback.
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — smoke test requires the RTX 5090")
device = "cuda"
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

# NLTK punkt is needed by the notebook's word_tokenize call.
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

import yaml
import torchaudio  # noqa: F401 — imported for parity with notebook (mel transform import)
from nltk.tokenize import word_tokenize
import phonemizer

# torch >= 2.6 flipped torch.load default to weights_only=True. The repo's helpers
# (models.load_ASR_models, models.load_F0_models, Utils/PLBERT/util.load_plbert,
# and our final epoch_2nd checkpoint load) all hit pickled training-state objects
# that aren't in the safe-globals allowlist. Force weights_only=False repo-wide.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

textcleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


t0 = time.time()

config = yaml.safe_load(open(CONFIG_PATH))

# Pretrained text aligner / pitch extractor / PL-BERT — config points at Utils/...
ASR_config = config.get("ASR_config", False)
ASR_path = config.get("ASR_path", False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

F0_path = config.get("F0_path", False)
pitch_extractor = load_F0_models(F0_path)

BERT_path = config.get("PLBERT_dir", False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config["model_params"]), text_aligner, pitch_extractor, plbert)
for key in model:
    model[key].eval()
    model[key].to(device)

# torch >= 2.6 defaults weights_only=True which rejects pickled training-state objects.
params_whole = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
params = params_whole["net"]

from collections import OrderedDict
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

t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s", flush=True)


def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    text = text.strip().replace('"', "")
    ps = global_phonemizer.phonemize([text])
    ps = " ".join(word_tokenize(ps[0]))

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        ).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )

    return out.squeeze().cpu().numpy()


sr = config["preprocess_params"]["sr"]  # 24000

t0 = time.time()
noise = torch.randn(1, 1, 256).to(device)
wav = inference(SMOKE_TEXT, noise, diffusion_steps=5, embedding_scale=1)
t_gen = time.time() - t0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), wav, sr)

duration = len(wav) / sr
print(f"generated {duration:.2f}s of audio in {t_gen:.2f}s  (RTF={t_gen/duration:.2f})", flush=True)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
