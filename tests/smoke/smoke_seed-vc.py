"""Smoke test for Seed-VC (V1, the default real-time-capable model).

Seed-VC is **voice conversion**, not TTS: given a source WAV and a target voice
reference WAV, it produces audio that "says the same thing in the target voice."
Zero-shot — no per-target training.

For this smoke test:
  * Source audio = the chatterbox smoke output (~5s of synthesized English).
  * Target reference = teio_0.wav from the repo's bundled reference clips
    (a clearly different voice — good signal that conversion actually happened).
  * Output is written to tests/outputs/smoke/seed-vc.wav.

Implementation note: we replicate the inference path from upstream's
inference.py (V1 / non-f0 branch) inline rather than shelling out, so we can
control timings and output path cleanly. Must chdir into the seed-vc repo root
because inference.py's load_models references config files via relative paths
(e.g. 'configs/hifigan.yml') and sets HF_HUB_CACHE='./checkpoints/hf_cache'
relative to cwd.
"""
import os
import sys
import time
from pathlib import Path

# --- Bootstrap: Seed-VC's load path uses relative file lookups. ---
SEED_VC_DIR = Path("/workspace/VoiceReplication/specialized/seed-vc")
os.chdir(SEED_VC_DIR)
sys.path.insert(0, str(SEED_VC_DIR))
# Mirror upstream's HF cache layout so weights land inside the repo (~3 GB).
os.environ.setdefault("HF_HUB_CACHE", str(SEED_VC_DIR / "checkpoints" / "hf_cache"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import librosa  # noqa: E402
import yaml  # noqa: E402

# Repo-local imports (require sys.path + cwd configured above)
from modules.commons import build_model, load_checkpoint, recursive_munch  # noqa: E402
from hf_utils import load_custom_model_from_hf  # noqa: E402

SMOKE_DIR = Path("/workspace/VoiceReplication/tests/outputs/smoke")
SOURCE_PATH = SMOKE_DIR / "chatterbox.wav"
TARGET_REF_PATH = SEED_VC_DIR / "examples" / "reference" / "teio_0.wav"  # bundled distinct voice
OUTPUT_PATH = SMOKE_DIR / "seed-vc.wav"

# Inference hyperparameters — match upstream inference.py defaults.
DIFFUSION_STEPS = 30
LENGTH_ADJUST = 1.0
INFERENCE_CFG_RATE = 0.7
FP16 = True

# --- Device check (fail loudly per smoke contract) ---
if not torch.cuda.is_available():
    print("CUDA not available — failing loudly per smoke contract", flush=True)
    sys.exit(1)

device = torch.device("cuda")
print(f"device={device}  torch={torch.__version__}", flush=True)
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

if not SOURCE_PATH.exists():
    print(f"Source audio not found at {SOURCE_PATH}; run smoke_chatterbox.py first.", flush=True)
    sys.exit(2)
if not TARGET_REF_PATH.exists():
    print(f"Target reference audio not found at {TARGET_REF_PATH}.", flush=True)
    sys.exit(2)

# --- Model load (V1 / non-f0 branch from inference.py) ---
t0 = time.time()

dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
)

config = yaml.safe_load(open(dit_config_path, "r"))
model_params = recursive_munch(config["model_params"])
model_params.dit_type = "DiT"
model = build_model(model_params, stage="DiT")
sr = config["preprocess_params"]["sr"]

model, _, _, _ = load_checkpoint(
    model,
    None,
    dit_checkpoint_path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
)
for key in model:
    model[key].eval()
    model[key].to(device)
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# Speaker-embedding model (CAMPPlus)
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402

campplus_ckpt_path = load_custom_model_from_hf(
    "funasr/campplus", "campplus_cn_common.bin", config_filename=None
)
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model.eval().to(device)

# Vocoder (V1 default = bigvgan)
vocoder_type = model_params.vocoder.type
assert vocoder_type == "bigvgan", f"Expected bigvgan vocoder for V1; got {vocoder_type}"
from modules.bigvgan import bigvgan  # noqa: E402

bigvgan_model = bigvgan.BigVGAN.from_pretrained(model_params.vocoder.name, use_cuda_kernel=False)
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.eval().to(device)
vocoder_fn = bigvgan_model

# Semantic tokenizer (V1 default = whisper)
speech_tokenizer_type = model_params.speech_tokenizer.type
assert speech_tokenizer_type == "whisper", f"Expected whisper tokenizer for V1; got {speech_tokenizer_type}"
from transformers import AutoFeatureExtractor, WhisperModel  # noqa: E402

whisper_name = model_params.speech_tokenizer.name
whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
del whisper_model.decoder
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)


def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
    ori_inputs = whisper_feature_extractor(
        [waves_16k.squeeze(0).cpu().numpy()],
        return_tensors="pt",
        return_attention_mask=True,
    )
    ori_input_features = whisper_model._mask_input_features(
        ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
    ).to(device)
    with torch.no_grad():
        ori_outputs = whisper_model.encoder(
            ori_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    S_ori = ori_outputs.last_hidden_state.to(torch.float32)
    S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
    return S_ori


# Mel spectrogram fn
from modules.audio import mel_spectrogram  # noqa: E402

mel_fn_args = {
    "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
    "win_size": config["preprocess_params"]["spect_params"]["win_length"],
    "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
    "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
    "sampling_rate": sr,
    "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
    "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
    "center": False,
}
mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

t_load = time.time() - t0
print(f"loaded in {t_load:.1f}s  (sr={sr})", flush=True)

# --- Conversion (mirrors inference.main, V1 / non-f0 path) ---
hop_length = 256  # V1 / non-f0
max_context_window = sr // hop_length * 30
overlap_frame_len = 16
overlap_wave_len = overlap_frame_len * hop_length

source_audio = librosa.load(str(SOURCE_PATH), sr=sr)[0]
ref_audio = librosa.load(str(TARGET_REF_PATH), sr=sr)[0]
source_duration = len(source_audio) / sr
print(
    f"source={SOURCE_PATH.name} ({source_duration:.2f}s)  "
    f"target_ref={TARGET_REF_PATH.name} ({len(ref_audio)/sr:.2f}s)",
    flush=True,
)

source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

t_vc_start = time.time()

with torch.no_grad():
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # Source is short (~5s) so whisper handles in one forward; skip the chunking branch.
    assert converted_waves_16k.size(-1) <= 16000 * 30, "Source > 30s — chunking branch not exercised in smoke."
    S_alt = semantic_fn(converted_waves_16k)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * LENGTH_ADJUST)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond, *_ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, *_ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if FP16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                DIFFUSION_STEPS,
                inference_cfg_rate=INFERENCE_CFG_RATE,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            # crossfade
            fade_out = np.cos(np.linspace(0, np.pi / 2, overlap_wave_len)) ** 2
            fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap_wave_len)) ** 2
            cur = vc_wave[0].cpu().numpy()
            cur[:overlap_wave_len] = cur[:overlap_wave_len] * fade_in + previous_chunk.cpu().numpy() * fade_out
            generated_wave_chunks.append(cur)
            break
        else:
            fade_out = np.cos(np.linspace(0, np.pi / 2, overlap_wave_len)) ** 2
            fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap_wave_len)) ** 2
            cur = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            cur[:overlap_wave_len] = cur[:overlap_wave_len] * fade_in + previous_chunk.cpu().numpy() * fade_out
            generated_wave_chunks.append(cur)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave = np.concatenate(generated_wave_chunks).astype(np.float32)

t_vc = time.time() - t_vc_start

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_PATH), vc_wave, sr)

out_duration = len(vc_wave) / sr
rtf = t_vc / source_duration
print(
    f"converted {source_duration:.2f}s source -> {out_duration:.2f}s output in {t_vc:.2f}s  "
    f"(RTF={rtf:.2f} vs source)",
    flush=True,
)
print(f"wrote {OUTPUT_PATH} (sr={sr})", flush=True)
sys.exit(0)
