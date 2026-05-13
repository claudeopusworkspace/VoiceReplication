"""Harness adapter for Seed-VC (V1, non-f0 model).

Seed-VC is **voice conversion**, not TTS. For each row we read:
  * `source` — pre-generated wav (chatterbox default voice speaking the sentence)
  * `ref_path` — character reference clip (target voice / timbre)
  * `output`  — where to write the converted wav

Loads the model exactly once, then iterates over the manifest. Mirrors the V1 /
non-f0 inference path from upstream's inference.py (see smoke_seed-vc.py for the
annotated walkthrough). Notable quirks preserved here:
  * chdir into the seed-vc repo so the in-repo config lookups work
  * HF_HUB_CACHE override so weights land inside the repo
  * 30 diffusion steps, length_adjust=1.0, inference_cfg_rate=0.7, fp16 autocast

Manifest rows have a `source` field that the shared `_common.Row` doesn't model,
so we parse the JSON manifest directly here.
"""
import argparse
import json
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


# Inference hyperparameters — match upstream inference.py defaults.
DIFFUSION_STEPS = 30
LENGTH_ADJUST = 1.0
INFERENCE_CFG_RATE = 0.7
FP16 = True


def load_seed_vc_v1(device: torch.device):
    """Load the V1 (non-f0) Seed-VC stack. Returns a dict of components."""
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
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)

    # Vocoder (V1 default = bigvgan)
    vocoder_type = model_params.vocoder.type
    assert vocoder_type == "bigvgan", f"Expected bigvgan vocoder for V1; got {vocoder_type}"
    from modules.bigvgan import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(model_params.vocoder.name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    # Semantic tokenizer (V1 default = whisper)
    speech_tokenizer_type = model_params.speech_tokenizer.type
    assert speech_tokenizer_type == "whisper", f"Expected whisper tokenizer for V1; got {speech_tokenizer_type}"
    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    # Mel spectrogram fn
    from modules.audio import mel_spectrogram

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

    return {
        "model": model,
        "campplus": campplus_model,
        "vocoder": bigvgan_model,
        "whisper": whisper_model,
        "whisper_fe": whisper_feature_extractor,
        "mel_fn": mel_fn,
        "sr": sr,
        "device": device,
    }


def make_semantic_fn(whisper_model, whisper_fe, device):
    def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
        ori_inputs = whisper_fe(
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
    return semantic_fn


def convert(components, semantic_fn, source_path: Path, ref_path: Path) -> np.ndarray:
    """Run V1 / non-f0 voice conversion. Returns float32 wav as numpy array."""
    model = components["model"]
    campplus_model = components["campplus"]
    vocoder_fn = components["vocoder"]
    mel_fn = components["mel_fn"]
    sr = components["sr"]
    device = components["device"]

    hop_length = 256  # V1 / non-f0
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = librosa.load(str(source_path), sr=sr)[0]
    ref_audio = librosa.load(str(ref_path), sr=sr)[0]

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    with torch.no_grad():
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        # Chunk the source through whisper if it's longer than 30s.
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5  # seconds
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                S_alt_chunk = semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt_chunk)
                else:
                    S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

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

        return np.concatenate(generated_wave_chunks).astype(np.float32)


def load_vc_manifest(path: Path) -> list[dict]:
    """Read the manifest as raw dicts (VC rows have an extra `source` field)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def emit_row(row: dict, *, load_s=None, gen_s=None, duration_s=None, sr=None,
             status="ok", error=None) -> None:
    rec = {
        "model": "seed-vc",
        "ref_id": row.get("ref_id"),
        "sentence_id": row.get("sentence_id"),
        "output": row.get("output"),
        "status": status,
    }
    if load_s is not None: rec["load_s"] = round(load_s, 3)
    if gen_s is not None: rec["gen_s"] = round(gen_s, 3)
    if duration_s is not None: rec["duration_s"] = round(duration_s, 3)
    if sr is not None: rec["sr"] = sr
    if error is not None: rec["error"] = error
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to JSON-lines manifest from the orchestrator")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('{"status": "fail", "error": "CUDA not available"}', flush=True)
        sys.exit(1)

    rows = load_vc_manifest(args.manifest)
    if not rows:
        print('{"status": "fail", "error": "empty manifest"}', flush=True)
        sys.exit(1)

    device = torch.device("cuda")

    t0 = time.time()
    components = load_seed_vc_v1(device)
    semantic_fn = make_semantic_fn(components["whisper"], components["whisper_fe"], device)
    load_s = time.time() - t0
    sr = components["sr"]

    for i, row in enumerate(rows):
        try:
            source_path = Path(row["source"])
            ref_path = Path(row["ref_path"])
            output_path = Path(row["output"])

            if not source_path.exists():
                emit_row(row, status="fail", error=f"source missing: {source_path}")
                continue
            if not ref_path.exists():
                emit_row(row, status="fail", error=f"ref missing: {ref_path}")
                continue

            t1 = time.time()
            wav = convert(components, semantic_fn, source_path, ref_path)
            gen_s = time.time() - t1

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), wav, sr)
            duration_s = len(wav) / sr

            emit_row(
                row,
                load_s=load_s if i == 0 else None,
                gen_s=gen_s,
                duration_s=duration_s,
                sr=sr,
                status="ok",
            )
        except Exception as e:
            emit_row(row, status="fail", error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
