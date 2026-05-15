#!/usr/bin/env bash
# Drive the RVC training pipeline from CLI (reproduces what infer-web.py's
# preprocess_dataset / extract_f0_feature / extract_feature_print / click_train
# do, minus the gradio plumbing).
#
# Usage: tests/finetune/rvc/run_train.sh [exp_name]
#
# Outputs land at specialized/rvc/logs/<exp_name>/ (weights also copied to
# assets/weights/ by train.py via the -sw flag).
set -euo pipefail

EXP_NAME="${1:-diana_rvc}"
SR="48k"        # match our source rate (48kHz mono)
SR_INT=48000    # preprocess.py wants the integer
VERSION="v2"
BATCH_SIZE=8
TOTAL_EPOCH=100
SAVE_EVERY=10
N_PROCESS=8

REPO_ROOT="/workspace/VoiceReplication/specialized/rvc"
PY="${REPO_ROOT}/.venv/bin/python"
TRAINSET="/workspace/VoiceReplication/reference_voice/ch05000_base_dialogue__en"
EXP_DIR="${REPO_ROOT}/logs/${EXP_NAME}"

mkdir -p "$EXP_DIR"
cd "$REPO_ROOT"

# Many RVC modules expect cwd == repo root and read .env. .env has paths for
# pretrained_v2 etc.; make sure it exists or fall back to defaults.
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || true
fi

# --- Step 1: preprocess (slice + normalize) ---
echo "=== step 1: preprocess (sr=${SR_INT}) ==="
"$PY" infer/modules/train/preprocess.py "$TRAINSET" "$SR_INT" "$N_PROCESS" "$EXP_DIR" False 3.7

# --- Step 2a: extract f0 (pitch) with rmvpe ---
echo "=== step 2a: extract f0 (rmvpe) ==="
"$PY" infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 "$EXP_DIR" True

# --- Step 2b: extract HuBERT v2 features ---
# The stock script calls fairseq.checkpoint_utils.load_model_ensemble_and_task,
# which internally does torch.load — and torch 2.6+ defaults to weights_only=True,
# rejecting the fairseq HuBERT checkpoint. Use a wrapper that monkey-patches
# torch.load first.
echo "=== step 2b: extract HuBERT v2 features (torch.load monkey-patched) ==="
"$PY" /workspace/VoiceReplication/tests/finetune/rvc/run_feature_extract_wrapped.py \
  cuda:0 1 0 0 "$EXP_DIR" "$VERSION" True

# --- Step 3: generate filelist.txt ---
echo "=== step 3: generate filelist ==="
"$PY" - <<EOF
import os, random
from pathlib import Path

exp_dir = Path("${EXP_DIR}")
sr = "${SR}"
spk_id = 0
fea_dim = 768  # v2

gt_wavs = exp_dir / "0_gt_wavs"
feature = exp_dir / "3_feature768"
f0 = exp_dir / "2a_f0"
f0nsf = exp_dir / "2b-f0nsf"

names = (set(p.stem for p in gt_wavs.iterdir())
         & set(p.stem for p in feature.iterdir())
         & set(p.stem for p in f0.iterdir())
         & set(p.stem for p in f0nsf.iterdir()))

opt = []
for name in names:
    opt.append(f"{gt_wavs}/{name}.wav|{feature}/{name}.npy|{f0}/{name}.wav.npy|{f0nsf}/{name}.wav.npy|{spk_id}")

now_dir = "${REPO_ROOT}"
for _ in range(2):
    opt.append(f"{now_dir}/logs/mute/0_gt_wavs/mute{sr}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id}")

random.shuffle(opt)
(exp_dir / "filelist.txt").write_text("\n".join(opt))
print(f"filelist.txt: {len(opt)} entries (incl. 2 mute)")

# Write training config from configs/v2/<sr>.json (copy into exp_dir as config.json)
import json
cfg_src = Path("${REPO_ROOT}/configs/${VERSION}/${SR}.json")
if cfg_src.exists():
    cfg = json.loads(cfg_src.read_text())
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=4))
    print(f"config.json copied from {cfg_src}")
else:
    print(f"WARNING: {cfg_src} missing — training will fail")
EOF

# --- Step 4: train ---
echo "=== step 4: train (epochs=${TOTAL_EPOCH}, batch=${BATCH_SIZE}, sr=${SR}) ==="
"$PY" infer/modules/train/train.py \
  -e "$EXP_NAME" \
  -sr "$SR" \
  -f0 1 \
  -bs "$BATCH_SIZE" \
  -g 0 \
  -te "$TOTAL_EPOCH" \
  -se "$SAVE_EVERY" \
  -pg "assets/pretrained_v2/f0G48k.pth" \
  -pd "assets/pretrained_v2/f0D48k.pth" \
  -l 1 -c 0 -sw 1 -v "$VERSION"

echo "=== train done ==="
ls -la "${REPO_ROOT}/assets/weights/" | tail -10

# --- Step 5: build the retrieval index (optional but improves quality) ---
# The upstream tools/infer/train-index-v2.py is template-style with a hardcoded
# './logs/anz/...' path. Use our parameterized replacement.
echo "=== step 5: build retrieval index ==="
"$PY" /workspace/VoiceReplication/tests/finetune/rvc/build_index.py "$EXP_NAME"

echo "=== ALL DONE for ${EXP_NAME} ==="
