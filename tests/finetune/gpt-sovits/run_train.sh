#!/usr/bin/env bash
# Drive GPT-SoVITS s2 (SoVITS) and s1 (GPT) training from CLI.
# Mirrors the env+config setup that webui.py's open1Ba/open1Bb do.
#
# Usage: tests/finetune/gpt-sovits/run_train.sh <version> <exp_name>
#   version: v2 | v2Pro | v4
#   exp_name: must match what was passed to run_prep.sh (e.g. diana_v2)
#
# Prereq: prep artifacts present at REPO_ROOT/logs/<exp_name>/{2-name2text.txt,
#         4-cnhubert/, 5-wav32k/, 6-name2semantic.tsv}.
set -euo pipefail

VERSION="${1:-v2}"
EXP_NAME="${2:-diana_${VERSION}}"

REPO_ROOT="/workspace/VoiceReplication/generators/gpt-sovits"
PY="${REPO_ROOT}/.venv/bin/python"
S2_DIR="${REPO_ROOT}/logs/${EXP_NAME}"

# SoVITS pretrained paths per version (s2D is `s2G` filename with G→D)
case "$VERSION" in
  v2)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    PRETRAINED_S2D="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    PRETRAINED_S1="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    S2_CONFIG_TEMPLATE="${REPO_ROOT}/GPT_SoVITS/configs/s2.json"
    S2_SCRIPT="GPT_SoVITS/s2_train.py"
    SOVITS_WEIGHTS_DIR="SoVITS_weights_v2"
    GPT_WEIGHTS_DIR="GPT_weights_v2"
    ;;
  v2Pro)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    PRETRAINED_S2D="${REPO_ROOT}/GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"
    PRETRAINED_S1="${REPO_ROOT}/GPT_SoVITS/pretrained_models/s1v3.ckpt"
    S2_CONFIG_TEMPLATE="${REPO_ROOT}/GPT_SoVITS/configs/s2v2Pro.json"
    S2_SCRIPT="GPT_SoVITS/s2_train.py"
    SOVITS_WEIGHTS_DIR="SoVITS_weights_v2Pro"
    GPT_WEIGHTS_DIR="GPT_weights_v2Pro"
    ;;
  v4)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    PRETRAINED_S2D=""  # v4 uses LoRA, doesn't take a discriminator pretrained
    PRETRAINED_S1="${REPO_ROOT}/GPT_SoVITS/pretrained_models/s1v3.ckpt"
    S2_CONFIG_TEMPLATE="${REPO_ROOT}/GPT_SoVITS/configs/s2.json"
    S2_SCRIPT="GPT_SoVITS/s2_train_v3_lora.py"
    SOVITS_WEIGHTS_DIR="SoVITS_weights_v4"
    GPT_WEIGHTS_DIR="GPT_weights_v4"
    ;;
  *) echo "unknown version: $VERSION"; exit 2;;
esac

cd "$REPO_ROOT"
mkdir -p "${S2_DIR}/logs_s2_${VERSION}" "${S2_DIR}/logs_s1_${VERSION}" "$SOVITS_WEIGHTS_DIR" "$GPT_WEIGHTS_DIR" TEMP

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/GPT_SoVITS:${PYTHONPATH:-}"
export _CUDA_VISIBLE_DEVICES="0"
export hz="25hz"

# ---------- SoVITS (s2) training ----------
SOVITS_BATCH=12        # 5090 has 32GB; 16 default is fine but 12 leaves headroom for GPT-train cache
SOVITS_EPOCHS=8        # community guidance for few-shot fine-tune
SOVITS_SAVE_EVERY=4    # save at epoch 4 and 8
SOVITS_TMP_CONFIG="${REPO_ROOT}/TEMP/tmp_s2_${VERSION}.json"

"$PY" - <<EOF
import json
with open("${S2_CONFIG_TEMPLATE}") as f:
    d = json.load(f)
d["train"]["batch_size"] = ${SOVITS_BATCH}
d["train"]["epochs"] = ${SOVITS_EPOCHS}
d["train"]["text_low_lr_rate"] = 0.4
d["train"]["pretrained_s2G"] = "${PRETRAINED_S2G}"
d["train"]["pretrained_s2D"] = "${PRETRAINED_S2D}"
d["train"]["if_save_latest"] = True
d["train"]["if_save_every_weights"] = True
d["train"]["save_every_epoch"] = ${SOVITS_SAVE_EVERY}
d["train"]["gpu_numbers"] = "0"
d["train"]["grad_ckpt"] = False
d["train"]["lora_rank"] = 32  # ignored by s2_train.py (v1/v2/v2Pro), used by v3/v4 LoRA
d["model"]["version"] = "${VERSION}"
d["data"]["exp_dir"] = "${S2_DIR}"
d["s2_ckpt_dir"] = "${S2_DIR}"
d["save_weight_dir"] = "${SOVITS_WEIGHTS_DIR}"
d["name"] = "${EXP_NAME}"
d["version"] = "${VERSION}"
with open("${SOVITS_TMP_CONFIG}", "w") as f:
    json.dump(d, f, indent=2)
print("wrote ${SOVITS_TMP_CONFIG}")
EOF

echo "=== SoVITS (s2) training: ${VERSION}, ${SOVITS_EPOCHS} epochs, batch ${SOVITS_BATCH} ==="
"$PY" -s "$S2_SCRIPT" --config "$SOVITS_TMP_CONFIG"

# ---------- GPT (s1) training ----------
GPT_BATCH=8
GPT_EPOCHS=15
GPT_SAVE_EVERY=5
GPT_TMP_CONFIG="${REPO_ROOT}/TEMP/tmp_s1_${VERSION}.yaml"
S1_CONFIG_TEMPLATE="${REPO_ROOT}/GPT_SoVITS/configs/s1longer-v2.yaml"

"$PY" - <<EOF
import yaml
with open("${S1_CONFIG_TEMPLATE}") as f:
    d = yaml.safe_load(f)
d["train"]["batch_size"] = ${GPT_BATCH}
d["train"]["epochs"] = ${GPT_EPOCHS}
d["train"]["save_every_n_epoch"] = ${GPT_SAVE_EVERY}
d["train"]["if_save_every_weights"] = True
d["train"]["if_save_latest"] = True
d["train"]["if_dpo"] = False
d["train"]["half_weights_save_dir"] = "${GPT_WEIGHTS_DIR}"
d["train"]["exp_name"] = "${EXP_NAME}"
d["pretrained_s1"] = "${PRETRAINED_S1}"
d["train_semantic_path"] = "${S2_DIR}/6-name2semantic.tsv"
d["train_phoneme_path"] = "${S2_DIR}/2-name2text.txt"
d["output_dir"] = "${S2_DIR}/logs_s1_${VERSION}"
with open("${GPT_TMP_CONFIG}", "w") as f:
    yaml.safe_dump(d, f)
print("wrote ${GPT_TMP_CONFIG}")
EOF

echo "=== GPT (s1) training: ${GPT_EPOCHS} epochs, batch ${GPT_BATCH} ==="
"$PY" -s GPT_SoVITS/s1_train.py --config_file "$GPT_TMP_CONFIG"

echo "=== TRAINING COMPLETE for ${EXP_NAME} ==="
echo "SoVITS weights: ${REPO_ROOT}/${SOVITS_WEIGHTS_DIR}/"
ls -la "${REPO_ROOT}/${SOVITS_WEIGHTS_DIR}/" || true
echo "GPT weights: ${REPO_ROOT}/${GPT_WEIGHTS_DIR}/"
ls -la "${REPO_ROOT}/${GPT_WEIGHTS_DIR}/" || true
