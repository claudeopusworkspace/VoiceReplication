#!/usr/bin/env bash
# Drive GPT-SoVITS prepare_datasets pipeline (steps 1, 2, 3) from CLI.
# Mirrors the env-var contract that webui.py's open1a/open1b/open1c set up.
#
# Usage: tests/finetune/gpt-sovits/run_prep.sh <version> <exp_name>
#   version: v2 | v2Pro | v4
#   exp_name: e.g. diana_v2
#
# Outputs land under generators/gpt-sovits/logs/<exp_name>/
set -euo pipefail

VERSION="${1:-v2}"
EXP_NAME="${2:-diana_${VERSION}}"

REPO_ROOT="/workspace/VoiceReplication/generators/gpt-sovits"
PY="${REPO_ROOT}/.venv/bin/python"
LIST_FILE="/workspace/VoiceReplication/tests/finetune/gpt-sovits/diana.list"
WAV_DIR="/workspace/VoiceReplication/reference_voice/ch05000_base_dialogue__en"
OPT_DIR="${REPO_ROOT}/logs/${EXP_NAME}"

case "$VERSION" in
  v2)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    S2_CONFIG="GPT_SoVITS/configs/s2.json"
    ;;
  v2Pro)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    S2_CONFIG="GPT_SoVITS/configs/s2v2Pro.json"
    ;;
  v4)
    PRETRAINED_S2G="${REPO_ROOT}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    S2_CONFIG="GPT_SoVITS/configs/s2.json"
    ;;
  *)
    echo "unknown version: $VERSION"; exit 2;;
esac

mkdir -p "$OPT_DIR"
cd "$REPO_ROOT"

# The prepare_datasets scripts import `text.*`, `feature_extractor.*`, `tools.*`
# expecting them to be at top level — the WebUI gets this by importing config.py
# (which sys.path.insert(0, now_dir)) and PYTHONPATH inheritance. Replicate that.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/GPT_SoVITS:${PYTHONPATH:-}"

# Common env
export inp_text="$LIST_FILE"
export inp_wav_dir="$WAV_DIR"
export exp_name="$EXP_NAME"
export opt_dir="$OPT_DIR"
export i_part="0"
export all_parts="1"
export _CUDA_VISIBLE_DEVICES="0"
export is_half="True"
export version="$VERSION"

# --- Step 1: text + BERT features ---
echo "=== step 1: get-text (BERT phonemes) ==="
export bert_pretrained_dir="${REPO_ROOT}/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
"$PY" -s GPT_SoVITS/prepare_datasets/1-get-text.py
# WebUI concatenates per-part outputs into a single 2-name2text.txt
if [ -f "${OPT_DIR}/2-name2text-0.txt" ]; then
  mv "${OPT_DIR}/2-name2text-0.txt" "${OPT_DIR}/2-name2text.txt"
fi

# --- Step 2: hubert features (and SV if Pro) ---
echo "=== step 2: get-hubert-wav32k ==="
export cnhubert_base_dir="${REPO_ROOT}/GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="${REPO_ROOT}/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
"$PY" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

if [[ "$VERSION" == *Pro* ]]; then
  echo "=== step 2b: get-sv (speaker embeddings, torchaudio.load monkey-patched) ==="
  "$PY" /workspace/VoiceReplication/tests/finetune/gpt-sovits/run_sv_wrapped.py
fi

# --- Step 3: semantic tokens ---
echo "=== step 3: get-semantic ==="
export pretrained_s2G="$PRETRAINED_S2G"
export s2config_path="$S2_CONFIG"
"$PY" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
# Concatenate per-part semantic tsv into a single 6-name2semantic.tsv with header
SEM_OUT="${OPT_DIR}/6-name2semantic.tsv"
{
  printf "item_name\tsemantic_audio\n"
  cat "${OPT_DIR}/6-name2semantic-0.tsv"
} > "$SEM_OUT"
rm -f "${OPT_DIR}/6-name2semantic-0.tsv"

echo "=== prep complete for ${EXP_NAME} ==="
echo "Artifacts in ${OPT_DIR}:"
ls -la "$OPT_DIR"
