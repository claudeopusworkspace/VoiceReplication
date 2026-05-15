"""NeuTTS-Air fine-tune driver for the Diana dataset.

Variant of `specialized/neutts-air/examples/finetune.py` that loads our
locally-encoded dataset (from tests/finetune/neutts-air/diana_dataset/) instead
of streaming Emilia-YODAS, and uses scale-appropriate hyperparameters for 250
clips (~17 min) instead of the 10k-step default targeted at 10 hr.

Output: tests/finetune/neutts-air/diana_ckpt/
"""
import os
import re
import sys
import warnings
from functools import partial
from pathlib import Path

import phonemizer
import torch
from datasets import load_from_disk
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

warnings.filterwarnings("ignore")

# Reuse the upstream preprocess function — it's the same chat format we want.
NEUTTS_DIR = Path("/workspace/VoiceReplication/specialized/neutts-air")
sys.path.insert(0, str(NEUTTS_DIR / "examples"))
from finetune import preprocess_sample, data_filter  # noqa: E402

ROOT = Path("/workspace/VoiceReplication")
DATASET_DIR = ROOT / "tests/finetune/neutts-air/diana_dataset"
CKPT_DIR = ROOT / "tests/finetune/neutts-air/diana_ckpt"


def main():
    restore_from = "neuphonic/neutts-air"
    max_seq_len = 2048

    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    g2p = phonemizer.backend.EspeakBackend(
        language="en-us",
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )
    partial_preprocess = partial(preprocess_sample, tokenizer=tokenizer, max_len=max_seq_len, g2p=g2p)

    ds = load_from_disk(str(DATASET_DIR))
    print(f"loaded {len(ds)} rows from {DATASET_DIR}")

    # The upstream filter rejects samples with digits or shouted acronyms — fine
    # for Emilia but might drop more of our 250 than we'd like. Apply it anyway
    # to stay faithful to the recipe.
    ds = ds.filter(data_filter)
    print(f"after data_filter: {len(ds)} rows")

    ds = ds.map(partial_preprocess, remove_columns=["text", "codes", "__key__"])
    # Drop rows where preprocess returned None (empty phonemes).
    ds = ds.filter(lambda r: r["input_ids"] is not None)
    print(f"after preprocess: {len(ds)} rows")

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        do_train=True,
        learning_rate=4e-5,
        max_steps=500,
        bf16=True,
        per_device_train_batch_size=4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_steps=600,            # > max_steps → only save final via save_model
        logging_steps=25,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        torch_compile=False,        # turn off — adds JIT overhead, marginal benefit for 500 steps
        dataloader_num_workers=2,
        report_to="none",
        seed=1337,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(str(CKPT_DIR))
    print(f"saved fine-tuned weights to {CKPT_DIR}")


if __name__ == "__main__":
    main()
