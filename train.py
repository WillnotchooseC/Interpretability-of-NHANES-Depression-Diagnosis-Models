"""
train.py - SFT Instruction Tuning for Qwen3-8B-Base
=====================================================
Pure transformers.Trainer implementation, no trl dependency.

Usage:
    python train.py --exp 1              # single experiment
    python train.py --exp 1 2 3 4 5      # all experiments
    python train.py --exp 1 --epochs 3 --lr 1e-4 --no_wandb

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes wandb
"""

import os
import json
import copy
import argparse
import torch
from datetime import datetime
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "nhanes_model")

MODEL_NAME = "Qwen/Qwen3-8B-Base"

EXP_TRAIN_FILES = {
    1: "exp1_base_train.json",
    2: "exp2_mix_replace_train.json",
    3: "exp3_domain_expand_train.json",
    4: "exp4_sqa_expand_train.json",
    5: "exp5_sqa_large_train.json",
}
VAL_FILE = "shared_val.json"

SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)

# Hyperparameters
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH = 4
PER_DEVICE_EVAL_BATCH = 4
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3
SEED = 42

# Label ignore index for cross-entropy loss
IGNORE_INDEX = -100


# ====================================================================
# Data Processing
# ====================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(sample):
    """Convert sample to chat messages list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]


def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize chat messages and create labels that mask everything
    except the assistant's response (instruction tuning style).

    Only the assistant tokens contribute to the loss.
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    # <|im_end|> token id - must be included in labels for model to learn to stop
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]

        # Tokenize full conversation
        # enable_thinking=False: disable Qwen3's <think> tags
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenizer(
            full_text, truncation=True, max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        # Ensure the sequence ends with <|im_end|>
        if full_ids[-1] != im_end_id:
            full_ids.append(im_end_id)

        # Tokenize prompt only (system + user), to find where assistant starts
        prompt_messages = messages[:-1]  # exclude assistant
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        # Labels: IGNORE_INDEX for prompt tokens, real ids for assistant tokens
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Ensure same length
        assert len(labels) == len(full_ids), (
            f"Length mismatch: labels={len(labels)}, ids={len(full_ids)}"
        )

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_mask_list.append([1] * len(full_ids))

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


def build_dataset(filepath, tokenizer, max_length):
    """Load JSON and tokenize into training dataset."""
    raw = load_json(filepath)
    records = [{"messages": build_messages(s)} for s in raw]
    ds = Dataset.from_list(records)

    ds = ds.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["messages"],
        desc="Tokenizing",
    )
    return ds


# ====================================================================
# Model
# ====================================================================

def load_model_and_tokenizer(model_name):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # or "eager" if sdpa not supported
    )
    model.config.use_cache = False

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ====================================================================
# Training
# ====================================================================

def train_one_experiment(exp_id, args):
    exp_name = f"exp{exp_id}"
    output_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id}: {EXP_TRAIN_FILES[exp_id]}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Data
    train_path = os.path.join(DATA_DIR, EXP_TRAIN_FILES[exp_id])
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    train_dataset = build_dataset(train_path, tokenizer, args.max_seq_len)
    val_dataset = build_dataset(val_path, tokenizer, args.max_seq_len)
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # Data collator (dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_seq_len,
        label_pad_token_id=IGNORE_INDEX,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen3-8b-sft-{exp_name}-{datetime.now().strftime('%m%d_%H%M')}" if args.use_wandb else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nTraining...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save best model
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Cleanup GPU memory
    del trainer, model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

    return best_dir


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3-8B-Base")
    parser.add_argument("--exp", type=int, nargs="+", required=True, help="Experiment IDs (1-5)")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging (disabled by default)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    for exp_id in args.exp:
        assert exp_id in [1, 2, 3, 4, 5]
        train_one_experiment(exp_id, args)

    print(f"\nAll {len(args.exp)} experiment(s) completed!")


if __name__ == "__main__":
    main()