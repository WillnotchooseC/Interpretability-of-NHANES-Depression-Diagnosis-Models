"""
grpo_train_decompv1.py - GRPO Training for decomp_v1 Stability at High Temperature
====================================================================================
Continues decomp_v1 (SFT-only so far, never RL'd) with Group Relative Policy
Optimization. Rollouts are sampled at TEMPERATURE=1.0 (not the usual 0.7) to
directly target the instability measured in the temp=1.0 statistical eval:
sub-question count drift (Cohen's h=0.88) and especially self-consistency
across independent samples (Cohen's d=-1.64, driven by the model not stopping
cleanly after a complete decomposition and looping into a second attempt).

Reward = mean(count_accuracy, type_accuracy, clean_stop) — deliberately just
these 3 terms, not the larger 6-term reward exp1's GRPO run used: count/type
are the two structural properties that must stay correct, and clean_stop
directly targets the dominant failure mode found above. No GPT calls (cheaper,
faster, and removes the reward function's dependency on the OpenAI key).

Base model: decomp_v1 merged model (nhanes_model/decomp_v1/merged_model —
the fused form of decomp_v1/best_model's LoRA adapter; GRPO applies its own
new LoRA on top, same pattern as the existing exp1-based GRPO run)
Training data: data/train_dataset_split_train.json (958 samples, the same
split decomp_v1's SFT run trained on) / data/train_dataset_split_val.json (50)

Output: nhanes_model/grpo_decompv1/ — deliberately NOT nhanes_model/grpo/,
which already holds the exp1-based GRPO run's final_model/merged_model;
overwriting that would destroy a model that isn't regenerable without
retraining from scratch.

Usage:
    python grpo_train.py
    python grpo_train.py --epochs 2 --lr 5e-7
    python grpo_train.py --resume ./nhanes_model/grpo_decompv1/checkpoint-100

Requirements:
    pip install trl transformers peft accelerate bitsandbytes
"""

import os
import re
import gc
import json
import time
import torch
import numpy as np
import logging
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "nhanes_model", "grpo_decompv1")

# Model
SFT_MODEL_PATH = os.path.join(BASE_DIR, "nhanes_model", "decomp_v1", "merged_model")

# Data
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset_split_train.json")
VAL_FILE = os.path.join(DATA_DIR, "train_dataset_split_val.json")

# Training hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
NUM_GENERATIONS = 4  # 每个prompt生成4个候选
# MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 1.0  # matches the eval temperature this run targets (see module docstring)

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# GRPO specific
KL_COEF = 0.05
# MISSING_EOS_PENALTY = -1.0
SEED = 42

SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)


# ====================================================================
# Data Preparation
# ====================================================================

# Global ground truth map: prompt -> reference output
ground_truth_map = {}


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(data, tokenizer):
    """Prepare GRPO dataset: only needs prompts (not completions)."""
    global ground_truth_map

    formatted = []
    for sample in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ground_truth_map[prompt] = sample["output"]
        formatted.append({"prompt": prompt})

    dataset = Dataset.from_list(formatted)
    logger.info(f"  Dataset size: {len(dataset)}, Ground truth map: {len(ground_truth_map)}")
    return dataset


# ====================================================================
# Parsing Helpers
# ====================================================================

def parse_sub_questions(text):
    """Extract sub-questions and types from output text."""
    subs = []
    lines = text.strip().split("\n")
    for line in lines:
        # Match the literal type keyword, not `\w+`: under sampling the model
        # sometimes glues a stray token directly onto this line with no
        # separator (e.g. "aggregationكُمْ"), and since Python's \w is
        # Unicode-aware, `(\w+)` would swallow that into the captured type.
        m = re.match(r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(searching|aggregation)", line.strip())
        if m:
            subs.append({"question": m.group(1).strip(), "type": m.group(2).strip().lower()})
    return subs


def count_steps(text):
    return len(re.findall(r"Step \d+:", text))


# ====================================================================
# Reward Functions
# ====================================================================

def reward_count(pred_text, ref_text):
    """
    Count accuracy reward.
    Perfect match: 1.0, off by 1: 0.7, off by 2: 0.4, else: 0.0
    """
    pred_count = count_steps(pred_text)
    ref_count = count_steps(ref_text)
    diff = abs(pred_count - ref_count)

    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.7
    elif diff == 2:
        return 0.4
    else:
        return 0.0


def reward_type(pred_text, ref_text):
    """Type accuracy: fraction of matching types at each position."""
    pred_subs = parse_sub_questions(pred_text)
    ref_subs = parse_sub_questions(ref_text)

    if not pred_subs or not ref_subs:
        return 0.0

    min_len = min(len(pred_subs), len(ref_subs))
    max_len = max(len(pred_subs), len(ref_subs))

    matches = sum(1 for i in range(min_len) if pred_subs[i]["type"] == ref_subs[i]["type"])
    return matches / max_len if max_len > 0 else 0.0


def reward_stop(text):
    """
    Clean-stop reward: did generation end right after the last complete
    decomposition step, instead of continuing into garbage or looping back
    into a second (possibly hallucinated) attempt? This targets the failure
    mode found in the temp=1.0 statistical eval (self-consistency Cohen's
    d=-1.64, the largest effect size measured) — under sampling, decomp_v1
    frequently produces the correct decomposition and then keeps generating:
    stray tokens glued onto the last line, or a full restart from "Step 1:".

    1.0 - nothing follows the last valid "# Type:" line (clean stop)
    0.5 - some trailing content follows, but no full loop restart (stray
          tokens after an otherwise-complete, correct decomposition)
    0.0 - no valid decomposition line at all, OR a "Step" number that resets
          /repeats after the last valid line (a genuine loop/re-attempt)
    """
    lines = text.strip().split("\n")

    # Loop/restart detection FIRST, over the whole text: a "Step" number that
    # resets/repeats anywhere means the model looped back into a second
    # attempt, regardless of where the last "# Type:" line happens to fall
    # (a loop's second attempt commonly re-emits its own valid "# Type:"
    # line too, which would otherwise look like a clean stop if checked only
    # relative to the *last* such line).
    last_step_num = 0
    for line in lines:
        m = re.match(r"\s*Step\s+(\d+):", line)
        if m:
            step_num = int(m.group(1))
            if step_num <= last_step_num:
                return 0.0
            last_step_num = step_num

    last_valid_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line.strip()):
            last_valid_idx = i
    if last_valid_idx == -1:
        return 0.0

    # trailing content on the same line as the last valid marker (e.g. a
    # stray token glued directly onto "...# Type: aggregationXYZ")
    same_line_tail = re.sub(
        r"^.*#\s*Type:\s*(searching|aggregation)", "", lines[last_valid_idx].strip()
    ).strip()
    tail = "\n".join(lines[last_valid_idx + 1:]).strip()

    if not same_line_tail and not tail:
        return 1.0
    return 0.5


# ====================================================================
# Combined Reward Function (for GRPOTrainer)
# ====================================================================

def combined_reward_func(completions, **kwargs):
    """
    Combined reward function for GRPO.

    Args:
        completions: list of list of dicts, e.g. [[{"role": "assistant", "content": "..."}]]

    Returns:
        list of float rewards
    """
    prompts = kwargs.get("prompts", [None] * len(completions))
    rewards = []

    for i, completion in enumerate(completions):
        # Extract text
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        # Get ground truth
        prompt = prompts[i] if i < len(prompts) else None
        ref = ground_truth_map.get(prompt, "")

        # Equal-weighted 3-term reward: sub-question count accuracy, type
        # (searching/aggregation) alignment, and clean-stop (see reward_stop's
        # docstring for why this last one is here — it's the dominant failure
        # mode this run is targeting).
        r_count = reward_count(text, ref)
        r_type = reward_type(text, ref)
        r_stop = reward_stop(text)

        total = (r_count + r_type + r_stop) / 3.0

        # Scale to [-1, 1] range for GRPO stability
        total = total * 2 - 1

        rewards.append(float(total))

        if i < 2:  # Log first 2 for debugging
            logger.debug(f"  Sample {i}: cnt={r_count:.2f} typ={r_type:.2f} "
                         f"stop={r_stop:.2f} total={total:.3f}")

    return rewards


# ====================================================================
# Main Training
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--model_path", type=str, default=SFT_MODEL_PATH)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Cap total optimizer steps (overrides --epochs) so trainer.train() "
                             "completes and saves a proper final_model within a fixed time budget, "
                             "rather than being killed mid-run.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Checkpoint (and eval, if enabled) every N steps.")
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "no"],
                        help="Set to 'no' to skip eval entirely (eval takes ~25-30min and isn't "
                             "worth it for a short time-boxed run — just save raw step checkpoints).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- GPU Info ----
    logger.info("=" * 60)
    logger.info("GPU Info")
    logger.info("=" * 60)
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                     f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    # ---- Load Tokenizer ----
    logger.info(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load Data ----
    logger.info(f"\nLoading training data: {TRAIN_FILE}")
    train_data = load_data(TRAIN_FILE)
    train_dataset = prepare_dataset(train_data, tokenizer)

    logger.info(f"Loading validation data: {VAL_FILE}")
    val_data = load_data(VAL_FILE)
    val_dataset = prepare_dataset(val_data, tokenizer)

    # ---- Test Reward Function ----
    logger.info("\nTesting reward function...")
    test_prompt = list(ground_truth_map.keys())[0]
    test_ref = ground_truth_map[test_prompt]
    test_completions = [[{"role": "assistant", "content": test_ref}]]
    test_reward = combined_reward_func(test_completions, prompts=[test_prompt])
    logger.info(f"  Ground truth reward: {test_reward[0]:.4f}")

    # Bad output test
    test_bad = [[{"role": "assistant", "content": "I don't know how to help."}]]
    test_bad_reward = combined_reward_func(test_bad, prompts=[test_prompt])
    logger.info(f"  Bad output reward: {test_bad_reward[0]:.4f}")
    logger.info(f"  Reward gap: {test_reward[0] - test_bad_reward[0]:.4f}")

    # ---- Model ----
    logger.info(f"\nLoading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ---- GRPO Config ----
    logger.info("\nConfiguring GRPO...")
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,

        # GRPO specific
        num_generations=args.num_generations,
        # max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        beta=KL_COEF,
        # missing_eos_penalty=MISSING_EOS_PENALTY,

        # Training
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging & saving
        logging_steps=5,
        eval_strategy=args.eval_strategy,
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        # NOTE: "eval_reward" looks like the right metric_for_best_model (it's
        # what we actually care about, not eval_loss) and IS present in the
        # printed/logged metrics — but GRPOTrainer merges reward metrics into
        # the log via a separate path from the plain dict `evaluate()` returns,
        # which is the dict `_determine_best_metric` actually inspects. Setting
        # metric_for_best_model="eval_reward" raises KeyError there ('eval_loss'
        # is the only metric that dict actually has) and crashes training right
        # after the eval completes, before the checkpoint save — confirmed the
        # hard way when this took down a run at step 100. Stick to eval_loss,
        # or set load_best_model_at_end=False and pick the best checkpoint
        # manually from training_history.json's log_history afterward.
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="eval_loss" if args.eval_strategy != "no" else None,
        greater_is_better=False,
        report_to="none",

        seed=SEED,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    logger.info("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        reward_funcs=combined_reward_func,
        peft_config=lora_config,
    )
    logger.info("GRPOTrainer initialized!")

    # ---- Train ----
    logger.info("\n" + "=" * 60)
    logger.info("Starting GRPO Training")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"  Generations per prompt: {args.num_generations}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Rollout temperature: {TEMPERATURE}")
    logger.info(f"  Reward: mean(count, type, clean_stop), equal-weighted")
    logger.info("=" * 60)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ---- Save ----
    logger.info("\nSaving final model...")
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"  Saved to {final_dir}")

    # ---- Save training history ----
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # ---- Cleanup ----
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("\nGRPO training complete!")


if __name__ == "__main__":
    main()