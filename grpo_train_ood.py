"""
grpo_train_ood.py - GRPO Training for decomp_v1_ood Coverage on OOD Templates
================================================================================
Continues decomp_v1_ood (SFT-only so far, never RL'd) with Group Relative
Policy Optimization, targeting the failure mode found by eval_decomp_v1_ood.py:
on ood_data's held-out test set (question templates never seen in ood_data's
train split), the model over-decomposes (86/120 raw samples predict MORE
sub-questions than the reference) and semantic similarity per sub-question
drops to 0.74 (vs 0.98 in-distribution for decomp_v1). Per-user decision, this
is fine as long as every reference sub-question is *covered* by some predicted
sub-question — an incomplete decomposition (missing a needed sub-question) is
the actual failure mode to fix, not an over-long one.

Reward = mean(coverage, type_accuracy, clean_stop):
  - coverage: entity-pair recall (patient/biomarker/algorithm/ratio), NOT the
    symmetric/penalized entity match from the original exp1 GRPO run — greedy-
    matches each REFERENCE sub-question to its best predicted sub-question and
    averages over the reference count, with NO penalty for extra predicted
    sub-questions (see reward_coverage). This directly targets "all needed
    sub-questions present, extras OK" rather than exact-count matching.
  - type_accuracy / clean_stop: unchanged from grpo_train.py (decomp_v1's
    stability run) — positional searching/aggregation match, and whether
    generation stops cleanly after the last complete step instead of looping.
Deliberately rule-based / no GPT calls, matching grpo_train.py's tradeoff:
cheaper, faster, no OpenAI dependency in the reward loop. (Entity vocabulary
below was checked against all of ood_data's biomarker mentions and matches
exactly — no new biomarker names appear in the OOD templates.)

Base model: decomp_v1_ood merged model (nhanes_model/decomp_v1_ood/merged_model)
Training data: ood_data/train_dataset_ood_rebalanced.json (1008 samples) /
ood_data/valid_dataset_ood_rebalanced.json (120)

Output: nhanes_model/grpo_decompv1_ood/ — kept separate from grpo/ (exp1-based)
and grpo_decompv1/ (decomp_v1 stability run) so none of the three overwrite
each other.

Usage:
    python grpo_train_ood.py
    python grpo_train_ood.py --epochs 2 --lr 5e-6
    python grpo_train_ood.py --resume ./nhanes_model/grpo_decompv1_ood/checkpoint-100

Requirements:
    pip install trl transformers peft accelerate bitsandbytes
"""

import os
import re
import gc
import json
import torch
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OOD_DATA_DIR = os.path.join(BASE_DIR, "ood_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "nhanes_model", "grpo_decompv1_ood")

# Model
SFT_MODEL_PATH = os.path.join(BASE_DIR, "nhanes_model", "decomp_v1_ood", "merged_model")

# Data
TRAIN_FILE = os.path.join(OOD_DATA_DIR, "train_dataset_ood_rebalanced.json")
VAL_FILE = os.path.join(OOD_DATA_DIR, "valid_dataset_ood_rebalanced.json")

# Training hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 0.7

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# GRPO specific
KL_COEF = 0.05
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
        m = re.match(r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(searching|aggregation)", line.strip())
        if m:
            subs.append({"question": m.group(1).strip(), "type": m.group(2).strip().lower()})
    return subs


def count_steps(text):
    return len(re.findall(r"Step \d+:", text))


# ====================================================================
# Entity Extraction & Coverage Reward
# (ported from the original exp1 GRPO run's extract_paired_entities /
#  match_sub_entities, with the extra-sub-question penalty removed — see
#  module docstring for why)
# ====================================================================

ALGORITHMS = ["random forest", "naive bayes", "decision tree", "logistic regression"]
BIOMARKERS = [
    "creatinine", "albumin", "blood manganese", "blood lead",
    "blood selenium", "blood mercury", "total cholesterol",
    "rbc folate", "glycohemoglobin", "direct hdl-cholesterol",
    "direct hdl cholesterol", "hdl-cholesterol", "hdl cholesterol",
    "vitamin d2", "vitamin d3",
]
RATIO_PATTERN = re.compile(r"\b1:\d+\b")
PATIENT_PATTERN = re.compile(r"patient\s+(\d+)", re.IGNORECASE)


def extract_paired_entities(text):
    """Extract entity-sets per sub-question, preserving pairings (which
    algorithm goes with which ratio, which patient with which biomarker)."""
    subs = []
    lines = text.strip().split("\n")

    for line in lines:
        m = re.match(r"Sub-question:\s*(.+?)(?:\s*#\s*Type:|\s*$)", line.strip())
        if not m:
            continue
        sq = m.group(1).lower()

        algos = {algo for algo in ALGORITHMS if algo in sq}
        patients = set(PATIENT_PATTERN.findall(sq))
        ratios = set(RATIO_PATTERN.findall(sq))
        bios = {bio for bio in BIOMARKERS if bio in sq}

        subs.append({
            "algos": algos, "ratios": ratios, "patients": patients, "bios": bios,
            "algo_ratio_pairs": frozenset((a, r) for a in algos for r in ratios) if algos and ratios else frozenset(),
            "patient_bio_pairs": frozenset((p, b) for p in patients for b in bios) if patients and bios else frozenset(),
        })

    return subs


def match_sub_entities(pred_sub, ref_sub):
    """Score how well a predicted sub-question's entities match a reference
    sub-question's, 0-1 (F1 over each entity-pair/singleton group)."""
    scores = []

    if pred_sub["algo_ratio_pairs"] or ref_sub["algo_ratio_pairs"]:
        pred_pairs, ref_pairs = pred_sub["algo_ratio_pairs"], ref_sub["algo_ratio_pairs"]
        if not pred_pairs and not ref_pairs:
            scores.append(1.0)
        elif not pred_pairs or not ref_pairs:
            scores.append(0.0)
        else:
            tp = len(pred_pairs & ref_pairs)
            p = tp / len(pred_pairs) if pred_pairs else 0
            r = tp / len(ref_pairs) if ref_pairs else 0
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    else:
        for key in ["algos", "ratios"]:
            ps, rs = pred_sub[key], ref_sub[key]
            if not ps and not rs:
                continue
            if not ps or not rs:
                scores.append(0.0)
                continue
            tp = len(ps & rs)
            p = tp / len(ps) if ps else 0
            r = tp / len(rs) if rs else 0
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

    if pred_sub["patient_bio_pairs"] or ref_sub["patient_bio_pairs"]:
        pred_pairs, ref_pairs = pred_sub["patient_bio_pairs"], ref_sub["patient_bio_pairs"]
        if not pred_pairs and not ref_pairs:
            scores.append(1.0)
        elif not pred_pairs or not ref_pairs:
            scores.append(0.0)
        else:
            tp = len(pred_pairs & ref_pairs)
            p = tp / len(pred_pairs) if pred_pairs else 0
            r = tp / len(ref_pairs) if ref_pairs else 0
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    else:
        for key in ["patients", "bios"]:
            ps, rs = pred_sub[key], ref_sub[key]
            if not ps and not rs:
                continue
            if not ps or not rs:
                scores.append(0.0)
                continue
            tp = len(ps & rs)
            p = tp / len(ps) if ps else 0
            r = tp / len(rs) if rs else 0
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

    return sum(scores) / len(scores) if scores else 1.0


def reward_coverage(pred_text, ref_text):
    """Coverage/recall reward: for each REFERENCE sub-question, is there some
    predicted sub-question that matches it well? Averaged over the reference
    count. Unlike the original exp1 reward_entity, extra predicted
    sub-questions are NOT penalized — only missing reference coverage is."""
    pred_subs = extract_paired_entities(pred_text)
    ref_subs = extract_paired_entities(ref_text)

    if not ref_subs:
        return 1.0
    if not pred_subs:
        return 0.0

    used_pred = set()
    total_score = 0.0
    for ref_sub in ref_subs:
        best_score, best_idx = 0.0, -1
        for j, pred_sub in enumerate(pred_subs):
            if j in used_pred:
                continue
            score = match_sub_entities(pred_sub, ref_sub)
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx >= 0:
            used_pred.add(best_idx)
        total_score += best_score

    return total_score / len(ref_subs)


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
    """Clean-stop reward (see grpo_train.py's reward_stop for full rationale):
    1.0 clean stop, 0.5 trailing garbage after a complete decomposition,
    0.0 no valid decomposition or a genuine loop/re-attempt."""
    lines = text.strip().split("\n")

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
    prompts = kwargs.get("prompts", [None] * len(completions))
    rewards = []

    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        prompt = prompts[i] if i < len(prompts) else None
        ref = ground_truth_map.get(prompt, "")

        r_coverage = reward_coverage(text, ref)
        r_type = reward_type(text, ref)
        r_stop = reward_stop(text)

        total = (r_coverage + r_type + r_stop) / 3.0
        total = total * 2 - 1  # scale to [-1, 1] for GRPO stability

        rewards.append(float(total))

        if i < 2:
            logger.debug(f"  Sample {i}: cov={r_coverage:.2f} typ={r_type:.2f} "
                         f"stop={r_stop:.2f} total={total:.3f}")

    return rewards


# ====================================================================
# Main Training
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Training (decomp_v1_ood coverage)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--model_path", type=str, default=SFT_MODEL_PATH)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Cap total optimizer steps (overrides --epochs).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GPU Info")
    logger.info("=" * 60)
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                     f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    logger.info(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"\nLoading training data: {TRAIN_FILE}")
    train_data = load_data(TRAIN_FILE)
    train_dataset = prepare_dataset(train_data, tokenizer)

    logger.info(f"Loading validation data: {VAL_FILE}")
    val_data = load_data(VAL_FILE)
    val_dataset = prepare_dataset(val_data, tokenizer)

    logger.info("\nTesting reward function...")
    test_prompt = list(ground_truth_map.keys())[0]
    test_ref = ground_truth_map[test_prompt]
    test_completions = [[{"role": "assistant", "content": test_ref}]]
    test_reward = combined_reward_func(test_completions, prompts=[test_prompt])
    logger.info(f"  Ground truth reward: {test_reward[0]:.4f}")

    test_over = [[{"role": "assistant", "content": test_ref + "\nStep 99: extra\nSub-question: irrelevant filler question # Type: searching"}]]
    test_over_reward = combined_reward_func(test_over, prompts=[test_prompt])
    logger.info(f"  Ground truth + 1 extra sub-Q reward: {test_over_reward[0]:.4f} (should stay close to ground-truth reward)")

    test_bad = [[{"role": "assistant", "content": "I don't know how to help."}]]
    test_bad_reward = combined_reward_func(test_bad, prompts=[test_prompt])
    logger.info(f"  Bad output reward: {test_bad_reward[0]:.4f}")

    logger.info(f"\nLoading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

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

        num_generations=args.num_generations,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        beta=KL_COEF,

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",

        seed=SEED,
        remove_unused_columns=False,
    )

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

    logger.info("\n" + "=" * 60)
    logger.info("Starting GRPO Training")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"  Generations per prompt: {args.num_generations}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Rollout temperature: {TEMPERATURE}")
    logger.info(f"  Reward: mean(coverage, type, clean_stop), equal-weighted, no extra-subQ penalty")
    logger.info("=" * 60)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    logger.info("\nSaving final model...")
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"  Saved to {final_dir}")

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("\nGRPO training complete!")


if __name__ == "__main__":
    main()
