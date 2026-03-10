"""
train_grpo.py - GRPO Training for Question Decomposition
==========================================================
Optimizes Exp1's SFT model using Group Relative Policy Optimization.
Rewards focus on: (1) Count Accuracy, (2) Semantic Similarity via GPT.

Base model: Exp1 merged model (nhanes_model/exp1/merged_model)
Training data: exp1_base_train.json (996 domain samples)

Usage:
    python train_grpo.py
    python train_grpo.py --epochs 2 --lr 5e-7
    python train_grpo.py --resume ./nhanes_model/grpo/checkpoint-100

Requirements:
    pip install trl transformers peft accelerate bitsandbytes openai
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
from openai import OpenAI
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "nhanes_model", "grpo")

# Model
SFT_MODEL_PATH = os.path.join(BASE_DIR, "nhanes_model", "exp1", "merged_model")

# Data
TRAIN_FILE = os.path.join(DATA_DIR, "exp1_base_train.json")
VAL_FILE = os.path.join(DATA_DIR, "shared_val.json")

# OpenAI (for GPT-based semantic reward)
OPENAI_API_KEY = "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"  # ← 换成你的key
OPENAI_API_BASE = None  # 代理地址，不用代理就留None
GPT_MODEL = "gpt-4o-mini"  # 用mini省钱，训练中会调很多次

# Training hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
NUM_GENERATIONS = 4  # 每个prompt生成4个候选
# MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 0.7

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Reward weights
REWARD_WEIGHT_COUNT = 0.25       # 步数精确匹配
REWARD_WEIGHT_ENTITY = 0.30     # 关键实体精确匹配（算法/病人/比例/特征）
REWARD_WEIGHT_REPETITION = 0.15 # 重复惩罚（子问题去重后的unique率）
REWARD_WEIGHT_SEMANTIC = 0.10   # GPT整体语义覆盖
REWARD_WEIGHT_FORMAT = 0.10     # 格式正确性
REWARD_WEIGHT_TYPE = 0.10       # searching/aggregation类型匹配

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
        m = re.match(r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(\w+)", line.strip())
        if m:
            subs.append({"question": m.group(1).strip(), "type": m.group(2).strip().lower()})
    return subs


def count_steps(text):
    return len(re.findall(r"Step \d+:", text))


# ====================================================================
# OpenAI Client
# ====================================================================

_openai_client = None

def get_client():
    global _openai_client
    if _openai_client is None:
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        _openai_client = OpenAI(**kwargs)
    return _openai_client


# ====================================================================
# Reward Functions
# ====================================================================

# Key entities (used by both reward_repetition and entity matching)
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


def reward_format(text):
    """Format correctness: 0 or 1."""
    score = 0.0
    if re.search(r"Step \d+:", text):
        score += 0.4
    if re.search(r"Sub-question:", text):
        score += 0.3
    if re.search(r"#\s*Type:\s*(searching|aggregation)", text):
        score += 0.3
    return score


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


def reward_repetition(text):
    """
    Repetition penalty: checks if sub-questions are unique.
    
    If model outputs 3 identical sub-questions, score = 1/3 = 0.33
    If all sub-questions are unique, score = 1.0
    
    Uses entity pairs for comparison (not exact string match),
    so "What is patient 12's vitamin d3?" appearing twice counts as duplicate
    even if worded slightly differently.
    """
    subs = parse_sub_questions(text)
    if len(subs) <= 1:
        return 1.0  # no repetition possible
    
    # Method 1: Entity-based dedup (catches semantic duplicates)
    seen_signatures = set()
    unique_count = 0
    
    for sub in subs:
        q = sub["question"].lower().strip()
        # Build a signature from key entities in this sub-question
        algos = frozenset(a for a in ALGORITHMS if a in q)
        patients = frozenset(PATIENT_PATTERN.findall(q))
        ratios = frozenset(RATIO_PATTERN.findall(q))
        bios = frozenset(b for b in BIOMARKERS if b in q)
        
        sig = (algos, patients, ratios, bios)
        
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_count += 1
    
    # unique ratio: 3 unique out of 5 = 0.6
    return unique_count / len(subs)

# ====================================================================
# Entity Extraction & Matching Reward (Paired)
# ====================================================================


def extract_paired_entities(text):
    """
    Extract paired entities from each sub-question separately.
    Returns a list of entity-sets, one per sub-question.
    This preserves which algorithm goes with which ratio, etc.

    Example:
      "How does random forest at 1:4 predict patient 12?"
      → {"algos": {"random forest"}, "ratios": {"1:4"}, "patients": {"12"}, "bios": set()}

      The PAIR "random forest + 1:4" is preserved because they're in the same sub-question.
    """
    subs = []
    lines = text.strip().split("\n")

    for line in lines:
        m = re.match(r"Sub-question:\s*(.+?)(?:\s*#\s*Type:|\s*$)", line.strip())
        if not m:
            continue
        sq = m.group(1).lower()

        algos = set()
        for algo in ALGORITHMS:
            if algo in sq:
                algos.add(algo)

        patients = set(PATIENT_PATTERN.findall(sq))
        ratios = set(RATIO_PATTERN.findall(sq))

        bios = set()
        for bio in BIOMARKERS:
            if bio in sq:
                bios.add(bio)

        # Build a frozen signature for this sub-question
        subs.append({
            "algos": algos,
            "ratios": ratios,
            "patients": patients,
            "bios": bios,
            # Combined pairs for strict matching
            "algo_ratio_pairs": frozenset(
                (a, r) for a in algos for r in ratios
            ) if algos and ratios else frozenset(),
            "patient_bio_pairs": frozenset(
                (p, b) for p in patients for b in bios
            ) if patients and bios else frozenset(),
        })

    return subs


def match_sub_entities(pred_sub, ref_sub):
    """
    Score how well a predicted sub-question's entities match a reference sub-question.
    Returns 0-1.
    """
    scores = []

    # 1. Algorithm+Ratio pairs (strict: "random forest 1:4" must match as a pair)
    if pred_sub["algo_ratio_pairs"] or ref_sub["algo_ratio_pairs"]:
        pred_pairs = pred_sub["algo_ratio_pairs"]
        ref_pairs = ref_sub["algo_ratio_pairs"]
        if not pred_pairs and not ref_pairs:
            scores.append(1.0)
        elif not pred_pairs or not ref_pairs:
            scores.append(0.0)
        else:
            tp = len(pred_pairs & ref_pairs)
            p = tp / len(pred_pairs) if pred_pairs else 0
            r = tp / len(ref_pairs) if ref_pairs else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            scores.append(f1)
    else:
        # No pairs, match algorithms and ratios separately
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

    # 2. Patient+Biomarker pairs
    if pred_sub["patient_bio_pairs"] or ref_sub["patient_bio_pairs"]:
        pred_pairs = pred_sub["patient_bio_pairs"]
        ref_pairs = ref_sub["patient_bio_pairs"]
        if not pred_pairs and not ref_pairs:
            scores.append(1.0)
        elif not pred_pairs or not ref_pairs:
            scores.append(0.0)
        else:
            tp = len(pred_pairs & ref_pairs)
            p = tp / len(pred_pairs) if pred_pairs else 0
            r = tp / len(ref_pairs) if ref_pairs else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            scores.append(f1)
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


def reward_entity(pred_text, ref_text):
    """
    Paired entity matching reward.

    Extracts entities PER sub-question, then uses greedy best-match
    to align pred sub-questions to ref sub-questions.

    Catches errors like:
    - "random forest 1:4" predicted as "random forest 1:1" → pair mismatch
    - "patient 12 vitamin d3" predicted as "patient 12 vitamin d2" → pair mismatch
    """
    pred_subs = extract_paired_entities(pred_text)
    ref_subs = extract_paired_entities(ref_text)

    if not pred_subs and not ref_subs:
        return 1.0
    if not pred_subs or not ref_subs:
        return 0.0

    # Greedy matching: for each ref sub, find best matching pred sub
    used_pred = set()
    total_score = 0.0

    for ref_sub in ref_subs:
        best_score = 0.0
        best_idx = -1
        for j, pred_sub in enumerate(pred_subs):
            if j in used_pred:
                continue
            score = match_sub_entities(pred_sub, ref_sub)
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx >= 0:
            used_pred.add(best_idx)
        total_score += best_score

    # Penalize extra predicted sub-questions
    n_extra = len(pred_subs) - len(ref_subs)
    if n_extra > 0:
        penalty = 0.1 * n_extra  # small penalty per extra sub-question
        total_score = max(0, total_score - penalty)

    return total_score / len(ref_subs) if ref_subs else 0.0


def reward_semantic_gpt(pred_text, ref_text, input_text=""):
    """
    GPT-based semantic similarity scoring.
    Returns a score between 0 and 1.
    """
    pred_subs = parse_sub_questions(pred_text)
    ref_subs = parse_sub_questions(ref_text)

    if not pred_subs:
        return 0.0

    pred_qs = [s["question"] for s in pred_subs]
    ref_qs = [s["question"] for s in ref_subs]

    prompt = f"""Score the semantic similarity between predicted and reference sub-questions for this decomposition task.

Original question: "{input_text}"

Reference sub-questions:
{chr(10).join(f'  R{i+1}: {q}' for i, q in enumerate(ref_qs))}

Predicted sub-questions:
{chr(10).join(f'  P{i+1}: {q}' for i, q in enumerate(pred_qs))}

Score from 0.0 to 1.0 based on:
- Do the predicted sub-questions cover the same information needs as the reference?
- Are they semantically equivalent even if worded differently?
- Penalize missing or extra sub-questions slightly.

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip()
        score = float(re.search(r"(\d+\.?\d*)", raw).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"  GPT scoring failed: {e}")
        return 0.5  # fallback neutral score


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

        # Extract original input from prompt for GPT context
        input_text = ""
        if prompt:
            # Extract user message from chat template
            m = re.search(r"user\n(.+?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
            if m:
                input_text = m.group(1).strip()

        # Calculate component rewards
        r_format = reward_format(text)
        r_count = reward_count(text, ref)
        r_type = reward_type(text, ref)
        r_entity = reward_entity(text, ref)
        r_repetition = reward_repetition(text) 

        # GPT semantic scoring (expensive, only for decent outputs)
        if r_format >= 0.5 and r_entity >= 0.3 and len(text.strip()) > 20:
            r_semantic = reward_semantic_gpt(text, ref, input_text)
        else:
            r_semantic = 0.0  # skip GPT for clearly bad outputs

        # Weighted combination (0 to 1 scale)
        total = (
            REWARD_WEIGHT_COUNT * r_count +
            REWARD_WEIGHT_ENTITY * r_entity +
            REWARD_WEIGHT_REPETITION * r_repetition +
            REWARD_WEIGHT_SEMANTIC * r_semantic +
            REWARD_WEIGHT_FORMAT * r_format +
            REWARD_WEIGHT_TYPE * r_type
        )

        # Scale to [-1, 1] range for GRPO stability
        total = total * 2 - 1

        rewards.append(float(total))

        if i < 2:  # Log first 2 for debugging
            logger.debug(f"  Sample {i}: fmt={r_format:.2f} cnt={r_count:.2f} "
                         f"ent={r_entity:.2f} rep={r_repetition:.2f} "
                         f"sem={r_semantic:.2f} typ={r_type:.2f} "
                         f"total={total:.3f}")

    return rewards


# ====================================================================
# Main Training
# ====================================================================

def main():
    global GPT_MODEL

    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--model_path", type=str, default=SFT_MODEL_PATH)
    parser.add_argument("--gpt_model", type=str, default=GPT_MODEL)
    args = parser.parse_args()

    GPT_MODEL = args.gpt_model

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
    logger.info(f"  Reward: count({REWARD_WEIGHT_COUNT}) + entity({REWARD_WEIGHT_ENTITY}) "
                 f"+ repetition({REWARD_WEIGHT_REPETITION}) + semantic({REWARD_WEIGHT_SEMANTIC}) "
                 f"+ format({REWARD_WEIGHT_FORMAT}) + type({REWARD_WEIGHT_TYPE})")
    logger.info(f"  GPT model: {GPT_MODEL}")
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