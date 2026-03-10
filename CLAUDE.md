# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML research project for **question decomposition** using fine-tuned Qwen3-8B-Base models on NHANES (National Health and Nutrition Examination Survey) domain data. The model learns to break complex questions into simpler sub-questions labeled as "searching" or "aggregation" types.

The project runs two training approaches:
1. **SFT (Supervised Fine-Tuning)** with QLoRA across 5 data experiments
2. **GRPO (Group Relative Policy Optimization)** reinforcement learning on top of the best SFT model (Exp1)

## Key Commands

### SFT Training
```bash
python train.py --exp 1              # single experiment
python train.py --exp 1 2 3 4 5      # all experiments
python train.py --exp 1 --epochs 3 --lr 1e-4 --use_wandb
```

### GRPO Training
```bash
python grpo_train.py                              # defaults
python grpo_train.py --epochs 2 --lr 5e-7
python grpo_train.py --resume ./nhanes_model/grpo/checkpoint-100
```

### Merge LoRA Adapters (required before inference)
```bash
python merge_lora.py --exp 1          # SFT experiments
python merge_grpo.py                  # GRPO (merges onto exp1's merged model)
```

### Inference & Evaluation
```bash
python evaluate.py --exp 1                        # inference + evaluate
python evaluate.py --exp 1 2 3 4 5                # all + cross-experiment comparison
python evaluate.py --exp 1 --skip_inference        # evaluate existing predictions only
python evaluate.py --exp 1 --batch                 # concurrent batch inference (faster)
python evaluate.py --exp 1 --both                  # evaluate both raw and truncated outputs
```

### Compare Results
```bash
python compare.py                     # compare all 5 SFT experiments (raw + truncated)
python compare_raw.py --exp 1 2 3 4 5 6   # include GRPO as exp6
```

## Architecture

### Training Pipeline
- **`train.py`** — SFT with QLoRA (4-bit NF4 quantization, LoRA on all linear layers). Uses `transformers.Trainer` directly, no TRL dependency. Masks prompt tokens in labels so only assistant responses contribute to loss. Disables Qwen3's `<think>` tags via `enable_thinking=False`.
- **`grpo_train.py`** — GRPO via `trl.GRPOTrainer`. Builds on Exp1's merged SFT model. Uses a 6-component weighted reward: count accuracy (0.25), entity matching (0.30), repetition penalty (0.15), GPT semantic scoring (0.10), format correctness (0.10), type matching (0.10). Calls OpenAI GPT-4o-mini for semantic reward scoring during training.
- **`merge_lora.py`** / **`merge_grpo.py`** — Merge LoRA adapters into base model for SGLang-compatible inference.

### Evaluation Pipeline
- **`evaluate.py`** — Launches an SGLang server for fast inference, then evaluates on 3 dimensions:
  1. **Semantic Similarity** — OpenAI `text-embedding-3-large` embeddings + Hungarian matching alignment
  2. **Sub-question Count Accuracy** — exact match and diff analysis
  3. **Redundancy Detection** — embedding cosine similarity flagging + GPT-4o verification
- Also includes output truncation logic to handle model generation artifacts (garbage tokens, looping).

### Data Layout
- `data/` — Training/validation/test JSON files. Each sample has `input` (complex question) and `output` (decomposed sub-questions).
  - `exp1_base_train.json` — 996 NHANES domain samples (base)
  - `exp2-5` — Variations: mix-replace, domain-expand, SQA-expand, SQA-large
  - `shared_val.json`, `shared_test.json` — Shared across all experiments
- `nhanes_model/` — Output directory for all checkpoints, merged models, predictions, and metrics.
  - `exp{1-5}/` — SFT experiment outputs (best_model/, merged_model/, test_predictions.json, metrics.json)
  - `exp6/` — GRPO evaluation results
  - `grpo/` — GRPO training outputs (final_model/, merged_model/)

### Expected Output Format
The model produces structured text:
```
Step 1: [description]
Sub-question: [question text] # Type: searching
Step 2: [description]
Sub-question: [question text] # Type: aggregation
```

## Dependencies
`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`, `openai`, `sglang`, `scipy`, `numpy`, `torch`, `wandb` (optional)

## Environment
- Runs on Lightning AI Studios (GPU environment)
- Base model: `Qwen/Qwen3-8B-Base` (downloaded via HuggingFace)
- OpenAI API required for evaluation (embeddings + GPT verification) and GRPO semantic reward
