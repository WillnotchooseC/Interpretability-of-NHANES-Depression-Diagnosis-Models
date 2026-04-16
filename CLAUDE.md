# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML research project for **question decomposition + knowledge-graph QA** on NHANES (National Health and Nutrition Examination Survey) data, aimed at interpretable depression diagnosis. Fine-tuned Qwen3-8B-Base decomposes complex questions into atomic sub-questions (labeled `searching` or `aggregation`); `searching` sub-questions are then translated to Cypher via few-shot prompting and executed against a Neo4j knowledge graph.

The project runs two training approaches:
1. **SFT (Supervised Fine-Tuning)** with QLoRA across 5 data experiments
2. **GRPO (Group Relative Policy Optimization)** reinforcement learning on top of the best SFT model (Exp1, evaluated as Exp6)

End-to-end pipeline (training → Cypher → answer) is distinct from training-only commands — see "Cypher / KG Evaluation" and "End-to-End Demo" below.

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

### Cypher / KG Evaluation
```bash
python few_shot_cypher.py --files exp1_test.json exp5_test.json exp6_test.json           # generate + execute Cypher
python few_shot_cypher.py --files exp1_test.json --sglang                                  # auto-launch SGLang for fast gen
python few_shot_cypher.py --files exp1_test.json --no_execute                              # dry run, no Neo4j exec
python calc_hit_rate.py                                                                     # two-step funnel: hit rate + exec rate
```

### End-to-End Demo (full QA pipeline)
```bash
python app.py                         # Gradio UI (decompose → Cypher → Neo4j → GPT synthesis), http://localhost:7860
python run_demo.py                    # GPU-free variant: GPT-4o-mini replaces the local SFT model
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
- Also includes output truncation logic to handle model generation artifacts (garbage tokens, looping). `check_prediction.py` is an ad-hoc script for inspecting these artifacts.

### Cypher / KG Pipeline
- **`few_shot_cypher.py`** — Few-shot English→Cypher translation over each extracted sub-question, then executes against Neo4j and records whether results are non-empty. Can auto-launch SGLang for fast generation. Output lands in `cypher_results/exp{N}_test_predictions_cypher_eval.json` and a reference baseline.
- **`calc_hit_rate.py`** — Two-step funnel over Cypher eval outputs:
  1. **Hit Rate**: ref `searching` sub-Qs matched 1-to-1 by a unique prediction sub-Q (cosine ≥ 0.90, Hungarian assignment, predictions deduplicated first).
  2. **Exec Rate**: among hit prediction sub-Qs, fraction whose Cypher returned non-empty results.
- **`connect_neo4j.py`** — Shared Neo4j driver/session helpers. Config via `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/`NEO4J_DATABASE` env vars (defaults to `bolt://localhost:7687`). Note: `few_shot_cypher.py` hardcodes a different URI (ngrok tunnel) — override with `--neo4j_password` / edit constants when running locally.

### Demo / UI
- **`app.py`** — Gradio UI at `:7860`. Loads the merged SFT model (defaults to `nhanes_model/exp5/merged_model`), runs decomposition → few-shot Cypher → Neo4j → GPT-4o-mini answer synthesis.
- **`run_demo.py`** — GPU-free variant of the same pipeline; GPT-4o-mini replaces the local model for decomposition. Useful for sanity-checking the Cypher+Neo4j path without a GPU.

### Data Layout
- `data/` — Training/validation/test JSON files. Each sample has `input` (complex question) and `output` (decomposed sub-questions).
  - `exp1_base_train.json` — 996 NHANES domain samples (base)
  - `exp2-5` — Variations: mix-replace, domain-expand, SQA-expand, SQA-large
  - `shared_val.json`, `shared_test.json` — Shared across all experiments
- `nhanes_model/` — Output directory for all checkpoints, merged models, predictions, and metrics.
  - `exp{1-5}/` — SFT experiment outputs (best_model/, merged_model/, test_predictions.json, metrics.json). `raw/` vs `truncated/` subdirs separate pre/post truncation metrics.
  - `exp6/` — GRPO evaluation results (same layout as SFT exps)
  - `grpo/` — GRPO training outputs (final_model/, merged_model/)
- `cypher_results/` — Cypher eval outputs per experiment plus `reference_baseline.json` (upper-bound exec rate using ground-truth sub-questions) and `combined_metrics.json`.

### Expected Output Format
The model produces structured text:
```
Step 1: [description]
Sub-question: [question text] # Type: searching
Step 2: [description]
Sub-question: [question text] # Type: aggregation
```

## Dependencies
`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`, `openai`, `sglang`, `scipy`, `numpy`, `torch`, `neo4j`, `gradio`, `wandb` (optional)

## Environment
- Runs on Lightning AI Studios (GPU environment)
- Base model: `Qwen/Qwen3-8B-Base` (downloaded via HuggingFace)
- OpenAI API required for evaluation (embeddings + GPT verification), GRPO semantic reward, and demo answer synthesis
- Neo4j required for Cypher execution and the end-to-end demo (local `bolt://localhost:7687` by default; override via env vars)
