# Interpretability of NHANES Depression Diagnosis Models

An ML research project for **question decomposition** using fine-tuned Qwen3-8B-Base models on NHANES (National Health and Nutrition Examination Survey) domain data. The model learns to break complex questions into simpler sub-questions labeled as "searching" or "aggregation" types, which are then converted to Cypher queries for knowledge graph retrieval.

## Overview

This project explores how to make depression diagnosis models more interpretable by decomposing complex clinical questions into structured, executable sub-questions. The pipeline consists of:

1. **Question Decomposition** — Fine-tuned LLM breaks complex questions into atomic sub-questions
2. **Cypher Translation** — Sub-questions are converted to Cypher queries for Neo4j knowledge graph
3. **Knowledge Retrieval** — Cypher queries execute against the NHANES database to retrieve answers

## Training Approaches

| Experiment | Training Data | Approach |
|------------|--------------|----------|
| Exp1 | 996 NHANES base samples | SFT (QLoRA) |
| Exp2 | Mix-replace augmentation | SFT (QLoRA) |
| Exp3 | Domain-expanded data | SFT (QLoRA) |
| Exp4 | SQA-expanded data | SFT (QLoRA) |
| Exp5 | SQA-large data | SFT (QLoRA) |
| Exp6 | Exp1 + GRPO reinforcement learning | SFT + GRPO |

- **SFT**: Supervised Fine-Tuning with QLoRA (4-bit NF4 quantization, LoRA on all linear layers)
- **GRPO**: Group Relative Policy Optimization with 6-component weighted reward (count accuracy, entity matching, repetition penalty, GPT semantic scoring, format correctness, type matching)

## Results

### Question Decomposition Quality (Truncated Output)

| Exp | Semantic Mean | Count Exact Rate (%) | Redundancy Rate (%) | Format Rate (%) | Type Accuracy (%) |
|-----|--------------|---------------------|--------------------|-----------------|--------------------|
| Exp1 | 0.9800 | 100.0 | 4.0 | 100.0 | 100.0 |
| Exp2 | 0.9642 | 47.0 | 25.0 | 100.0 | 99.2 |
| Exp3 | 0.9645 | 48.0 | 28.0 | 100.0 | 100.0 |
| Exp4 | 0.9645 | 47.0 | 27.0 | 100.0 | 100.0 |
| Exp5 | 0.9821 | 99.0 | 3.0 | 100.0 | 100.0 |
| Exp6 | 0.9786 | 99.0 | 4.0 | 100.0 | 100.0 |

### Hit Rate & Cypher Execution (cosine >= 0.92, Hungarian 1-to-1 matching)

Evaluates end-to-end effectiveness: whether the model's decomposed sub-questions correctly match reference questions and can be successfully executed as Cypher queries against the knowledge graph.

| Exp | Ref (Searching) | Hit | Hit+Exec | Hit Rate | Exec/Hit |
|-----|----------------|-----|----------|----------|----------|
| Exp1 | 241 | 210 | 153 | 87.1% | 72.9% |
| Exp5 | 241 | 216 | 157 | 89.6% | 72.7% |
| Exp6 | 241 | 210 | 153 | 87.1% | 72.9% |

- **Ref (Searching)**: Number of reference sub-questions with type=searching (denominator)
- **Hit**: Prediction sub-questions that match a reference sub-question (cosine similarity >= 0.92, deduplicated, 1-to-1 Hungarian assignment)
- **Hit+Exec**: Hit sub-questions whose Cypher queries executed successfully and returned non-empty results
- **Hit Rate**: Hit / Ref (Searching) — measures question decomposition coverage
- **Exec/Hit**: Hit+Exec / Hit — measures Cypher translation quality among matched questions

## Expected Output Format

The model produces structured text:
```
Step 1: [description]
Sub-question: [question text] # Type: searching
Step 2: [description]
Sub-question: [question text] # Type: aggregation
```

## Usage

### Training
```bash
python train.py --exp 1              # SFT single experiment
python train.py --exp 1 2 3 4 5      # SFT all experiments
python grpo_train.py                  # GRPO on top of Exp1
```

### Merge & Inference
```bash
python merge_lora.py --exp 1          # merge LoRA adapters
python evaluate.py --exp 1            # inference + evaluation
python evaluate.py --exp 1 --both     # evaluate raw and truncated
```

### Cypher Evaluation
```bash
python few_shot_cypher.py             # convert sub-questions to Cypher and execute
python calc_hit_rate.py               # calculate hit rate and execution rate
```

## Project Structure

```
├── train.py                 # SFT training with QLoRA
├── grpo_train.py            # GRPO reinforcement learning
├── merge_lora.py            # Merge SFT LoRA adapters
├── merge_grpo.py            # Merge GRPO LoRA adapters
├── evaluate.py              # Inference + evaluation (semantic, count, redundancy)
├── compare.py               # Compare SFT experiments
├── compare_raw.py           # Compare all experiments including GRPO
├── calc_hit_rate.py         # Hit rate & Cypher execution analysis
├── few_shot_cypher.py       # Sub-question to Cypher translation
├── connect_neo4j.py         # Neo4j database connection
├── data/                    # Training/validation/test data
│   ├── exp1_base_train.json
│   ├── exp2-5_*.json
│   ├── shared_val.json
│   └── shared_test.json
├── nhanes_model/            # Metrics and evaluation results
│   └── exp{1-6}/raw|truncated/metrics.json
└── cypher_results/          # Cypher execution evaluation results
```

## Dependencies

`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`, `openai`, `sglang`, `scipy`, `numpy`, `torch`, `wandb` (optional)

## Environment

- Base model: [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- GPU environment (Lightning AI Studios)
- OpenAI API required for evaluation (embeddings + GPT verification) and GRPO semantic reward
