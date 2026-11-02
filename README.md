# Interpretability of NHANES Depression Diagnosis Models

An end-to-end ML and GraphRAG research project for interpretable question answering over NHANES depression-model explanations. The system fine-tunes Qwen3-8B-Base models to decompose complex questions into structured sub-questions, translates searchable steps into Cypher, retrieves SHAP-derived evidence from Neo4j, and synthesizes an evidence-grounded response through an interactive Gradio application.

> **Research-use disclaimer:** This repository evaluates model explanations and knowledge-graph question answering. It is not a clinical diagnostic system and must not be used to make patient-care decisions.

## Overview

The project studies whether decomposition and graph grounding can make questions about depression-prediction models more transparent and executable. The current pipeline consists of:

1. **Question decomposition** — a fine-tuned LLM breaks a complex question into atomic `searching` and `aggregation` sub-questions.
2. **Entity and schema grounding** — graph vocabulary, entity identities, properties, and relationship directions are retrieved from Neo4j.
3. **Cypher generation** — each searchable sub-question is translated into a single executable Cypher query.
4. **Knowledge retrieval** — Cypher queries retrieve SHAP values, feature importance, model performance, patient-level explanations, and related evidence.
5. **Answer synthesis** — retrieved evidence is composed into a final response.
6. **Interactive application** — a Gradio interface runs the full decomposition-to-answer pipeline through SGLang-backed local models and configured API services.

The knowledge graph stores explanation evidence derived from multiple binary depression-classification models, allowing users to inspect why a prediction was made and compare evidence across patients, biomarkers, algorithms, and training settings.

## Training Approaches

| Experiment | Training Data | Approach |
|---|---|---|
| Exp1 | 996 NHANES base samples | SFT (QLoRA) |
| Exp2 | Mix-replace augmentation | SFT (QLoRA) |
| Exp3 | Domain-expanded data | SFT (QLoRA) |
| Exp4 | SQA-expanded data | SFT (QLoRA) |
| Exp5 | SQA-large data | SFT (QLoRA) |
| Exp6 | Exp1 + GRPO reinforcement learning | SFT + GRPO |

- **SFT:** supervised fine-tuning with QLoRA, 4-bit NF4 quantization, and LoRA adapters on linear layers.
- **GRPO:** Group Relative Policy Optimization with a six-component reward covering count accuracy, entity matching, repetition, semantic quality, format correctness, and type matching.

## Original Experimental Results

The following results are retained from the original repository evaluation.

### Question Decomposition Quality (Truncated Output)

| Exp | Semantic Mean | Count Exact Rate (%) | Redundancy Rate (%) | Format Rate (%) | Type Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Exp1 | 0.9800 | 100.0 | 4.0 | 100.0 | 100.0 |
| Exp2 | 0.9642 | 47.0 | 25.0 | 100.0 | 99.2 |
| Exp3 | 0.9645 | 48.0 | 28.0 | 100.0 | 100.0 |
| Exp4 | 0.9645 | 47.0 | 27.0 | 100.0 | 100.0 |
| Exp5 | 0.9821 | 99.0 | 3.0 | 100.0 | 100.0 |
| Exp6 | 0.9786 | 99.0 | 4.0 | 100.0 | 100.0 |

### Hit Rate and Cypher Execution

Original evaluation at cosine similarity >= 0.92 with Hungarian one-to-one matching:

| Exp | Ref (Searching) | Hit | Hit+Exec | Hit Rate | Exec/Hit |
|---|---:|---:|---:|---:|---:|
| Exp1 | 241 | 210 | 153 | 87.1% | 72.9% |
| Exp5 | 241 | 216 | 157 | 89.6% | 72.7% |
| Exp6 | 241 | 210 | 153 | 87.1% | 72.9% |

- **Ref (Searching):** number of reference searching sub-questions.
- **Hit:** predicted sub-questions matched to a unique reference at the cosine threshold.
- **Hit+Exec:** matched sub-questions whose Cypher executed and returned non-empty results.
- **Hit Rate:** `Hit / Ref (Searching)`, measuring decomposition coverage.
- **Exec/Hit:** `Hit+Exec / Hit`, measuring Cypher quality among matched questions.

> The current `calc_hit_rate.py` default is 0.90, while the table above was generated at 0.92. Re-running the script at the current threshold will not exactly reproduce this historical table.

## Updated End-to-End GraphRAG Evaluation

### Evaluation Design

The updated comparison evaluates Cypher-generation strategies on `ood_data/curriculum_test_200_gold.json`:

- **200 samples:** 100 seen-template and 100 unseen-template examples.
- **289 searching sub-questions** produced by the corrected progressive decomposition model.
- **Few-shot baseline:** the original question is included, but no graph grounding is used.
- **GraphRAG v1:** schema/entity grounding without the later prompt and repair fixes.
- **GraphRAG v2:** answer-value redaction, demo-first hard rules, and deterministic repair for concatenated queries.
- **Paired evaluation:** all arms use the same fixed decompositions and are compared sample by sample.

Three temperature-zero replays of each principal arm produced byte-identical Cypher and returned rows when fresh servers processed identical prompts in identical order. This demonstrates replay determinism under the tested configuration, not the absence of model noise under changed seeds, order, sampling, or infrastructure.

### Test-200 Graph Coverage

| Method | Overall | Seen 100 | Unseen 100 |
|---|---:|---:|---:|
| Few-shot baseline | 0.751 | **0.954** | 0.548 |
| GraphRAG v1 | 0.806 | 0.906 | 0.705 |
| GraphRAG v2 | **0.850** | **0.954** | **0.745** |

GraphRAG v2 corrected the seen-template regression of v1 while improving unseen-template generalization:

- **v2 - v1 overall:** +0.044, Wilcoxon `p = 8e-5`, 19 wins / 2 losses.
- **v2 - v1 seen:** +0.048, `p = 0.002`, 10 wins / 0 losses.
- **v2 - v1 unseen:** +0.040, `p = 0.008`, 9 wins / 2 losses.
- **v2 vs few-shot seen:** exact aggregate tie at 0.954.
- **v2 vs few-shot unseen:** +0.197, Cohen's `d = 0.61`.
- **v2 execution rate:** 99.3%.
- **v2 non-empty result rate:** 99.0%.

The v2 gains came from three targeted changes:

1. **Answer-value redaction:** grounding exposes identities and relevant property names without leaking target answer values into the generation prompt.
2. **Hard generation rules:** exactly one `RETURN`, relationship types cannot be used as node labels, answer properties belong in `RETURN`/`ORDER BY` unless explicitly constrained, and matching demonstrations should be reused structurally.
3. **Deterministic repair:** multi-query outputs are split and the block with the strongest entity filtering is retained.

Prompt and redaction alone reached 0.837 overall coverage; deterministic repair raised it to 0.850. Nine steps were repaired, all executed with non-empty results, and five seen samples improved from 0.5 to 1.0 coverage.

### Neighborhood-Rescue Ablation

An additional arm added pre-generation and execution-feedback neighborhood rescue on top of GraphRAG v2:

- 25 of 289 steps triggered rescue: 18 before generation and 7 after an empty/error result.
- Execution rate increased from 99.3% to 100%.
- Four of seven post-execution retries converted an empty/error result into rows.
- Gold coverage changed from 0.8495 to 0.8460.
- Paired Wilcoxon `p = 0.47` with 1 better / 3 worse / 196 unchanged samples.
- An execution-feedback-only counterfactual improved overall coverage by only +0.0004.

**Conclusion:** neighborhood rescue did not produce a measurable coverage gain and is not carried forward in the preferred v2 pipeline. The durable contribution from this experiment was the corrected directed-edge representation.

### Prompt Rule vs Graph Grounding

The no-truncation ablation separates one important prompt rule from GraphRAG retrieval. The rule prevents truncating `local_max_abs` results when later sub-questions refer to the earlier result set.

| Method | Overall | Seen 100 | Unseen 100 | 49 Coreference Chains |
|---|---:|---:|---:|---:|
| Few-shot | 0.7513 | 0.9542 | 0.5484 | 0.3521 |
| Few-shot + no-truncation rule | 0.7850 | 0.9542 | 0.6158 | 0.4898 |
| GraphRAG v2 | **0.8495** | **0.9542** | **0.7449** | **0.7531** |

- Rule-only unseen gain: +0.0675, 95% interval `[+0.032, +0.109]`, `p = 0.0014`, `d = 0.34`.
- GraphRAG v2 over rule-only: +0.1290 unseen, 95% interval `[+0.079, +0.185]`, `p = 5e-5`, `d = 0.48`.
- The prompt rule explains **34%** of GraphRAG v2's improvement over few-shot.
- The graph-grounding machinery explains the remaining **66%**.
- On slices where the rule cannot fire, the rule-only arm ties the original few-shot baseline exactly.

These results support the narrower claim that GraphRAG improves unseen-template, multi-step coreference questions, while also showing that one-third of the raw margin came from a prompt rule rather than retrieval itself.

## Expected Decomposition Format

```text
Step 1: [description]
Sub-question: [question text] # Type: searching
Step 2: [description]
Sub-question: [question text] # Type: aggregation
```

## Usage

### Training

```bash
python train.py --exp 1
python train.py --exp 1 2 3 4 5
python grpo_train.py
```

### Merge and Decomposition Evaluation

```bash
python merge_lora.py --exp 1
python evaluate.py --exp 1
python evaluate.py --exp 1 --both
```

### Original Cypher Evaluation

```bash
python few_shot_cypher.py
python calc_hit_rate.py
```

### Updated GraphRAG Evaluation

```bash
# Run the test-200 few-shot and GraphRAG arms
python run_test200_fewshot_vs_graphrag_3reps.py

# Score few-shot, GraphRAG v1, and GraphRAG v2
python score_test200_arms.py --rep 1

# Run the no-truncation ablation
python score_test200_notruncate_ablation.py
```

The full application is implemented in `app.py`. Runtime secrets and endpoints must be supplied through environment variables; credentials are intentionally excluded from this repository.

## Project Structure

```text
├── app.py                                  # Gradio end-to-end application
├── train.py                                # SFT training with QLoRA
├── grpo_train.py                           # GRPO reinforcement learning
├── merge_lora.py                           # Merge SFT LoRA adapters
├── merge_grpo.py                           # Merge GRPO adapters
├── evaluate.py                             # Decomposition inference and evaluation
├── compare.py                              # Compare SFT experiments
├── calc_hit_rate.py                        # Historical hit-rate and execution analysis
├── few_shot_cypher.py                      # Few-shot English-to-Cypher baseline
├── graph_schema.py                         # Live Neo4j schema-card construction
├── entity_retrieval.py                     # GraphRAG entity and relationship grounding
├── connect_neo4j.py                        # Shared Neo4j connection helpers
├── run_finetuned_pipeline_eval.py          # End-to-end decomposition/Cypher evaluation
├── run_cypher_qa_comparison.py             # Few-shot vs schema/grounding comparison
├── experiment1_5_baseline_original_input.py
├── experiment1_6_baseline_notruncate.py
├── experiment2_fewshot_graphrag.py
├── experiment2_fewshot_graphrag_v2.py
├── experiment3_fewshot_graphrag_v2_rescue.py
├── run_test200_fewshot_vs_graphrag_3reps.py
├── score_test200_arms.py
├── score_test200_notruncate_ablation.py
├── data/                                    # Original training/validation/test data
├── ood_data/                                # OOD curriculum and gold evaluation data
├── nhanes_model/                            # Source/configuration; weights are excluded
└── cypher_results*/                         # Generated evaluation outputs (not committed)
```

## Dependencies

Core dependencies include:

`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`, `openai`, `sglang`, `scipy`, `numpy`, `torch`, `neo4j`, `gradio`, and optional `wandb`.

## Environment

- Base model: [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- Training environment: GPU-enabled Lightning AI Studio
- Knowledge graph: Neo4j
- Local serving: SGLang
- OpenAI services: embeddings, selected evaluation judges, and configured synthesis/Cypher components
- Sensitive configuration: environment variables such as `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and API credentials

## Repository and Artifact Policy

This repository contains source code, experiment definitions, small configuration files, and documented aggregate results. It intentionally excludes:

- downloaded base-model caches;
- merged model weights and checkpoints;
- generated training and evaluation outputs;
- Neo4j runtime databases and transaction logs;
- local application certificates;
- API credentials and service tokens.

Large model artifacts should be stored in a dedicated model registry or object store rather than regular Git history.

