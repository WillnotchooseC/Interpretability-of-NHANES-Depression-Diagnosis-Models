# Explainable Depression-Risk Model Analysis with GraphRAG and Fine-Tuned LLMs

> **Research artifact | NHANES · SHAP · Neo4j · GraphRAG · Qwen3-8B · QLoRA/SFT · SGLang · Gradio**
>
> **Publication status:** The associated paper, *Enhancing Interpretability of Depression Diagnosis Models with LLM Explanations*, has been accepted for publication. Citation and DOI will be added after the publisher releases the final record.

This repository presents a research prototype for **evidence-grounded explanations of machine-learning models applied to depression-related NHANES data**. It combines model predictions and SHAP feature attributions with a Neo4j knowledge graph, an instruction-tuned language model, and a multi-step GraphRAG question-answering workflow.

Instead of relying on unconstrained free-form model responses, the system decomposes a question, grounds relevant entities in a graph schema, generates and executes Cypher queries, and uses retrieved evidence to construct an explanation. The objective is to make model behavior more **inspectable, testable, and reproducible**—not to make independent clinical diagnoses.

## Project at a glance

| Area | Implementation / reported outcome |
| --- | --- |
| Data and explanation graph | Approximately **3,000 NHANES records** analyzed; graph artifacts connecting predictions and SHAP explanations for **200+ patient records** |
| LLM adaptation | **Qwen3-8B** with **4-bit QLoRA / supervised fine-tuning (SFT)** on approximately **1,000 curated instruction examples** |
| Reasoning and retrieval | Question decomposition, entity/schema grounding, multi-query Cypher generation and repair, graph evidence retrieval, answer synthesis |
| GraphRAG v2, paired evaluation | **85.0% gold-step coverage**, compared with **75.1%** for the few-shot baseline (200 paired questions; 289 search sub-questions) |
| GraphRAG v2, unseen templates | **74.5% coverage**, compared with **54.8%** for the baseline (**+19.7 percentage points** across 100 unseen-template questions) |
| Query reliability | **99.3% Cypher execution success** and **99.0% non-empty result rate** in the reported updated GraphRAG v2 evaluation |
| RL research | Investigated **Group Relative Policy Optimization (GRPO)** with a custom multi-component reward, but **the final pipeline uses SFT, not GRPO** |

> **Evaluation note:** The rows above describe distinct reported evaluation sets and metrics. They should not be interpreted as a single end-to-end clinical accuracy score. All figures are project-reported experimental results, not production-service guarantees.

## 1. Motivation

Predictive classifiers can provide a risk estimate while leaving users uncertain about **which features influenced the prediction**, **how individuals compare**, and **what data supports a particular explanation**. A standalone LLM may produce fluent but unsupported explanations, especially for questions that require several linked facts.

This project explores an alternative: **graph-grounded explanations backed by model-output and feature-attribution data**. The system is designed to support questions such as:

- Which features most strongly contributed to a particular model prediction?
- How do feature contributions vary across selected records or models?
- Which evidence should be retrieved to answer a multi-step question about a prediction?

These are questions about **model interpretability and evidence retrieval**, not medical treatment or diagnostic recommendations.

## 2. System architecture

```text
                 User question
                       |
                       v
           Fine-tuned Qwen3-8B (SFT)
             Question decomposition
                       |
                       v
            Entity and schema grounding
                       |
                       v
               Cypher generation
                       |
            +----------+----------+
            |                     |
            v                     v
       Query validation      Deterministic
       / execution           query repair
            |                     |
            +----------+----------+
                       |
                       v
               Neo4j evidence graph
          (NHANES + predictions + SHAP)
                       |
                       v
            Retrieved answer evidence
                       |
                       v
             Evidence-based synthesis
                       |
                       v
                 Gradio interface
```

**Major engineering components**

1. **Data and attribution layer:** Prepare NHANES records and link classifier outputs with SHAP explanations in Neo4j for patient-level analysis and comparison.
2. **Question decomposition:** Fine-tune an LLM to split complex requests into structured, answerable sub-questions.
3. **Grounded graph retrieval:** Resolve entities and graph schema references before producing executable Cypher queries.
4. **Reliability controls:** Apply Cypher-oriented prompt constraints, selected-value redaction, and deterministic repair of unsuccessful multi-query steps.
5. **Application layer:** Connect model inference and graph retrieval to an interactive Gradio workflow, using SGLang in the inference stack.
6. **Evaluation:** Compare baselines, training curricula, retrieval configurations, and repair strategies using controlled question sets and ablations.

## 3. Fine-tuning: SFT as the final approach

The production-path **research prototype uses supervised fine-tuning (SFT)** rather than reinforcement-learning fine-tuning.

- **Base model:** Qwen3-8B.
- **Adaptation:** Parameter-efficient 4-bit QLoRA / SFT.
- **Training data:** Approximately 1,000 curated instruction examples for question decomposition and related structured outputs.
- **Design exploration:** Multiple dataset mixes and curricula, including easy-to-hard training, were compared rather than assuming that all instruction mixtures were equally effective.
- **Reported decomposition benchmark:** The strongest SFT configuration achieved **99% exact sub-question-count matching**, **100% format compliance**, and **3% redundancy** under the documented decomposition evaluation.

The rationale for SFT is pragmatic: it provided an effective quality/compute trade-off for a constrained, structured decomposition task.

### GRPO: earlier reinforcement-learning research

An **earlier experimental version** explored **Group Relative Policy Optimization (GRPO)** following SFT. The prototype investigated a six-part reward design targeting:

1. Correct sub-question count;
2. Appropriate entity matching;
3. Reduced repetition;
4. Semantic quality of the decomposition;
5. Output-format compliance;
6. Question-type consistency.

This work provided hands-on experience with **reward engineering, policy-optimization experiments, and the training-cost trade-offs of RL for structured LLM outputs**. Because GRPO training time was too costly for the observed development needs, the project **did not adopt GRPO in its final model pipeline**. **No result in this README should be interpreted as a GRPO production or deployment gain.**

## 4. GraphRAG and evaluation

### GraphRAG v2: paired question comparison

| Metric | Few-shot baseline | GraphRAG v2 | Difference |
| --- | ---: | ---: | ---: |
| Gold-step coverage | 75.1% | **85.0%** | **+9.9 pp** |

Evaluation scope: **200 paired questions / 289 search sub-questions**, as reported in the project's evaluation summary.

### Generalization to unseen question templates

| Metric | Baseline | GraphRAG v2 | Difference |
| --- | ---: | ---: | ---: |
| Coverage on 100 unseen-template questions | 54.8% | **74.5%** | **+19.7 pp** |

The updated pipeline also reported **99.3% Cypher execution success** and **99.0% non-empty results**. These are query-path reliability measures; they do **not**, by themselves, prove that every final natural-language explanation is correct.

### Ablation-driven decisions

The experiments studied the effects of:

- Graph/schema grounding and entity disambiguation;
- Prompt and Cypher constraints;
- Deterministic multi-query repair;
- Additional neighborhood-rescue strategies.

Reported ablation findings attributed roughly **two-thirds of the unseen-template coverage gain to graph grounding**. Rescue mechanisms that failed to improve coverage were not retained solely for architectural complexity.

## 5. Engineering takeaways

This repository is intended to demonstrate more than a model fine-tuning result:

- **Structured LLM engineering:** Constrain model outputs to graph-executable sub-tasks instead of depending on unverified narrative generation.
- **Tool and data integration:** Connect an LLM inference service, Neo4j/Cypher, SHAP signals, and a Gradio UI into an end-to-end workflow.
- **Measurable reliability:** Evaluate retrieval coverage separately from query execution and non-empty-result rates.
- **Performance/compute trade-offs:** Compare curricula and fine-tuning methods; choose SFT over a more expensive GRPO prototype where justified.
- **Controlled experimentation:** Use paired evaluations, unseen templates, and ablation studies to identify which engineering changes contributed measurable benefit.

The system is a **research prototype**, not a clinically validated or production-deployed medical device.

## 6. Reproducibility and usage

The codebase contains experimental components for dataset preparation, model adaptation, graph-backed question answering, evaluation, and the application interface. Reproducing the complete results requires the original experiment configuration, relevant NHANES features, model checkpoints, and a populated Neo4j graph.

**Before running the repository:**

1. Prepare the NHANES inputs according to the data-processing code and respect any dataset-use requirements.
2. Provision a Neo4j instance and populate the graph using the repository's data/graph preparation workflow.
3. Obtain the supported Qwen3 model weights and configure the inference/fine-tuning environment.
4. Configure credentials **locally**, using ignored environment files or a secrets manager. Never commit tokens, passwords, notebook secrets, or local Shell histories.
5. Run the repository's evaluation workflow using the corresponding experiment settings and compare against the documented baseline/splits.
6. Start the Gradio UI only after the inference and graph dependencies are available.

> **Setup note:** Exact executable commands, dependency versions, dataset paths, and checkpoint names should be taken from the code and environment used for the accepted-paper experiments. They are intentionally not invented here. Keep any already-verified installation or run commands from the repository's previous README when publishing this revision.

## 7. Limitations and responsible use

- This research explains **machine-learning predictions and associated attribution evidence**; it does not diagnose depression or replace professional medical judgment.
- SHAP scores describe model attribution, **not causal effects** or a patient's clinical condition.
- Retrieval coverage, Cypher execution rate, and answer correctness measure different system properties and should be reported separately.
- Reported outcomes reflect specific model versions, question sets, graph content, and experimental settings; other deployments require their own validation.
- Do not commit protected health information, active credentials, private configuration files, caches, or local runtime histories to this public repository.

## 8. Publication and citation

**Associated paper (accepted; publication details pending):**

*Enhancing Interpretability of Depression Diagnosis Models with LLM Explanations.*

A complete venue citation, final author list, DOI, and BibTeX entry will be added when the paper is officially published. Please use the publisher's final bibliographic record when available.

## 9. Contact

For questions about the research artifact, evaluation methodology, or reproducibility, please open a GitHub issue in this repository.
