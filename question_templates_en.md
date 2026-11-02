# Complex-Question Skeleton Templates — `shared_test.json`

This document classifies the 100 complex questions in [data/shared_test.json](data/shared_test.json) into **11 skeleton templates**, based on the semantic structure of their decomposed sub-questions.

Each sub-question belongs to one of 14 fine-grained semantic categories (8 `searching` + 6 `aggregation`); the categories are defined in the appendix.

---

## Overview

| Cluster | # Templates | # Samples | Contains aggregation? |
|---|---|---|---|
| A. Pure lookup | 7 | 76 | No |
| B. Comparison / aggregation | 5 | 24 | Yes |
| **Total** | **11** | **100** | — |

| Template ID | Name | Samples | Share |
|---|---|---|---|
| A1 | Single-fact lookup | 26 | 26% |
| A2 | Definition + patient measurement | 7 | 7% |
| A3 | Patient measurement + prediction correctness | 8 | 8% |
| A4 | Full interpretation: definition + measurement + effect / importance | 12 | 12% |
| A5 | Cross-model SHAP fan-out | 7 | 7% |
| A6 | Probability + correctness | 4 | 4% |
| A7 | Long-tail lookup combinations | 12 | 12% |
| B1 | Cross-ratio probability comparison | 7 | 7% |
| B2 | Multi-model prediction consistency | 5 | 5% |
| B3 | Ratio-vs-accuracy analysis | 4 | 4% |
| B4 | Cross-model / cross-ratio importance comparison | 4 | 4% |
| B5 | Cross-model sensitivity / common-feature comparison | 4 | 4% |

---

## Cluster A — Pure Lookup (76 samples, no aggregation)

### A1 · Single-fact lookup (26)

A single atomic question. The user wants exactly one piece of information — could be `measurement`, `definition`, `probability`, `correctness`, `feature_effect`, or `importance`.

**Structure:** `SRC_*` (one of any)

**Representative example:**

```
INPUT:
What's the likelihood of patient 113 being diagnosed with depression
using logistic regression at 1:4?

OUTPUT:
Let's solve this methodically:
Step 1: Retrieve the predicted probability of depression for patient
        113 from the logistic regression model with a 1:4 ratio.
Sub-question: What is the probability of patient 113 having depression
in the logistic regression algorithm with a 1:4 ratio?  # Type: searching
```

---

### A2 · Definition + patient measurement (7)

Define a clinical feature, then look up that feature's value for a specific patient.

**Structure:** `SRC_definition → SRC_measurement`

**Representative example:**

```
INPUT:
Explain creatinine to me first, then I need patient 184's creatinine level.

OUTPUT:
I'll figure this out:
Step 1: Gather the medical definition of creatinine.
Sub-question: Can you explain what creatinine is?  # Type: searching
Step 2: Get the creatinine value for patient 184.
Sub-question: What is patient 184's creatinine value?  # Type: searching
```

---

### A3 · Patient measurement + prediction correctness (8)

Fetch a patient's lab value (or a feature definition), then check whether a specific (model, ratio) made a correct prediction for that patient.

**Structure:** `SRC_measurement → SRC_correctness`  *or*  `SRC_definition → SRC_correctness`

**Representative example:**

```
INPUT:
Get patient 34's blood mercury, then check if logistic regression (1:3)
was accurate.

OUTPUT:
Step by step:
Step 1: Fetch patient 34's blood mercury value from the database.
Sub-question: What is patient 34's blood mercury level?  # Type: searching
Step 2: Verify logistic regression's prediction accuracy with a 1:3
        ratio for patient 34.
Sub-question: Was logistic regression's prediction at 1:3 accurate for
patient 34?  # Type: searching
```

---

### A4 · Full interpretation: definition + measurement + effect / importance (12)

The most complete "feature interpretation" template: define the feature, retrieve the patient's value, and then explain how that feature affects the model's prediction (`feature_effect` or `importance`).

**Structure:** `SRC_definition + SRC_measurement + (SRC_feature_effect | SRC_importance)` — 3 to 4 sub-questions

**Representative example:**

```
INPUT:
Can you tell me patient 237's HDL level, define HDL, and show how
raising it impacts decision tree at 1:4?

OUTPUT:
Here's my systematic approach:
Step 1: Obtain patient 237's HDL measurement.
Sub-question: What is patient 237's HDL level?  # Type: searching
Step 2: Define HDL in clinical terms.
Sub-question: What is the clinical definition of HDL?  # Type: searching
Step 3: Analyze how raising HDL would impact the decision tree model
        trained with a 1:4 ratio.
Sub-question: How would an increase in HDL affect the decision tree
model's predictions at a 1:4 training ratio?  # Type: searching
```

---

### A5 · Cross-model SHAP fan-out (7)

After defining and/or measuring a feature, retrieve its SHAP value across multiple (model, ratio) combinations. This is the only pure-lookup template that intentionally repeats the same query type N times.

**Structure:** `(SRC_definition | SRC_measurement) → SRC_shap × N`  (N = 3 or 4)

**Representative example:**

```
INPUT:
What is the definition of vitamin d2? What's patient 26's vitamin d2
level? In random forest, decision tree, and logistic regression models
with 1:1 ratio, what are the SHAP values of vitamin d2 for patient 26?

OUTPUT:
I'll process this as follows:
Step 1: Retrieve the definition of vitamin d2.
Sub-question: What is the definition of vitamin d2?  # Type: searching
Step 2: Query patient 26's vitamin d2 measurement.
Sub-question: What is the vitamin d2 level of patient 26?  # Type: searching
Step 3: Extract the SHAP value of vitamin d2 for patient 26 in the
        random forest model trained with a 1:1 ratio.
Sub-question: What is the SHAP value of vitamin d2 for patient 26 in
the random forest model with a 1:1 training ratio?  # Type: searching
Step 4: Extract the SHAP value of vitamin d2 for patient 26 in the
        decision tree model trained with a 1:1 ratio.
Sub-question: What is the SHAP value of vitamin d2 for patient 26 in
the decision tree model with a 1:1 training ratio?  # Type: searching
Step 5: Extract the SHAP value of vitamin d2 for patient 26 in the
        logistic regression model trained with a 1:1 ratio.
Sub-question: What is the SHAP value of vitamin d2 for patient 26 in
the logistic regression model with a 1:1 training ratio?  # Type: searching
```

---

### A6 · Probability + correctness (4)

Retrieve the predicted probability for a given (patient, model, ratio), then check whether that prediction was correct.

**Structure:** `SRC_probability → SRC_correctness`

**Representative example:**

```
INPUT:
What is the predicted probability of depression for patient 42 in the
random forest model trained with a 1:1 ratio, and was this prediction
accurate?

OUTPUT:
Let's reason step by step:
Step 1: Retrieve the predicted probability of depression for patient 42
        from the random forest model trained with a 1:1 ratio.
Sub-question: What is the probability that patient 42 is predicted to
have depression by the random forest model with a 1:1 ratio?
# Type: searching
Step 2: Check whether the prediction made by the random forest model
        with a 1:1 ratio for patient 42 was correct.
Sub-question: Was the prediction for patient 42 accurate in the random
forest model trained with a 1:1 ratio?  # Type: searching
```

---

### A7 · Long-tail lookup combinations (12)

Multi-step lookups that don't fit any of the six standard templates above. Common shapes include `definition + importance`, `measurement + feature_effect`, `definition + other`, or chains of correctness checks without a final aggregation step.

**Representative example:**

```
INPUT:
I'd like to know what HDL is and how influential it is in logistic
regression (1:2).

OUTPUT:
I'll approach this methodically:
Step 1: Identify what HDL is.
Sub-question: What is HDL?  # Type: searching
Step 2: Determine how influential HDL is in the logistic regression
        model with a 1:2 ratio.
Sub-question: How important is HDL in logistic regression at 1:2?
# Type: searching
```

> **Note:** A7 is the most fragmented bucket — each sub-pattern within it appears only 1–2 times in the test set, making per-pattern evaluation noisy.

---

## Cluster B — Comparison / Aggregation (24 samples, aggregation required)

Each sample in this cluster ends with an `aggregation` sub-question that compares or summarizes the results of the preceding `searching` sub-questions.

### B1 · Cross-ratio probability comparison (7)

Same model, different ratios — which ratio yields the higher predicted probability?

**Structure:** `SRC_probability × N → AGG_compare_prob`

**Representative example:**

```
INPUT:
Between 1:1 and 1:2 ratios in logistic regression, which produces a
higher diagnosis probability for patient 81?

OUTPUT:
Processing this query:
Step 1: Look up the predicted probability for patient 81 from the
        logistic regression model with a 1:1 ratio.
Sub-question: What is patient 81's diagnosis probability in the
logistic regression model at a 1:1 ratio?  # Type: searching
Step 2: Retrieve the predicted probability for patient 81 from the
        logistic regression model with a 1:2 ratio.
Sub-question: What is patient 81's diagnosis probability in the
logistic regression model at a 1:2 ratio?  # Type: searching
Step 3: Assess the probabilities to determine which is higher.
Sub-question: Which ratio (1:1 or 1:2) gives a higher diagnosis
probability for patient 81 in logistic regression?  # Type: aggregation
```

> **Note:** A few B1 samples contain degenerate inputs such as *"Between 1:1 and 1:1 ratios..."* — clear data noise that affects evaluation.

---

### B2 · Multi-model prediction consistency (5)

Fixed ratio, multiple models — did **all** of them predict correctly for the patient?

**Structure:** `SRC_correctness × N → AGG_all_correct`

**Representative example:**

```
INPUT:
At 1:1, did logistic regression, decision tree, and random forest all
make correct predictions for patient 103?

OUTPUT:
Here's my verification strategy:
Step 1: Verify if the logistic regression model made a correct
        prediction for patient 103 at a 1:1 ratio.
Sub-question: Did the logistic regression model make a correct
prediction for patient 103 at a 1:1 training ratio?  # Type: searching
Step 2: Verify if the decision tree model made a correct prediction
        for patient 103 at a 1:1 ratio.
Sub-question: Did the decision tree model make a correct prediction
for patient 103 at a 1:1 training ratio?  # Type: searching
Step 3: Verify if the random forest model made a correct prediction
        for patient 103 at a 1:1 ratio.
Sub-question: Did the random forest model make a correct prediction
for patient 103 at a 1:1 training ratio?  # Type: searching
Step 4: Summarize whether all models achieved correct predictions.
Sub-question: Did logistic regression, decision tree, and random forest
all make correct predictions for patient 103 at a 1:1 ratio?
# Type: aggregation
```

---

### B3 · Ratio-vs-accuracy analysis (4)

Fixed model, sweep multiple ratios — analyze how class-imbalance ratio impacts prediction correctness for one patient.

**Structure:** `SRC_correctness × N → AGG_ratio_impact`  (N is typically 3–5)

**Representative example:**

```
INPUT:
Check prediction correctness for patient 280 in the random forest model
at 1:1, 1:4, and 1:5, then assess the relationship between training
ratio and accuracy.

OUTPUT:
Let me solve this:
Step 1: Verify whether the random forest model trained with a 1:1 ratio
        made an accurate prediction for patient 280.
Sub-question: Was the prediction for patient 280 correct in the random
forest model trained with a 1:1 ratio?  # Type: searching
Step 2: Verify whether the random forest model trained with a 1:4 ratio
        made an accurate prediction for patient 280.
Sub-question: Was the prediction for patient 280 correct in the random
forest model trained with a 1:4 ratio?  # Type: searching
Step 3: Verify whether the random forest model trained with a 1:5 ratio
        made an accurate prediction for patient 280.
Sub-question: Was the prediction for patient 280 correct in the random
forest model trained with a 1:5 ratio?  # Type: searching
Step 4: Analyze the influence of data imbalance on the prediction
        accuracy across different ratios.
Sub-question: How does data imbalance (1:1 vs 1:4 vs 1:5) affect the
random forest model's prediction accuracy for patient 280?
# Type: aggregation
```

> **Note:** The ratio list in some B3 inputs contains duplicates (e.g. `1:3, 1:5, 1:3, 1:4, 1:5`) — also data noise.

---

### B4 · Cross-model / cross-ratio importance comparison (4)

Compare a feature's importance across different (model, ratio) settings, often with the feature definition included.

**Structure:** `(SRC_importance | SRC_other) × N [+ SRC_definition] → AGG_compare_importance`

**Representative example:**

```
INPUT:
Between 1:1 and 1:3 in logistic regression, where does rbc folate rank
higher? What is it, by the way?

OUTPUT:
Processing this query:
Step 1: Collect the importance of rbc folate in the logistic regression
        model at a 1:1 ratio.
Sub-question: How important is rbc folate in the logistic regression
model with a 1:1 ratio?  # Type: searching
Step 2: Fetch the importance of rbc folate in the logistic regression
        model at a 1:3 ratio.
Sub-question: How important is rbc folate in the logistic regression
model with a 1:3 ratio?  # Type: searching
Step 3: Retrieve the medical definition of rbc folate.
Sub-question: What is the definition of rbc folate?  # Type: searching
Step 4: Compare the importance levels across both ratios.
Sub-question: At which ratio (1:1 or 1:3) is rbc folate more important
in the logistic regression model?  # Type: aggregation
```

---

### B5 · Cross-model sensitivity / common-feature comparison (4)

Merges two low-frequency aggregation intents:

- **Sensitivity comparison** (`AGG_sensitivity`): which model is more affected when the feature increases?
- **Common / different features** (`AGG_common_diff`): which features did two models share when predicting depression?

**Structure:** `SRC_feature_effect × N → AGG_sensitivity`  *or*  `SRC_other × N → AGG_common_diff`

**Representative example:**

```
INPUT:
When rbc folate goes up, which model is more affected: logistic
regression at 1:1 or naive bayes at 1:3?

OUTPUT:
Analysis incoming:
Step 1: Obtain the effect of increasing rbc folate on the logistic
        regression model with a 1:1 ratio.
Sub-question: How does increasing rbc folate affect the logistic
regression model's prediction at a 1:1 ratio?  # Type: searching
Step 2: Gather the effect of increasing rbc folate on the naive bayes
        model with a 1:3 ratio.
Sub-question: How does increasing rbc folate affect the naive bayes
model's prediction at a 1:3 ratio?  # Type: searching
Step 3: Contrast the effects across both models.
Sub-question: Which model is more sensitive to increases in rbc folate
- logistic regression at 1:1 or naive bayes at 1:3?  # Type: aggregation
```

---

## How to use this taxonomy

1. **Training-data balancing.** A1 (26%), A4 (12%), and A7 (12%) dominate the test set, while every B template has only 4–7 samples. When doing stratified sampling in [train.py](train.py), use the 11 templates above as the strata.
2. **GRPO reward debugging.** Entity-matching and type-matching rewards in [grpo_train.py](grpo_train.py) have higher variance on the B cluster, because aggregation sub-questions reference more entities. Inspecting reward-per-template reveals issues that an overall mean hides.
3. **Per-template evaluation.** [evaluate.py](evaluate.py) currently reports global metrics. Adding per-template accuracy buckets — especially for A7, B4, B5 — makes regressions much easier to spot.
4. **Cypher few-shot pool.** [few_shot_cypher.py](few_shot_cypher.py) currently samples few-shots from the entire searching pool. Stratifying by the 8 SRC sub-categories (see appendix) is likely to improve KG-hit rate.

---

## Appendix — Sub-question Semantic Categories (14)

### Searching (8 categories / 241 sub-questions)

| Category | Meaning |
|---|---|
| `SRC_definition` | Asking for the clinical definition of a feature / lab indicator |
| `SRC_measurement` | Looking up a specific patient's value for a feature |
| `SRC_probability` | Predicted depression probability for (patient, model, ratio) |
| `SRC_correctness` | Whether (model, ratio) predicted correctly for a patient |
| `SRC_shap` | SHAP value of (feature, patient, model, ratio) |
| `SRC_importance` | Feature importance for (model, ratio) |
| `SRC_feature_effect` | Effect of increasing a feature on (model, ratio)'s prediction |
| `SRC_other` | Edge phrasings that don't fit any of the above |

### Aggregation (6 categories / 24 sub-questions)

| Category | Meaning |
|---|---|
| `AGG_compare_prob` | Which (ratio / model) gives the higher probability |
| `AGG_all_correct` | Did **all** specified models / ratios predict correctly |
| `AGG_ratio_impact` | Overall effect of multiple ratios on accuracy |
| `AGG_compare_importance` | Compare a feature's importance across (model, ratio) |
| `AGG_sensitivity` | Which model is more sensitive to a feature change |
| `AGG_common_diff` | Common / different driving features across models |

---

*Counts derived from [data/shared_test.json](data/shared_test.json): 100 samples, 265 sub-questions.*
