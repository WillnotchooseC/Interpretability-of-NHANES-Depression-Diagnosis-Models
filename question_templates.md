# `shared_test.json` 复杂问题骨架模板

本文档对 [data/shared_test.json](data/shared_test.json) 中的 100 条复杂问题（input）按"分解后的子问题语义结构"进行归并，得到 **11 个骨架模板**。

每个模板由若干个子问题语义类别组合而成，子问题语义类别共 14 种（8 种 searching + 6 种 aggregation），定义见文末附录。

---

## 总览

| 簇 | 模板数 | 样本数 | 是否含 aggregation |
|---|---|---|---|
| A. 纯查找型 | 7 | 76 | 否 |
| B. 比较 / 汇总型 | 5 | 24 | 是 |
| **合计** | **11** | **100** | — |

| 模板 ID | 名称 | 样本数 | 占比 |
|---|---|---|---|
| A1 | 单点查询 | 26 | 26% |
| A2 | 定义 + 患者值 | 7 | 7% |
| A3 | 患者值 + 模型正确性 | 8 | 8% |
| A4 | 完整解读：定义 + 取值 + 影响 / 重要性 | 12 | 12% |
| A5 | 跨模型 SHAP 多连查 | 7 | 7% |
| A6 | 概率 + 正确性 | 4 | 4% |
| A7 | 长尾查找组合 | 12 | 12% |
| B1 | 比例间概率比较 | 7 | 7% |
| B2 | 多模型预测一致性 | 5 | 5% |
| B3 | 比例对准确率影响 | 4 | 4% |
| B4 | 跨模型 / 比例 重要性比较 | 4 | 4% |
| B5 | 跨模型 敏感性 / 共同特征比较 | 4 | 4% |

---

## A. 纯查找型（76 条，无 aggregation）

### A1 · 单点查询（26 条）

只问一个事实，对应 1 个 searching 子问题。覆盖 `measurement / definition / probability / correctness / feature_effect / importance` 等任一意图。

**结构**：`SRC_*`（任一）

**例**：
- *What's the likelihood of patient 113 being diagnosed with depression using logistic regression at 1:4?*
- *Analyze the impact of increasing glycohemoglobin on naive bayes at 1:4.*

---

### A2 · 定义 + 患者值（7 条）

先定义某个生化特征，再查某位患者的该特征值。

**结构**：`SRC_definition → SRC_measurement`

**例**：
- *Explain creatinine to me first, then I need patient 184's creatinine level.*
- *Can you explain what blood manganese measures, and then show me the value for patient 72?*

---

### A3 · 患者值 + 模型正确性（8 条）

先获取患者指标（或定义），再核对某模型/某比例对该患者的预测是否正确。

**结构**：`SRC_measurement → SRC_correctness` 或 `SRC_definition → SRC_correctness`

**例**：
- *Get patient 34's blood mercury, then check if logistic regression (1:3) was accurate.*
- *Get random forest's prediction for patient 128 at 1:3 and tell me if it was correct.*

---

### A4 · 完整解读：定义 + 取值 + 影响 / 重要性（12 条）

数据集中最完整的"特征解读"模板：定义 + 该患者的值 + 该特征对模型预测的影响（feature_effect 或 importance）。

**结构**：`SRC_definition + SRC_measurement + (SRC_feature_effect | SRC_importance)`，3–4 个子问题

**例**：
- *Can you tell me patient 237's HDL level, define HDL, and show how raising it impacts decision tree at 1:4?*
- *I need to know: Patient 85's rbc folate - what is it, what does this test measure, and how does elevating it affect logistic regression at 1:1?*

---

### A5 · 跨模型 SHAP 多连查（7 条）

定义 / 测量该特征后，分别查多个（模型 × 比例）下的 SHAP 值。这是数据集中唯一会"逐模型重复同一类查询"的纯查找模板。

**结构**：`(SRC_definition | SRC_measurement) → SRC_shap × N`（N=3~4）

**例**：
- *What is the definition of vitamin d2? What's patient 26's vitamin d2 level? In random forest, decision tree, and logistic regression models with 1:1 ratio, what are the SHAP values of vitamin d2 for patient 26?*
- *What's the direct hdl-cholesterol level of patient 133? In the models random forest, logistic regression, decision tree, and naive bayes trained at 1:1 ratio, what are the SHAP values?*

---

### A6 · 概率 + 正确性（4 条）

先取某模型对该患者的预测概率，再核对预测是否正确。

**结构**：`SRC_probability → SRC_correctness`

**例**：
- *What is the predicted probability of depression for patient 42 in the random forest model trained with a 1:1 ratio, and was this prediction correct?*
- *Patient 195 - what's their depression probability in the naive bayes model at 1:2 ratio, and was it predicted correctly?*

---

### A7 · 长尾查找组合（12 条）

不属于以上 6 个标准模板的小众多步查找。常见 2–3 步组合：定义 + 重要性、取值 + 特征影响、定义 + 其他、连续多次正确性核查（不带聚合）等。

**例**：
- *I'd like to know what HDL is and how influential it is in logistic regression (1:2).*
- *I need to know patient 219's albumin reading and what happens to the random forest prediction (1:5 ratio) if we increase it.*

> 备注：A7 是建模时最容易出问题的部分——子问题搭配方式分散，每种组合在测试集里只出现 1–2 次。

---

## B. 比较 / 汇总型（24 条，必含 aggregation）

特点：前面若干 searching 子问题为聚合做铺垫，最后一个 aggregation 子问题做比较 / 汇总。

### B1 · 比例间概率比较（7 条）

同一模型在不同比例下，比较哪个比例给出的概率更高。

**结构**：`SRC_probability × N → AGG_compare_prob`

**例**：
- *Between 1:1 and 1:2 ratios in logistic regression, which produces a higher diagnosis probability for patient 81?*

> 注意：测试集里有退化输入 *Between 1:1 and 1:1 ratios...*（patient 170 / patient 281），属数据噪声，对评估结果有干扰。

---

### B2 · 多模型预测一致性（5 条）

固定比例下，问几个模型是否**全部**预测正确。

**结构**：`SRC_correctness × N → AGG_all_correct`

**例**：
- *At 1:1, did logistic regression, decision tree, and random forest all make correct predictions for patient 103?*
- *Check if random forest, naive bayes, and decision tree predictions were all correct for patient 290 with 1:2 training.*

---

### B3 · 比例对准确率影响（4 条）

固定模型，遍历多个比例，分析不平衡比例对该患者预测准确率的影响。

**结构**：`SRC_correctness × N → AGG_ratio_impact`（N≈5）

**例**：
- *Evaluate the prediction accuracy of the decision tree model for patient 227 under 1:3, 1:2, 1:5, 1:3, and 1:5 class distribution scenarios.*

> 注意：比例列表常出现重复（如 `1:3, 1:5, 1:3, 1:4, 1:5`），属数据噪声。

---

### B4 · 跨模型 / 比例 重要性比较（4 条）

比较同一特征在不同（模型, 比例）组合下的重要性，常附带特征定义。

**结构**：`(SRC_importance | SRC_other) × N [+ SRC_definition] → AGG_compare_importance`

**例**：
- *Between 1:1 and 1:3 in logistic regression, where does rbc folate rank higher? What is it, by the way?*
- *Compare blood manganese importance in logistic regression (1:5) vs naive bayes (1:3), define blood manganese, then summarize.*

---

### B5 · 跨模型 敏感性 / 共同特征比较（4 条）

合并了两个低频聚合意图：

- **敏感性比较**（`AGG_sensitivity`）：哪个模型对特征上升更敏感
- **共同 / 差异特征**（`AGG_common_diff`）：两个模型作出预测共用了哪些特征

**结构**：`SRC_feature_effect × N → AGG_sensitivity` 或 `SRC_other × N → AGG_common_diff`

**例**：
- *When rbc folate goes up, which model is more affected: logistic regression at 1:1 or naive bayes at 1:3?*
- *Blood Mercury: does decision tree (1:1) or random forest (1:5) show more sensitivity to increases?*

---

## 使用建议

1. **训练数据均衡**：A1（26%）和 A4 / A7（12% 各）权重过大；B 簇模板各只有 4–7 条。在 [train.py](train.py) 里若做分层抽样，建议按本表 11 个模板分桶。
2. **GRPO 奖励调试**：[grpo_train.py](grpo_train.py) 中的 entity matching / type matching 在 B 簇上方差更大，因为聚合句的实体提取更难——分桶看 reward 比看整体均值更有意义。
3. **评估分桶**：在 [evaluate.py](evaluate.py) 输出指标时，建议按本表 ID 输出 per-template 准确率，特别关注 A7 / B4 / B5 这几类样本量低的桶。
4. **Cypher 阶段**：[few_shot_cypher.py](few_shot_cypher.py) 的 few-shot 池目前是按 searching 整体抽取的，按本文档的 8 种 SRC 子类做均衡可能提升 KG 命中率。

---

## 附录：子问题语义类别（14 类）

### Searching 类（8 种 / 241 条）

| 类别 | 含义 |
|---|---|
| `SRC_definition` | 询问某特征 / 指标的临床定义 |
| `SRC_measurement` | 查询某患者某特征的具体测量值 |
| `SRC_probability` | 某（模型, 比例）对某患者预测的抑郁概率 |
| `SRC_correctness` | 某（模型, 比例）对某患者的预测是否正确 |
| `SRC_shap` | 某（模型, 比例, 患者, 特征）下的 SHAP 值 |
| `SRC_importance` | 某（模型, 比例）下某特征的重要性 |
| `SRC_feature_effect` | 增加某特征对某（模型, 比例）预测的影响 |
| `SRC_other` | 边缘措辞，难以归入上述任一类 |

### Aggregation 类（6 种 / 24 条）

| 类别 | 含义 |
|---|---|
| `AGG_compare_prob` | 比较不同（比例 / 模型）下哪个概率更高 |
| `AGG_all_correct` | 多个模型 / 比例是否**全部**预测正确 |
| `AGG_ratio_impact` | 多比例对预测准确率的整体影响 |
| `AGG_compare_importance` | 跨（模型, 比例）比较特征重要性 |
| `AGG_sensitivity` | 哪个模型对特征变化更敏感 |
| `AGG_common_diff` | 跨模型预测的共同 / 差异驱动特征 |

---

*统计基于 [data/shared_test.json](data/shared_test.json)，100 条样本，265 个子问题。*
