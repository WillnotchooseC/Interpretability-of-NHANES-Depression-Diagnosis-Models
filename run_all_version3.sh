#!/bin/bash
set -e
python experiment1_baseline_fewshot_version3.py --files ood_data/progressive_decomp_for_experiments.json --train_dataset ood_data/curriculum_oppose_sft.json
echo "=== EXP1_VERSION3 DONE ==="
python experiment2_fewshot_graphrag_version3.py --files ood_data/progressive_decomp_for_experiments.json --train_dataset ood_data/curriculum_oppose_sft.json
echo "=== EXP2_VERSION3 DONE ==="
python experiment3_fewshot_graphrag_rescue_version3.py --files ood_data/progressive_decomp_for_experiments.json --train_dataset ood_data/curriculum_oppose_sft.json
echo "=== EXP3_VERSION3 DONE ==="
echo "=== ALL_VERSION3_DONE ==="
