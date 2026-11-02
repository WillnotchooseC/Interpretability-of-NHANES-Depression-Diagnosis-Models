#!/bin/bash
# GraphRAG v2: two extra passes with permuted sample order (seeds 42, 43 -- the same
# permutations used for the few-shot arm, so rep2/rep3 are order-paired across arms),
# strictly sequential, fresh SGLang per pass; then merge rep1(orig)+s42+s43.
cd /teamspace/studios/this_studio
set -e
for SEED in 42 43; do
  echo "########## graphrag_v2 shuffled seed $SEED  $(date -u +%FT%TZ) ##########"
  python -u run_test200_shuffled_pass.py --arm graphrag_v2 --seed $SEED
done
python -u merge_test200_shuffled_reps.py --arm graphrag_v2 --seeds 42 43
echo "ALL DONE $(date -u +%FT%TZ)"
