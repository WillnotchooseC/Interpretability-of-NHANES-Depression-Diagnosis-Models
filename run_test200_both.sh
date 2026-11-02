#!/bin/bash
# one greedy pass each, strictly sequential: progressive_corrected then baseline_corrected
cd /teamspace/studios/this_studio
set -e
T=ood_data/curriculum_test_200.json
for RUN in ood_curriculum_progressive_corrected ood_curriculum_baseline_mixed1076_corrected; do
  python -u gen_test250_reps.py --run $RUN --test_file $T --reps 1
  python -u export_test250_reps_for_judging.py --run $RUN --test_file $T --reps 1
done
echo "ALL DONE"
