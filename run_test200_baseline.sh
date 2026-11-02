#!/bin/bash
cd /teamspace/studios/this_studio
set -e
T=ood_data/curriculum_test_200.json
RUN=ood_curriculum_baseline_mixed1076_corrected
python -u gen_test250_reps.py --run $RUN --test_file $T --reps 1
python -u export_test250_reps_for_judging.py --run $RUN --test_file $T --reps 1
echo "ALL DONE"
