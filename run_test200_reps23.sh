#!/bin/bash
# reps 2 and 3 of both *_corrected models on curriculum_test_200, strictly sequential.
# Each rep launches its own SGLang server (gen_test250_reps.py); between passes we
# wait until :30000 is actually closed and the GPU is released, so a lingering
# server can never be reused by the next pass (see CLAUDE.md Gotchas, 2026-09-12).
cd /teamspace/studios/this_studio
set -e
T=ood_data/curriculum_test_200.json
PORT=30000

wait_port_free() {
  for i in $(seq 1 60); do
    if ! curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && ! pgrep -f "sglang.launch_server" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "!! port $PORT / sglang still up after 120s -- killing leftovers"
  pkill -9 -f "sglang.launch_server" || true
  pkill -9 -f "sglang::" || true
  sleep 5
  if curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1; then
    echo "!! port $PORT STILL busy, aborting"; exit 1
  fi
}

wait_port_free
for RUN in ood_curriculum_progressive_corrected ood_curriculum_baseline_mixed1076_corrected; do
  for REP in 2 3; do
    echo; echo "########## $RUN rep$REP  $(date -u +%FT%TZ) ##########"
    python -u gen_test250_reps.py --run $RUN --test_file $T --reps $REP
    wait_port_free
  done
  # one export per model, all 3 reps in the same yaml/json
  python -u export_test250_reps_for_judging.py --run $RUN --test_file $T --reps 1 2 3
done
echo "ALL DONE $(date -u +%FT%TZ)"
