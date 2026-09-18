#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

run() {
    local params="$1"
    local scenario
    local name
    local log_file

    scenario="$(basename "$(dirname "$params")")"
    name="$(basename "$params" .py)"
    log_file="$LOG_DIR/${scenario}_${name}.log"

    echo
    echo "=================================================="
    echo "Running: $params"
    echo "Log:     $log_file"
    echo "=================================================="

    python main/feature_grid/main.py --params "$params" 2>&1 | tee "$log_file"
}

# --------------------------------------------------
# Intra-subject
# --------------------------------------------------
run "main/feature_grid/params/intra_subject/params_bci2a.py"
run "main/feature_grid/params/intra_subject/params_eegmmidb.py"
run "main/feature_grid/params/intra_subject/params_weibo.py"
run "main/feature_grid/params/intra_subject/params_zhou.py"

# --------------------------------------------------
# Cross-session
# --------------------------------------------------
run "main/feature_grid/params/cross_session/params_bci2a.py"
run "main/feature_grid/params/cross_session/params_zhou.py"

# --------------------------------------------------
# Cross-subject
# --------------------------------------------------
run "main/feature_grid/params/cross_subject/params_bci2a.py"
run "main/feature_grid/params/cross_subject/params_eegmmidb.py"
run "main/feature_grid/params/cross_subject/params_weibo.py"
run "main/feature_grid/params/cross_subject/params_zhou.py"

echo
echo "All runs finished successfully."
echo "Logs saved to: $LOG_DIR"