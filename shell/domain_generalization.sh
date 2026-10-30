
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/domain_generalization/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

run() {
    local params="$1"
    local name
    local log_file

    name="$(basename "$params" .py)"
    log_file="$LOG_DIR/${name}.log"

    echo
    echo "=================================================="
    echo "Running: $params"
    echo "Log:     $log_file"
    echo "=================================================="

    python main/domain_generalization/main.py --params "$params" 2>&1 | tee "$log_file"
}

# Cross-subject
run "main/domain_generalization/params/cross_subject/params_bci2a.py"
#run "main/domain_generalization/params/cross_subject/params_eegmmidb.py"
#run "main/domain_generalization/params/cross_subject/params_weibo.py"
#run "main/domain_generalization/params/cross_subject/params_zhou.py"

echo
echo "All runs finished successfully."
echo "Logs saved to: $LOG_DIR"
