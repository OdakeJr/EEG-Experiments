
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/domain_generalization_vision/$(date +"%Y%m%d_%H%M%S")"
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

    python main/domain_generalization_vision/main.py \
        --params "$params" 2>&1 | tee "$log_file"
}

# Vision datasets
run "main/domain_generalization_vision/params/vision_dataset/params_pacs.py"
#run "main/domain_generalization_vision/params/vision_dataset/params_vlcs.py"
#run "main/domain_generalization_vision/params/vision_dataset/params_office_home.py"

echo
echo "All runs finished successfully."
echo "Logs saved to: $LOG_DIR"
