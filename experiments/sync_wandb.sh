#!/bin/bash
#SBATCH --job-name=wandb-sync
#SBATCH --output=wandb-sync-%j.out
#SBATCH --error=wandb-sync-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# ------------------------------------------------------------
# Usage:
#   sbatch wandb_sync.sbatch MT_Jan_14_16_00_Results_Pendulum_Test_7_offline
# ------------------------------------------------------------

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch wandb_sync.sbatch <PROJECT_DIR>"
    exit 1
fi

PROJECT_DIR="$1"
BASE_DIR="/cluster/scratch/kiten"
TARGET="${BASE_DIR}/${PROJECT_DIR}"

if [ ! -d "$TARGET" ]; then
    echo "ERROR: directory does not exist: $TARGET"
    exit 1
fi

echo "============================================================"
echo "W&B offline sync job"
echo "Project directory : $TARGET"
echo "Started at        : $(date)"
echo "============================================================"

export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

NUM_RUNS=$(find "$TARGET" -type d -name "offline-run-*" | wc -l)

if [ "$NUM_RUNS" -eq 0 ]; then
    echo "No offline runs found. Nothing to sync."
    exit 0
fi

echo "Found $NUM_RUNS offline runs to sync."
echo

COUNTER=0

find "$TARGET" -type d -name "offline-run-*" | while read -r RUN_DIR; do
    COUNTER=$((COUNTER + 1))
    echo "------------------------------------------------------------"
    echo "[$COUNTER / $NUM_RUNS] Syncing:"
    echo "  $RUN_DIR"
    echo "  Time: $(date)"
    wandb sync "$RUN_DIR"
done

echo "============================================================"
echo "All runs synced successfully."
echo "Finished at : $(date)"
echo "============================================================"
