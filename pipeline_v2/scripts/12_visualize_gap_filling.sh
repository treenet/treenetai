#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Stage 12: Visualize Gap Filling
#===============================================================================
# This script generates visualizations of model gap-filling performance.
#
# HOW TO USE:
#   1. Edit the USER CONFIGURATION section below
#   2. Run: ./scripts/12_visualize_gap_filling.sh
#   3. Or dry-run first: ./scripts/12_visualize_gap_filling.sh --dry-run
#
# DENORMALIZATION MODES (2026-01-12):
#   normalized   - Keep outputs in [0,1] range (relative values)
#   ideal        - Denormalize using LM params (reference only - NOT available in production!)
#   operational  - Denormalize using input params (realistic deployment scenario)
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the visualization process               ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# DATA/OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Run name: determines input folder
RUN_NAME="swiss_segment_norm_all_combos"

# Output root directory (for processed data)
OUTPUT_ROOT="/storage/lukovic/Data/FORWARDS/treenet/processed"

# Visualization output directory
# NOTE: Images must NOT be stored in /home/lukovic/codes/
VIZ_OUTPUT_ROOT="/home/lukovic/data/treenet"

# ─────────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Path to trained model
# Leave empty to use the latest experiment's best model
MODEL_PATH=""
# Example: "/storage/lukovic/.../experiments/20260111_152352_segment_norm_attention/best_model.keras"

# ─────────────────────────────────────────────────────────────────────────────────
# DENORMALIZATION MODE (NEW 2026-01-12)
# ─────────────────────────────────────────────────────────────────────────────────
# Options: normalized, ideal, operational
#
# normalized  - Keep [0,1] values, no denormalization
# ideal       - Use LM (output) normalization params
#               ⚠️ REFERENCE ONLY - NOT available in production!
# operational - Use input normalization params
#               ✅ RECOMMENDED - realistic deployment scenario
DENORM_MODE="operational"

# ─────────────────────────────────────────────────────────────────────────────────
# VISUALIZATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Number of samples to visualize
N_SAMPLES=10

# Data split to use: train or test
SPLIT="test"

# Gap injection settings
GAP_DAYS=7
N_GAPS=2

# ─────────────────────────────────────────────────────────────────────────────────
# HARDWARE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# GPU ID to use
GPU_ID=2

# ─────────────────────────────────────────────────────────────────────────────────
# PYTHON ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────────
PYTHON="/home/lukovic/pyenv/lamella/bin/python"

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                    END OF USER CONFIGURATION                                   ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# Override from environment if set
RUN_NAME="${RUN_NAME_ENV:-$RUN_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT_ENV:-$OUTPUT_ROOT}"
VIZ_OUTPUT_ROOT="${VIZ_OUTPUT_ROOT_ENV:-$VIZ_OUTPUT_ROOT}"
MODEL_PATH="${MODEL_PATH_ENV:-$MODEL_PATH}"
DENORM_MODE="${DENORM_MODE_ENV:-$DENORM_MODE}"
N_SAMPLES="${N_SAMPLES_ENV:-$N_SAMPLES}"
SPLIT="${SPLIT_ENV:-$SPLIT}"
GAP_DAYS="${GAP_DAYS_ENV:-$GAP_DAYS}"
N_GAPS="${N_GAPS_ENV:-$N_GAPS}"
GPU_ID="${GPU_ID_ENV:-$GPU_ID}"
PYTHON="${PYTHON_ENV:-$PYTHON}"

# Derived paths
DATA_DIR="${OUTPUT_ROOT}/${RUN_NAME}/model_data"

# Set output directory based on denorm mode
case "${DENORM_MODE}" in
    normalized)
        VIZ_OUTPUT_DIR="${VIZ_OUTPUT_ROOT}/gap_filling_visualization"
        ;;
    ideal)
        VIZ_OUTPUT_DIR="${VIZ_OUTPUT_ROOT}/gap_filling_visualization_denorm"
        ;;
    operational)
        VIZ_OUTPUT_DIR="${VIZ_OUTPUT_ROOT}/gap_filling_visualization_operational"
        ;;
    *)
        echo "ERROR: Invalid DENORM_MODE: ${DENORM_MODE}"
        echo "Valid options: normalized, ideal, operational"
        exit 1
        ;;
esac

# Find latest experiment if MODEL_PATH not specified
if [ -z "${MODEL_PATH}" ]; then
    EXPERIMENT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/experiments"
    LATEST_EXP=$(ls -td ${EXPERIMENT_DIR}/*/ 2>/dev/null | head -1)
    if [ -n "${LATEST_EXP}" ]; then
        MODEL_PATH="${LATEST_EXP}best_model.keras"
    fi
fi

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log "========================================="
log "TreeNet AI Pipeline v2 - Stage 12: Visualize Gap Filling"
log "========================================="

# Create output directory
mkdir -p "${VIZ_OUTPUT_DIR}"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${PIPELINE_DIR}/12_visualize_gap_filling.py"
CMD="${CMD} --data-dir ${DATA_DIR}"
CMD="${CMD} --output-dir ${VIZ_OUTPUT_DIR}"
CMD="${CMD} --model-path ${MODEL_PATH}"
CMD="${CMD} --denorm ${DENORM_MODE}"
CMD="${CMD} --n-samples ${N_SAMPLES}"
CMD="${CMD} --split ${SPLIT}"
CMD="${CMD} --gap-days ${GAP_DAYS}"
CMD="${CMD} --n-gaps ${N_GAPS}"

# Handle dry-run
if [ "$1" = "--dry-run" ]; then
    log "DRY RUN - Would execute:"
    echo "$CMD"
    exit 0
fi

# Execute
log "Starting visualization..."
log "GPU: ${GPU_ID}"
log "Model: ${MODEL_PATH}"
log "Data: ${DATA_DIR}"
log "Output: ${VIZ_OUTPUT_DIR}"
log "Denorm Mode: ${DENORM_MODE}"
log "Samples: ${N_SAMPLES} (${SPLIT} split)"
log "Gap: ${GAP_DAYS} days, ${N_GAPS} gaps/segment"

# Run
cd "${PIPELINE_DIR}"
eval $CMD

log "========================================="
log "Visualization complete!"
log "Output: ${VIZ_OUTPUT_DIR}"
log "========================================="
