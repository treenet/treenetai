#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Stage 6: Reconstruct Time Series
#===============================================================================
# This script reconstructs/gap-fills complete time series using a trained model.
#
# HOW TO USE:
#   1. Edit the USER CONFIGURATION section below
#   2. Run: ./scripts/6_reconstruct_timeseries.sh
#   3. Or dry-run first: ./scripts/6_reconstruct_timeseries.sh --dry-run
#
# NEW FLAGS (2026-01-12):
#   --align-stem: Compute scale/offset from Nov-Dec overlap for correct stem scale
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the reconstruction process              ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# DATA/OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Run name: determines input folder
RUN_NAME="swiss_segment_norm_all_combos"

# Output root directory
OUTPUT_ROOT="/storage/lukovic/Data/FORWARDS/treenet/processed"

# ─────────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Path to trained model
# Leave empty to use the latest experiment's best model
MODEL_PATH=""
# Example: "/storage/lukovic/.../experiments/20260111_152352_segment_norm_attention/best_model.keras"

# ─────────────────────────────────────────────────────────────────────────────────
# SITE SELECTION
# ─────────────────────────────────────────────────────────────────────────────────
# Site ID to reconstruct (leave empty to process all available sites)
SITE_ID=""
# Example: SITE_ID="3"

# ─────────────────────────────────────────────────────────────────────────────────
# ALIGNMENT OPTIONS (NEW 2026-01-12)
# ─────────────────────────────────────────────────────────────────────────────────
# Enable stem alignment: true/false
# Uses Nov-Dec overlap period to compute optimal scale/offset for stem channel
# Recommended: true (ensures correct absolute scale for stem)
ALIGN_STEM="true"

# ─────────────────────────────────────────────────────────────────────────────────
# HARDWARE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# GPU ID to use (0-5 for 6-GPU system)
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
MODEL_PATH="${MODEL_PATH_ENV:-$MODEL_PATH}"
SITE_ID="${SITE_ID_ENV:-$SITE_ID}"
ALIGN_STEM="${ALIGN_STEM_ENV:-$ALIGN_STEM}"
GPU_ID="${GPU_ID_ENV:-$GPU_ID}"
PYTHON="${PYTHON_ENV:-$PYTHON}"

# Derived paths
DATA_DIR="${OUTPUT_ROOT}/${RUN_NAME}/model_data"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/reconstructed"

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
log "TreeNet AI Pipeline v2 - Stage 6: Reconstruct Time Series"
log "========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${PIPELINE_DIR}/6_reconstruct_timeseries.py"
CMD="${CMD} --data-dir ${DATA_DIR}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --model-path ${MODEL_PATH}"

if [ -n "${SITE_ID}" ]; then
    CMD="${CMD} --site-id ${SITE_ID}"
fi

# NEW FLAG (2026-01-12)
if [ "${ALIGN_STEM}" = "true" ]; then
    CMD="${CMD} --align-stem"
fi

# Handle dry-run
if [ "$1" = "--dry-run" ]; then
    log "DRY RUN - Would execute:"
    echo "$CMD"
    exit 0
fi

# Execute
log "Starting reconstruction..."
log "GPU: ${GPU_ID}"
log "Model: ${MODEL_PATH}"
log "Data: ${DATA_DIR}"
log "Output: ${OUTPUT_DIR}"
log "Align Stem: ${ALIGN_STEM}"
if [ -n "${SITE_ID}" ]; then
    log "Site ID: ${SITE_ID}"
fi

# Run
cd "${PIPELINE_DIR}"
eval $CMD

log "========================================="
log "Reconstruction complete!"
log "Output: ${OUTPUT_DIR}"
log "========================================="
