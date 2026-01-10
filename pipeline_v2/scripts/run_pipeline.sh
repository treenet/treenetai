#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Master Pipeline Runner
#===============================================================================
# This script runs all pipeline stages in sequence.
#
# HOW TO USE:
#   1. Configure each stage in its respective script:
#      - scripts/1_build_segments.sh
#      - scripts/2_train_model.sh
#      - scripts/3_evaluate_model.sh
#   2. Select which stages to run below
#   3. Run: ./scripts/run_pipeline.sh
#   4. Or dry-run first: ./scripts/run_pipeline.sh --dry-run
#
# NOTE: For full control, run each stage script individually!
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the pipeline execution                  ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE SELECTION
# ─────────────────────────────────────────────────────────────────────────────────
# Which stages to run: true/false
# Set to "true" to run, "false" to skip
RUN_STAGE_1="true"   # Build segments
RUN_STAGE_2="true"   # Train model
RUN_STAGE_3="true"   # Evaluate model

# ─────────────────────────────────────────────────────────────────────────────────
# PAUSE BETWEEN STAGES
# ─────────────────────────────────────────────────────────────────────────────────
# Wait for user confirmation between stages: true/false
PAUSE_BETWEEN_STAGES="false"

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                    END OF USER CONFIGURATION                                   ║
#║                                                                                 ║
#║  NOTE: All other parameters should be configured in the individual stage       ║
#║        scripts: 1_build_segments.sh, 2_train_model.sh, 3_evaluate_model.sh     ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# Override from environment if set
RUN_STAGE_1="${RUN_STAGE_1_ENV:-$RUN_STAGE_1}"
RUN_STAGE_2="${RUN_STAGE_2_ENV:-$RUN_STAGE_2}"
RUN_STAGE_3="${RUN_STAGE_3_ENV:-$RUN_STAGE_3}"
PAUSE_BETWEEN_STAGES="${PAUSE_BETWEEN_STAGES_ENV:-$PAUSE_BETWEEN_STAGES}"

# ============================================================================
# DERIVED PATHS
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║ [$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

pause_if_enabled() {
    if [ "${PAUSE_BETWEEN_STAGES}" = "true" ]; then
        echo ""
        read -p "Press Enter to continue to the next stage, or Ctrl+C to abort... "
        echo ""
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log "TreeNet AI Pipeline v2 - Full Pipeline"

echo "Stage selection:"
echo "  [1] Build segments: ${RUN_STAGE_1}"
echo "  [2] Train model:    ${RUN_STAGE_2}"
echo "  [3] Evaluate model: ${RUN_STAGE_3}"
echo ""
echo "Pause between stages: ${PAUSE_BETWEEN_STAGES}"
echo ""

# Handle dry-run
DRY_RUN_FLAG=""
if [ "$1" = "--dry-run" ]; then
    DRY_RUN_FLAG="--dry-run"
    echo "DRY RUN MODE - Commands will be shown but not executed"
    echo ""
fi

# ============================================================================
# STAGE 1: BUILD SEGMENTS
# ============================================================================
if [ "${RUN_STAGE_1}" = "true" ]; then
    log "STAGE 1: Building Segments"
    "${SCRIPT_DIR}/1_build_segments.sh" ${DRY_RUN_FLAG}
    pause_if_enabled
else
    log "STAGE 1: Skipped (RUN_STAGE_1=false)"
fi

# ============================================================================
# STAGE 2: TRAIN MODEL
# ============================================================================
if [ "${RUN_STAGE_2}" = "true" ]; then
    log "STAGE 2: Training Model"
    "${SCRIPT_DIR}/2_train_model.sh" ${DRY_RUN_FLAG}
    pause_if_enabled
else
    log "STAGE 2: Skipped (RUN_STAGE_2=false)"
fi

# ============================================================================
# STAGE 3: EVALUATE MODEL
# ============================================================================
if [ "${RUN_STAGE_3}" = "true" ]; then
    log "STAGE 3: Evaluating Model"
    "${SCRIPT_DIR}/3_evaluate_model.sh" ${DRY_RUN_FLAG}
else
    log "STAGE 3: Skipped (RUN_STAGE_3=false)"
fi

# ============================================================================
# COMPLETE
# ============================================================================
log "Pipeline Complete!"

echo "Summary:"
echo "  Stage 1 (Build segments): ${RUN_STAGE_1}"
echo "  Stage 2 (Train model):    ${RUN_STAGE_2}"
echo "  Stage 3 (Evaluate model): ${RUN_STAGE_3}"
echo ""
echo "For detailed logs, check the logs/ directory in your output folder."
