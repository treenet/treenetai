#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Stage 3: Evaluate Model
#===============================================================================
# This script evaluates a trained model and generates visualizations.
#
# HOW TO USE:
#   1. Edit the USER CONFIGURATION section below
#   2. Run: ./scripts/3_evaluate_model.sh
#   3. Or dry-run first: ./scripts/3_evaluate_model.sh --dry-run
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the evaluation process                  ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# DATA/OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Run name: determines input/output folder
RUN_NAME="swiss_full_yearly_norm"

# Output root directory
OUTPUT_ROOT="/storage/lukovic/Data/FORWARDS/treenet/processed"

# Experiment to evaluate: "latest" or specific folder name
# Examples: "latest", "20250110_122927", "baseline_20250110"
EXPERIMENT="latest"

# ─────────────────────────────────────────────────────────────────────────────────
# EVALUATION OPTIONS
# ─────────────────────────────────────────────────────────────────────────────────
# Number of test samples to evaluate
# Set to -1 to evaluate all test samples
N_SAMPLES=-1

# Generate visualizations: true/false
GENERATE_PLOTS="true"

# Number of plots to generate
# Only used if GENERATE_PLOTS=true
N_PLOTS=20

# ─────────────────────────────────────────────────────────────────────────────────
# HARDWARE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# GPU ID to use (0-5 for 6-GPU system)
# Set to -1 for CPU-only evaluation
GPU_ID=0

# ─────────────────────────────────────────────────────────────────────────────────
# PROCESSING OPTIONS
# ─────────────────────────────────────────────────────────────────────────────────
# Enable verbose output: true/false
VERBOSE="true"

# ─────────────────────────────────────────────────────────────────────────────────
# PYTHON ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────────
# Path to Python interpreter
PYTHON="/home/lukovic/pyenv/lamella/bin/python"

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                    END OF USER CONFIGURATION                                   ║
#║            Do not edit below this line unless you know what you're doing       ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# Override from environment if set
RUN_NAME="${RUN_NAME_ENV:-$RUN_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT_ENV:-$OUTPUT_ROOT}"
EXPERIMENT="${EXPERIMENT_ENV:-$EXPERIMENT}"
N_SAMPLES="${N_SAMPLES_ENV:-$N_SAMPLES}"
GENERATE_PLOTS="${GENERATE_PLOTS_ENV:-$GENERATE_PLOTS}"
N_PLOTS="${N_PLOTS_ENV:-$N_PLOTS}"
GPU_ID="${GPU_ID_ENV:-$GPU_ID}"
VERBOSE="${VERBOSE_ENV:-$VERBOSE}"
PYTHON="${PYTHON_ENV:-$PYTHON}"

# Derived paths
DATA_DIR="${OUTPUT_ROOT}/${RUN_NAME}/model_data"
EXPERIMENT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/experiments"

# ============================================================================
# DERIVED PATHS
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="${OUTPUT_ROOT}/${RUN_NAME}/logs"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

resolve_experiment() {
    if [ "${EXPERIMENT}" = "latest" ]; then
        RESOLVED_EXP=$(ls -td ${EXPERIMENT_DIR}/*/ 2>/dev/null | head -1)
        if [ -z "${RESOLVED_EXP}" ]; then
            log "ERROR: No experiments found in ${EXPERIMENT_DIR}"
            exit 1
        fi
    else
        RESOLVED_EXP="${EXPERIMENT_DIR}/${EXPERIMENT}"
        if [ ! -d "${RESOLVED_EXP}" ]; then
            log "ERROR: Experiment not found: ${RESOLVED_EXP}"
            exit 1
        fi
    fi
    echo "${RESOLVED_EXP}"
}

save_config() {
    local config_file="$1"
    local model_dir="$2"
    cat > "$config_file" << CONFIGEOF
#===============================================================================
# Evaluation Configuration
# Generated: $(date '+%Y-%m-%d %H:%M:%S')
#===============================================================================

[paths]
run_name = ${RUN_NAME}
data_dir = ${DATA_DIR}
experiment_dir = ${EXPERIMENT_DIR}
model_dir = ${model_dir}

[evaluation]
n_samples = ${N_SAMPLES}
generate_plots = ${GENERATE_PLOTS}
n_plots = ${N_PLOTS}

[hardware]
gpu_id = ${GPU_ID}

[processing]
verbose = ${VERBOSE}

[environment]
python = ${PYTHON}
script = ${PIPELINE_DIR}/3_evaluate_model.py
CONFIGEOF
    log "Configuration saved to: $config_file"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log "========================================="
log "TreeNet AI Pipeline v2 - Stage 3: Evaluate Model"
log "========================================="

# Resolve experiment path
MODEL_DIR=$(resolve_experiment)
log "Model directory: ${MODEL_DIR}"

# Create directories
mkdir -p "${LOGS_DIR}"

# Check for model file
MODEL_PATH="${MODEL_DIR}/best_model.keras"
if [ ! -f "${MODEL_PATH}" ]; then
    log "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Save configuration
CONFIG_FILE="${MODEL_DIR}/evaluation_config.ini"
save_config "${CONFIG_FILE}" "${MODEL_DIR}"

# Build command
if [ "${GPU_ID}" = "-1" ]; then
    CMD="${PYTHON} ${PIPELINE_DIR}/3_evaluate_model.py"
else
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${PIPELINE_DIR}/3_evaluate_model.py"
fi

CMD="${CMD} --model-path ${MODEL_PATH}"
CMD="${CMD} --data-dir ${DATA_DIR}"
CMD="${CMD} --output-dir ${MODEL_DIR}"

if [ "${N_SAMPLES}" != "-1" ]; then
    CMD="${CMD} --n-samples ${N_SAMPLES}"
fi

if [ "${GENERATE_PLOTS}" = "true" ]; then
    CMD="${CMD} --generate-plots"
    CMD="${CMD} --n-plots ${N_PLOTS}"
fi

if [ "${VERBOSE}" = "true" ]; then
    CMD="${CMD} --verbose"
fi

# Handle dry-run
if [ "$1" = "--dry-run" ]; then
    log "DRY RUN - Would execute:"
    echo "$CMD"
    echo ""
    log "Configuration:"
    cat "${CONFIG_FILE}"
    exit 0
fi

# Execute
log "Starting model evaluation..."
log "GPU: ${GPU_ID}"
log "Model: ${MODEL_PATH}"
log "Output: ${MODEL_DIR}"
log "Samples: ${N_SAMPLES} (-1 = all)"
log "Plots: ${GENERATE_PLOTS} (n=${N_PLOTS})"

# Run with logging
cd "${PIPELINE_DIR}"
eval $CMD 2>&1 | tee "${LOGS_DIR}/evaluation_console.log"

log "========================================="
log "Evaluation complete!"
log "Config: ${CONFIG_FILE}"
log "Log: ${LOGS_DIR}/evaluation_console.log"
log "Results: ${MODEL_DIR}/"
log "========================================="
