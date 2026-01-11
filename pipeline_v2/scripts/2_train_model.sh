#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Stage 2: Train Model
#===============================================================================
# This script trains the TCN model for gap filling.
#
# HOW TO USE:
#   1. Edit the USER CONFIGURATION section below
#   2. Run: ./scripts/2_train_model.sh
#   3. Or dry-run first: ./scripts/2_train_model.sh --dry-run
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the training process                    ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# DATA/OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Run name: determines input/output folder
# Should match the run_name used in segment building
RUN_NAME="swiss_segment_norm_all_combos"

# Output root directory
OUTPUT_ROOT="/storage/lukovic/Data/FORWARDS/treenet/processed"

# Optional: custom experiment name (leave empty for timestamp only)
# Examples: "baseline", "larger_model", "test_1"
EXPERIMENT_NAME="segment_norm_attention"

# ─────────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────────
# Number of convolutional filters
# Options: 32, 64, 128, 256
# Higher = more capacity but slower training
N_FILTERS=128

# Convolutional kernel size
# Options: 3, 5, 7
# Larger = larger receptive field per layer
KERNEL_SIZE=3

# Number of TCN blocks (depth of network)
# Options: 2, 3, 4, 5, 6
# More blocks = larger receptive field, but more parameters
N_BLOCKS=5

# Dropout rate for regularization
# Options: 0.1, 0.2, 0.3, 0.4
# Higher = more regularization (helps prevent overfitting)
DROPOUT=0.2

# ─────────────────────────────────────────────────────────────────────────────────
# ATTENTION MECHANISM (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────────
# Enable multi-head attention after TCN encoder: true/false
# Attention adds global context - can "see" entire time series
# Recommended: try both with and without, compare results
USE_ATTENTION="true"

# Number of attention heads
# Options: 2, 4, 8
# More heads = more diverse attention patterns
N_ATTENTION_HEADS=8

# Attention key dimension (per head)
# Options: 16, 32, 64
# Higher = more expressive but more parameters
ATTENTION_KEY_DIM=64

# ─────────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────────
# Number of training epochs
# Typical: 50-200
EPOCHS=100

# Batch size
# Options: 8, 16, 32, 64
# GPU memory constraints: 16-32 for single RTX 3090
BATCH_SIZE=16

# Initial learning rate
# Options: 0.001, 0.0003, 0.0001
# Lower = slower but more stable learning
LEARNING_RATE=0.0003

# ─────────────────────────────────────────────────────────────────────────────────
# GAP INJECTION SETTINGS (DATA AUGMENTATION)
# ─────────────────────────────────────────────────────────────────────────────────
# Enable gap injection during training: true/false
# Recommended: true (improves gap-filling performance)
ENABLE_GAPS="true"

# Minimum gap length in days
MIN_GAP_DAYS=1

# Maximum gap length in days
# Typical: 7, 12, 15
MAX_GAP_DAYS=12

# ─────────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING & LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────────
# Early stopping patience (epochs without improvement before stopping)
EARLY_STOP_PATIENCE=10

# Reduce LR patience (epochs without improvement before reducing LR)
REDUCE_LR_PATIENCE=4

# Minimum learning rate (stop reducing below this)
MIN_LR=0.000001

# ─────────────────────────────────────────────────────────────────────────────────
# HARDWARE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# GPU ID to use (0-5 for 6-GPU system)
# Check availability with: nvidia-smi
GPU_ID=2

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
EXPERIMENT_NAME="${EXPERIMENT_NAME_ENV:-$EXPERIMENT_NAME}"
N_FILTERS="${N_FILTERS_ENV:-$N_FILTERS}"
KERNEL_SIZE="${KERNEL_SIZE_ENV:-$KERNEL_SIZE}"
N_BLOCKS="${N_BLOCKS_ENV:-$N_BLOCKS}"
DROPOUT="${DROPOUT_ENV:-$DROPOUT}"
USE_ATTENTION="${USE_ATTENTION_ENV:-$USE_ATTENTION}"
N_ATTENTION_HEADS="${N_ATTENTION_HEADS_ENV:-$N_ATTENTION_HEADS}"
ATTENTION_KEY_DIM="${ATTENTION_KEY_DIM_ENV:-$ATTENTION_KEY_DIM}"
EPOCHS="${EPOCHS_ENV:-$EPOCHS}"
BATCH_SIZE="${BATCH_SIZE_ENV:-$BATCH_SIZE}"
LEARNING_RATE="${LEARNING_RATE_ENV:-$LEARNING_RATE}"
ENABLE_GAPS="${ENABLE_GAPS_ENV:-$ENABLE_GAPS}"
MIN_GAP_DAYS="${MIN_GAP_DAYS_ENV:-$MIN_GAP_DAYS}"
MAX_GAP_DAYS="${MAX_GAP_DAYS_ENV:-$MAX_GAP_DAYS}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE_ENV:-$EARLY_STOP_PATIENCE}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE_ENV:-$REDUCE_LR_PATIENCE}"
MIN_LR="${MIN_LR_ENV:-$MIN_LR}"
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

save_config() {
    local config_file="$1"
    cat > "$config_file" << CONFIGEOF
#===============================================================================
# Training Configuration
# Generated: $(date '+%Y-%m-%d %H:%M:%S')
#===============================================================================

[paths]
run_name = ${RUN_NAME}
data_dir = ${DATA_DIR}
experiment_dir = ${EXPERIMENT_DIR}
experiment_name = ${EXPERIMENT_NAME}

[model_architecture]
model_type = tcn
n_filters = ${N_FILTERS}
kernel_size = ${KERNEL_SIZE}
n_blocks = ${N_BLOCKS}
dropout_rate = ${DROPOUT}

[attention]
use_attention = ${USE_ATTENTION}
n_attention_heads = ${N_ATTENTION_HEADS}
attention_key_dim = ${ATTENTION_KEY_DIM}

[training]
epochs = ${EPOCHS}
batch_size = ${BATCH_SIZE}
learning_rate = ${LEARNING_RATE}

[gap_injection]
enabled = ${ENABLE_GAPS}
min_gap_days = ${MIN_GAP_DAYS}
max_gap_days = ${MAX_GAP_DAYS}

[callbacks]
early_stop_patience = ${EARLY_STOP_PATIENCE}
reduce_lr_patience = ${REDUCE_LR_PATIENCE}
min_lr = ${MIN_LR}

[hardware]
gpu_id = ${GPU_ID}

[processing]
verbose = ${VERBOSE}

[environment]
python = ${PYTHON}
script = ${PIPELINE_DIR}/2_train_model.py
CONFIGEOF
    log "Configuration saved to: $config_file"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log "========================================="
log "TreeNet AI Pipeline v2 - Stage 2: Train Model"
log "========================================="

# Create directories
mkdir -p "${LOGS_DIR}"
mkdir -p "${EXPERIMENT_DIR}"

# Save configuration (both in logs and experiment dir)
CONFIG_FILE="${OUTPUT_ROOT}/${RUN_NAME}/training_config.ini"
save_config "${CONFIG_FILE}"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${PIPELINE_DIR}/2_train_model.py"
CMD="${CMD} --data-dir ${DATA_DIR}"
CMD="${CMD} --output-dir ${EXPERIMENT_DIR}"
CMD="${CMD} --n-filters ${N_FILTERS}"
CMD="${CMD} --kernel-size ${KERNEL_SIZE}"
CMD="${CMD} --n-blocks ${N_BLOCKS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --learning-rate ${LEARNING_RATE}"
CMD="${CMD} --min-gap-days ${MIN_GAP_DAYS}"
CMD="${CMD} --max-gap-days ${MAX_GAP_DAYS}"

if [ -n "${EXPERIMENT_NAME}" ]; then
    CMD="${CMD} --experiment-name ${EXPERIMENT_NAME}"
fi

if [ "${ENABLE_GAPS}" = "false" ]; then
    CMD="${CMD} --no-gaps"
fi

if [ "${USE_ATTENTION}" = "true" ]; then
    CMD="${CMD} --use-attention"
    CMD="${CMD} --n-attention-heads ${N_ATTENTION_HEADS}"
    CMD="${CMD} --attention-key-dim ${ATTENTION_KEY_DIM}"
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
log "Starting model training..."
log "GPU: ${GPU_ID}"
log "Data: ${DATA_DIR}"
log "Output: ${EXPERIMENT_DIR}"

# Print hyperparameters summary
log "Hyperparameters:"
log "  Architecture: ${N_FILTERS} filters, ${N_BLOCKS} blocks, kernel=${KERNEL_SIZE}, dropout=${DROPOUT}"
log "  Attention: ${USE_ATTENTION} (heads=${N_ATTENTION_HEADS}, key_dim=${ATTENTION_KEY_DIM})"
log "  Training: ${EPOCHS} epochs, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
log "  Gaps: ${MIN_GAP_DAYS}-${MAX_GAP_DAYS} days (enabled: ${ENABLE_GAPS})"

# Run with logging
cd "${PIPELINE_DIR}"
eval $CMD 2>&1 | tee "${LOGS_DIR}/training_console.log"

# Copy config to experiment directory (after experiment dir is created)
LATEST_EXP=$(ls -td ${EXPERIMENT_DIR}/*/ 2>/dev/null | head -1)
if [ -n "${LATEST_EXP}" ]; then
    cp "${CONFIG_FILE}" "${LATEST_EXP}/training_config.ini"
    log "Config copied to: ${LATEST_EXP}/training_config.ini"
fi

log "========================================="
log "Training complete!"
log "Config: ${CONFIG_FILE}"
log "Log: ${LOGS_DIR}/training_console.log"
log "========================================="
