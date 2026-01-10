#!/bin/bash
# ==============================================================================
# 8_visualize_predictions.sh - Visualize Model Predictions
# ==============================================================================
# Visualizes model predictions compared to input and ground truth.
# Shows input signal, ground truth, and model predictions with error metrics.
#
# Usage:
#   ./scripts/8_visualize_predictions.sh
#
# Prerequisites:
#   - Trained model (best_model.keras) in experiment directory
#   - Processed segment data
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# USER CONFIGURATION - Modify these values as needed
# ==============================================================================

# ---- PATH CONFIGURATION ----
# Experiment directory (contains best_model.keras)
EXPERIMENT_DIR="/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/experiments/20260110_145342_with_attention"

# Data directory (default: parent of experiments/)
DATA_DIR=""  # Leave empty to auto-detect from experiment dir

# Output root directory for all figures
OUTPUT_ROOT="/home/lukovic/data/treenet"

# Output subdirectory structure: OUTPUT_ROOT/visualizations/predictions/<experiment_name>/
# Leave empty to use default experiment name from EXPERIMENT_DIR
OUTPUT_SUBDIR=""  # e.g., "attention_v1" or leave empty for auto

# ---- SAMPLING OPTIONS ----
# Number of samples to visualize
N_SAMPLES=5

# Dataset split to use
SPLIT="test"  # Options: "train", "test"

# Random seed for reproducible sample selection
SEED=42

# ---- VIEW OPTIONS ----
# Create zoomed view of first N days (leave empty to skip)
ZOOM_DAYS=7

# Channels to visualize (space-separated)
CHANNELS="local_T local_RH stem"  # Options: "local_T", "local_RH", "stem"

# ---- PLOT OPTIONS ----
# Figure size (width height in inches)
FIGSIZE_WIDTH=16
FIGSIZE_HEIGHT=10

# Figure DPI
DPI=150

# Show error band around predictions
SHOW_ERROR_BAND=true  # Options: true, false

# ==============================================================================
# END OF USER CONFIGURATION
# ==============================================================================

# Determine the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python command
PYTHON_CMD="${PYTHON_CMD:-python}"

# Build command
CMD="$PYTHON_CMD ${PROJECT_DIR}/8_visualize_predictions.py"
CMD="$CMD --experiment-dir ${EXPERIMENT_DIR}"
CMD="$CMD --n-samples ${N_SAMPLES}"
CMD="$CMD --split ${SPLIT}"
CMD="$CMD --seed ${SEED}"
CMD="$CMD --channels ${CHANNELS}"
CMD="$CMD --figsize ${FIGSIZE_WIDTH} ${FIGSIZE_HEIGHT}"
CMD="$CMD --dpi ${DPI}"

# Optional: data directory
if [ -n "$DATA_DIR" ]; then
    CMD="$CMD --data-dir ${DATA_DIR}"
fi

# Build output directory
if [ -n "$OUTPUT_ROOT" ]; then
    if [ -n "$OUTPUT_SUBDIR" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/visualizations/predictions/${OUTPUT_SUBDIR}"
    else
        # Extract experiment name from EXPERIMENT_DIR
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        OUTPUT_DIR="${OUTPUT_ROOT}/visualizations/predictions/${EXPERIMENT_NAME}"
    fi
    CMD="$CMD --output-dir ${OUTPUT_DIR}"
fi

# Optional: zoom days
if [ -n "$ZOOM_DAYS" ]; then
    CMD="$CMD --zoom-days ${ZOOM_DAYS}"
fi

# Optional: error band
if [ "$SHOW_ERROR_BAND" = false ]; then
    CMD="$CMD --no-error-band"
fi

# Print configuration
echo "=============================================================================="
echo "Prediction Visualization"
echo "=============================================================================="
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ${N_SAMPLES}"
echo "Split: ${SPLIT}"
echo "Channels: ${CHANNELS}"
if [ -n "$ZOOM_DAYS" ]; then
    echo "Zoom: First ${ZOOM_DAYS} days"
fi
echo "=============================================================================="
echo ""

# Run
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=============================================================================="
echo "Visualization complete!"
echo "=============================================================================="
