#!/bin/bash
# ==============================================================================
# 4_visualize_segments.sh - Visualize Processed Segments
# ==============================================================================
# Visualizes processed segments showing input and target channels.
# Creates plots for specific sites/years or summary statistics.
#
# Usage:
#   ./scripts/4_visualize_segments.sh
#
# Prerequisites:
#   - Processed segments in model_data directory
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# USER CONFIGURATION - Modify these values as needed
# ==============================================================================

# ---- PATH CONFIGURATION ----
# Data directory containing processed segments
DATA_DIR="/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/model_data"

# Output root directory for all figures
OUTPUT_ROOT="/home/lukovic/data/treenet"

# Output subdirectory structure: OUTPUT_ROOT/visualizations/segments/<dataset_name>/
OUTPUT_SUBDIR="swiss_full_yearly_norm"

# ---- MODE CONFIGURATION ----
# Choose ONE mode: "site", "summary", or "all"
MODE="site"  # Options: "site", "summary", "all"

# For site mode: specific site and year to visualize
SITE_ID=1
YEAR=2021

# ---- DATA SPLIT ----
SPLIT="train"  # Options: "train", "test"

# ---- OPTIONS ----
# Maximum segments to plot per site
MAX_PER_SITE=10

# Show global channels on input plots
SHOW_GLOBALS=true  # Options: true, false

# ==============================================================================
# END OF USER CONFIGURATION
# ==============================================================================

# Determine the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python command
PYTHON_CMD="${PYTHON_CMD:-python}"

# Build output directory
OUTPUT_DIR="${OUTPUT_ROOT}/visualizations/segments/${OUTPUT_SUBDIR}"

# Build command
CMD="$PYTHON_CMD ${PROJECT_DIR}/4_visualize_segments.py"
CMD="$CMD --data-dir ${DATA_DIR}"
CMD="$CMD --output-dir ${OUTPUT_DIR}"
CMD="$CMD --split ${SPLIT}"
CMD="$CMD --max-per-site ${MAX_PER_SITE}"

# Add mode-specific arguments
case $MODE in
    "site")
        CMD="$CMD --site ${SITE_ID} --year ${YEAR}"
        ;;
    "summary")
        CMD="$CMD --summary"
        ;;
    "all")
        CMD="$CMD --all"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'. Use 'site', 'summary', or 'all'."
        exit 1
        ;;
esac

# Optional: hide global channels
if [ "$SHOW_GLOBALS" = false ]; then
    CMD="$CMD --no-globals"
fi

# Print configuration
echo "=============================================================================="
echo "Segment Visualization"
echo "=============================================================================="
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Mode: ${MODE}"
if [ "$MODE" = "site" ]; then
    echo "Site: ${SITE_ID}, Year: ${YEAR}"
fi
echo "Split: ${SPLIT}"
echo "Max per site: ${MAX_PER_SITE}"
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
