#!/bin/bash
# ==============================================================================
# 7_visualize_reconstruction.sh - Visualize Gap-Filling Results
# ==============================================================================
# Visualizes gap-filling reconstruction results by comparing original 
# and reconstructed time series.
#
# Usage:
#   ./scripts/7_visualize_reconstruction.sh
#
# Prerequisites:
#   - Reconstructed time series .ftr file from model predictions
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# USER CONFIGURATION - Modify these values as needed
# ==============================================================================

# ---- PATH CONFIGURATION ----
# Path to reconstructed time series file (REQUIRED)
RECONSTRUCTED_PATH="/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/reconstructed/reconstructed_site3.ftr"

# Directory with original time series files
ORIGINAL_DIR="/storage/lukovic/Data/FORWARDS/treenet/processed/swiss_full_yearly_norm/model_data/intermediate_timeseries"

# Directory with raw server data (for gap detection)
RAW_DATA_DIR="/storage/lukovic/Data/FORWARDS/treenet/server_data"

# Output root directory for all figures
OUTPUT_ROOT="/home/lukovic/data/treenet"

# Output subdirectory
OUTPUT_SUBDIR="reconstruction"  # Full path: OUTPUT_ROOT/visualizations/reconstruction/

# ---- SENSOR IDS ----
# Site configuration
SITE_ID=3
THERMO_ID=9      # Thermometer series ID
HYGRO_ID=7       # Hygrometer series ID
DENDRO_ID=18     # Dendrometer series ID

# ---- DATE RANGE ----
# Optional: filter visualization to date range (leave empty for full range)
START_DATE=""    # Format: YYYY-MM-DD
END_DATE=""      # Format: YYYY-MM-DD

# ---- OPTIONS ----
# Show filled gap regions highlighted
SHOW_GAPS=true   # Options: true, false

# ==============================================================================
# END OF USER CONFIGURATION
# ==============================================================================

# Determine the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python command
PYTHON_CMD="${PYTHON_CMD:-python}"

# Build output directory
OUTPUT_DIR="${OUTPUT_ROOT}/visualizations/${OUTPUT_SUBDIR}"

# Build command
CMD="$PYTHON_CMD ${PROJECT_DIR}/7_visualize_reconstruction.py"
CMD="$CMD --reconstructed-path ${RECONSTRUCTED_PATH}"
CMD="$CMD --original-dir ${ORIGINAL_DIR}"
CMD="$CMD --raw-data-dir ${RAW_DATA_DIR}"
CMD="$CMD --output-dir ${OUTPUT_DIR}"
CMD="$CMD --site-id ${SITE_ID}"
CMD="$CMD --thermo-id ${THERMO_ID}"
CMD="$CMD --hygro-id ${HYGRO_ID}"
CMD="$CMD --dendro-id ${DENDRO_ID}"

# Optional: date range
if [ -n "$START_DATE" ]; then
    CMD="$CMD --start-date ${START_DATE}"
fi
if [ -n "$END_DATE" ]; then
    CMD="$CMD --end-date ${END_DATE}"
fi

# Optional: show gaps
if [ "$SHOW_GAPS" = true ]; then
    CMD="$CMD --show-gaps"
fi

# Print configuration
echo "=============================================================================="
echo "Gap-Filling Reconstruction Visualization"
echo "=============================================================================="
echo "Reconstructed: ${RECONSTRUCTED_PATH}"
echo "Original: ${ORIGINAL_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Site: ${SITE_ID} (T:${THERMO_ID}, H:${HYGRO_ID}, D:${DENDRO_ID})"
if [ -n "$START_DATE" ]; then
    echo "Date Range: ${START_DATE} to ${END_DATE}"
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
