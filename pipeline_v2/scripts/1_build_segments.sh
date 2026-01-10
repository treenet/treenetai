#!/bin/bash
#===============================================================================
# TreeNet AI Pipeline v2 - Stage 1: Build Segments
#===============================================================================
# This script builds 30-day segments from raw TreeNet sensor data.
#
# HOW TO USE:
#   1. Edit the USER CONFIGURATION section below
#   2. Run: ./scripts/1_build_segments.sh
#   3. Or dry-run first: ./scripts/1_build_segments.sh --dry-run
#
#===============================================================================

set -e  # Exit on error

#╔═══════════════════════════════════════════════════════════════════════════════╗
#║                          USER CONFIGURATION                                    ║
#║         Edit these values to customize the segment building process            ║
#╚═══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Run name: determines output folder name
# Convention: {country}_{scope}_{normalization}
# Examples: swiss_full_yearly_norm, swiss_subset_segment_norm
RUN_NAME="swiss_full_yearly_norm"

# Output root directory (all results stored here)
OUTPUT_ROOT="/storage/lukovic/Data/FORWARDS/treenet/processed"

# ─────────────────────────────────────────────────────────────────────────────────
# DATA SOURCE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Root directory containing raw sensor data
DATA_ROOT="/storage/lukovic/Data/FORWARDS/treenet/server_data"

# Root directory containing meteo data
METEO_ROOT="/storage/lukovic/Data/FORWARDS/treenet/server_data/meteo_data"

# Country filter: "Switzerland", "Netherlands", or "all"
COUNTRY="Switzerland"

# ─────────────────────────────────────────────────────────────────────────────────
# PROCESSING LIMITS
# ─────────────────────────────────────────────────────────────────────────────────
# Maximum number of sites to process (-1 = unlimited)
MAX_SITES=-1

# Maximum number of sensor combinations to process (-1 = unlimited)
# Use small numbers (e.g., 5) for testing
MAX_COMBINATIONS=-1

# ─────────────────────────────────────────────────────────────────────────────────
# SEGMENT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────────
# Length of each segment in days
SEGMENT_DAYS=30

# Overlap stride in days (smaller = more segments, more overlap)
STRIDE_DAYS=10

# ─────────────────────────────────────────────────────────────────────────────────
# NORMALIZATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────────
# Normalization method: "minmax" or "zscore"
NORM_METHOD="minmax"

# Normalization scope: "year" (annual min/max) or "segment" (per-segment)
# "year" is recommended for consistent scaling
NORM_SCOPE="year"

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
DATA_ROOT="${DATA_ROOT_ENV:-$DATA_ROOT}"
METEO_ROOT="${METEO_ROOT_ENV:-$METEO_ROOT}"
COUNTRY="${COUNTRY_ENV:-$COUNTRY}"
MAX_SITES="${MAX_SITES_ENV:-$MAX_SITES}"
MAX_COMBINATIONS="${MAX_COMBINATIONS_ENV:-$MAX_COMBINATIONS}"
SEGMENT_DAYS="${SEGMENT_DAYS_ENV:-$SEGMENT_DAYS}"
STRIDE_DAYS="${STRIDE_DAYS_ENV:-$STRIDE_DAYS}"
NORM_METHOD="${NORM_METHOD_ENV:-$NORM_METHOD}"
NORM_SCOPE="${NORM_SCOPE_ENV:-$NORM_SCOPE}"
VERBOSE="${VERBOSE_ENV:-$VERBOSE}"
PYTHON="${PYTHON_ENV:-$PYTHON}"

# ============================================================================
# DERIVED PATHS
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOGS_DIR="${OUTPUT_DIR}/logs"

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
# Segment Building Configuration
# Generated: $(date '+%Y-%m-%d %H:%M:%S')
#===============================================================================

[paths]
run_name = ${RUN_NAME}
output_root = ${OUTPUT_ROOT}
output_dir = ${OUTPUT_DIR}
data_root = ${DATA_ROOT}
meteo_root = ${METEO_ROOT}

[data_source]
country = ${COUNTRY}
max_sites = ${MAX_SITES}
max_combinations = ${MAX_COMBINATIONS}

[segment_parameters]
segment_days = ${SEGMENT_DAYS}
stride_days = ${STRIDE_DAYS}
resolution_minutes = 10
hourly_target_resolution = true

[normalization]
method = ${NORM_METHOD}
scope = ${NORM_SCOPE}

[processing]
verbose = ${VERBOSE}

[environment]
python = ${PYTHON}
script = ${PIPELINE_DIR}/1_build_segments.py
CONFIGEOF
    log "Configuration saved to: $config_file"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log "========================================="
log "TreeNet AI Pipeline v2 - Stage 1: Build Segments"
log "========================================="

# Create output directories
mkdir -p "${LOGS_DIR}"

# Save configuration
CONFIG_FILE="${OUTPUT_DIR}/segment_build_config.ini"
save_config "${CONFIG_FILE}"

# Build command
CMD="${PYTHON} ${PIPELINE_DIR}/1_build_segments.py"
CMD="${CMD} --run-name ${RUN_NAME}"
CMD="${CMD} --country ${COUNTRY}"
CMD="${CMD} --max-sites ${MAX_SITES}"
CMD="${CMD} --max-combinations ${MAX_COMBINATIONS}"

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
log "Starting segment building..."
log "Command: $CMD"
log "Output: ${OUTPUT_DIR}"

# Run with logging
cd "${PIPELINE_DIR}"
$CMD 2>&1 | tee "${LOGS_DIR}/build_console.log"

log "========================================="
log "Segment building complete!"
log "Output: ${OUTPUT_DIR}"
log "Config: ${CONFIG_FILE}"
log "Log: ${LOGS_DIR}/build_console.log"
log "========================================="
