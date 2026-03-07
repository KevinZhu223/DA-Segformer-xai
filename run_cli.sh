#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_cli.sh — Run the XAI Attention Extractor CLI
#
# Usage:
#   ./run_cli.sh                          # single validation image (demo)
#   ./run_cli.sh --batch                  # first 10 val images
#   ./run_cli.sh --image /path/to/img.png # specific image
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Paths (adjust for bare host vs Docker) ─────────────────────────────
DATA_VOLUME="${DATA_VOLUME:-/media/volume/Data_Kevin_Zhu}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${DATA_VOLUME}/semseg_2d_code/semseg_2d/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0}"

IMG_DIR="${DATA_VOLUME}/RescueNet/val-org-img"
GT_DIR="${DATA_VOLUME}/RescueNet/val-label-img"
OUT_DIR="${SCRIPT_DIR}/xai_output"

# ── Parse mode ──────────────────────────────────────────────────────────
MODE="single"
CUSTOM_IMAGE=""
for arg in "$@"; do
    case "$arg" in
        --batch) MODE="batch" ;;
        --image) MODE="custom" ;;
        *)
            if [[ "$MODE" == "custom" && -z "$CUSTOM_IMAGE" ]]; then
                CUSTOM_IMAGE="$arg"
            fi
            ;;
    esac
done

# ── Environment ─────────────────────────────────────────────────────────
PYTHON_BIN="${SCRIPT_DIR}/xai_venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Virtual environment not found at $PYTHON_BIN"
    echo "Please run: python3 -m venv xai_venv && ./xai_venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  XAI Attention Extractor — DA-Segformer"
echo "  Checkpoint : ${MODEL_CHECKPOINT}"
echo "  Output     : ${OUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "batch" ]]; then
    echo "Mode: batch (first 10 images)"
    "$PYTHON_BIN" cli.py \
        --image-dir "$IMG_DIR" \
        --gt-dir    "$GT_DIR" \
        --output    "$OUT_DIR" \
        --checkpoint "$MODEL_CHECKPOINT" \
        --method     received \
        --limit      10

elif [[ "$MODE" == "custom" ]]; then
    echo "Mode: single image → $CUSTOM_IMAGE"
    "$PYTHON_BIN" cli.py \
        --image      "$CUSTOM_IMAGE" \
        --gt-dir     "$GT_DIR" \
        --output     "$OUT_DIR" \
        --checkpoint "$MODEL_CHECKPOINT" \
        --method     received

else
    # Default: pick the first image in the val set
    FIRST_IMG=$(find "$IMG_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' \) -print -quit 2>/dev/null)
    if [[ -z "$FIRST_IMG" ]]; then
        echo "ERROR: No images found in $IMG_DIR"
        exit 1
    fi
    echo "Mode: single image (demo) → $FIRST_IMG"
    "$PYTHON_BIN" cli.py \
        --image      "$FIRST_IMG" \
        --gt-dir     "$GT_DIR" \
        --output     "$OUT_DIR" \
        --checkpoint "$MODEL_CHECKPOINT" \
        --method     received
fi

echo ""
echo "✓ Results saved to ${OUT_DIR}"
