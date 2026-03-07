#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_dashboard.sh — Launch the Streamlit XAI Dashboard
#
# Usage:
#   ./run_dashboard.sh              # default settings
#   ./run_dashboard.sh --port 8502  # custom port
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Paths ──────────────────────────────────────────────────────────────
DATA_VOLUME="${DATA_VOLUME:-/media/volume/Data_Kevin_Zhu}"
export DATA_VOLUME
export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${DATA_VOLUME}/semseg_2d_code/semseg_2d/runs/rescuenet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7461-Ep255.0}"

PORT="${1:-8501}"
# Strip --port flag if provided
if [[ "$PORT" == "--port" ]]; then
    PORT="${2:-8501}"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  XAI Dashboard — DA-Segformer"
echo "  Checkpoint: ${MODEL_CHECKPOINT}"
echo "  URL:        http://localhost:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

./xai_venv/bin/streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
