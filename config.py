"""
Configuration for XAI Segformer Attention Extractor.

Centralises all paths, class definitions, architecture constants,
and visualisation parameters so every other module imports from here.
"""

import os
from typing import Dict, List, Tuple

# ── Runtime roots ─────────────────────────────────────────────────────────
# Inside Docker the volumes are /data and /working; on the bare host they
# live under DATA_VOLUME.  Override with environment variables if needed.
DATA_VOLUME = os.environ.get(
    "DATA_VOLUME", "/media/volume/Data_Kevin_Zhu"
)
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(DATA_VOLUME, ""))
WORKING_ROOT = os.environ.get(
    "WORKING_ROOT",
    os.path.join(DATA_VOLUME, "semseg_2d_code", "semseg_2d"),
)

# ── Best RescueNet DA-Segformer checkpoint (Epoch 255, mIoU 0.7461) ──────
DEFAULT_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    os.path.join(
        WORKING_ROOT,
        "runs",
        "rescuenet_final_b4_ohem_cosine_V2",
        "BEST_MODELS_ARCHIVE",
        "checkpoint-mIoU-0.7461-Ep255.0",
    ),
)

# ── RescueNet dataset ────────────────────────────────────────────────────
RESCUENET_ROOT = os.path.join(DATA_VOLUME, "RescueNet")
RESCUENET_SPLITS = {
    "train": {
        "images": os.path.join(RESCUENET_ROOT, "train-org-img"),
        "labels": os.path.join(RESCUENET_ROOT, "train-label-img"),
    },
    "val": {
        "images": os.path.join(RESCUENET_ROOT, "val-org-img"),
        "labels": os.path.join(RESCUENET_ROOT, "val-label-img"),
    },
    "test": {
        "images": os.path.join(RESCUENET_ROOT, "test-org-img"),
        "labels": os.path.join(RESCUENET_ROOT, "test-label-img"),
    },
}

# ── Class definitions (11 classes) ────────────────────────────────────────
NUM_CLASSES = 11
IGNORE_INDEX = 255

CLASS_NAMES: List[str] = [
    "Background",                  # 0
    "Water",                       # 1
    "Building_No_Damage",          # 2
    "Building_Minor_Damage",       # 3
    "Building_Major_Damage",       # 4
    "Building_Total_Destruction",  # 5
    "Vehicle",                     # 6
    "Road-Clear",                  # 7
    "Road-Blocked",                # 8
    "Tree",                        # 9
    "Pool",                        # 10
]

LABEL2ID: Dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
ID2LABEL: Dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}

# BGR palette (matches viz_smooth_stitch.py in the training codebase)
PALETTE_BGR: Dict[int, Tuple[int, int, int]] = {
    0:  (0,   0,   0),    # Background – Black
    1:  (255, 0,   0),    # Water – Blue
    2:  (20,  255, 20),   # Building No Damage – Green
    3:  (0,   215, 255),  # Building Minor Damage – Yellow
    4:  (0,   0,   255),  # Building Major Damage – Red
    5:  (0,   0,   139),  # Building Total Destruction – Dark Red
    6:  (128, 0,   128),  # Vehicle – Purple
    7:  (128, 128, 128),  # Road-Clear – Gray
    8:  (64,  64,  64),   # Road-Blocked – Dark Gray
    9:  (0,   100, 0),    # Tree – Dark Green
    10: (255, 128, 0),    # Pool – Orange-Blue
}

# RGB palette (for matplotlib / PIL / Streamlit)
PALETTE_RGB: Dict[int, Tuple[int, int, int]] = {
    k: (b, g, r) for k, (b, g, r) in PALETTE_BGR.items()
}

# Semantic class groups for focused XAI analysis
DAMAGE_CLASSES: List[int]   = [3, 4, 5]       # Minor / Major / Total Destruction
BUILDING_CLASSES: List[int] = [2, 3, 4, 5]    # All building classes
ROAD_CLASSES: List[int]     = [7, 8]           # Road clear + blocked

# ── SegFormer B4 architecture constants ───────────────────────────────────
IMAGE_SIZE = 1024                              # Default inference resolution

SEGFORMER_NUM_STAGES       = 4
SEGFORMER_DEPTHS           = [3, 8, 27, 3]    # Transformer blocks per stage
SEGFORMER_NUM_HEADS        = [1, 2, 5, 8]     # Attention heads per stage
SEGFORMER_SR_RATIOS        = [8, 4, 2, 1]     # Sequence-reduction ratios
SEGFORMER_HIDDEN_SIZES     = [64, 128, 320, 512]
SEGFORMER_SPATIAL_SCALES   = [4, 8, 16, 32]   # Downsample factor per stage

# ── Visualisation defaults ────────────────────────────────────────────────
HEATMAP_ALPHA    = 0.50          # Blend factor for overlay
HEATMAP_COLORMAP = "jet"         # cv2.applyColorMap identifier string
PANEL_BORDER_PX  = 4             # White border between panels
FONT_SCALE       = 0.7
FONT_THICKNESS   = 2
