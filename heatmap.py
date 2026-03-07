"""
Heatmap generation and multi-panel visualisation.

Converts normalised attention maps ([0, 1] float tensors) into
colour-mapped overlays suitable for papers and the Streamlit dashboard.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import (
    CLASS_NAMES,
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    NUM_CLASSES,
    PALETTE_BGR,
    PANEL_BORDER_PX,
    FONT_SCALE,
    FONT_THICKNESS,
)

# Map string colormap name → OpenCV constant
_CV2_COLORMAPS = {
    "jet":      cv2.COLORMAP_JET,
    "hot":      cv2.COLORMAP_HOT,
    "inferno":  cv2.COLORMAP_INFERNO,
    "magma":    cv2.COLORMAP_MAGMA,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "turbo":    cv2.COLORMAP_TURBO,
    "plasma":   cv2.COLORMAP_PLASMA,
    "bone":     cv2.COLORMAP_BONE,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core image utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip and convert a [0, 1] float array to uint8 [0, 255]."""
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def apply_colormap(
    attention_map: np.ndarray,
    colormap: str = HEATMAP_COLORMAP,
) -> np.ndarray:
    """Apply an OpenCV colour map to a [0, 1] single-channel float map.

    Parameters
    ----------
    attention_map : (H, W) float in [0, 1]
    colormap : name string (see ``_CV2_COLORMAPS``).

    Returns
    -------
    (H, W, 3) BGR uint8 heatmap.
    """
    cm = _CV2_COLORMAPS.get(colormap, cv2.COLORMAP_JET)
    gray = to_uint8(attention_map)
    return cv2.applyColorMap(gray, cm)


def blend(
    image_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """Alpha-blend a heatmap onto an image.

    Returns (H, W, 3) BGR uint8.
    """
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(hm, alpha, image_bgr, 1.0 - alpha, 0).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Segmentation mask → colour
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def colorize_mask(
    mask: np.ndarray,
    palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Convert integer class mask to a BGR colour image.

    Parameters
    ----------
    mask : (H, W) int array with values in [0, NUM_CLASSES-1].
    palette : class-id → (B, G, R).  Defaults to ``PALETTE_BGR``.

    Returns
    -------
    (H, W, 3) BGR uint8.
    """
    if palette is None:
        palette = PALETTE_BGR
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, colour in palette.items():
        canvas[mask == cid] = colour
    return canvas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Text overlay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def put_label(
    img: np.ndarray,
    text: str,
    position: str = "top",
) -> np.ndarray:
    """Burn a text label onto the image (modifies in-place & returns)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)
    pad = 6
    if position == "top":
        org = (pad, th + pad)
    else:
        org = (pad, img.shape[0] - pad)

    # Black outline for legibility
    cv2.putText(img, text, org, font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Multi-panel builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_panel(
    panels: List[Tuple[np.ndarray, str]],
    layout: str = "horizontal",
    border: int = PANEL_BORDER_PX,
) -> np.ndarray:
    """Compose labelled sub-images into a single panel.

    Parameters
    ----------
    panels : list of ``(image_bgr, title)`` tuples.
    layout : ``"horizontal"`` or ``"vertical"``.
    border : white border width in pixels.

    Returns
    -------
    (H, W, 3) BGR uint8 composite.
    """
    labelled: List[np.ndarray] = []
    target_h = panels[0][0].shape[0]

    for img, title in panels:
        # Resize all panels to the same height
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        frame = img.copy()
        put_label(frame, title)
        labelled.append(frame)

    if layout == "horizontal":
        sep = np.full((target_h, border, 3), 255, dtype=np.uint8)
        pieces: list = []
        for i, p in enumerate(labelled):
            if i > 0:
                pieces.append(sep)
            pieces.append(p)
        return np.concatenate(pieces, axis=1)
    else:
        target_w = max(p.shape[1] for p in labelled)
        sep = np.full((border, target_w, 3), 255, dtype=np.uint8)
        pieces = []
        for i, p in enumerate(labelled):
            if p.shape[1] < target_w:
                pad_w = target_w - p.shape[1]
                p = np.pad(p, ((0, 0), (0, pad_w), (0, 0)), constant_values=255)
            if i > 0:
                pieces.append(sep)
            pieces.append(p)
        return np.concatenate(pieces, axis=0)


def make_4panel(
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    attention_norm: np.ndarray,
    colormap: str = HEATMAP_COLORMAP,
    alpha: float = HEATMAP_ALPHA,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Standard 4- or 5-panel output.

    Panels: Original | [GT] | Prediction | Attention Heatmap | Overlay

    Parameters
    ----------
    image_bgr : (H, W, 3) original image in BGR.
    pred_mask : (H, W) integer class IDs.
    attention_norm : (H, W) float [0, 1] normalised attention.
    gt_mask : optional ground-truth mask for a 5th panel.

    Returns
    -------
    (H, W_total, 3) BGR uint8 composite image.
    """
    pred_colour = colorize_mask(pred_mask)
    heatmap = apply_colormap(attention_norm, colormap)
    heatmap_resized = cv2.resize(
        heatmap, (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = blend(image_bgr, heatmap_resized, alpha)

    panels: list = [
        (image_bgr, "Original"),
    ]
    if gt_mask is not None:
        panels.append((colorize_mask(gt_mask), "Ground Truth"))
    panels += [
        (pred_colour, "Prediction"),
        (heatmap_resized, "Attention Heatmap"),
        (overlay, "Overlay"),
    ]
    return build_panel(panels)


def make_stage_panel(
    image_bgr: np.ndarray,
    stage_maps_norm: Dict[str, np.ndarray],
    colormap: str = HEATMAP_COLORMAP,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """Per-stage attention panel: Original | Stage 0 | … | Stage 3.

    Parameters
    ----------
    stage_maps_norm : ``{"stage_0": (H, W), …}`` normalised [0, 1].

    Returns
    -------
    (H, W_total, 3) BGR composite.
    """
    h, w = image_bgr.shape[:2]
    panels: list = [(image_bgr, "Original")]
    for s in range(4):
        key = f"stage_{s}"
        if key not in stage_maps_norm:
            continue
        m = stage_maps_norm[key]
        hm = apply_colormap(m, colormap)
        hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = blend(image_bgr, hm_resized, alpha)
        panels.append((overlay, f"Stage {s}"))
    return build_panel(panels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BGR ↔ RGB helpers (for Streamlit / matplotlib / PIL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Legend builder (for dashboards)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_legend_image(
    height: int = 400,
    width: int = 250,
) -> np.ndarray:
    """Create a colour-legend image (BGR) for the RescueNet classes."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    row_h = height // (NUM_CLASSES + 1)
    for cid in range(NUM_CLASSES):
        y = cid * row_h + row_h // 2
        colour = PALETTE_BGR[cid]
        cv2.rectangle(canvas, (10, y - 8), (30, y + 8), colour, -1)
        cv2.rectangle(canvas, (10, y - 8), (30, y + 8), (0, 0, 0), 1)
        cv2.putText(
            canvas, CLASS_NAMES[cid], (38, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return canvas
