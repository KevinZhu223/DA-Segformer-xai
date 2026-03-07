"""
End-to-end XAI inference pipeline for DA-Segformer.

Uses **sliding-window (smooth-stitch) inference** — the same approach
that produced the validated 0.7461 mIoU results.  The model was trained
on 1024×1024 crops at native resolution, so we tile the full-resolution
image into overlapping 1024×1024 windows, run each tile through the
model (capturing attention per tile), and stitch both the softmax
predictions and the attention maps back together with overlap averaging.

Usage::

    from pipeline import XAIPipeline

    pipe = XAIPipeline()
    result = pipe.run("path/to/satellite.png")
    cv2.imwrite("panel.png", result.panel_main)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor

from attention_extractor import HookExtractor, StealthExtractor, build_extractor
from attention_processor import (
    class_masked_attention,
    compute_all_maps,
    normalize_map,
)
from config import (
    BUILDING_CLASSES,
    DAMAGE_CLASSES,
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    IGNORE_INDEX,
    IMAGE_SIZE,
    NUM_CLASSES,
    RESCUENET_SPLITS,
    SEGFORMER_NUM_STAGES,
)
from heatmap import (
    apply_colormap,
    bgr2rgb,
    blend,
    build_panel,
    colorize_mask,
    make_4panel,
    make_stage_panel,
    rgb2bgr,
)
from model_loader import load_model

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class XAIResult:
    """Container for all outputs from a single inference pass."""

    # Original image
    image_rgb: np.ndarray          # (H, W, 3) uint8 RGB
    image_bgr: np.ndarray          # (H, W, 3) uint8 BGR

    # Segmentation prediction
    pred_mask: np.ndarray           # (H, W) int — class IDs
    pred_colour_bgr: np.ndarray    # (H, W, 3) coloured mask

    # Attention maps (float32 [0, 1])
    attention_fused: np.ndarray     # (H, W) — combined across stages
    attention_stages: Dict[str, np.ndarray] = field(default_factory=dict)

    # Class-conditional attention (float32 [0, 1])
    attention_damage: Optional[np.ndarray] = None   # damage classes only
    attention_building: Optional[np.ndarray] = None  # all building classes

    # Pre-rendered panels (BGR uint8)
    panel_main: Optional[np.ndarray] = None      # 4-panel
    panel_stages: Optional[np.ndarray] = None    # per-stage panel
    panel_damage: Optional[np.ndarray] = None    # damage-focused panel

    # Metadata
    image_path: Optional[str] = None
    gt_mask: Optional[np.ndarray] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class XAIPipeline:
    """Full XAI pipeline with sliding-window inference.

    Parameters
    ----------
    checkpoint : path to HuggingFace checkpoint directory.
    device : ``"auto"`` / ``"cuda"`` / ``"cpu"``.
    image_size : tile size for sliding window (default 1024).
    stride : sliding window stride; default 768 (25% overlap).
    extraction_strategy : ``"hook"`` or ``"stealth"``.
    attention_method : ``"received"`` or ``"entropy"`` or ``"rollout"``.
    last_layer_only : process only the last block per stage (saves memory).
    colormap : colour map name for heatmaps.
    alpha : blend factor.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        image_size: int = IMAGE_SIZE,
        stride: int = 768,
        extraction_strategy: str = "hook",
        attention_method: str = "received",
        last_layer_only: bool = False,
        colormap: str = HEATMAP_COLORMAP,
        alpha: float = HEATMAP_ALPHA,
    ) -> None:
        self.image_size = image_size
        self.crop_size = image_size
        self.stride = stride
        self.attention_method = attention_method
        self.last_layer_only = last_layer_only
        self.colormap = colormap
        self.alpha = alpha

        # ── Load model ────────────────────────────────────────────────
        self.model, self._raw_processor, self.device = load_model(
            checkpoint=checkpoint, device=device, image_size=image_size,
        )

        # Build a tile-level processor: normalize only, NO resize
        # (we handle cropping/tiling ourselves at native resolution)
        self.tile_processor = SegformerImageProcessor(
            do_resize=False,
            do_normalize=True,
        )

        # ── Attach attention extractor ────────────────────────────────
        self.extractor = build_extractor(self.model, strategy=extraction_strategy)
        self._strategy = extraction_strategy
        logger.info(
            "Pipeline ready  [%s | %s | %s | crop=%d stride=%d].",
            self.device, extraction_strategy, attention_method,
            self.crop_size, self.stride,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2-D Hanning window for smooth tile blending
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _make_hanning_window(size: int) -> np.ndarray:
        """Create a 2-D Hanning (raised-cosine) window.

        The window tapers from 1.0 at the centre to ~0 at the edges,
        which eliminates hard tile-boundary artefacts when tiles
        overlap during stitching.
        """
        w1d = np.hanning(size)
        window = np.outer(w1d, w1d).astype(np.float64)
        # Normalise so peak = 1
        window /= window.max() + 1e-12
        return window

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Percentile + gamma normalisation for better contrast
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @staticmethod
    def _percentile_normalize(
        arr: np.ndarray,
        lo_pct: float = 1.0,
        hi_pct: float = 99.0,
        gamma: float = 0.7,
    ) -> np.ndarray:
        """Percentile-clipped min–max normalisation with gamma.

        1. Clip at the *lo_pct* / *hi_pct* percentiles so that rare
           extreme outliers don't crush the dynamic range.
        2. Min–max rescale to [0, 1].
        3. Apply ``x ** gamma`` (gamma < 1 → spreads low values, making
           the heatmap brighter; gamma = 0.7 balances visibility
           without washing out contrast).
        """
        lo = np.percentile(arr, lo_pct)
        hi = np.percentile(arr, hi_pct)
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        clipped = np.clip(arr, lo, hi)
        normed = ((clipped - lo) / (hi - lo)).astype(np.float32)
        return np.power(normed, gamma)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Single-tile inference (with attention capture)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @torch.no_grad()
    def _infer_tile(self, tile_np: np.ndarray):
        """Run one crop_size × crop_size tile.

        Returns
        -------
        probs : (C, crop, crop) tensor on device — softmax class probs.
        fused_raw : (crop, crop) numpy float64 — RAW (un-normalised)
            fused attention.  Normalisation happens only once after all
            tiles are stitched to avoid tile-boundary artefacts.
        stage_raws : dict of (crop, crop) numpy float64 per stage — RAW.
        """
        self.extractor.clear()

        inputs = self.tile_processor(images=tile_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        use_output_attentions = self._strategy == "hook"
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=use_output_attentions,
        )

        # Upsample logits to tile size and compute probabilities
        logits = outputs.logits  # (1, C, H/4, W/4)
        logits_up = F.interpolate(
            logits,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        probs = F.softmax(logits_up, dim=1).squeeze(0)  # (C, crop, crop)

        # Compute per-tile attention maps
        attn_by_stage = self.extractor.get_all_attentions_by_stage()
        attn_maps = compute_all_maps(
            attn_by_stage,
            image_size=self.crop_size,
            method=self.attention_method,
            last_layer_only=self.last_layer_only,
        )

        # ── Return RAW (un-normalised) attention ──────────────────────
        # Using "fused" (not "fused_norm") preserves relative magnitude
        # across tiles so that stitching doesn't create boundary artefacts.
        fused_raw = attn_maps["fused"].squeeze(0).cpu().numpy().astype(np.float64)
        # Upsample fused from (crop, crop) — it's already at crop_size
        # from fuse_multiscale, so no resize needed.

        stage_raws: Dict[str, np.ndarray] = {}
        for s in range(SEGFORMER_NUM_STAGES):
            key = f"stage_{s}"
            if key in attn_maps:
                smap = attn_maps[key].squeeze(0)
                smap_up = F.interpolate(
                    smap.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.crop_size, self.crop_size),
                    mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy().astype(np.float64)
                stage_raws[key] = smap_up

        return probs, fused_raw, stage_raws

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Sliding-window inference over full image
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @torch.no_grad()
    def _smooth_stitch(self, image_np: np.ndarray):
        """Sliding-window inference with overlapping tiles.

        Stitches both softmax probabilities AND attention maps
        using overlap-averaging, exactly like the validated
        viz_smooth_stitch.py approach.

        Parameters
        ----------
        image_np : (H, W, 3) uint8 RGB numpy array.

        Returns
        -------
        pred_mask : (H, W) int numpy array — argmax class IDs.
        fused_norm : (H, W) float32 numpy array — normalised [0, 1].
        stage_norms : dict of (H, W) float32 arrays per stage.
        """
        h, w, _ = image_np.shape
        crop = self.crop_size
        stride = self.stride

        # 2-D Hanning window for smooth attention blending
        hann = self._make_hanning_window(crop)

        # Accumulators
        prob_acc = torch.zeros((NUM_CLASSES, h, w), device=self.device)
        count_map = torch.zeros((1, h, w), device=self.device)

        # Attention accumulators use Hanning-weighted sums (CPU, float64)
        attn_acc_fused = np.zeros((h, w), dtype=np.float64)
        attn_acc_stages: Dict[str, np.ndarray] = {
            f"stage_{s}": np.zeros((h, w), dtype=np.float64)
            for s in range(SEGFORMER_NUM_STAGES)
        }
        attn_weight = np.zeros((h, w), dtype=np.float64)  # Hanning weight sum

        # ── Handle images smaller than one tile ───────────────────────
        if h <= crop and w <= crop:
            pad_h = max(0, crop - h)
            pad_w = max(0, crop - w)
            padded = cv2.copyMakeBorder(
                image_np, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            probs, fused_raw, stage_raws = self._infer_tile(padded)
            probs = probs[:, :h, :w]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

            fused_crop = fused_raw[:h, :w]
            fused_norm = self._percentile_normalize(fused_crop)
            stages: Dict[str, np.ndarray] = {}
            for key, smap in stage_raws.items():
                stages[key] = self._percentile_normalize(smap[:h, :w])
            return pred_mask, fused_norm, stages

        # ── Tile grid ─────────────────────────────────────────────────
        n_rows = math.ceil((h - crop) / stride) + 1
        n_cols = math.ceil((w - crop) / stride) + 1
        total_tiles = n_rows * n_cols
        logger.info(
            "Sliding window: %d×%d tiles (%d total), "
            "crop=%d stride=%d on %d×%d image.",
            n_rows, n_cols, total_tiles, crop, stride, h, w,
        )

        tile_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                y1 = int(r * stride)
                x1 = int(c * stride)
                y2 = min(y1 + crop, h)
                x2 = min(x1 + crop, w)

                # Clamp to keep tile exactly crop×crop
                if y2 - y1 < crop:
                    y1 = h - crop
                if x2 - x1 < crop:
                    x1 = w - crop
                y2, x2 = y1 + crop, x1 + crop

                tile = image_np[y1:y2, x1:x2]
                probs, fused_raw, stage_raws = self._infer_tile(tile)

                # Accumulate prediction probabilities (uniform weight)
                prob_acc[:, y1:y2, x1:x2] += probs
                count_map[:, y1:y2, x1:x2] += 1.0

                # Accumulate RAW attention with Hanning window weighting
                attn_acc_fused[y1:y2, x1:x2] += fused_raw * hann
                for key, smap in stage_raws.items():
                    attn_acc_stages[key][y1:y2, x1:x2] += smap * hann
                attn_weight[y1:y2, x1:x2] += hann

                tile_idx += 1
                if tile_idx % 5 == 0 or tile_idx == total_tiles:
                    logger.info("  tile %d/%d", tile_idx, total_tiles)

        # ── Average and argmax for prediction ─────────────────────────
        prob_acc /= count_map
        pred_mask = torch.argmax(prob_acc, dim=0).cpu().numpy().astype(np.uint8)

        # ── Hanning-weighted average → percentile normalisation ───────
        weight_safe = np.maximum(attn_weight, 1e-12)
        fused_avg = (attn_acc_fused / weight_safe).astype(np.float32)
        fused_norm = self._percentile_normalize(fused_avg)

        stage_norms: Dict[str, np.ndarray] = {}
        for s in range(SEGFORMER_NUM_STAGES):
            key = f"stage_{s}"
            smap = (attn_acc_stages[key] / weight_safe).astype(np.float32)
            stage_norms[key] = self._percentile_normalize(smap)

        return pred_mask, fused_norm, stage_norms

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def run(
        self,
        image_input,
        gt_mask: Optional[np.ndarray] = None,
        render_panels: bool = True,
    ) -> XAIResult:
        """Run the full XAI pipeline on a single image.

        Uses sliding-window inference at native resolution for accurate
        predictions (matching the validated training pipeline), while
        simultaneously extracting and stitching attention maps per tile.

        Parameters
        ----------
        image_input : str / Path / PIL.Image / ndarray (RGB)
            The input satellite image.
        gt_mask : optional (H, W) integer array — ground-truth labels.
        render_panels : whether to pre-render multi-panel composites.

        Returns
        -------
        XAIResult dataclass with all outputs.
        """
        image_path: Optional[str] = None

        # ── Load image ────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            pil_img = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        image_rgb = np.array(pil_img)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # ── Sliding-window inference + attention ──────────────────────
        pred_mask, fused_norm, stage_norms = self._smooth_stitch(image_rgb)

        pred_colour = colorize_mask(pred_mask)

        # ── Class-conditional attention ───────────────────────────────
        fused_t = torch.from_numpy(fused_norm)
        pred_t = torch.from_numpy(pred_mask.astype(np.int64))

        dmg_map = class_masked_attention(fused_t, pred_t, DAMAGE_CLASSES)
        bld_map = class_masked_attention(fused_t, pred_t, BUILDING_CLASSES)

        result = XAIResult(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            pred_mask=pred_mask,
            pred_colour_bgr=pred_colour,
            attention_fused=fused_norm,
            attention_stages=stage_norms,
            attention_damage=dmg_map.numpy(),
            attention_building=bld_map.numpy(),
            image_path=image_path,
            gt_mask=gt_mask,
        )

        # ── Render panels ─────────────────────────────────────────────
        if render_panels:
            result.panel_main = make_4panel(
                image_bgr, pred_mask, fused_norm,
                colormap=self.colormap, alpha=self.alpha,
                gt_mask=gt_mask,
            )
            result.panel_stages = make_stage_panel(
                image_bgr, stage_norms,
                colormap=self.colormap, alpha=self.alpha,
            )
            # Damage-focused panel
            dmg_np = dmg_map.numpy()
            if dmg_np.max() > 0:
                result.panel_damage = make_4panel(
                    image_bgr, pred_mask, dmg_np,
                    colormap=self.colormap, alpha=self.alpha,
                    gt_mask=gt_mask,
                )

        return result

    def run_batch(
        self,
        image_dir: str,
        output_dir: str,
        gt_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Process every image in a directory and save panels."""
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_dir_p = Path(image_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in image_dir_p.iterdir()
            if f.suffix.lower() in IMG_EXTS
        )
        if limit:
            files = files[:limit]

        saved: List[str] = []
        for i, fpath in enumerate(files):
            logger.info("[%d/%d] %s", i + 1, len(files), fpath.name)

            gt_mask = None
            if gt_dir:
                gt_path = Path(gt_dir) / f"{fpath.stem}_lab.png"
                if gt_path.exists():
                    gt_mask = np.array(Image.open(gt_path)).astype(np.int64)

            result = self.run(str(fpath), gt_mask=gt_mask)

            panel_path = out_path / f"{fpath.stem}_xai_panel.png"
            cv2.imwrite(str(panel_path), result.panel_main)
            saved.append(str(panel_path))

            stage_path = out_path / f"{fpath.stem}_xai_stages.png"
            cv2.imwrite(str(stage_path), result.panel_stages)
            saved.append(str(stage_path))

            if result.panel_damage is not None:
                dmg_path = out_path / f"{fpath.stem}_xai_damage.png"
                cv2.imwrite(str(dmg_path), result.panel_damage)
                saved.append(str(dmg_path))

            attn_path = out_path / f"{fpath.stem}_attention.npy"
            np.save(str(attn_path), result.attention_fused)

        logger.info("Saved %d files to %s.", len(saved), out_path)
        return saved
