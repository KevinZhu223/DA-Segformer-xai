#!/usr/bin/env python3
"""
Command-line interface for the XAI Attention Extractor.

Supports single-image and batch-directory modes.

Examples
--------
Single image::

    python cli.py --image /data/RescueNet/val-org-img/7635.jpg \\
                  --output ./xai_output

Batch::

    python cli.py --image-dir /data/RescueNet/val-org-img \\
                  --gt-dir   /data/RescueNet/val-label-img \\
                  --output   ./xai_output \\
                  --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# Ensure local imports work when invoked from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_CHECKPOINT,
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    IMAGE_SIZE,
    RESCUENET_SPLITS,
)
from pipeline import XAIPipeline

logger = logging.getLogger("xai_segformer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XAI Attention Extractor for DA-Segformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Input ─────────────────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", type=str, help="Path to a single image.")
    inp.add_argument("--image-dir", type=str, help="Directory of images (batch mode).")

    p.add_argument("--gt-dir", type=str, default=None,
                   help="Directory of ground-truth label masks (*_lab.png).")

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument("--output", "-o", type=str, default="./xai_output",
                   help="Output directory.")

    # ── Model ─────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="HuggingFace checkpoint directory.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device.")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help="Inference resolution (square).")

    # ── Extraction settings ───────────────────────────────────────────
    p.add_argument("--strategy", type=str, default="hook",
                   choices=["hook", "stealth"],
                   help="Attention extraction strategy.")
    p.add_argument("--method", type=str, default="received",
                   choices=["received", "entropy", "rollout"],
                   help="Attention aggregation method.")
    p.add_argument("--last-layer-only", action="store_true",
                   help="Only use the last transformer block per stage.")

    # ── Visualisation ─────────────────────────────────────────────────
    p.add_argument("--colormap", type=str, default=HEATMAP_COLORMAP,
                   help="Heatmap colour map (jet, hot, turbo, …).")
    p.add_argument("--alpha", type=float, default=HEATMAP_ALPHA,
                   help="Heatmap overlay blend factor.")

    # ── Batch ─────────────────────────────────────────────────────────
    p.add_argument("--limit", type=int, default=None,
                   help="Max images to process in batch mode.")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    logger.info("Initialising XAI pipeline …")
    pipe = XAIPipeline(
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        extraction_strategy=args.strategy,
        attention_method=args.method,
        last_layer_only=args.last_layer_only,
        colormap=args.colormap,
        alpha=args.alpha,
    )

    if args.image:
        # ── Single-image mode ─────────────────────────────────────────
        logger.info("Processing single image: %s", args.image)
        t0 = time.perf_counter()

        gt_mask = None
        if args.gt_dir:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            gt_path = os.path.join(args.gt_dir, f"{stem}_lab.png")
            if os.path.exists(gt_path):
                gt_mask = np.array(Image.open(gt_path)).astype(np.int64)
                logger.info("Loaded GT mask: %s", gt_path)

        result = pipe.run(args.image, gt_mask=gt_mask)
        elapsed = time.perf_counter() - t0
        logger.info("Inference + XAI completed in %.2f s.", elapsed)

        stem = os.path.splitext(os.path.basename(args.image))[0]

        # Save main panel
        main_path = os.path.join(args.output, f"{stem}_xai_panel.png")
        cv2.imwrite(main_path, result.panel_main)
        logger.info("Saved main panel  → %s", main_path)

        # Save stage panel
        stage_path = os.path.join(args.output, f"{stem}_xai_stages.png")
        cv2.imwrite(stage_path, result.panel_stages)
        logger.info("Saved stage panel → %s", stage_path)

        # Save damage panel
        if result.panel_damage is not None:
            dmg_path = os.path.join(args.output, f"{stem}_xai_damage.png")
            cv2.imwrite(dmg_path, result.panel_damage)
            logger.info("Saved damage panel → %s", dmg_path)

        # Save raw attention numpy
        npy_path = os.path.join(args.output, f"{stem}_attention.npy")
        np.save(npy_path, result.attention_fused)
        logger.info("Saved raw attention → %s", npy_path)

    else:
        # ── Batch mode ────────────────────────────────────────────────
        logger.info("Batch mode: %s", args.image_dir)
        t0 = time.perf_counter()
        saved = pipe.run_batch(
            image_dir=args.image_dir,
            output_dir=args.output,
            gt_dir=args.gt_dir,
            limit=args.limit,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Batch done: %d outputs in %.1f s.", len(saved), elapsed)


if __name__ == "__main__":
    main()
