"""
Attention tensor processing for SegFormer's hierarchical encoder.

Converts raw attention weight tensors (Q·Kᵀ after softmax) into
interpretable 2-D spatial maps.  Three complementary metrics are
provided:

* **Attention Received** — for each key token, how much total attention
  it receives from all queries.  Highlights the most-referenced regions.
* **Attention Entropy** — per-query entropy of the attention distribution.
  Low entropy → the model is highly focused at that spatial position.
* **Attention Rollout** — multiplicative propagation of attention across
  all layers within a stage (only valid for square attention, i.e. stage 3
  where the sequence-reduction ratio is 1).

SegFormer-B4 quick reference (1024×1024 input)
-----------------------------------------------
Stage | Blocks | Heads | Spatial (Q) | SR | Spatial (K)
  0   |   3    |   1   | 256 × 256   |  8 | 32 × 32
  1   |   8    |   2   | 128 × 128   |  4 | 32 × 32
  2   |  27    |   5   |  64 ×  64   |  2 | 32 × 32
  3   |   3    |   8   |  32 ×  32   |  1 | 32 × 32
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    SEGFORMER_DEPTHS,
    SEGFORMER_NUM_HEADS,
    SEGFORMER_SR_RATIOS,
    SEGFORMER_SPATIAL_SCALES,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Geometry helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def spatial_dims(image_size: int, stage: int):
    """Return (H_q, W_q, H_k, W_k) for a given input size and stage."""
    ds = SEGFORMER_SPATIAL_SCALES[stage]
    sr = SEGFORMER_SR_RATIOS[stage]
    h_q = w_q = image_size // ds
    h_k = w_k = h_q // sr
    return h_q, w_q, h_k, w_k


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Per-layer primitives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def head_average(attn: torch.Tensor) -> torch.Tensor:
    """Average attention across heads.

    (B, H, N_q, N_k) → (B, N_q, N_k)
    """
    return attn.float().mean(dim=1)


def attention_received(attn_avg: torch.Tensor, h_k: int, w_k: int) -> torch.Tensor:
    """Key-importance map: sum of attention each key token receives.

    Parameters
    ----------
    attn_avg : (B, N_q, N_k)
    h_k, w_k : spatial dimensions of the key grid

    Returns
    -------
    (B, h_k, w_k)
    """
    received = attn_avg.sum(dim=1)  # (B, N_k)
    return received.view(received.shape[0], h_k, w_k)


def attention_entropy(attn_avg: torch.Tensor, h_q: int, w_q: int) -> torch.Tensor:
    """Attention-focus map: inverted normalised entropy per query.

    Higher value → more focused (peaked) attention distribution.

    Parameters
    ----------
    attn_avg : (B, N_q, N_k)

    Returns
    -------
    (B, h_q, w_q)  values in [0, 1]
    """
    eps = 1e-8
    p = attn_avg.clamp(min=eps)
    ent = -(p * p.log()).sum(dim=-1)          # (B, N_q)
    max_ent = math.log(attn_avg.shape[-1])    # log(N_k)
    concentration = 1.0 - ent / max_ent       # invert & normalise
    return concentration.view(concentration.shape[0], h_q, w_q)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Attention Rollout (square attention only – Stage 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def attention_rollout(
    layer_attentions: List[torch.Tensor],
) -> torch.Tensor:
    """Multiplicative attention rollout for a single stage.

    Follows Abnar & Zuidema (2020): at each layer, add the identity
    matrix (modelling the residual / skip connection), re-normalise,
    then multiply.

    Parameters
    ----------
    layer_attentions : list of (B, H, N, N) — must be *square*.

    Returns
    -------
    (B, N, N) — rolled-out attention matrix.
    """
    result: Optional[torch.Tensor] = None
    for attn in layer_attentions:
        a = head_average(attn)  # (B, N, N)
        I = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype).unsqueeze(0)
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else torch.matmul(result, a)
    return result  # type: ignore[return-value]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Per-stage processor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_stage(
    stage_attns: List[torch.Tensor],
    stage_idx: int,
    image_size: int,
    method: str = "received",
    last_layer_only: bool = False,
) -> torch.Tensor:
    """Aggregate all layers in a stage into one spatial attention map.

    Parameters
    ----------
    stage_attns : list of (B, H, N_q, N_k) tensors.
    stage_idx : 0–3.
    image_size : input resolution (e.g. 1024).
    method : ``"received"`` | ``"entropy"`` | ``"rollout"``
        ``"rollout"`` falls back to ``"received"`` if attention is
        non-square (stages 0–2).
    last_layer_only : if True, only use the *last* transformer block
        in the stage (saves memory & often most informative).

    Returns
    -------
    (B, H_out, W_out) spatial map (un-normalised).
    """
    h_q, w_q, h_k, w_k = spatial_dims(image_size, stage_idx)
    sr = SEGFORMER_SR_RATIOS[stage_idx]

    if last_layer_only:
        stage_attns = [stage_attns[-1]]

    # ── Rollout (only for square attention, SR=1) ─────────────────────
    if method == "rollout":
        if sr == 1:
            rolled = attention_rollout(stage_attns)  # (B, N, N)
            # "Attention received" from the rolled-out matrix
            received = rolled.sum(dim=1)  # (B, N)
            return received.view(received.shape[0], h_q, w_q)
        else:
            logger.debug(
                "Stage %d has SR=%d (non-square attention); "
                "falling back to 'received'.",
                stage_idx, sr,
            )
            method = "received"

    # ── Layer-wise maps → average ─────────────────────────────────────
    maps: List[torch.Tensor] = []
    for attn in stage_attns:
        avg = head_average(attn)  # (B, N_q, N_k)
        if method == "received":
            maps.append(attention_received(avg, h_k, w_k))
        elif method == "entropy":
            maps.append(attention_entropy(avg, h_q, w_q))
        else:
            raise ValueError(f"Unknown method: {method!r}")

    # Average across layers
    return torch.stack(maps).mean(dim=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Multi-scale fusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fuse_multiscale(
    stage_maps: List[torch.Tensor],
    target_size: int,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Upsample per-stage maps and fuse into a single heatmap.

    Parameters
    ----------
    stage_maps : list of (B, H_s, W_s) tensors, one per stage.
    target_size : output resolution (e.g. 1024).
    weights : per-stage weights; default equal.

    Returns
    -------
    (B, target_size, target_size) — un-normalised fused map.
    """
    if weights is None:
        weights = [1.0] * len(stage_maps)
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    fused: Optional[torch.Tensor] = None
    for smap, w in zip(stage_maps, weights):
        up = F.interpolate(
            smap.unsqueeze(1).float(),      # (B, 1, H, W)
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                        # (B, H, W)
        fused = w * up if fused is None else fused + w * up
    return fused  # type: ignore[return-value]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Normalisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def normalize_map(m: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [0, 1].  Works on (H, W) or (B, H, W)."""
    if m.dim() == 2:
        lo, hi = m.min(), m.max()
        return torch.zeros_like(m) if (hi - lo) < 1e-8 else (m - lo) / (hi - lo)
    return torch.stack([normalize_map(m[i]) for i in range(m.shape[0])])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Class-conditional masking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def class_masked_attention(
    attention_map: torch.Tensor,
    prediction: torch.Tensor,
    class_ids: List[int],
) -> torch.Tensor:
    """Zero-out attention outside predicted class regions.

    Parameters
    ----------
    attention_map : (H, W) or (B, H, W), normalised [0, 1].
    prediction : (H, W) or (B, H, W), integer class IDs.
    class_ids : classes to keep.

    Returns
    -------
    Same shape as *attention_map*, re-normalised to [0, 1].
    """
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    for cid in class_ids:
        mask = mask | (prediction == cid)

    masked = attention_map * mask.float()
    return normalize_map(masked)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  High-level convenience
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_all_maps(
    attentions_by_stage: Dict[int, List[torch.Tensor]],
    image_size: int,
    method: str = "received",
    last_layer_only: bool = False,
    stage_weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """One-call processor: raw attention → per-stage + fused maps.

    Returns
    -------
    dict with keys:
        ``"stage_0"`` … ``"stage_3"`` : per-stage (B, H_s, W_s) maps
        ``"fused"``                    : (B, image_size, image_size)
        ``"fused_norm"``               : normalised fused map [0, 1]
    """
    stage_maps: List[torch.Tensor] = []
    result: Dict[str, torch.Tensor] = {}

    for s in sorted(attentions_by_stage):
        smap = process_stage(
            attentions_by_stage[s], s, image_size,
            method=method, last_layer_only=last_layer_only,
        )
        result[f"stage_{s}"] = smap
        stage_maps.append(smap)

    fused = fuse_multiscale(stage_maps, image_size, weights=stage_weights)
    result["fused"] = fused
    result["fused_norm"] = normalize_map(fused)
    return result
