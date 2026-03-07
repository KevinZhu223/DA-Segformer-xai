#!/usr/bin/env python3
"""
Streamlit dashboard for the XAI Attention Extractor.

Launch::

    streamlit run app.py -- --checkpoint /path/to/checkpoint

Designed for two audiences:
  - **Stakeholders**: Clear, high-level visual explanations of model behaviour.
  - **Researchers**: Quantitative metrics, downloadable data, per-stage analysis.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure local imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLASS_NAMES,
    DAMAGE_CLASSES,
    BUILDING_CLASSES,
    DEFAULT_CHECKPOINT,
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    IMAGE_SIZE,
    NUM_CLASSES,
    PALETTE_BGR,
    PALETTE_RGB,
    SEGFORMER_DEPTHS,
    SEGFORMER_NUM_HEADS,
    SEGFORMER_SR_RATIOS,
    SEGFORMER_SPATIAL_SCALES,
)
from heatmap import (
    apply_colormap,
    bgr2rgb,
    blend,
    build_legend_image,
    colorize_mask,
    rgb2bgr,
)
from pipeline import XAIPipeline, XAIResult

logger = logging.getLogger("xai_dashboard")


# ---------------------------------------------------------------------------
#  Custom CSS for professional appearance
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
    /* Tighten default Streamlit padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Section dividers */
    .section-divider {
        border: none;
        border-top: 1px solid #444;
        margin: 1.5rem 0 1rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #999;
        margin-top: 2px;
    }

    /* Make sidebar headings smaller */
    [data-testid="stSidebar"] h2 { font-size: 1rem; }

    /* Professional tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Table styling */
    .stTable table { font-size: 0.82rem; }
</style>
"""


# ---------------------------------------------------------------------------
#  Streamlit page config (no emoji)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Attention Extractor  |  DA-Segformer",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DA-Segformer model ...")
def get_pipeline(checkpoint: str, device: str, image_size: int) -> XAIPipeline:
    return XAIPipeline(
        checkpoint=checkpoint,
        device=device,
        image_size=image_size,
        extraction_strategy="hook",
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_png(arr_rgb: np.ndarray) -> bytes:
    """Encode an RGB numpy array as PNG bytes (for download buttons)."""
    img = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _metric_html(value: str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------
def sidebar_settings() -> dict:
    st.sidebar.markdown("## Configuration")

    checkpoint = st.sidebar.text_input(
        "Model checkpoint path",
        value=DEFAULT_CHECKPOINT,
    )
    device = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    image_size = st.sidebar.selectbox(
        "Tile resolution (px)",
        [512, 768, 1024],
        index=2,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Attention Extraction")
    method = st.sidebar.selectbox(
        "Aggregation method",
        ["received", "entropy", "rollout"],
        index=0,
        help=(
            "**received** -- key importance: highlights regions most referenced by queries.  \n"
            "**entropy** -- query focus: highlights where the model concentrates attention.  \n"
            "**rollout** -- multiplicative propagation across layers (stage 3 only, "
            "falls back to received for stages 0-2)."
        ),
    )
    last_layer_only = st.sidebar.checkbox(
        "Last layer only (faster, lower memory)", value=False,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Visualisation")
    colormap = st.sidebar.selectbox(
        "Colour map",
        ["jet", "turbo", "hot", "inferno", "magma", "viridis", "plasma"],
        index=0,
    )
    alpha = st.sidebar.slider(
        "Overlay blend factor", 0.10, 0.90, HEATMAP_ALPHA, 0.05,
    )

    return {
        "checkpoint": checkpoint,
        "device": device,
        "image_size": image_size,
        "method": method,
        "last_layer_only": last_layer_only,
        "colormap": colormap,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    settings = sidebar_settings()

    # Header
    st.markdown("# XAI Attention Extractor for DA-Segformer")
    st.markdown(
        "Upload a satellite image to visualise which spatial regions the "
        "DA-Segformer model attends to when classifying post-disaster "
        "structural damage."
    )

    # Load pipeline (cached)
    pipe = get_pipeline(
        settings["checkpoint"], settings["device"], settings["image_size"],
    )
    pipe.attention_method = settings["method"]
    pipe.last_layer_only = settings["last_layer_only"]
    pipe.colormap = settings["colormap"]
    pipe.alpha = settings["alpha"]

    # Image upload
    uploaded = st.file_uploader(
        "Upload a satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.markdown(
            '<hr class="section-divider">', unsafe_allow_html=True,
        )
        _show_landing_page()
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_w, img_h = pil_img.size

    # Run inference
    with st.spinner("Running inference and attention extraction ..."):
        t0 = time.perf_counter()
        result = pipe.run(pil_img, render_panels=True)
        elapsed = time.perf_counter() - t0

    # Summary metrics bar
    _render_summary_bar(result, elapsed, img_w, img_h)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs([
        "Overview",
        "Per-Stage Attention",
        "Damage Focus",
        "Statistics & Data",
        "Methodology",
    ])

    with tabs[0]:
        _tab_overview(result, settings)
    with tabs[1]:
        _tab_stages(result, settings)
    with tabs[2]:
        _tab_damage(result, settings)
    with tabs[3]:
        _tab_stats(result, settings)
    with tabs[4]:
        _tab_methodology()


# ---------------------------------------------------------------------------
#  Summary bar (top-level KPIs)
# ---------------------------------------------------------------------------
def _render_summary_bar(result: XAIResult, elapsed: float, w: int, h: int):
    pred = result.pred_mask

    # Compute damage pixel count
    damage_px = sum(int((pred == c).sum()) for c in DAMAGE_CLASSES)
    damage_pct = damage_px / pred.size * 100

    building_px = sum(int((pred == c).sum()) for c in BUILDING_CLASSES)
    building_pct = building_px / pred.size * 100

    n_classes = len(set(pred.flatten().tolist()))

    cols = st.columns(5)
    metrics = [
        (f"{elapsed:.1f}s", "Inference Time"),
        (f"{w} x {h}", "Image Resolution"),
        (f"{n_classes}", "Classes Detected"),
        (f"{building_pct:.1f}%", "Building Coverage"),
        (f"{damage_pct:.1f}%", "Damage Coverage"),
    ]
    for col, (val, label) in zip(cols, metrics):
        col.markdown(_metric_html(val, label), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Tab: Overview
# ---------------------------------------------------------------------------
def _tab_overview(result: XAIResult, settings: dict):
    st.markdown("### Segmentation Prediction and Attention Overlay")
    st.markdown(
        "The model segments every pixel into one of 11 damage/land-cover classes. "
        "The attention heatmap reveals which spatial regions the transformer "
        "encoder references most during inference."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            result.image_rgb,
            caption="Original Satellite Image",
            use_container_width=True,
        )
    with col2:
        pred_rgb = _to_rgb(result.pred_colour_bgr)
        st.image(
            pred_rgb,
            caption="Predicted Segmentation Mask",
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        hm = apply_colormap(result.attention_fused, settings["colormap"])
        hm_rgb = _to_rgb(
            cv2.resize(hm, (result.image_bgr.shape[1], result.image_bgr.shape[0]))
        )
        st.image(
            hm_rgb,
            caption="Global Attention Heatmap",
            use_container_width=True,
        )
    with col4:
        hm_full = apply_colormap(result.attention_fused, settings["colormap"])
        overlay = blend(result.image_bgr, hm_full, settings["alpha"])
        st.image(
            _to_rgb(overlay),
            caption="Attention Overlay on Original",
            use_container_width=True,
        )

    # Downloads
    with st.expander("Download outputs"):
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.download_button(
                "Download Prediction Mask (PNG)",
                data=_encode_png(pred_rgb),
                file_name="prediction_mask.png",
                mime="image/png",
            )
        with dcol2:
            st.download_button(
                "Download Attention Heatmap (PNG)",
                data=_encode_png(hm_rgb),
                file_name="attention_heatmap.png",
                mime="image/png",
            )
        with dcol3:
            st.download_button(
                "Download Overlay (PNG)",
                data=_encode_png(_to_rgb(overlay)),
                file_name="attention_overlay.png",
                mime="image/png",
            )

    # Colour legend
    with st.expander("Class colour legend"):
        legend = build_legend_image()
        st.image(_to_rgb(legend), width=280)


# ---------------------------------------------------------------------------
#  Tab: Per-Stage Attention
# ---------------------------------------------------------------------------
def _tab_stages(result: XAIResult, settings: dict):
    st.markdown("### Hierarchical Attention -- All 4 Encoder Stages")
    st.markdown(
        "SegFormer's Mix-Transformer encoder extracts features at "
        "4 spatial scales. Early stages capture fine-grained texture "
        "and edges; later stages capture whole-object and scene-level context."
    )

    # Architecture reference table
    with st.expander("Stage architecture reference"):
        st.markdown(
            "| Stage | Transformer Blocks | Attention Heads | Sequence Reduction | Feature Scale | Spatial Role |\n"
            "|:-----:|:------------------:|:---------------:|:------------------:|:-------------:|:------------:|\n"
            "| 0 | 3 | 1 | 8x | 1/4 | Edges, texture |\n"
            "| 1 | 8 | 2 | 4x | 1/8 | Object parts |\n"
            "| 2 | 27 | 5 | 2x | 1/16 | Whole objects |\n"
            "| 3 | 3 | 8 | 1x (full) | 1/32 | Scene context |"
        )

    cols = st.columns(4)
    for s in range(4):
        key = f"stage_{s}"
        with cols[s]:
            st.markdown(
                f"**Stage {s}**  \n"
                f"{SEGFORMER_DEPTHS[s]} blocks  |  "
                f"{SEGFORMER_NUM_HEADS[s]} heads  |  "
                f"SR {SEGFORMER_SR_RATIOS[s]}"
            )
            if key in result.attention_stages:
                smap = result.attention_stages[key]
                hm = apply_colormap(smap, settings["colormap"])
                hm_resized = cv2.resize(
                    hm,
                    (result.image_bgr.shape[1], result.image_bgr.shape[0]),
                )
                overlay = blend(result.image_bgr, hm_resized, settings["alpha"])
                st.image(_to_rgb(overlay), use_container_width=True)
            else:
                st.warning("No data for this stage.")

    # Download stage panel
    if result.panel_stages is not None:
        with st.expander("Composite stage panel"):
            st.image(
                _to_rgb(result.panel_stages), use_container_width=True,
            )
            st.download_button(
                "Download Stage Panel (PNG)",
                data=_encode_png(_to_rgb(result.panel_stages)),
                file_name="stage_attention_panel.png",
                mime="image/png",
            )


# ---------------------------------------------------------------------------
#  Tab: Damage Focus
# ---------------------------------------------------------------------------
def _tab_damage(result: XAIResult, settings: dict):
    st.markdown("### Class-Conditional Attention -- Structural Damage")
    st.markdown(
        "Attention values are masked to only show regions where the model "
        "predicts building damage (minor, major, or total destruction). "
        "This isolates the spatial cues the model uses specifically "
        "for damage assessment decisions."
    )

    has_damage = (
        result.attention_damage is not None
        and result.attention_damage.max() > 0
    )

    if has_damage:
        col1, col2 = st.columns(2)
        with col1:
            hm = apply_colormap(result.attention_damage, settings["colormap"])
            hm_r = cv2.resize(
                hm,
                (result.image_bgr.shape[1], result.image_bgr.shape[0]),
            )
            overlay_d = blend(result.image_bgr, hm_r, settings["alpha"])
            st.image(
                _to_rgb(overlay_d),
                caption="Damage Attention (Minor / Major / Total Destruction)",
                use_container_width=True,
            )
        with col2:
            if result.attention_building is not None:
                hm_b = apply_colormap(
                    result.attention_building, settings["colormap"],
                )
                hm_b_r = cv2.resize(
                    hm_b,
                    (result.image_bgr.shape[1], result.image_bgr.shape[0]),
                )
                overlay_b = blend(result.image_bgr, hm_b_r, settings["alpha"])
                st.image(
                    _to_rgb(overlay_b),
                    caption="All Buildings Attention (incl. No Damage)",
                    use_container_width=True,
                )

        # Analysis guidance
        with st.expander("Interpreting damage attention"):
            st.markdown(
                "**Warm colours** (red, yellow) indicate regions where "
                "the model concentrates attention within damage-classified "
                "pixels. Compare the damage attention map (left) against "
                "the all-buildings map (right) to see whether the model "
                "attends differently to damaged vs. undamaged structures.  \n\n"
                "If attention is concentrated at building edges, the model "
                "may be relying on structural outline cues. If it is "
                "concentrated at rooftop interiors, it may be using "
                "texture/colour cues from debris or material damage."
            )

        # Downloads
        with st.expander("Download damage analysis"):
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    "Download Damage Overlay (PNG)",
                    data=_encode_png(_to_rgb(overlay_d)),
                    file_name="damage_attention_overlay.png",
                    mime="image/png",
                )
            with dcol2:
                if result.panel_damage is not None:
                    st.download_button(
                        "Download Damage Panel (PNG)",
                        data=_encode_png(_to_rgb(result.panel_damage)),
                        file_name="damage_attention_panel.png",
                        mime="image/png",
                    )
    else:
        st.info(
            "No building damage was predicted in this image. "
            "Upload an image containing damaged structures to see "
            "damage-specific attention analysis."
        )


# ---------------------------------------------------------------------------
#  Tab: Statistics & Data (researcher-focused)
# ---------------------------------------------------------------------------
def _tab_stats(result: XAIResult, settings: dict):
    st.markdown("### Prediction Statistics")

    pred = result.pred_mask
    total_px = pred.size

    # Per-class breakdown
    rows = []
    for cid in range(NUM_CLASSES):
        count = int((pred == cid).sum())
        pct = count / total_px * 100
        rows.append({
            "Class ID": cid,
            "Class Name": CLASS_NAMES[cid],
            "Pixel Count": f"{count:,}",
            "Coverage (%)": f"{pct:.2f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Attention distribution
    st.markdown("### Attention Distribution")
    attn = result.attention_fused

    acol1, acol2, acol3, acol4 = st.columns(4)
    acol1.metric("Min", f"{attn.min():.4f}")
    acol2.metric("Max", f"{attn.max():.4f}")
    acol3.metric("Mean", f"{attn.mean():.4f}")
    acol4.metric("Std Dev", f"{attn.std():.4f}")

    # Percentile table
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = {f"p{p}": f"{np.percentile(attn, p):.4f}" for p in percentiles}
    st.markdown("**Percentile distribution:**")
    st.dataframe(
        [pct_vals],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Per-class attention breakdown
    st.markdown("### Per-Class Mean Attention")
    st.markdown(
        "Average attention value within pixels predicted as each class. "
        "Higher values suggest the model references those regions more "
        "during inference."
    )
    class_attn_rows = []
    for cid in range(NUM_CLASSES):
        mask = pred == cid
        count = int(mask.sum())
        if count > 0:
            mean_attn = float(attn[mask].mean())
            std_attn = float(attn[mask].std())
        else:
            mean_attn = 0.0
            std_attn = 0.0
        class_attn_rows.append({
            "Class": CLASS_NAMES[cid],
            "Pixels": f"{count:,}",
            "Mean Attention": f"{mean_attn:.4f}",
            "Std Attention": f"{std_attn:.4f}",
        })
    st.dataframe(class_attn_rows, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Raw data downloads
    st.markdown("### Download Raw Data")
    st.markdown(
        "Download the raw attention arrays for further analysis in "
        "Python, MATLAB, or any tool that reads NumPy `.npy` files."
    )

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        buf = io.BytesIO()
        np.save(buf, result.attention_fused)
        st.download_button(
            "Fused Attention (.npy)",
            data=buf.getvalue(),
            file_name="attention_fused.npy",
            mime="application/octet-stream",
        )
    with dcol2:
        buf = io.BytesIO()
        np.save(buf, result.pred_mask)
        st.download_button(
            "Prediction Mask (.npy)",
            data=buf.getvalue(),
            file_name="prediction_mask.npy",
            mime="application/octet-stream",
        )
    with dcol3:
        # Pack all stage attention into a single dict-of-arrays
        stage_buf = io.BytesIO()
        np.savez_compressed(stage_buf, **result.attention_stages)
        st.download_button(
            "Per-Stage Attention (.npz)",
            data=stage_buf.getvalue(),
            file_name="attention_stages.npz",
            mime="application/octet-stream",
        )


# ---------------------------------------------------------------------------
#  Tab: Methodology
# ---------------------------------------------------------------------------
def _tab_methodology():
    st.markdown("### Methodology")

    st.markdown(
        """
This tool extracts and visualises self-attention weights from a
SegFormer-B4 model fine-tuned for post-disaster semantic segmentation
on the RescueNet dataset.

---

**Model Architecture**

DA-Segformer uses the Mix-Transformer (MiT) encoder with a
lightweight MLP decoder. The encoder has 4 hierarchical stages,
each operating at a different spatial scale:

| Stage | Blocks | Heads | Seq. Reduction | Scale |
|:-----:|:------:|:-----:|:--------------:|:-----:|
| 0 | 3 | 1 | 8x | 1/4 |
| 1 | 8 | 2 | 4x | 1/8 |
| 2 | 27 | 5 | 2x | 1/16 |
| 3 | 3 | 8 | 1x | 1/32 |

Total parameters: 64M. Best validation mIoU: **0.7461** (Epoch 255).

---

**Attention Extraction**

During inference, forward hooks intercept the post-softmax
Q-K^T attention matrices from every `SegformerEfficientSelfAttention`
module (41 total across all stages). This is completely passive --
no gradients are computed and no model weights are modified.

---

**Aggregation Methods**

- **Attention Received**: For each key token, sum of attention it
  receives from all query tokens. Highlights regions the model
  references most.
- **Attention Entropy**: Per-query entropy of the attention
  distribution. Low entropy = the model is highly focused at that
  spatial position.
- **Attention Rollout**: Multiplicative propagation of attention
  across layers within a stage (Abnar & Zuidema, 2020). Only
  applicable to Stage 3 where attention is square (SR=1).

---

**Sliding-Window Inference**

The model was trained on 1024x1024 crops at native resolution.
To process full satellite images (up to 4000x4000), a sliding
window with 25% overlap (stride=768) tiles the image. Both
softmax prediction probabilities and attention maps are stitched
using overlap averaging with a 2D Hanning window to eliminate
tile boundary artefacts.

---

**Normalisation**

Attention maps are normalised using percentile clipping (1st-99th
percentile) followed by gamma correction (gamma=0.7) to produce
visually informative heatmaps without losing contrast to outlier
peaks.

---

**Dataset: RescueNet**

11-class post-disaster satellite imagery:
Background, Water, Building (No Damage), Building (Minor Damage),
Building (Major Damage), Building (Total Destruction), Vehicle,
Road (Clear), Road (Blocked), Tree, Pool.

---

**References**

- Xie, E. et al. "SegFormer: Simple and Efficient Design for
  Semantic Segmentation with Transformers." NeurIPS 2021.
- Abnar, S. & Zuidema, W. "Quantifying Attention Flow in
  Transformers." ACL 2020.
        """
    )


# ---------------------------------------------------------------------------
#  Landing page (before upload)
# ---------------------------------------------------------------------------
def _show_landing_page():
    st.markdown("### About This Tool")
    st.markdown(
        """
This dashboard provides **Explainable AI (XAI) analysis** of a
SegFormer-B4 model trained for post-disaster damage classification
on satellite imagery.

**For stakeholders and decision-makers:**
- See exactly what the model looks at when it classifies buildings
  as damaged or undamaged.
- Verify model behaviour on specific satellite images from your
  area of interest.
- Download high-resolution visualisation panels for reports and
  presentations.

**For researchers and engineers:**
- Inspect attention patterns at each of the 4 hierarchical encoder
  stages.
- Compare attention aggregation methods (received, entropy, rollout).
- Download raw attention tensors (`.npy`) for quantitative analysis.
- Examine per-class attention statistics to identify potential
  model biases.

---

**How to use:** Upload a satellite image using the file uploader
above. The model will run sliding-window inference at native
resolution and extract attention weights from all 41 transformer
blocks.

---
        """
    )

    # Architecture table
    st.markdown("#### Model Architecture: SegFormer-B4")
    st.markdown(
        "| Stage | Blocks | Heads | Seq. Reduction | Feature Scale |\n"
        "|:-----:|:------:|:-----:|:--------------:|:-------------:|\n"
        "| 0 | 3 | 1 | 8x | 1/4 |\n"
        "| 1 | 8 | 2 | 4x | 1/8 |\n"
        "| 2 | 27 | 5 | 2x | 1/16 |\n"
        "| 3 | 3 | 8 | 1x (full) | 1/32 |"
    )
    st.markdown(
        "**Parameters:** 64M  |  **Classes:** 11  |  "
        "**Best mIoU:** 0.7461 (Epoch 255)"
    )


if __name__ == "__main__":
    main()
