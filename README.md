# XAI Attention Extractor for DA-Segformer

**Explainable AI interpretability pipeline for post-disaster damage assessment.**

This tool extracts the internal self-attention weights from a pre-trained DA-Segformer model during inference, processes them into spatial heatmaps, and overlays them on satellite imagery — proving exactly which pixels the model focused on when making structural damage predictions.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. CLI — single image
python cli.py --image /path/to/satellite.png --output ./xai_output

# 3. Web dashboard
streamlit run app.py
```

---

## Architecture

```
DA-Segformer (SegFormer-B4)
├── Hierarchical Mix-Transformer Encoder
│   ├── Stage 0:  3 blocks ×  1 head, SR=8, features at 1/4  scale
│   ├── Stage 1:  8 blocks ×  2 heads, SR=4, features at 1/8  scale
│   ├── Stage 2: 27 blocks ×  5 heads, SR=2, features at 1/16 scale
│   └── Stage 3:  3 blocks ×  8 heads, SR=1, features at 1/32 scale
└── Lightweight MLP Decode Head (768-dim, 11-class output)
```

**Best checkpoint**: Epoch 255, mIoU **0.7461** on RescueNet validation.

### Attention Extraction Flow

```
Image → SegFormer Forward Pass → Forward Hooks Capture Q·Kᵀ Matrices
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             Head Average         Attention Rollout    Attention Entropy
            (key importance)     (info propagation)    (query focus)
                    │                    │                    │
                    └────────► Reshape 1D → 2D Spatial Grid ◄┘
                                         │
                               Bilinear Upsample to 1024²
                                         │
                               Normalise [0, 1] + JET Colormap
                                         │
                               Alpha-Blend with Original Image
                                         │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                      Multi-Panel PNG          Streamlit Dashboard
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Paths, class definitions, SegFormer-B4 architecture constants |
| `model_loader.py` | Load HuggingFace checkpoint, set eval mode, disable gradients |
| `attention_extractor.py` | Hook-based and stealth (monkey-patch) attention capture |
| `attention_processor.py` | Head averaging, entropy, rollout, multi-scale fusion |
| `heatmap.py` | Colourmap application, blending, multi-panel composites |
| `pipeline.py` | End-to-end orchestration (preprocessing → inference → visualisation) |
| `cli.py` | Command-line interface (single image + batch mode) |
| `app.py` | Streamlit interactive web dashboard |
| `run_cli.sh` | Shell launcher for CLI |
| `run_dashboard.sh` | Shell launcher for Streamlit |

---

## Attention Metrics

### 1. Attention Received (default: `--method received`)
For each key token, sums how much attention it receives from all queries.
Highlights spatial regions that are most frequently referenced by the model.

### 2. Attention Entropy (`--method entropy`)
Per-query entropy of the attention distribution.
Low entropy = focused attention (the model knows what to look at).
High entropy = diffuse attention (model aggregates broadly).

### 3. Attention Rollout (`--method rollout`)
Multiplicative propagation through all layers in a stage, following
[Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928).
Only valid for Stage 3 (SR=1, square attention); other stages fall back to "received."

---

## CLI Reference

```bash
# Single image
python cli.py \
    --image /data/RescueNet/val-org-img/7635.jpg \
    --gt-dir /data/RescueNet/val-label-img \
    --output ./xai_output \
    --method received \
    --colormap jet \
    --alpha 0.5

# Batch (first 20 validation images)
python cli.py \
    --image-dir /data/RescueNet/val-org-img \
    --gt-dir /data/RescueNet/val-label-img \
    --output ./xai_output \
    --limit 20

# Memory-efficient (last layer only per stage)
python cli.py \
    --image /path/to/image.png \
    --last-layer-only \
    --device cpu
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | — | Single image path |
| `--image-dir` | — | Batch directory |
| `--gt-dir` | — | Ground-truth labels (`*_lab.png`) |
| `--output` | `./xai_output` | Output directory |
| `--checkpoint` | Epoch 255 best | HuggingFace checkpoint dir |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--image-size` | `1024` | Inference resolution |
| `--strategy` | `hook` | `hook` or `stealth` |
| `--method` | `received` | `received` / `entropy` / `rollout` |
| `--last-layer-only` | `false` | Use only last block per stage |
| `--colormap` | `jet` | `jet`, `turbo`, `hot`, `inferno`, etc. |
| `--alpha` | `0.5` | Heatmap overlay blend |
| `--limit` | — | Max images in batch mode |

---

## Streamlit Dashboard

```bash
./run_dashboard.sh              # http://localhost:8501
./run_dashboard.sh --port 8502  # custom port
```

**Features:**
- Drag-and-drop image upload
- Live inference + attention extraction
- 4-tab layout: Overview / Per-Stage / Damage Focus / Metrics
- Adjustable colour map, blend alpha, and aggregation method
- Class colour legend and pixel-count statistics

---

## RescueNet Classes (11)

| ID | Class | Colour |
|----|-------|--------|
| 0 | Background | Black |
| 1 | Water | Blue |
| 2 | Building No Damage | Green |
| 3 | Building Minor Damage | Yellow |
| 4 | Building Major Damage | Red |
| 5 | Building Total Destruction | Dark Red |
| 6 | Vehicle | Purple |
| 7 | Road-Clear | Gray |
| 8 | Road-Blocked | Dark Gray |
| 9 | Tree | Dark Green |
| 10 | Pool | Orange-Blue |

---

## Key Design Decisions

1. **No retraining / no backpropagation**: Model weights are frozen; extraction is purely passive via forward hooks.
2. **Dual extraction strategy**: `HookExtractor` uses `register_forward_hook()`; `StealthExtractor` monkey-patches `forward()` for environments where you can't pass `output_attentions=True`.
3. **Memory-aware**: `--last-layer-only` reduces peak memory from ~4 GB to ~500 MB for attention tensors.
4. **Hierarchical handling**: Correctly accounts for SegFormer's non-square attention matrices (Efficient Self-Attention with sequence reduction ratios 8/4/2/1).
5. **Class-conditional masking**: Attention can be filtered to specific class regions (e.g., only building damage) for targeted interpretability.

---

## Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- HuggingFace Transformers ≥ 4.35
- OpenCV (headless)
- Streamlit ≥ 1.30
- NumPy, Pillow
