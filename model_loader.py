"""
Model loader for DA-Segformer.

Loads a HuggingFace SegformerForSemanticSegmentation checkpoint
(SafeTensors or pytorch_model.bin) and its paired image processor.
Sets the model to evaluation mode for inference-only use.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from config import DEFAULT_CHECKPOINT, IMAGE_SIZE

logger = logging.getLogger(__name__)


def load_model(
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
) -> Tuple[SegformerForSemanticSegmentation, SegformerImageProcessor, torch.device]:
    """Load the DA-Segformer checkpoint and image processor.

    Parameters
    ----------
    checkpoint : str, optional
        Path to the HuggingFace-format checkpoint directory.
        Must contain ``config.json`` and ``model.safetensors``
        (or ``pytorch_model.bin``).  Defaults to the best RescueNet
        checkpoint at Epoch 255.
    device : str, optional
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).  ``"auto"``
        picks CUDA when available.
    image_size : int
        Target resolution for the image processor.  Default 1024.

    Returns
    -------
    model : SegformerForSemanticSegmentation
        Model in ``eval()`` mode with gradients disabled.
    processor : SegformerImageProcessor
        Matched image preprocessor.
    device : torch.device
        Resolved device.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    ckpt_path = Path(checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set MODEL_CHECKPOINT env-var or pass --checkpoint."
        )
    if not (ckpt_path / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json in {ckpt_path}.  "
            "Expected a HuggingFace-format checkpoint directory."
        )

    # ── Resolve device ────────────────────────────────────────────────
    if device is None or device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    logger.info("Using device: %s", resolved)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Load model ────────────────────────────────────────────────────
    logger.info("Loading model from %s …", ckpt_path)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(ckpt_path),
        torch_dtype=torch.float32,
    )
    model.to(resolved)
    model.eval()
    # Globally disable gradient computation — purely passive inference
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "Model loaded (%s params, %d classes).",
        f"{sum(p.numel() for p in model.parameters()):,}",
        model.config.num_labels,
    )

    # ── Load image processor ──────────────────────────────────────────
    # The fine-tuned checkpoint may not ship a preprocessor_config.json,
    # so try the checkpoint first, then fall back to the base model name
    # stored in the config, and finally to the well-known HF hub name.
    processor = None
    for source in [
        str(ckpt_path),
        getattr(model.config, "_name_or_path", None),
        "nvidia/segformer-b4-finetuned-ade-512-512",
    ]:
        if source is None:
            continue
        try:
            processor = SegformerImageProcessor.from_pretrained(source)
            logger.info("Image processor loaded from: %s", source)
            break
        except OSError:
            logger.debug("No preprocessor_config.json in %s, trying next …", source)

    if processor is None:
        raise RuntimeError(
            "Could not load SegformerImageProcessor from any known source."
        )

    # Override the resize target so inference resolution matches training
    processor.do_resize = True
    processor.size = {"height": image_size, "width": image_size}
    logger.info("Image processor configured for %d×%d.", image_size, image_size)

    return model, processor, resolved
