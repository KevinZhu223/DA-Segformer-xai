"""
Attention extraction from SegFormer via PyTorch forward hooks.

Two strategies are provided:

1. **HookExtractor** (recommended) — registers ``register_forward_hook``
   callbacks on every ``SegformerEfficientSelfAttention`` module.  When
   inference is run with ``output_attentions=True``, the hooks silently
   intercept the post-softmax Q·Kᵀ matrices, detach them, and move them
   to CPU so GPU memory is freed immediately.

2. **StealthExtractor** — monkey-patches the ``forward()`` method of each
   attention module so that attention weights are captured *without*
   requiring ``output_attentions=True``.  This is useful when calling
   code you cannot modify.

Both extractors expose the same ``attention_store`` dictionary keyed by
``(stage_idx, layer_idx)`` tuples.

Attention tensor shapes for SegFormer-B4, 1024×1024 input
----------------------------------------------------------
Stage 0 : (B, 1,  65 536, 1 024)   SR = 8
Stage 1 : (B, 2,  16 384, 1 024)   SR = 4
Stage 2 : (B, 5,   4 096, 1 024)   SR = 2
Stage 3 : (B, 8,   1 024, 1 024)   SR = 1
"""

from __future__ import annotations

import logging
import types
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger(__name__)

# Type alias for the storage dict
AttnKey = Tuple[int, int]  # (stage_idx, layer_idx)
AttnStore = Dict[AttnKey, torch.Tensor]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy 1 — Clean hook-based extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HookExtractor:
    """Capture attention weights via ``register_forward_hook``.

    Usage::

        extractor = HookExtractor(model)
        with torch.no_grad():
            outputs = model(pixel_values=pv, output_attentions=True)
        stage3_attns = extractor.get_stage_attentions(3)
        extractor.remove_hooks()
    """

    def __init__(
        self,
        model: SegformerForSemanticSegmentation,
        capture_keys: Optional[Set[AttnKey]] = None,
    ) -> None:
        self.model = model
        self.capture_keys = capture_keys
        self.attention_store: AttnStore = OrderedDict()
        self._hooks: list = []
        self._register()

    # ── internal ──────────────────────────────────────────────────────
    def _register(self) -> None:
        encoder = self.model.segformer.encoder
        num_stages = len(encoder.block)
        total_hooks = 0
        for s in range(num_stages):
            for l in range(len(encoder.block[s])):
                module = encoder.block[s][l].attention.self
                key: AttnKey = (s, l)
                if self.capture_keys is not None and key not in self.capture_keys:
                    continue

                def _hook(
                    _mod: torch.nn.Module,
                    _inp,
                    output,
                    _key: AttnKey = key,
                ) -> None:
                    # SegformerEfficientSelfAttention.forward returns
                    # (context,) or (context, attn_probs).
                    if isinstance(output, tuple) and len(output) >= 2:
                        self.attention_store[_key] = output[1].detach().cpu()

                handle = module.register_forward_hook(_hook)
                self._hooks.append(handle)
                total_hooks += 1

        logger.info("Registered %d attention hooks across %d stages.",
                     total_hooks, num_stages)

    # ── public API ────────────────────────────────────────────────────
    def clear(self) -> None:
        """Discard all stored attention tensors."""
        self.attention_store.clear()

    def remove_hooks(self) -> None:
        """De-register all hooks (idempotent)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("All attention hooks removed.")

    def get_stage_attentions(self, stage_idx: int) -> List[torch.Tensor]:
        """Return attention tensors for one stage, ordered by layer."""
        return [
            attn
            for (s, _l), attn in sorted(self.attention_store.items())
            if s == stage_idx
        ]

    def get_all_attentions_by_stage(self) -> Dict[int, List[torch.Tensor]]:
        """Return ``{stage_idx: [attn_layer_0, attn_layer_1, …]}``."""
        out: Dict[int, List[torch.Tensor]] = {}
        for (s, _l), attn in sorted(self.attention_store.items()):
            out.setdefault(s, []).append(attn)
        return out

    @property
    def num_captured(self) -> int:
        return len(self.attention_store)

    def __del__(self) -> None:
        self.remove_hooks()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy 2 — Stealth (monkey-patch) extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StealthExtractor:
    """Capture attention by patching ``forward()`` — *no*
    ``output_attentions=True`` needed.

    The patched forward calls the *original* forward with
    ``output_attentions=True`` internally, stores the attention tensor,
    then returns the output *without* the attention payload so the rest
    of the model behaves identically to the unpatched version.

    Usage::

        extractor = StealthExtractor(model)
        with torch.no_grad():
            outputs = model(pixel_values=pv)   # no output_attentions!
        stage3_attns = extractor.get_stage_attentions(3)
        extractor.restore()
    """

    def __init__(
        self,
        model: SegformerForSemanticSegmentation,
        capture_keys: Optional[Set[AttnKey]] = None,
    ) -> None:
        self.model = model
        self.capture_keys = capture_keys
        self.attention_store: AttnStore = OrderedDict()
        self._originals: Dict[AttnKey, object] = {}
        self._patch()

    def _patch(self) -> None:
        encoder = self.model.segformer.encoder
        for s in range(len(encoder.block)):
            for l in range(len(encoder.block[s])):
                module = encoder.block[s][l].attention.self
                key: AttnKey = (s, l)
                if self.capture_keys is not None and key not in self.capture_keys:
                    continue

                # save the *bound* original
                self._originals[key] = module.forward

                # build patched forward as a closure
                store_ref = self.attention_store
                orig_fwd = self._originals[key]

                def _make_patched(_orig_fwd, _key):
                    """Factory to capture closure variables correctly."""

                    def _patched_forward(
                        hidden_states,
                        height,
                        width,
                        output_attentions: bool = False,
                    ):
                        # Always request attention internally
                        out = _orig_fwd(
                            hidden_states, height, width, output_attentions=True
                        )
                        context, attn_probs = out
                        store_ref[_key] = attn_probs.detach().cpu()

                        if output_attentions:
                            return context, attn_probs
                        return (context,)

                    return _patched_forward

                module.forward = _make_patched(orig_fwd, key)

        logger.info(
            "Stealth-patched %d attention modules.", len(self._originals)
        )

    def restore(self) -> None:
        """Undo all monkey-patches, restoring original forward methods."""
        encoder = self.model.segformer.encoder
        for (s, l), orig in self._originals.items():
            encoder.block[s][l].attention.self.forward = orig
        self._originals.clear()
        logger.info("All attention patches restored.")

    # ── public API (mirrors HookExtractor) ────────────────────────────
    def clear(self) -> None:
        self.attention_store.clear()

    def get_stage_attentions(self, stage_idx: int) -> List[torch.Tensor]:
        return [
            attn
            for (s, _l), attn in sorted(self.attention_store.items())
            if s == stage_idx
        ]

    def get_all_attentions_by_stage(self) -> Dict[int, List[torch.Tensor]]:
        out: Dict[int, List[torch.Tensor]] = {}
        for (s, _l), attn in sorted(self.attention_store.items()):
            out.setdefault(s, []).append(attn)
        return out

    @property
    def num_captured(self) -> int:
        return len(self.attention_store)

    def __del__(self) -> None:
        # best-effort cleanup
        if self._originals:
            try:
                self.restore()
            except Exception:
                pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Convenience factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_extractor(
    model: SegformerForSemanticSegmentation,
    strategy: str = "hook",
    capture_keys: Optional[Set[AttnKey]] = None,
) -> HookExtractor | StealthExtractor:
    """Factory: create an extractor by name.

    Parameters
    ----------
    strategy : ``"hook"`` | ``"stealth"``
    """
    if strategy == "hook":
        return HookExtractor(model, capture_keys=capture_keys)
    if strategy == "stealth":
        return StealthExtractor(model, capture_keys=capture_keys)
    raise ValueError(f"Unknown strategy {strategy!r}. Use 'hook' or 'stealth'.")
