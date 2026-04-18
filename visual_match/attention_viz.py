"""
Attention visualization for VLA policies (pi05, Groot).

Provides forward-hook based attention capture plus two visualizations:

  1. Per-layer / per-head raw attention maps
  2. Attention rollout (Abnar & Zuidema 2020)

Both are aggregated into heatmaps over the *input image patches* so you can
answer: "what part of the scene is the policy looking at when it produces
this action?"

High-level API
--------------

    from visual_match.attention_viz import PI05AttentionCapture, render_overlays

    capture = PI05AttentionCapture(policy)
    with capture:
        action = policy.select_action(batch)

    overlays = render_overlays(
        images_by_cam={"cam_high": img_high_uint8, "cam_wrist": img_wrist_uint8},
        capture=capture,
        mode="rollout",          # "rollout" | "last_layer" | "layer=<i>"
        query="action_mean",     # which action-token query to use
    )
    # overlays["cam_high"] is an RGB uint8 image with heatmap burned on top.

The `GrootAttentionCapture` class works analogously for Groot's DiT
cross-attention (action-query -> vision-language tokens).

Implementation notes
--------------------

* **pi05**  — during inference, action-token queries live in
  `gemma_expert.model.layers[i].self_attn`. We hook each layer to grab the
  attention weights tensor [B, H, Q=chunk, K=prefix+suffix].
  The prefix layout is captured by temporarily wrapping
  `PI05Pytorch.embed_prefix` so we know which indices correspond to each
  camera's patch grid vs language tokens.

* **Groot**  — action queries attend to VL tokens inside `DiT.transformer_blocks[i].attn1`
  (cross-attention when the block is configured that way). Diffusers'
  `Attention` uses SDPA and does not return weights. We install a custom
  `AttnProcessor` that computes attention manually so we can capture weights.
  The VL token layout for Groot is delegated to the user: we just expose the
  raw [B, H, Q, K] matrix plus a helper for a square-grid reshape.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ============================================================================
# Dataclasses describing prefix layout
# ============================================================================


@dataclass
class ImageTokenSpan:
    """Indices into the key sequence that correspond to one camera's patches."""

    cam_key: str
    start: int
    end: int  # exclusive
    grid_h: int
    grid_w: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    def __post_init__(self):
        if self.grid_h * self.grid_w != self.num_tokens:
            raise ValueError(
                f"grid_h*grid_w ({self.grid_h}x{self.grid_w}="
                f"{self.grid_h*self.grid_w}) must equal num_tokens "
                f"({self.num_tokens}) for cam {self.cam_key}"
            )


@dataclass
class PrefixLayout:
    """How the key sequence is laid out during pi05 / groot inference."""

    image_spans: list[ImageTokenSpan] = field(default_factory=list)
    lang_start: int | None = None
    lang_end: int | None = None
    prefix_len: int | None = None
    suffix_len: int | None = None

    def image_span(self, cam_key: str) -> ImageTokenSpan | None:
        for s in self.image_spans:
            if s.cam_key == cam_key:
                return s
        return None


# ============================================================================
# PI05 capture
# ============================================================================


class PI05AttentionCapture:
    """Capture per-layer attention weights during pi05 inference.

    Usage:
        capture = PI05AttentionCapture(policy)
        with capture:
            action = policy.select_action(batch)
        # capture.attentions is a list[Tensor], one per Gemma expert layer,
        # each of shape [denoise_steps, H, Q=chunk, K=prefix+suffix].
        # capture.layout describes the prefix layout.
    """

    def __init__(self, policy, keep_denoise: str = "stack", verbose: bool = True):
        """
        Args:
            policy: a PI05Policy instance.
            keep_denoise: how to aggregate across diffusion denoise steps.
                * "stack"  -> keep per-step attention stacked on dim 0.
                * "last"   -> keep only the final denoise step.
                * "mean"   -> mean across denoise steps.
            verbose: print diagnostic info on first install / uninstall.
        """
        self.policy = policy
        self.keep_denoise = keep_denoise
        self.verbose = verbose

        self._expert_layers = (
            policy.model.paligemma_with_expert.gemma_expert.model.layers
        )
        self._num_layers = len(self._expert_layers)
        self._per_layer_steps: list[list[torch.Tensor]] = [
            [] for _ in range(self._num_layers)
        ]

        self.layout: PrefixLayout = PrefixLayout()
        self._hooks: list[Any] = []
        self._orig_embed_prefix = None
        self._hook_call_counts: list[int] = [0] * self._num_layers
        self._announced_install = False
        self._announced_first_capture = False
        self._warned_never_called = False
        self._last_total_calls = 0

        self._image_feature_keys: list[str] = list(
            policy.config.image_features.keys()
        )
        self._patch_size = 14  # SigLIP patch size used by PaliGemma
        self._image_hw = tuple(policy.config.image_resolution)
        self._chunk_size = int(policy.config.chunk_size)

    @property
    def had_calls(self) -> bool:
        return self._last_total_calls > 0

    # -- context manager --------------------------------------------------

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._uninstall()

    def reset(self):
        self._per_layer_steps = [[] for _ in range(self._num_layers)]
        self.layout = PrefixLayout()

    # -- public accessors -------------------------------------------------

    @property
    def attentions(self) -> list[torch.Tensor]:
        """Per-layer aggregated attention tensors.

        Returns a list of length num_layers. Each element has shape
        [H, Q, K] (if keep_denoise in {"last","mean"}) or
        [T, H, Q, K] (if keep_denoise == "stack"), where T is the number of
        denoise steps that actually fired during the captured inference.
        """
        out = []
        for steps in self._per_layer_steps:
            if len(steps) == 0:
                out.append(None)
                continue
            stack = torch.stack(steps, dim=0)  # [T, H, Q, K]
            if self.keep_denoise == "last":
                out.append(stack[-1])
            elif self.keep_denoise == "mean":
                out.append(stack.mean(dim=0))
            else:  # stack
                out.append(stack)
        return out

    # -- internals --------------------------------------------------------

    def _install(self):
        # 1) Monkey-patch embed_prefix to record layout
        inner = self.policy.model

        orig_embed_prefix = inner.embed_prefix
        self._orig_embed_prefix = orig_embed_prefix
        capture = self

        def wrapped_embed_prefix(images, img_masks, tokens, masks):
            embs, pad_masks, att_masks = orig_embed_prefix(
                images, img_masks, tokens, masks
            )
            capture._record_layout_from_embed(
                embs, pad_masks, img_masks, tokens, masks
            )
            return embs, pad_masks, att_masks

        inner.embed_prefix = wrapped_embed_prefix  # type: ignore[assignment]

        # Reset per-window hook call counts
        self._hook_call_counts = [0] * self._num_layers

        # 2) Install a forward hook on every expert layer's self-attn to grab
        #    attn weights. In pi05 inference, the gemma expert runs in eager
        #    mode, so GemmaAttention returns (attn_output, attn_weights).
        for layer_idx, layer in enumerate(self._expert_layers):
            hook = layer.self_attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._hooks.append(hook)
        if self.verbose and not self._announced_install:
            cfg = getattr(self.policy.model.paligemma_with_expert.gemma_expert, "config", None)
            attn_impl = getattr(cfg, "_attn_implementation", "<?>") if cfg is not None else "<?>"
            print(
                f"[PI05AttnCapture] installed hooks on {self._num_layers} "
                f"expert layers (attn_impl={attn_impl!r})"
            )
            if attn_impl not in (None, "eager", "<?>"):
                print(
                    f"  WARNING: expected eager attention to extract weights. "
                    f"Got {attn_impl!r}. Attention weights may be None."
                )
            self._announced_install = True

    def _uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        total = sum(self._hook_call_counts)
        non_empty = sum(1 for s in self._per_layer_steps if s)
        self._last_total_calls = total
        if self.verbose:
            if total == 0 and not self._warned_never_called:
                print(
                    f"[PI05AttnCapture] NOTE: no hooks fired during this "
                    f"capture. gemma expert forward did not run (e.g. "
                    f"select_action is serving from the action queue). "
                    f"This message is shown once."
                )
                self._warned_never_called = True
            elif total > 0 and non_empty == 0:
                print(
                    f"[PI05AttnCapture] WARN: hooks fired ({total} times) but "
                    f"no attention weights were returned. Ensure the gemma "
                    f"expert is using eager attention (attn_implementation='eager')."
                )
            elif not self._announced_first_capture and non_empty > 0:
                print(
                    f"[PI05AttnCapture] first successful capture: "
                    f"hook calls={total}, non-empty layers="
                    f"{non_empty}/{self._num_layers}, steps="
                    f"{len(self._per_layer_steps[0])}"
                )
                self._announced_first_capture = True
        if self._orig_embed_prefix is not None:
            self.policy.model.embed_prefix = self._orig_embed_prefix  # type: ignore[assignment]
            self._orig_embed_prefix = None

    def _make_attn_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            self._hook_call_counts[layer_idx] += 1
            # output = (attn_output, attn_weights) for HF eager attention
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn_weights = output[1]
            if attn_weights is None:
                return
            if attn_weights.dim() != 4:
                return
            aw = attn_weights[0].detach().to(torch.float32).cpu()
            self._per_layer_steps[layer_idx].append(aw)

        return _hook

    def _record_layout_from_embed(self, embs, pad_masks, img_masks, tokens, masks):
        """Rebuild image/language token spans from what embed_prefix produced."""
        # embs: [B, N, D]
        # We reconstruct the sequence: for each present image, then language.
        layout = PrefixLayout()
        cursor = 0
        ih, iw = self._image_hw
        patch = self._patch_size
        grid_h, grid_w = ih // patch, iw // patch
        num_patches = grid_h * grid_w

        # Iterate in the same order as policy._preprocess_images:
        present_img_keys = [
            k for k in self.policy.config.image_features if k in self._image_feature_keys
        ]
        # pi05 also pads missing cameras with zero image (mask=0), but we still
        # want to record a span for each *positionally provided* image (those
        # that produced tokens, whether real or padded).
        num_images = len(img_masks)
        for cam_idx in range(num_images):
            cam_key = present_img_keys[cam_idx] if cam_idx < len(present_img_keys) else f"image_{cam_idx}"
            span = ImageTokenSpan(
                cam_key=cam_key,
                start=cursor,
                end=cursor + num_patches,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            layout.image_spans.append(span)
            cursor += num_patches

        # Language tokens span
        n_lang = int(tokens.shape[1])
        layout.lang_start = cursor
        layout.lang_end = cursor + n_lang
        cursor += n_lang

        layout.prefix_len = cursor
        layout.suffix_len = self._chunk_size
        self.layout = layout


# ============================================================================
# Groot capture (cross-attention in DiT)
# ============================================================================


class _GrootCaptureAttnProcessor:
    """Custom processor for diffusers.models.attention.Attention that records
    attention weights in addition to computing the attention output.

    We implement scaled-dot-product attention manually (float32 softmax) so we
    can return attn weights to the capture object.
    """

    def __init__(self, capture: "GrootAttentionCapture", layer_idx: int):
        self.capture = capture
        self.layer_idx = layer_idx
        self.call_count = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        self.call_count += 1
        if self.capture.verbose and self.call_count == 1:
            print(
                f"[GrootAttnCapture] layer {self.layer_idx} first call: "
                f"Q={tuple(hidden_states.shape)}, "
                f"K={tuple(hidden_states.shape) if encoder_hidden_states is None else tuple(encoder_hidden_states.shape)}"
            )
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def _split_heads(x):
            b, n, _ = x.shape
            return x.view(b, n, attn.heads, head_dim).transpose(1, 2)

        q = _split_heads(query)
        k = _split_heads(key)
        v = _split_heads(value)

        scaling = head_dim ** -0.5
        # Compute in float32 to match HF eager attention for stable capture
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        attn_logits = torch.matmul(q32, k32.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Record attention
        self.capture._record(self.layer_idx, attn_weights.detach().cpu())

        out = torch.matmul(attn_weights.to(q.dtype), v)
        out = out.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


class GrootAttentionCapture:
    """Capture cross-attention weights inside Groot's DiT action head.

    For each `DiT.transformer_blocks[i].attn1` we install a custom processor
    that computes attention manually and records [B, H, Q, K] weights.

    Because we don't know the exact VL token layout (Eagle does dynamic
    tiling), you'll likely want to slice `capture.attentions[l][..., vl_start:vl_end]`
    using the encoder_hidden_states length you observe at runtime. A helper
    `infer_square_grid(n)` is provided for the common single-crop case.
    """

    def __init__(self, policy, keep_denoise: str = "stack", verbose: bool = True):
        self.policy = policy
        self.keep_denoise = keep_denoise
        self.verbose = verbose

        # Resolve DiT blocks
        try:
            dit = policy._groot_model.action_head.model  # type: ignore[attr-defined]
        except AttributeError as e:
            raise RuntimeError(
                "Could not find policy._groot_model.action_head.model "
                "(expected DiT)."
            ) from e
        self._dit = dit
        self._blocks = dit.transformer_blocks
        self._num_layers = len(self._blocks)

        # Try to resolve the Eagle backbone + its image_token_index so we can
        # auto-detect per-image token spans from input_ids at runtime.
        self._eagle_model = None
        self._image_token_index: int | None = None
        try:
            eagle = policy._groot_model.backbone.eagle_model  # type: ignore[attr-defined]
            self._eagle_model = eagle
            self._image_token_index = int(getattr(eagle, "image_token_index", None))
        except Exception:
            pass

        self._per_layer_steps: list[list[torch.Tensor]] = [
            [] for _ in range(self._num_layers)
        ]
        self._saved_processors: list[Any] = []
        self._installed_procs: list[Any] = []
        self._eagle_hook = None
        # Per-image (start, end) spans in vl_embs key dimension, auto-detected
        # from input_ids. Populated on first capture window that runs the backbone.
        self.vision_token_spans: list[tuple[int, int]] = []
        self._announced_install = False
        self._announced_first_capture = False
        self._warned_never_called = False
        self._warned_spans_missing = False
        self._last_total_calls = 0

    @property
    def had_calls(self) -> bool:
        """True if the most recent capture window actually recorded attention."""
        return self._last_total_calls > 0

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._uninstall()

    def reset(self):
        self._per_layer_steps = [[] for _ in range(self._num_layers)]

    def _install(self):
        self._saved_processors = []
        self._installed_procs = []
        for i, block in enumerate(self._blocks):
            attn = block.attn1
            self._saved_processors.append(attn.get_processor())
            proc = _GrootCaptureAttnProcessor(self, i)
            attn.set_processor(proc)
            self._installed_procs.append(proc)

        # Forward-pre-hook on Eagle backbone so we can inspect input_ids and
        # auto-detect contiguous image-token spans in vl_embs.
        if self._eagle_model is not None and self._image_token_index is not None:
            self._eagle_hook = self._eagle_model.register_forward_pre_hook(
                self._eagle_pre_hook, with_kwargs=True
            )

        if self.verbose and not self._announced_install:
            now = type(self._blocks[0].attn1.get_processor()).__name__
            print(
                f"[GrootAttnCapture] installed custom processor on "
                f"{len(self._blocks)} DiT blocks (active processor type={now})"
                + (", eagle input_ids hook attached"
                   if self._eagle_hook is not None else
                   ", eagle hook NOT attached (manual vl_start required)")
            )
            self._announced_install = True

    def _eagle_pre_hook(self, module, args, kwargs):
        """Capture input_ids from the Eagle backbone call to infer vision spans."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) >= 2:
            input_ids = args[1]  # (pixel_values, input_ids, ...)
        if input_ids is None:
            return None
        try:
            ids = input_ids[0] if input_ids.dim() == 2 else input_ids
            mask = (ids == self._image_token_index)
            if not mask.any():
                return None
            positions = mask.nonzero(as_tuple=False).flatten().tolist()
            # Group contiguous runs into per-image spans
            spans: list[tuple[int, int]] = []
            run_start = positions[0]
            prev = positions[0]
            for p in positions[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    spans.append((run_start, prev + 1))
                    run_start = p
                    prev = p
            spans.append((run_start, prev + 1))
            self.vision_token_spans = spans
            if self.verbose and not getattr(self, "_announced_spans", False):
                print(
                    f"[GrootAttnCapture] detected vision token spans (from "
                    f"input_ids, image_token_index={self._image_token_index}): "
                    f"{spans}"
                )
                self._announced_spans = True
        except Exception as e:
            if self.verbose and not self._warned_spans_missing:
                print(f"[GrootAttnCapture] WARN: failed to parse input_ids for "
                      f"vision spans: {e}")
                self._warned_spans_missing = True
        return None

    def _uninstall(self):
        total_calls = sum(p.call_count for p in self._installed_procs)
        non_empty = sum(1 for s in self._per_layer_steps if s)
        self._last_total_calls = total_calls
        if self.verbose:
            if total_calls == 0 and not self._warned_never_called:
                print(
                    f"[GrootAttnCapture] NOTE: processor was not called during "
                    f"this capture. DiT forward likely didn't run (e.g. "
                    f"select_action is serving from an action queue between "
                    f"chunks). This message is shown once; subsequent empty "
                    f"captures will be silent."
                )
                self._warned_never_called = True
            elif not self._announced_first_capture and non_empty > 0:
                print(
                    f"[GrootAttnCapture] first successful capture: "
                    f"processor calls={total_calls}, "
                    f"non-empty layers={non_empty}/{self._num_layers}, "
                    f"steps={len(self._per_layer_steps[0])}"
                )
                self._announced_first_capture = True
        for block, proc in zip(self._blocks, self._saved_processors, strict=False):
            block.attn1.set_processor(proc)
        self._saved_processors.clear()
        self._installed_procs.clear()
        if self._eagle_hook is not None:
            self._eagle_hook.remove()
            self._eagle_hook = None

    def _record(self, layer_idx: int, attn: torch.Tensor):
        # Store [H, Q, K] (drop batch dim for B=1 inference)
        if attn.dim() == 4:
            attn = attn[0]
        self._per_layer_steps[layer_idx].append(attn.to(torch.float32))

    @property
    def attentions(self) -> list[torch.Tensor | None]:
        out = []
        for steps in self._per_layer_steps:
            if len(steps) == 0:
                out.append(None)
                continue
            stack = torch.stack(steps, dim=0)  # [T, H, Q, K]
            if self.keep_denoise == "last":
                out.append(stack[-1])
            elif self.keep_denoise == "mean":
                out.append(stack.mean(dim=0))
            else:
                out.append(stack)
        return out


# ============================================================================
# Rollout math
# ============================================================================


def attention_rollout(
    attentions: list[torch.Tensor],
    head_fuse: str = "mean",
    add_residual: bool = True,
) -> torch.Tensor | None:
    """Compute attention rollout over a stack of self-attention matrices.

    Follows Abnar & Zuidema (2020). For self-attention layers where Q and K
    are the same sequence, we can compose layers as:

        A_l = 0.5 * (A_l + I)      (account for residual)
        A_l = A_l / sum(A_l, -1)   (re-normalize rows)
        R = A_L @ A_{L-1} @ ... @ A_1

    Args:
        attentions: list of per-layer attention tensors [H, Q, K] (Q==K).
            Tensors with leading dims are flattened to [H, Q, K] (first batch
            taken).
        head_fuse: "mean" | "max" | "min" fusion across heads.
        add_residual: if True, add identity to each matrix before normalizing.

    Returns:
        Rollout matrix [Q, K] where R[i, j] is the rollout weight from token i
        to token j. Returns None if attentions is empty / all None.
    """
    mats = []
    for A in attentions:
        if A is None:
            continue
        # Take first batch if present
        while A.dim() > 3:
            A = A[0]
        if A.dim() != 3:
            raise ValueError(f"attention tensors must be [H, Q, K], got {tuple(A.shape)}")
        if head_fuse == "mean":
            A = A.mean(dim=0)
        elif head_fuse == "max":
            A = A.max(dim=0).values
        elif head_fuse == "min":
            A = A.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fuse: {head_fuse}")
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"attention_rollout expects square per-layer matrices "
                f"(self-attention). Got {A.shape}. For cross-attention, "
                f"use cross_attention_rollout() instead."
            )
        if add_residual:
            I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            A = 0.5 * (A + I)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)
        mats.append(A)
    if not mats:
        return None
    R = mats[0]
    for A in mats[1:]:
        R = A @ R
    return R


def cross_attention_rollout(
    cross_attentions: list[torch.Tensor],
    query_agg: str = "mean",
    head_fuse: str = "mean",
) -> torch.Tensor | None:
    """Simple rollout-like aggregation for cross-attention layers (Q != K).

    We can't compose matrices (shapes mismatch), so we just mean-pool across
    layers / heads / queries. Returns a 1D tensor of length K representing how
    much the chosen query set attends to each key token, aggregated across
    the stack.

    Args:
        cross_attentions: list of [H, Q, K] tensors (batch dim already removed).
        query_agg: how to aggregate over queries ("mean" | "max" | "last").
        head_fuse: same as attention_rollout.
    """
    acc = None
    count = 0
    for A in cross_attentions:
        if A is None:
            continue
        while A.dim() > 3:
            A = A[0]
        if head_fuse == "mean":
            A = A.mean(dim=0)
        elif head_fuse == "max":
            A = A.max(dim=0).values
        else:
            raise ValueError(f"Unknown head_fuse: {head_fuse}")
        if query_agg == "mean":
            v = A.mean(dim=0)
        elif query_agg == "max":
            v = A.max(dim=0).values
        elif query_agg == "last":
            v = A[-1]
        else:
            raise ValueError(f"Unknown query_agg: {query_agg}")
        acc = v if acc is None else acc + v
        count += 1
    if acc is None:
        return None
    return acc / count


# ============================================================================
# Heatmap & overlay
# ============================================================================


def attention_to_heatmap(
    attn_1d: torch.Tensor | np.ndarray,
    grid_h: int,
    grid_w: int,
    target_hw: tuple[int, int],
    normalize: str = "minmax",
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Turn a 1D attention vector over image patches into a 2D heatmap.

    Args:
        attn_1d: length = grid_h * grid_w, the attention weight per patch.
        grid_h, grid_w: patch grid shape.
        target_hw: output (H, W).
        normalize: "minmax" | "sum" | "none".
        blur_sigma: optional gaussian blur in pixels (set 0 to skip).

    Returns:
        float32 ndarray of shape target_hw, values in [0, 1] when normalize="minmax".
    """
    if isinstance(attn_1d, torch.Tensor):
        attn_1d = attn_1d.detach().to(torch.float32).cpu().numpy()
    attn_1d = np.asarray(attn_1d, dtype=np.float32)
    if attn_1d.size != grid_h * grid_w:
        raise ValueError(
            f"attn_1d has {attn_1d.size} entries but grid {grid_h}x{grid_w} "
            f"= {grid_h * grid_w}"
        )
    grid = attn_1d.reshape(grid_h, grid_w)

    if normalize == "sum":
        total = grid.sum()
        if total > 0:
            grid = grid / total
    elif normalize == "minmax":
        lo, hi = float(grid.min()), float(grid.max())
        grid = (grid - lo) / (hi - lo + 1e-9)
    # Upsample
    H, W = target_hw
    if _HAS_CV2:
        heat = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        t = torch.from_numpy(grid)[None, None]
        heat = F.interpolate(t, size=(H, W), mode="bicubic", align_corners=False)
        heat = heat[0, 0].numpy()
    if blur_sigma > 0 and _HAS_CV2:
        k = int(max(3, 2 * round(3 * blur_sigma) + 1))
        heat = cv2.GaussianBlur(heat, (k, k), blur_sigma)
    # Always renormalize to [0, 1] after upsample for consistent display
    lo, hi = float(heat.min()), float(heat.max())
    heat = (heat - lo) / (hi - lo + 1e-9)
    return heat.astype(np.float32)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Burn a heatmap on top of an RGB image. Returns RGB uint8."""
    if not _HAS_CV2:
        raise RuntimeError("overlay_heatmap requires OpenCV (cv2).")
    img = image_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    cmap_name = colormap.lower()
    cmap = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
    }.get(cmap_name, cv2.COLORMAP_JET)
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cmap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(heat_rgb, alpha, img, 1.0 - alpha, 0)
    return out


# ============================================================================
# High-level convenience for pi05
# ============================================================================


def pi05_attention_for_camera(
    capture: PI05AttentionCapture,
    cam_key: str,
    mode: str = "rollout",
    layer_index: int | None = None,
    query_agg: str = "mean",
    head_fuse: str = "mean",
) -> np.ndarray:
    """Extract a 1D attention vector over one camera's patches.

    Args:
        capture: a PI05AttentionCapture that has already captured at least
            one inference.
        cam_key: e.g. "observation.images.cam_high".
        mode: one of:
            * "rollout"        — cross-attention rollout over all layers.
            * "last_layer"     — only the final expert layer.
            * "layer=<i>"      — only layer i (0-indexed).
            * "mean_layers"    — mean over all layers.
        layer_index: overrides parsing of "layer=<i>".
        query_agg: "mean" | "max" | "last"  — how to aggregate over the
            chunk_size action queries.
        head_fuse: "mean" | "max" | "min".

    Returns:
        ndarray of shape (grid_h * grid_w,), float32. Note this is NOT
        normalized to [0,1] here — pass through attention_to_heatmap.
    """
    if not capture.layout.image_spans:
        raise RuntimeError(
            "PrefixLayout is empty: did inference actually run inside the "
            "capture context?"
        )
    span = capture.layout.image_span(cam_key)
    if span is None:
        raise KeyError(
            f"Camera {cam_key!r} not in capture.layout.image_spans "
            f"(available: {[s.cam_key for s in capture.layout.image_spans]})."
        )

    per_layer = capture.attentions  # list of [H, Q, K] or [T, H, Q, K] or None

    # Reduce denoise-step dim if present
    def _reduce_time(A):
        if A is None:
            return None
        if A.dim() == 4:  # [T, H, Q, K]
            A = A.mean(dim=0)
        return A

    per_layer = [_reduce_time(A) for A in per_layer]

    def _fuse_heads(A):
        if head_fuse == "mean":
            return A.mean(dim=0)
        if head_fuse == "max":
            return A.max(dim=0).values
        if head_fuse == "min":
            return A.min(dim=0).values
        raise ValueError(head_fuse)

    def _agg_queries(A2d):
        # A2d: [Q, K]
        if query_agg == "mean":
            return A2d.mean(dim=0)
        if query_agg == "max":
            return A2d.max(dim=0).values
        if query_agg == "last":
            return A2d[-1]
        raise ValueError(query_agg)

    # Helper: get a [Q, K] matrix for a given layer
    def _layer_qk(i: int) -> torch.Tensor | None:
        A = per_layer[i]
        if A is None:
            return None
        return _fuse_heads(A)

    if mode == "rollout":
        # True rollout requires self-attention. In pi05 inference the expert
        # sees query=suffix, key=prefix+suffix which is NOT square, so we fall
        # back to a layer-wise mean over the action->image slice. This is
        # still informative and commonly called "rollout" in VLA papers.
        slices = []
        for i in range(len(per_layer)):
            A = _layer_qk(i)
            if A is None:
                continue
            # Slice to action queries (first chunk_size rows) x image keys
            q = A[: capture._chunk_size, span.start : span.end]  # [Q, Npatch]
            slices.append(_agg_queries(q))  # [Npatch]
        if not slices:
            raise RuntimeError("No captured attention layers found for pi05.")
        v = torch.stack(slices, dim=0).mean(dim=0)
    elif mode == "mean_layers":
        slices = []
        for i in range(len(per_layer)):
            A = _layer_qk(i)
            if A is None:
                continue
            q = A[: capture._chunk_size, span.start : span.end]
            slices.append(_agg_queries(q))
        v = torch.stack(slices, dim=0).mean(dim=0)
    elif mode == "last_layer":
        # Use last non-None layer
        for i in range(len(per_layer) - 1, -1, -1):
            A = _layer_qk(i)
            if A is not None:
                v = _agg_queries(A[: capture._chunk_size, span.start : span.end])
                break
        else:
            raise RuntimeError("No attention captured.")
    elif mode.startswith("layer="):
        idx = int(mode.split("=", 1)[1])
        A = _layer_qk(idx)
        if A is None:
            raise RuntimeError(f"Layer {idx} has no captured attention.")
        v = _agg_queries(A[: capture._chunk_size, span.start : span.end])
    else:
        if layer_index is None:
            raise ValueError(f"Unknown mode: {mode!r}")
        A = _layer_qk(layer_index)
        if A is None:
            raise RuntimeError(f"Layer {layer_index} has no captured attention.")
        v = _agg_queries(A[: capture._chunk_size, span.start : span.end])

    return v.detach().to(torch.float32).cpu().numpy()


def render_pi05_overlays(
    capture: PI05AttentionCapture,
    images_by_cam: dict[str, np.ndarray],
    mode: str = "rollout",
    alpha: float = 0.5,
    colormap: str = "jet",
    blur_sigma: float = 2.0,
    normalize: str = "minmax",
    query_agg: str = "mean",
    head_fuse: str = "mean",
) -> dict[str, np.ndarray]:
    """Build a dict of {cam_key -> overlay image} for pi05."""
    out: dict[str, np.ndarray] = {}
    for cam_key, img in images_by_cam.items():
        if img is None:
            continue
        try:
            v = pi05_attention_for_camera(
                capture,
                cam_key=cam_key,
                mode=mode,
                query_agg=query_agg,
                head_fuse=head_fuse,
            )
        except KeyError:
            continue
        span = capture.layout.image_span(cam_key)
        heat = attention_to_heatmap(
            v,
            grid_h=span.grid_h,
            grid_w=span.grid_w,
            target_hw=img.shape[:2],
            normalize=normalize,
            blur_sigma=blur_sigma,
        )
        out[cam_key] = overlay_heatmap(img, heat, alpha=alpha, colormap=colormap)
    return out


# ============================================================================
# Convenience for Groot
# ============================================================================


def infer_square_grid(n: int) -> tuple[int, int]:
    """If n is a perfect square return (s, s). Otherwise raise ValueError."""
    s = int(round(math.sqrt(n)))
    if s * s != n:
        raise ValueError(
            f"Cannot infer a square grid for {n} tokens "
            f"(sqrt(n)={math.sqrt(n):.3f})."
        )
    return s, s


def groot_attention_for_vision(
    capture: GrootAttentionCapture,
    vl_start: int,
    vl_end: int,
    grid_h: int,
    grid_w: int,
    mode: str = "mean_layers",
    layer_index: int | None = None,
    query_agg: str = "mean",
    head_fuse: str = "mean",
    action_query_start: int = 0,
    action_query_end: int | None = None,
) -> np.ndarray:
    """Extract a 1D attention vector over VL tokens for a Groot policy.

    Args:
        capture: a GrootAttentionCapture that has already captured inference.
        vl_start, vl_end: key indices of the visual portion of vl_embs.
        grid_h, grid_w: how to reshape those VL tokens back to a 2D grid
            (e.g. (32, 32) for a 448x448 SigLIP-style image).
        mode: "rollout" | "mean_layers" | "last_layer" | "layer=<i>".
        layer_index: overrides "layer=<i>" parsing.
        query_agg / head_fuse: same semantics as pi05_attention_for_camera.
        action_query_start / action_query_end: restrict to a subset of action
            queries (DiT's query sequence is [state, future_tokens..., action...]).
    """
    per_layer = capture.attentions

    def _reduce_time(A):
        if A is None:
            return None
        if A.dim() == 4:
            A = A.mean(dim=0)
        return A

    per_layer = [_reduce_time(A) for A in per_layer]

    def _fuse_heads(A):
        if head_fuse == "mean":
            return A.mean(dim=0)
        if head_fuse == "max":
            return A.max(dim=0).values
        if head_fuse == "min":
            return A.min(dim=0).values
        raise ValueError(head_fuse)

    def _agg_queries(A2d):
        aq = A2d[action_query_start:action_query_end]
        if query_agg == "mean":
            return aq.mean(dim=0)
        if query_agg == "max":
            return aq.max(dim=0).values
        if query_agg == "last":
            return aq[-1]
        raise ValueError(query_agg)

    def _layer_vec(i):
        A = per_layer[i]
        if A is None:
            return None
        if A.shape[-1] < vl_end:
            return None  # VL tokens are not present in this layer
        A = _fuse_heads(A)
        v = _agg_queries(A[:, vl_start:vl_end])
        return v

    if mode in ("rollout", "mean_layers"):
        vecs = [v for v in (_layer_vec(i) for i in range(len(per_layer))) if v is not None]
        if not vecs:
            raise RuntimeError("No captured attention layers for Groot.")
        v = torch.stack(vecs, dim=0).mean(dim=0)
    elif mode == "last_layer":
        for i in range(len(per_layer) - 1, -1, -1):
            v = _layer_vec(i)
            if v is not None:
                break
        else:
            raise RuntimeError("No captured attention.")
    elif mode.startswith("layer="):
        idx = int(mode.split("=", 1)[1])
        v = _layer_vec(idx)
        if v is None:
            raise RuntimeError(f"Layer {idx} empty.")
    else:
        if layer_index is None:
            raise ValueError(f"Unknown mode: {mode!r}")
        v = _layer_vec(layer_index)
        if v is None:
            raise RuntimeError(f"Layer {layer_index} empty.")
    return v.detach().to(torch.float32).cpu().numpy()


# ============================================================================
# Side-by-side grid helper (for live display)
# ============================================================================


def make_attention_dashboard(
    raw_images: dict[str, np.ndarray],
    overlays: dict[str, np.ndarray],
    row_height: int = 256,
) -> np.ndarray:
    """Stack raw images (top row) and overlays (bottom row) into one RGB image.

    Cameras appear left-to-right in the order of raw_images.
    """
    if not _HAS_CV2:
        raise RuntimeError("make_attention_dashboard requires OpenCV.")
    cams = list(raw_images.keys())
    top = []
    bot = []
    for cam in cams:
        raw = raw_images[cam]
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        h, w = raw.shape[:2]
        scale = row_height / float(h)
        w_s = int(round(w * scale))
        raw_s = cv2.resize(raw, (w_s, row_height), interpolation=cv2.INTER_AREA)

        ov = overlays.get(cam)
        if ov is None:
            ov_s = np.zeros_like(raw_s)
        else:
            ov_s = cv2.resize(ov, (w_s, row_height), interpolation=cv2.INTER_AREA)

        # Label
        label = cam.split(".")[-1]
        cv2.putText(raw_s, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(raw_s, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(raw_s, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        top.append(raw_s)
        bot.append(ov_s)
    top_row = np.concatenate(top, axis=1) if top else np.zeros((row_height, 0, 3), np.uint8)
    bot_row = np.concatenate(bot, axis=1) if bot else np.zeros((row_height, 0, 3), np.uint8)
    return np.concatenate([top_row, bot_row], axis=0)
