"""Backbone feature extraction. First model: V-JEPA-2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_rgb_uint8(pixels: np.ndarray) -> np.ndarray:
    """(T, H, W) grayscale uint8 -> (T, H, W, 3) uint8. (T, H, W, 3) passes
    through. We expand grayscale once on the full movie (cheap: ~1 GB for a
    3600-frame, 304-px movie) instead of materializing every overlapping
    clip up front (was ~60 GB for natural_movie_three at stride 1)."""
    if pixels.ndim == 3:
        return np.repeat(pixels[..., None], 3, axis=-1).astype(np.uint8, copy=False)
    if pixels.ndim == 4 and pixels.shape[-1] == 3:
        return pixels.astype(np.uint8, copy=False)
    raise ValueError(f"unexpected pixel shape {pixels.shape}")


def vjepa2_extract_features(
    model: Any,
    processor: Any,
    pixels: np.ndarray,
    clip_frames: int = 64,
    stride: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
    pool: str = "mean",
) -> np.ndarray:
    """Extract per-clip features from V-JEPA-2.

    pixels: (T, H, W) uint8 grayscale OR (T, H, W, 3) uint8 RGB.
    Returns: (n_clips, hidden_size) float32 array.

    Per-frame assignment downstream: clip k spans pixels[k : k+clip_frames];
    we assign its feature to frame `k + clip_frames - 1` (the last frame of
    the clip). The first `clip_frames - 1` frames have no feature and are
    skipped by the encoder.

    Clips are built lazily inside the batch loop — never materialized
    upfront — so memory stays bounded at one (T, H, W, 3) movie copy plus
    one batch of clips.
    """
    import torch

    T = pixels.shape[0]
    if T < clip_frames:
        raise ValueError(f"Movie too short ({T}) for clip size {clip_frames}.")
    n_clips = (T - clip_frames) // stride + 1

    pixels_rgb = _ensure_rgb_uint8(pixels)  # (T, H, W, 3) uint8 — single copy
    print(f"[features] {n_clips} clips of {clip_frames} frames "
          f"({T} movie frames, stride {stride}); "
          f"movie buffer {pixels_rgb.nbytes / 1e9:.2f} GB")

    model.eval().to(device)
    model_dtype = next(model.parameters()).dtype

    feats = []
    with torch.no_grad():
        for i in range(0, n_clips, batch_size):
            j = min(i + batch_size, n_clips)
            batch = [pixels_rgb[s * stride : s * stride + clip_frames]
                     for s in range(i, j)]
            inputs = processor(batch, return_tensors="pt")
            inputs = {
                k: (v.to(device, dtype=model_dtype)
                    if torch.is_floating_point(v) else v.to(device))
                for k, v in inputs.items()
            }
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, n_patches, D)
            if pool == "mean":
                pooled = hidden.mean(dim=1)
            else:
                raise ValueError(f"unknown pool {pool!r}")
            feats.append(pooled.float().cpu().numpy())
            del outputs, hidden, pooled, inputs
            if device == "cuda":
                torch.cuda.empty_cache()
            if (i // batch_size) % 50 == 0:
                print(f"[features]   clip {i}/{n_clips}")

    return np.concatenate(feats, axis=0)


def load_vjepa2(
    hf_id: str,
    init: str = "pretrained",
    seed: int = 0,
    dtype: str = "float16",
):
    """Load (model, processor). `init='random'` returns a freshly-initialized
    model with the same architecture but no pretrained weights, for the
    Brain-Score-style noise-floor control.

    Defaults to fp16 weights + SDPA attention to fit ViT-L 64-frame 256² on
    a 16 GB T4. Eager attention materializes the full 8192×8192 attention
    matrix per layer, which OOMs on T4.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoVideoProcessor

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    processor = AutoVideoProcessor.from_pretrained(hf_id)

    common_kwargs = {"attn_implementation": "sdpa"}

    if init == "random":
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(hf_id)
        try:
            model = AutoModel.from_config(config, **common_kwargs)
        except TypeError:
            model = AutoModel.from_config(config)
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
    elif init == "pretrained":
        try:
            model = AutoModel.from_pretrained(
                hf_id, dtype=torch_dtype, **common_kwargs,
            )
        except TypeError:
            # Older transformers: `dtype` was `torch_dtype`, no
            # `attn_implementation` kwarg on this model class.
            model = AutoModel.from_pretrained(hf_id, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"unknown init {init!r}")

    return model, processor
